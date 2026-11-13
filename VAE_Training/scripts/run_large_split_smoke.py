"""Two-update full-encoder smoke test with one backbone on each GPU."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

from omegaconf import OmegaConf
import psutil
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from safetensors.torch import save_file,safe_open

from vae.model import VAEConfig
from vae.reconstruction import reconstruct_rows
from vae.split_training import SplitReasoningVAE,StreamedCPUAdamW

ROOT=Path(__file__).resolve().parents[1]


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def frozen_hash(model):
    h=hashlib.sha256()
    for name,p in model.decoder_model.named_parameters():
        h.update(name.encode())
        for chunk in p.detach().view(torch.uint8).flatten().split(32*1024*1024):
            h.update(chunk.cpu().numpy().tobytes())
    return h.hexdigest()


def save_trainable(model,folder):
    folder.mkdir()
    shard,index,files={}, {}, []
    used=0
    def flush():
        nonlocal shard,used
        if not shard:return
        name=f'trainable-{len(files):03d}.safetensors'
        save_file(shard,folder/name)
        files.append(name)
        index.update({key:name for key in shard})
        shard={};used=0
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        size=p.numel()*p.element_size()
        if used+size>1024**3:flush()
        shard[name]=p.detach().cpu().contiguous();used+=size
    flush()
    write(folder/'index.json',dict(format='split_vae_trainable_v1',weight_map=index,
                                  frozen_decoder_included=False))
    # Verify every serialized trainable tensor against the current live weights.
    named=dict(model.named_parameters())
    verified=0
    for filename in files:
        with safe_open(folder/filename,framework='pt',device='cpu') as stream:
            for name in stream.keys():
                if not torch.equal(stream.get_tensor(name),named[name].detach().cpu()):
                    raise RuntimeError(f'Checkpoint tensor mismatch: {name}')
                verified+=1
    return dict(shards=len(files),tensors_verified=verified)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',type=Path,required=True)
    args=ap.parse_args();out=args.directory.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    def status(stage,**fields):
        data=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-started,**fields)
        write(out/'status.json',data);print(json.dumps(data),flush=True)
    cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'configs/vae_large_split_smoke.yaml'),resolve=True)
    write(out/'config.json',cfg)
    torch.manual_seed(42)
    vc=VAEConfig(**cfg['model'])
    status('loading_models',backend='explicit_model_parallel_streamed_cpu_adam')
    tok=AutoTokenizer.from_pretrained(vc.encoder_name,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    encoder=AutoModelForCausalLM.from_pretrained(vc.encoder_name,dtype=torch.bfloat16,
                 attn_implementation='sdpa',device_map={'':'cuda:1'},local_files_only=True)
    decoder=AutoModelForCausalLM.from_pretrained(vc.decoder_name,dtype=torch.bfloat16,
                 attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True)
    model=SplitReasoningVAE(vc,encoder,decoder)
    model.fc_mu.to('cuda:1');model.fc_var.to('cuda:1')
    model.latents_query.data=model.latents_query.data.to('cuda:1')
    model.latent_to_decoder.to('cuda:0');model.ae_speical_token.data=model.ae_speical_token.data.to('cuda:0')
    assert all(not p.requires_grad for p in model.decoder_model.parameters())
    counts=dict(encoder=sum(p.numel() for p in encoder.parameters()),decoder=sum(p.numel() for p in decoder.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    status('initializing_optimizer',parameter_counts=counts)
    optimizer=StreamedCPUAdamW(model.named_parameters(),cfg['train']['lr'],tuple(cfg['train']['betas']),cfg['train']['weight_decay'])
    named=dict(model.named_parameters())
    chosen=['encoder_model.model.layers.0.self_attn.q_proj.weight','latents_query','fc_mu.weight',
            'fc_var.weight','latent_to_decoder.weight','ae_speical_token']
    masters=dict(zip([n for n,_ in optimizer.named],optimizer.masters))
    before={n:masters[n].detach().flatten()[:1024].clone() for n in chosen}
    initial_hash=frozen_hash(model)
    fixtures=[json.loads(l) for l in (ROOT/'data/processed/vae-large-comparable-v1/smoke.jsonl').open()]
    rows=[fixtures[i%len(fixtures)] for i in range(cfg['train']['micro_batch_size'])]
    length=max(len(r['input_ids']) for r in rows)
    ids=torch.full((len(rows),length),tok.pad_token_id,dtype=torch.long,device='cuda:1')
    mask=torch.zeros_like(ids)
    for i,r in enumerate(rows):
        ids[i,:len(r['input_ids'])]=torch.tensor(r['input_ids'],device=ids.device);mask[i,:len(r['input_ids'])]=1
    for device in [0,1]:torch.cuda.reset_peak_memory_stats(device)
    losses=[];step_seconds=[]
    for update in range(1,cfg['train']['updates']+1):
        t=time.time();status('forward_backward',update=update,total_updates=cfg['train']['updates'])
        model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(input_ids=ids,attention_mask=mask,kl_weight=cfg['train']['kl_weight'])
        assert torch.isfinite(result['loss'])
        losses.append(float(result['loss']));result['loss'].backward()
        gradients={}
        for name in chosen:
            grad=named[name].grad
            gradients[name]=dict(finite=bool(grad is not None and torch.isfinite(grad).all()),
                                 absolute_sum=float(grad.float().abs().sum()) if grad is not None else 0)
            assert gradients[name]['finite'] and gradients[name]['absolute_sum']>0,name
        status('optimizer_step',update=update,total_updates=cfg['train']['updates'],gradients=gradients)
        grad_norm=optimizer.step();del result
        for device in [0,1]:torch.cuda.synchronize(device)
        step_seconds.append(time.time()-t)
        status('update_completed',update=update,total_updates=cfg['train']['updates'],seconds=step_seconds[-1],grad_norm=grad_norm)
    updates={n:float((masters[n].detach().flatten()[:1024]-before[n]).abs().max()) for n in chosen}
    assert all(v>0 for v in updates.values())
    final_hash=frozen_hash(model);assert final_hash==initial_hash
    status('reconstruction')
    reconstructed=reconstruct_rows(model,tok,[rows[0]],max_new_tokens=32)
    status('saving_checkpoint')
    ckpt=out/f'checkpoint_step_{optimizer.steps}';ckpt.mkdir()
    tensor_check=save_trainable(model,ckpt/'trainable')
    torch.save(optimizer.state_dict(),ckpt/'optimizer.pt')
    torch.save(dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),ckpt/'rng.pt')
    write(ckpt/'metadata.json',dict(config=cfg,updates=optimizer.steps,backend='explicit_model_parallel_streamed_cpu_adam',
                                  frozen_decoder_path=vc.decoder_name))
    measurements=[dict(rank=i,device=i,peak_allocated_gib=torch.cuda.max_memory_allocated(i)/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(i)/2**30) for i in [0,1]]
    report=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=optimizer.steps,
                smoke_batch_size=len(rows),unique_source_targets=len(fixtures),
                repeated_fixtures_for_memory_test=True,
                backend='explicit_model_parallel_streamed_cpu_adam',parameter_counts=counts,losses=losses,
                update_seconds=step_seconds,gradients=gradients,master_parameter_changes=updates,
                decoder_unchanged=True,decoder_hash_before=initial_hash,decoder_hash_after=final_hash,
                reconstruction=reconstructed,checkpoint_saved=True,checkpoint_tensor_check=tensor_check,
                checkpoint_resume_verified=False,ranks=measurements,seconds=time.time()-started,
                test_used=False,full_training_started=False,process_rss_gib=psutil.Process().memory_info().rss/2**30)
    write(out/'summary.json',report);status('completed',summary=str(out/'summary.json'))


if __name__=='__main__':main()
