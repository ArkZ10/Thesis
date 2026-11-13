"""Bounded ZeRO-3 VAE update, gradient/frozen-weight checks and checkpoint saving.

Run with torchrun --nproc_per_node=2. --tiny validates the same distributed
infrastructure with random tiny models before loading the real checkpoints.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.utils import safe_get_local_grad, safe_get_local_fp32_param
from omegaconf import OmegaConf
import psutil
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config
from transformers.integrations import HfDeepSpeedConfig

from vae.model import ReasoningVAE, VAEConfig, build_vae
from vae.reconstruction import reconstruct_rows
from vae.hybrid_checkpoint import save_hybrid_state

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def count(p):
    return getattr(p, 'ds_numel', p.numel())


class SmokeVAE(ReasoningVAE):
    def forward(self, *args, reconstruction_rows=None, tokenizer=None, **kwargs):
        # Enter through the engine/root forward so ZeRO's module hooks also
        # surround free reconstruction, rather than bypassing the root module.
        if reconstruction_rows is not None:
            return reconstruct_rows(self, tokenizer, reconstruction_rows, max_new_tokens=32)
        return super().forward(*args, **kwargs)


class TinyTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    def __call__(self, text, **kwargs):
        return {'input_ids': [2, 3, 4]}
    def decode(self, ids, **kwargs):
        return ' '.join(str(i) for i in ids if i not in (0, 1))


def decoder_hash(model):
    h = hashlib.sha256()
    for name, p in model.decoder_model.named_parameters():
        h.update(name.encode())
        part = getattr(p, 'ds_tensor', p)
        raw=part.detach().contiguous().view(torch.uint8).flatten()
        for block in raw.split(32*1024*1024):
            h.update(block.cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--tiny', action='store_true')
    args = ap.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend='nccl')
    rank = dist.get_rank()
    started = time.time()
    out = args.directory.resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    def status(stage, **fields):
        if rank == 0:
            data = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.time()-started, **fields)
            write(out/'status.json', data)
            print(json.dumps(data), flush=True)

    cfg = OmegaConf.to_container(OmegaConf.load(ROOT/'configs/vae_large_smoke.yaml'), resolve=True)
    ds_cfg = json.loads((ROOT/'configs/zero3_large_smoke.json').read_text())
    torch.manual_seed(cfg['train']['seed'])
    if rank == 0:
        write(out/'config.json', cfg)
        write(out/'deepspeed.json', ds_cfg)
    status('loading_models', tiny=args.tiny)
    hf_ds = HfDeepSpeedConfig(ds_cfg)  # retained for the lifetime of HF loading
    vc = VAEConfig(**cfg['model'])
    if args.tiny:
        ds_cfg['zero_optimization']['sub_group_size']=10000
        vc.latent_dim = 32
        vc.num_latents = 3
        tok = TinyTokenizer()
        with deepspeed.zero.Init(config_dict_or_path=ds_cfg, dtype=torch.bfloat16):
            conf = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                              num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
            enc = AutoModelForCausalLM.from_config(conf, dtype=torch.bfloat16, attn_implementation='sdpa')
            dec = AutoModelForCausalLM.from_config(conf, dtype=torch.bfloat16, attn_implementation='sdpa')
            model = SmokeVAE(vc, enc, dec)
        rows = [dict(parent_id='synthetic', problem_id='synthetic', sentence_index=0,
                     reasoning_block='2 3 4', input_ids=[2,3,4,1])]*2
    else:
        data = ROOT/'data/processed/vae-large-comparable-v1'
        tok = AutoTokenizer.from_pretrained(vc.encoder_name, local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        rows = [json.loads(l) for l in (data/'smoke.jsonl').open()]
        with deepspeed.zero.Init(config_dict_or_path=ds_cfg, dtype=torch.bfloat16):
            model = build_vae(vc, dtype=torch.bfloat16)
        model.__class__ = SmokeVAE
    model.set_trainable()
    counts = {name: sum(count(p) for p in mod.parameters()) for name, mod in
              [('encoder', model.encoder_model), ('decoder', model.decoder_model)]}
    counts['trainable'] = sum(count(p) for p in model.parameters() if p.requires_grad)
    assert all(not p.requires_grad for p in model.decoder_model.parameters())
    status('initializing_optimizer', parameter_counts=counts)
    opt = DeepSpeedCPUAdam([p for p in model.parameters() if p.requires_grad],
                          lr=cfg['train']['lr'], betas=tuple(cfg['train']['betas']),
                          weight_decay=cfg['train']['weight_decay'])
    engine, opt, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=ds_cfg)
    model = engine.module
    allocation = {device:sum(p.numel() for i,p in enumerate(engine.optimizer.fp32_partitioned_groups_flat)
                            if engine.optimizer.subgroup_to_device[i]==device) for device in ['cpu','cuda']}
    status('optimizer_ready', master_parameter_elements_by_device=allocation,
           gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
           ram_available_gib=psutil.virtual_memory().available/2**30)
    named = dict(model.named_parameters())
    chosen = [next(n for n in named if n.startswith('encoder_model.model.layers.0.self_attn.q_proj')),
              'latents_query', 'fc_mu.weight', 'fc_var.weight', 'latent_to_decoder.weight', 'ae_speical_token']
    before = {n: safe_get_local_fp32_param(named[n]).detach().flatten()[:1024].clone() for n in chosen}
    frozen_before = decoder_hash(model)
    row = rows[rank % len(rows)]
    ids = torch.tensor([row['input_ids']], device=engine.device)
    mask = torch.ones_like(ids)
    torch.manual_seed(cfg['train']['seed']+rank)
    torch.cuda.reset_peak_memory_stats()
    losses=[]
    steps=cfg['train']['updates']
    for update in range(1,steps+1):
        status('forward_backward', update=update,total_updates=steps,source_tokens=ids.shape[1],
               ram_available_gib=psutil.virtual_memory().available/2**30)
        engine.train()
        result=engine(input_ids=ids,attention_mask=mask,kl_weight=cfg['train']['kl_weight'])
        assert torch.isfinite(result['loss'])
        losses.append(float(result['loss']))
        engine.backward(result['loss'])
        gradients={}
        for name in chosen:
            g=safe_get_local_grad(named[name])
            values=torch.tensor([float(g is not None and torch.isfinite(g).all()),
                                 float(g.float().abs().sum()) if g is not None else 0.],
                                device=engine.device,dtype=torch.float64)
            dist.all_reduce(values,op=dist.ReduceOp.SUM)
            gradients[name]=dict(all_ranks_finite=values[0].item()==dist.get_world_size(),
                                 global_absolute_sum=values[1].item())
            assert gradients[name]['all_ranks_finite'] and gradients[name]['global_absolute_sum']>0,name
        status('optimizer_step',update=update,total_updates=steps,gradients=gradients)
        engine.step()
        del result
    updates = {}
    for n in chosen:
        after = safe_get_local_fp32_param(named[n]).detach().flatten()[:1024]
        v = torch.tensor(float((after-before[n]).abs().max()),device=engine.device)
        dist.all_reduce(v,op=dist.ReduceOp.MAX)
        updates[n] = v.item()
        assert v.item()>0, n
    frozen_after = decoder_hash(model)
    unchanged = torch.tensor(int(frozen_before==frozen_after),device=engine.device)
    dist.all_reduce(unchanged,op=dist.ReduceOp.MIN)
    assert unchanged.item()==1
    status('reconstruction', updates=updates)
    engine.eval()
    # Identical input on both ranks ensures identical decoding loop lengths and
    # matching ZeRO collectives. A bounded smoke test is not an overfit test.
    with torch.no_grad():
        reconstructed = engine(reconstruction_rows=[rows[0]],tokenizer=tok)
    status('saving_checkpoint')
    saved = engine.save_checkpoint(str(out/'checkpoints'), tag=f'step_{steps}',
                                   client_state={'updates':steps,'config':cfg,'tiny':args.tiny},
                                   exclude_frozen_parameters=True)
    assert saved
    hybrid=save_hybrid_state(engine.optimizer,out/'checkpoints'/f'step_{steps}'/f'hybrid_rank_{rank}')
    torch.cuda.synchronize()
    measurement = dict(rank=rank, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                       process_rss_gib=psutil.Process().memory_info().rss/2**30,
                       decoder_shard_hash_before=frozen_before, decoder_shard_hash_after=frozen_after)
    write(out/f'rank_{rank}.json',measurement)
    dist.barrier()
    if rank==0:
        files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size) for p in (out/'checkpoints').rglob('*') if p.is_file()]
        assert len(files)>=3 and sum(p['bytes'] for p in files)>0
        report=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(), tiny=args.tiny,
                    optimizer_updates=steps, parameter_counts=counts, losses=losses,
                    gradients=gradients, master_parameter_changes=updates, decoder_unchanged=True,
                    reconstruction=reconstructed, checkpoint_saved=True, checkpoint_files=files,
                    hybrid_gpu_optimizer_saved=bool(hybrid['subgroups']),
                    checkpoint_resume_verified=False, seconds=time.time()-started,
                    ranks=[json.loads((out/f'rank_{i}.json').read_text()) for i in range(dist.get_world_size())],
                    test_used=False, full_training_started=False)
        write(out/'summary.json',report)
        status('completed', summary=str(out/'summary.json'))
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
