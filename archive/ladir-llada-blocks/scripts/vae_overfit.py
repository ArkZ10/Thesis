"""Eight-trace VAE overfit gate; no diffusion or question-only reasoning.

Commands: check, launch, worker --directory PATH, finish --directory PATH.
Encoder: last four existing LoRA layers + compression modules, FP32 master params.
Decoder: separate frozen copy of the checkpoint base on GPU 1 (never adapters).
Train with stochastic posterior and KL 1e-5; evaluate deterministic posterior mean.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import torch

from checkpoint_runtime import ROOT
from latent_smoke import write,append
from vae_reconstruction import load_vae,encode

STEPS=128
LR=1e-4
BETA=1e-5
TRAIN=[(2,3,4),(3,4,2),(4,5,3),(5,2,6),(6,3,2),(7,2,3),(8,4,2),(9,3,4)]
DEV=[(2,4,3),(3,5,2),(4,2,5),(5,3,4),(6,2,3),(7,3,2),(8,2,4),(9,2,3)]


def data():
    result={};seen=set()
    for split,values in [('train',TRAIN),('development',DEV)]:
        rows=[]
        for a,b,c in values:
            key=(min(a,b),max(a,b),c)
            assert key not in seen;seen.add(key)
            subtotal=sum((a,b));answer=sum(subtotal for _ in range(c))
            text=f'{a}+{b}={subtotal},{subtotal}*{c}={answer}'
            # Two independent operator checks, plus readable exact trace target.
            assert subtotal==a+b and answer==subtotal*c
            qid='trace-'+hashlib.sha256(str(key).encode()).hexdigest()[:12]
            rows.append({'id':qid,'split':split,'operands':[a,b,c],
                'encoder_text':text,'target_text':text,'verified_answer':answer})
        result[split]=rows
    return result


def training_order(rows):
    rng=random.Random(42);order=[]
    while len(order)<STEPS:
        indices=list(range(len(rows)));rng.shuffle(indices);order.extend(indices)
    return order[:STEPS]


def selected(name):
    if name.startswith(('mean.','log_var.','decompress_layer.','memory_token_embed.')):return True
    match=re.search(r'\.layers\.(\d+)\.',name)
    return bool(match and int(match[1])>=28 and ('.lora_A.' in name or '.lora_B.' in name))


def configure_encoder(ae):
    for name,p in ae.named_parameters():
        p.requires_grad=selected(name)
        if p.requires_grad:p.data=p.data.float()
    ae.eval()
    return {n:p for n,p in ae.named_parameters() if p.requires_grad}


def frozen_decoder(ae):
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import LlamaForCausalLM
    cfg=copy.deepcopy(ae.icae.get_base_model().config);cfg._attn_implementation='eager'
    with init_empty_weights():decoder=LlamaForCausalLM(cfg)
    source={}
    for k,v in ae.icae.state_dict().items():
        if '.lora_A.' in k or '.lora_B.' in k:continue
        assert k.startswith('base_model.model.'),k
        mapped=k.removeprefix('base_model.model.').replace('.base_layer.','.')
        source[mapped]=v
    assert source.keys()==decoder.state_dict().keys()
    for key,value in source.items():
        set_module_tensor_to_device(decoder,key,'cuda:1',value=value.detach(),dtype=torch.bfloat16)
        assert torch.equal(decoder.get_parameter(key).detach().cpu(),value.detach().cpu()),key
    for module in decoder.modules():
        for key,buffer in list(module._buffers.items()):
            if buffer is not None:module._buffers[key]=buffer.to('cuda:1')
        if hasattr(module,'original_inv_freq'):module.original_inv_freq=module.inv_freq
    for p in decoder.parameters():p.requires_grad=False
    decoder.eval()
    return decoder


def reconstruction_loss(ae,decoder,t,row,rng=None):
    # Encoder receives the complete known trace, intentionally: a codec task.
    mu,logvar,_=encode(ae,t,row['encoder_text'])
    mu=mu.float();logvar=logvar.float()
    z=mu
    if rng is not None:
        noise=torch.randn(mu.shape,device=mu.device,generator=rng)
        z=mu+torch.exp(.5*logvar)*noise
    latent=ae.decompress_layer(z).to('cuda:1')
    marker=ae.decompress_layer(ae.memory_token_embed(torch.tensor([[3]],device=ae.device))).to('cuda:1')
    target=t(row['target_text'],return_tensors='pt')['input_ids'].to('cuda:1')
    target=torch.cat((target,torch.tensor([[2]],device='cuda:1')),dim=1)
    target_embeddings=decoder.get_input_embeddings()(target)
    prefix=torch.cat((latent,marker),dim=1)
    full=torch.cat((prefix,target_embeddings),dim=1)
    logits=decoder(inputs_embeds=full,use_cache=False).logits
    # Causal teacher forcing: the last prefix position predicts the first target.
    prediction=logits[:,prefix.shape[1]-1:prefix.shape[1]-1+target.shape[1]]
    ce=torch.nn.functional.cross_entropy(prediction.float().reshape(-1,prediction.shape[-1]),target.reshape(-1))
    kl=-.5*(1+logvar-mu.square()-logvar.exp()).mean()
    # Loss terms originate on different GPUs; the copy preserves encoder gradients.
    loss=ce+BETA*kl.to(ce.device)
    assert torch.isfinite(loss)
    return loss,{'ce':ce,'kl':kl,'mu':mu,'logvar':logvar}


def greedy_reconstruct(ae,decoder,t,row):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        mu,_,_=encode(ae,t,row['encoder_text'])
        latent=ae.decompress_layer(mu.float()).to('cuda:1')
        marker=ae.decompress_layer(ae.memory_token_embed(torch.tensor([[3]],device=ae.device))).to('cuda:1')
        x=torch.cat((latent,marker),dim=1);cache=None;ids=[];stop=None
        for _ in range(64):
            output=decoder(inputs_embeds=x,past_key_values=cache,use_cache=True)
            scores=output.logits[:,-1,:128256]
            assert torch.isfinite(scores).all()
            chosen=scores.argmax(-1);token=int(chosen.item());cache=output.past_key_values
            if token in (2,t.eos_token_id):stop=token;break
            ids.append(token);x=decoder.get_input_embeddings()(chosen).unsqueeze(1)
        text=t.decode(ids,skip_special_tokens=True).strip()
    normalize=lambda x:' '.join(x.split())
    return {**row,'decoded_text':text,'token_ids':ids,'stop_id':stop,
        'exact_match':normalize(text)==normalize(row['target_text']),'budget_exhausted':stop is None}


def evaluate(ae,decoder,t,dataset,directory,label,status):
    ae.eval();decoder.eval();results={}
    for split,rows in dataset.items():
        evaluated=[]
        for i,row in enumerate(rows):
            status(stage=label,split=split,completed=i,total=len(rows))
            record=greedy_reconstruct(ae,decoder,t,row)
            append(directory/f'{label}.jsonl',record);evaluated.append(record)
        results[split]={'exact':sum(r['exact_match'] for r in evaluated),'total':len(evaluated),
            'budget_exhausted':sum(r['budget_exhausted'] for r in evaluated)}
    write(directory/f'{label}-summary.json',results)
    return results


def train_modes(ae,decoder):
    # Separate decoder avoids PEFT enable/disable context during checkpointed
    # backward. Train mode enables activation checkpointing, not weight updates.
    ae.eval();ae.icae.train();decoder.train()
    for module in list(ae.modules())+list(decoder.modules()):
        if isinstance(module,torch.nn.Dropout):module.eval()


def gradients(ae,decoder,params):
    result={}
    for prefix in ('encoder_adapters','mean','log_var','decompress_layer','memory_token_embed'):
        chosen=[p for n,p in params.items() if ('.lora_' in n if prefix=='encoder_adapters' else n.startswith(prefix+'.'))]
        grads=[p.grad for p in chosen if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),prefix
        result[prefix]={'tensors':len(grads),'nonzero':sum(bool(torch.count_nonzero(g)) for g in grads)}
        assert result[prefix]['nonzero']>0,prefix
    assert all(p.grad is None for n,p in ae.named_parameters() if n not in params)
    assert all(p.grad is None for p in decoder.parameters())
    return result


def check():
    rows=data();order=training_order(rows['train'])
    assert len(order)==STEPS and set(Counter(order).values())=={16}
    assert selected('icae.base_model.model.model.layers.28.self_attn.q_proj.lora_A.default.weight')
    assert not selected('icae.base_model.model.model.layers.27.self_attn.q_proj.lora_A.default.weight')
    assert not selected('icae.base_model.model.model.layers.30.self_attn.q_proj.base_layer.weight')
    assert not selected('icae.base_model.model.lm_head.weight')
    assert selected('log_var.weight')
    return {'status':'passed','train':8,'development':8,'arithmetic_verified':True,
        'canonical_problem_groups_disjoint':True,'presentations_per_training_trace':16,
        'parameter_selection_checked':True}


def worker(directory):
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    status=lambda **kw:write(directory/'status.json',kw)
    cfg=json.loads((directory/'configuration.json').read_text())
    for rel,digest in cfg['code_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    dataset=json.loads((directory/'dataset.json').read_text())
    ae,t,loaded=load_vae(lambda n,total:status(stage='loading_vae',loaded=n,total=total))
    write(directory/'loading_verification.json',loaded)
    status(stage='copying_frozen_decoder')
    decoder=frozen_decoder(ae)
    # Baseline uses the exact same dedicated frozen decoder and decoding procedure.
    baseline=evaluate(ae,decoder,t,dataset,directory,'baseline',status)
    params=configure_encoder(ae)
    assert params and all(not p.requires_grad for p in decoder.parameters())
    write(directory/'trainable_parameters.json',{'count':sum(p.numel() for p in params.values()),
        'by_name':{n:p.numel() for n,p in params.items()},'master_dtype':'FP32',
        'decoder_frozen':True,'base_encoder_frozen':True})
    frozen_versions={n:p._version for n,p in ae.named_parameters() if n not in params}
    decoder_versions={n:p._version for n,p in decoder.named_parameters()}
    initial={n:p.detach().cpu().clone() for n,p in params.items()}
    ae.icae.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    train_modes(ae,decoder)
    rng=torch.Generator(device=ae.device).manual_seed(42)
    for i in range(2):torch.cuda.reset_peak_memory_stats(i)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss,parts=reconstruction_loss(ae,decoder,t,dataset['train'][0],rng)
    loss.backward()
    report=gradients(ae,decoder,params)
    write(directory/'preflight.json',{'status':'passed','loss':float(loss.detach()),
        'gradient_groups':report,'frozen_decoder_gradient_free':True,
        'gpu_peak_allocated_gib':{str(i):torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)}})
    ae.zero_grad(set_to_none=True);del loss,parts
    rng.manual_seed(42)
    optimizer=torch.optim.AdamW(params.values(),lr=LR,weight_decay=0)
    order=training_order(dataset['train'])
    from safetensors.torch import save_file
    for step,index in enumerate(order,1):
        train_modes(ae,decoder);optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,parts=reconstruction_loss(ae,decoder,t,dataset['train'][index],rng)
        loss.backward()
        gradient_report=gradients(ae,decoder,params)
        norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.0,error_if_nonfinite=True)
        optimizer.step()
        append(directory/'metrics.jsonl',{'step':step,'total_steps':STEPS,
            'training_id':dataset['train'][index]['id'],'loss':float(loss.detach()),
            'ce':float(parts['ce'].detach()),'kl':float(parts['kl'].detach()),'grad_norm_before_clipping':float(norm)})
        status(stage='training',step=step,total_steps=STEPS)
        if step in (64,128):
            checkpoint=directory/f'checkpoints/checkpoint-{step}'
            checkpoint.mkdir(parents=True,exist_ok=False)
            save_file({n:p.detach().cpu().contiguous() for n,p in params.items()},str(checkpoint/'trainable.safetensors'))
            torch.save({'step':step,'optimizer':optimizer.state_dict(),'posterior_rng':rng.get_state().cpu()},checkpoint/'training_state.pt')
    assert frozen_versions=={n:p._version for n,p in ae.named_parameters() if n not in params}
    assert decoder_versions=={n:p._version for n,p in decoder.named_parameters()}
    changes={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in params.items()}
    assert any(v>0 for v in changes.values())
    write(directory/'training_verification.json',{'status':'passed','optimizer_steps':STEPS,
        'frozen_parameter_versions_unchanged':True,'decoder_parameter_versions_unchanged':True,
        'trainable_parameter_max_changes':changes,'gradient_groups':gradient_report})
    ae.eval();decoder.eval()
    final=evaluate(ae,decoder,t,dataset,directory,'after-training',status)
    # Save deterministic encoder output for fresh-process checkpoint restoration.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        mu,lv,_=encode(ae,t,dataset['train'][0]['encoder_text'])
    save_file({'mean':mu.cpu().contiguous(),'logvar':lv.cpu().contiguous()},str(directory/'reload_expected.safetensors'))
    write(directory/'summary.json',{'baseline':baseline,'after_training':final,'training_steps':STEPS,
        'purpose':'Reconstruct known inputs; not question-only reasoning or a diffusion evaluation.',
        'decoder_frozen':True,'development_used_for_optimization':False,
        'reload_verification':'pending'})
    status(stage='training_and_evaluation_complete_reload_pending',step=STEPS)


def finish(directory):
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from safetensors.torch import load_file
    status=lambda **kw:write(directory/'status.json',kw)
    ae,t,_=load_vae(lambda n,total:status(stage='reload_verification',loaded=n,total=total))
    params=configure_encoder(ae);ae.eval()
    saved=load_file(str(directory/'checkpoints/checkpoint-128/trainable.safetensors'))
    assert saved.keys()==params.keys()
    result=ae.load_state_dict(saved,strict=False)
    assert not result.unexpected_keys and set(result.missing_keys)==set(ae.state_dict())-set(saved)
    assert all(torch.equal(p.detach().cpu(),saved[n]) for n,p in params.items())
    row=json.loads((directory/'dataset.json').read_text())['train'][0]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        mu,lv,_=encode(ae,t,row['encoder_text'])
    expected=load_file(str(directory/'reload_expected.safetensors'))
    assert torch.equal(mu.cpu(),expected['mean']) and torch.equal(lv.cpu(),expected['logvar'])
    write(directory/'reload_verification.json',{'status':'passed','fresh_process':True,'encoder_output_bitwise_equal':True})
    summary=json.loads((directory/'summary.json').read_text());summary['reload_verification']='passed'
    write(directory/'summary.json',summary)
    status(stage='complete',step=STEPS,next='Report results before any diffusion training or branching.')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','launch','worker','finish'])
    parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    if args.command=='check':print(json.dumps(check(),indent=2));return
    if args.command=='launch':
        report=check();directory=ROOT/'result/vae-overfit'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory.mkdir(parents=True,exist_ok=False)
        write(directory/'dataset.json',data());write(directory/'data_verification.json',report)
        write(directory/'configuration.json',{'training_steps':STEPS,'learning_rate':LR,'seed':42,
            'train_traces':8,'development_traces':8,'beta_kl':BETA,
            'target':'Known trace reconstruction, including BOS and legacy terminator 2.',
            'encoder_trainable':'Existing LoRA in layers 28..31 + mean/logvar/decompress/memory embeddings.',
            'base_encoder_frozen':True,'separate_decoder_frozen':True,'encoder_decoder_dropout':0,
            'latent_shape':[1,3,128],'training_posterior':'Sampled, no extra 0.3 noise.',
            'evaluation_posterior':'Mean','decoder_devices':{'encoder':0,'frozen_decoder':1},
            'checkpoint_source':json.loads((ROOT/'result/setup/checkpoint.json').read_text()),
            'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                Path(__file__),ROOT/'scripts/vae_reconstruction.py',ROOT/'scripts/checkpoint_runtime.py')},
            'training_order':training_order(data()['train']),'diffusion_training':False})
        write(directory/'status.json',{'stage':'loading'})
        write(ROOT/'result/vae-overfit/latest.json',{'directory':str(directory)})
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0,1'
        with (directory/'worker.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--directory',str(directory)],
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=ROOT)
        write(directory/'process.json',{'pid':proc.pid});print(directory,flush=True);return
    try:
        if args.command=='worker':
            worker(args.directory)
            os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),'finish','--directory',str(args.directory)])
        else:finish(args.directory)
    except Exception:
        import traceback
        write(args.directory/'status.json',{'stage':'failed','traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
