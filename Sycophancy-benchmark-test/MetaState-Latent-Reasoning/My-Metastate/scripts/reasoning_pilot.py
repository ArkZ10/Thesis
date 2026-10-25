"""Fixed-input latent-workspace pilot. Commands: check, run, reload.
Four internal forwards (including bootstrap), then a supervised readout.
All five forwards see identical prompt + eight MASK tokens. No target reveals.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

from phase3_pilot import (ROOT, MODULES, dependencies, write, append, sha,
                          hash_parameters, module_snapshot, grad_report, memory)

CANVAS = 8
PASSES = 4
STEPS = 128
DATA = ROOT/'data/reasoning-pilot'


def collate(rows, mask_id, eos_id):
    import torch
    if len(rows)!=1:
        raise ValueError('Pilot supports batch size one only; avoids padding ambiguities')
    row=rows[0]
    boundary=next(i for i,v in enumerate(row['labels']) if v!=-100)
    assert all(v==-100 for v in row['labels'][:boundary])
    target=row['labels'][boundary:]
    assert target and len(target)<=CANVAS, 'Target too long; never truncate'
    # Answer length cannot change the model input; EOS fills unused answer slots.
    ids=row['input_ids'][:boundary]+[mask_id]*CANVAS
    labels=[-100]*boundary+target+[eos_id]*(CANVAS-len(target))
    return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long),
            'labels':torch.tensor([labels])}


def latent_forward(model, inputs, passes=PASSES):
    import torch
    state=cond=None
    # Time stays fully masked because no tokens are revealed.
    for _ in range(passes):
        out=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],
                  state=state,past_cond=cond,t=torch.ones(inputs['input_ids'].shape[0],device=inputs['input_ids'].device),
                  use_cache=False,return_dict=True)
        state,cond=out.new_state,out.past_cond
        del out
    out=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],
              state=state,past_cond=cond,t=torch.ones(inputs['input_ids'].shape[0],device=inputs['input_ids'].device),
              use_cache=False,return_dict=True)
    return out


def objective(model,inputs):
    import torch
    out=latent_forward(model,inputs)
    chosen=inputs['labels']!=-100
    loss=torch.nn.functional.cross_entropy(out.logits[chosen].float(),inputs['labels'][chosen])
    assert torch.isfinite(loss), 'Nonfinite loss'
    return loss


def check():
    import torch
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w=torch.nn.Parameter(torch.tensor(.2));self.calls=[]
        def forward(self,input_ids,state=None,**kw):
            self.calls.append(input_ids.clone())
            s=self.w if state is None else state*self.w+1
            scores=torch.arange(20).float()*s
            return SimpleNamespace(new_state=s,past_cond=s,logits=scores.expand(1,input_ids.shape[1],20))
    row={'input_ids':[1,2,3,4],'labels':[-100,-100,3,4]}
    a=collate([row],19,0)
    b=collate([{'input_ids':[1,2,8,9,4],'labels':[-100,-100,8,9,4]}],19,0)
    assert torch.equal(a['input_ids'],b['input_ids']), 'Answer length leaked'
    m=Toy();objective(m,a).backward()
    assert len(m.calls)==PASSES+1
    assert all(torch.equal(x,a['input_ids']) for x in m.calls)
    assert all((x[:,-CANVAS:]==19).all() for x in m.calls)
    assert m.w.grad is not None and m.w.grad.abs()>0
    report={'status':'passed','forwards':len(m.calls),'identical_masked_inputs':True,
            'target_content_and_length_hidden':True,'gradient_through_internal_passes':True}
    print(json.dumps(report),flush=True)
    return report


def run(directory):
    torch,sft=dependencies()
    from datasets import load_from_disk
    from transformers import TrainerCallback, set_seed
    from dllm.utils.models import get_model,get_tokenizer
    from dllm.core.trainers.recurrent_mdlm import RecurrentMDLMTrainer
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert json.loads((DATA/'validation.json').read_text())['status']=='passed'
    for rel,digest in manifest['files'].items(): assert sha(DATA/rel)==digest
    cfg=json.loads((directory/'configuration.json').read_text())
    assert cfg['script_sha256']==sha(Path(__file__))
    backbone=cfg['backbone']['model_name_or_path']
    args=sft.ModelArguments(model_name_or_path=backbone,backbone_name_or_path=backbone,
        freeze_backbone=True,dtype='bfloat16',num_memory_slots=64,state_hidden_size=1024,
        time_cond_dim=1024,mixer_dim=768,injector_dim=768,mixer_slot_attn=True,
        unroll_steps=PASSES,state_norm_weight=0,slot_diversity_weight=0,debug_stats=False)
    sft.prepare_recurrent_model_args(args)
    model=get_model(args,is_recurrent=True,training=True)
    tokenizer=get_tokenizer(args);tokenizer.pad_token=tokenizer.eos_token
    assert all(not p.requires_grad for p in model.model.parameters())
    assert all(n.split('.')[0] in MODULES for n,p in model.named_parameters() if p.requires_grad)
    initial=module_snapshot(model)
    before=hash_parameters(model.model)
    write(directory/'backbone_before_sha256.json',before)
    ds=load_from_disk(str(DATA/'tokenized'))
    fn=lambda rows:collate(rows,tokenizer.mask_token_id,tokenizer.eos_token_id)
    trainargs=sft.TrainingArguments(output_dir=str(directory/'checkpoints'),max_steps=STEPS,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=1,
        learning_rate=2e-5,lr_scheduler_type='constant',logging_steps=1,
        eval_strategy='no',save_strategy='steps',save_steps=32,save_total_limit=4,
        report_to='none',bf16=True,seed=42,data_seed=42,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},dataloader_num_workers=0,
        disable_tqdm=True,remove_unused_columns=False)
    trainargs.backbone_name_or_path=backbone
    class Trainer(RecurrentMDLMTrainer):
        def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
            loss=objective(model,inputs)
            return (loss,{'loss':loss}) if return_outputs else loss
    class Audit(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model=None,**kw):
            append(directory/'gradients.jsonl',{'step':state.global_step+1,'modules':grad_report(model)})
        def on_log(self,args,state,control,logs=None,**kw):
            append(directory/'metrics.jsonl',{'step':state.global_step,**(logs or {})})
            write(directory/'status.json',{'stage':'training','step':state.global_step,'total_steps':STEPS})
    trainer=Trainer(model=model,processing_class=tokenizer,args=trainargs,
                    train_dataset=ds['train'],data_collator=fn,callbacks=[Audit()])
    trainer.save_model(str(directory/'initial-checkpoint'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    longest=max(range(len(ds['train'])),key=lambda i:len(ds['train'][i]['input_ids']))
    batch={k:v.to('cuda') for k,v in fn([ds['train'][longest]]).items()}
    observed=[]
    def observe(module,args,kwargs):
        observed.append(kwargs['input_ids'].detach().cpu().clone())
    hook=model.register_forward_pre_hook(observe,with_kwargs=True)
    model.train();torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=objective(model,batch)
    loss.backward();hook.remove()
    assert len(observed)==PASSES+1 and all(torch.equal(x,observed[0]) for x in observed)
    assert all((x[:,-CANVAS:]==tokenizer.mask_token_id).all() for x in observed)
    profile={'status':'passed','loss':float(loss.detach()),'forwards':len(observed),
             'identical_masked_inputs':True,'memory':memory(torch),'gradients':grad_report(model)}
    # At zero-bridge initialization upstream memory gradients may initially be zero.
    write(directory/'preflight.json',profile)
    assert all(torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters() if n in initial)
    model.zero_grad(set_to_none=True);del loss,batch,observed;gc.collect();torch.cuda.empty_cache()
    set_seed(42)
    write(directory/'status.json',{'stage':'training','step':0,'total_steps':STEPS,'preflight':'passed'})
    trainer.train();trainer.save_state()
    after=hash_parameters(model.model);assert before==after,'Backbone changed'
    gradients=[json.loads(x) for x in (directory/'gradients.jsonl').read_text().splitlines()]
    assert len(gradients)==STEPS
    for name in MODULES:
        assert any(g['modules'][name]['nonzero_gradient_tensors']>0 for g in gradients),name
    changes={m:sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters()
                   if n in initial and n.startswith(m+'.')) for m in MODULES}
    assert all(changes.values())
    # Save a deterministic readout for fresh-process reload verification.
    batch={k:v.to('cuda') for k,v in fn([ds['train'][0]]).items()}
    model.eval()
    from safetensors.torch import save_file,load_file
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=latent_forward(model,batch)
    save_file({'logits':out.logits[:,-CANVAS:].contiguous().cpu(),
               'state':out.new_state.contiguous().cpu()},str(directory/'reload_expected.safetensors'))
    weights=load_file(str(directory/f'checkpoints/checkpoint-{STEPS}/recurrent_modules.safetensors'))
    live={f'{m}.{n}':v.detach().cpu() for m in MODULES for n,v in getattr(model,m).state_dict().items()}
    assert set(weights)==set(live) and all(torch.equal(weights[k],live[k]) for k in live)
    write(directory/'training_verification.json',{'status':'passed','steps':STEPS,
        'backbone_unchanged':True,'changed_tensors':changes,'saved_weights_exact':True,
        'all_modules_received_gradients':True,'accuracy_evaluation':'not yet performed'})
    write(directory/'status.json',{'stage':'training_complete_reload_pending','step':STEPS})


def reload(directory):
    torch,_=dependencies()
    from dllm.utils.models import get_model,get_tokenizer
    from datasets import load_from_disk
    from safetensors.torch import load_file
    cfg=json.loads((directory/'configuration.json').read_text())
    args=SimpleNamespace(model_name_or_path=str(directory/f'checkpoints/checkpoint-{STEPS}'),
        backbone_name_or_path=cfg['backbone']['model_name_or_path'],dtype='bfloat16',
        attn_implementation=None,freeze_backbone=True)
    model=get_model(args,is_recurrent=True,training=False);model.eval()
    t=get_tokenizer(args)
    row=load_from_disk(str(DATA/'tokenized'))['train'][0]
    batch={k:v.to('cuda') for k,v in collate([row],t.mask_token_id,t.eos_token_id).items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):out=latent_forward(model,batch)
    expected=load_file(str(directory/'reload_expected.safetensors'))
    assert torch.equal(expected['logits'],out.logits[:,-CANVAS:].cpu())
    assert torch.equal(expected['state'],out.new_state.cpu())
    write(directory/'reload_verification.json',{'status':'passed','fresh_process':True,'bitwise_equal':True})
    write(directory/'status.json',{'stage':'complete','step':STEPS,'next':'Stage 4 evaluation requires user go-ahead'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','launch','run','reload'])
    p.add_argument('--directory',type=Path)
    a=p.parse_args()
    if a.command=='check': check();return
    if a.command=='launch':
        import subprocess
        report=check()
        directory=ROOT/'result/reasoning-pilot'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory.mkdir(parents=True,exist_ok=False)
        write(directory/'configuration.json',{'seed':42,'steps':STEPS,'learning_rate':2e-5,
            'internal_passes_including_bootstrap':PASSES,'readout_passes':1,'answer_canvas':CANVAS,
            'initialization':'Fresh recurrent modules, seed 42; not SciQ checkpoint',
            'backbone':json.loads((ROOT/'setup/backbone.json').read_text()),
            'script_sha256':sha(Path(__file__)),'data_manifest_sha256':sha(DATA/'manifest.json'),
            'target_policy':'Answer, EOT, then EOS to fixed width; all eight positions supervised',
            'decoding_policy':'Single parallel greedy readout after internal updates; no token reveals',
            'training_split':'train only; development and test unused during this stage'})
        write(directory/'cpu_verification.json',report)
        write(directory/'status.json',{'stage':'loading'})
        write(ROOT/'result/reasoning-pilot/latest.json',{'directory':str(directory)})
        # Detached coordinator runs training then a fresh-process reload, and stops.
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1'
        with (directory/'worker.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--directory',str(directory)],
                                  stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=ROOT)
        write(directory/'process.json',{'pid':proc.pid})
        print(str(directory));return
    try:
        if a.command=='run':
            run(a.directory)
            # Replace this process so training allocations cannot occupy the GPU during reload.
            os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),'reload','--directory',str(a.directory)])
        else:reload(a.directory)
    except Exception:
        write(a.directory/'status.json',{'stage':'failed','traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
