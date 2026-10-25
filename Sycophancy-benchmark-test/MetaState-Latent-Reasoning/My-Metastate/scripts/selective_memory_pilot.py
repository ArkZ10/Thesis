"""Small selective-update training pilot; check, launch, train, reload commands.

Bootstrap + 3 masked internal updates + 4 predict-before-reveal diffusion rounds.
At each round two positions are revealed. Training reveals GT at positions chosen
by model confidence; inference reveals model predictions with the same utility.
No intermediate-target supervision, candidate search, or automatic evaluation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

from phase3_pilot import (ROOT, MODULES, dependencies, write, append, sha,
    hash_parameters, module_snapshot, grad_report, memory)
from reasoning_pilot import DATA, CANVAS, collate
from candidate_trajectory.selective_memory import ACTIONS, forward_with_memory_action

STEPS = 128
INTERNAL = 3  # In addition to one bootstrap.
DIFFUSION = 4
USED_ACTIONS = INTERNAL + DIFFUSION - 1


def schedules():
    # Six consumed selective writes per optimizer step. The final returned memory
    # is unused by the loss, so its write is `all` and excluded from action counts.
    pool = (list(ACTIONS) * ((STEPS*USED_ACTIONS+4)//5))[:STEPS*USED_ACTIONS]
    random.Random(42).shuffle(pool)
    return [pool[i:i+USED_ACTIONS] for i in range(0,len(pool),USED_ACTIONS)]


def trajectory(model, batch, plan, mask_id, training, transfer=None, audit=None):
    """No in-place token/state writes; every selected update retains gradients."""
    import torch
    if transfer is None:
        from dllm.core.samplers.utils import get_transfer_index
        transfer = get_transfer_index
    assert len(plan) == USED_ACTIONS and all(a in ACTIONS for a in plan)
    x = batch['input_ids'].clone()
    region = torch.zeros_like(x,dtype=torch.bool);region[:,-CANVAS:] = True
    assert (x[region] == mask_id).all()
    state = cond = None
    calls = 0
    def step(action, bootstrap=False):
        nonlocal state,cond,calls
        t = torch.ones(x.shape[0],device=x.device) if bootstrap else ((x==mask_id)&region).sum(-1).float()/(region.sum(-1).float()+1e-6)
        previous = state
        out = forward_with_memory_action(model, state=state, action=action,
            input_ids=x,attention_mask=batch['attention_mask'],past_cond=cond,t=t,
            use_cache=False,return_dict=True)
        if audit is not None:
            audit.append({'action':action,'tokens':x.detach().cpu().clone(),
                'previous':None if previous is None else previous.detach().cpu().clone(),
                'accepted':out.new_state.detach().cpu().clone(), 'bootstrap':bootstrap})
        state,cond = out.new_state,out.past_cond
        calls += 1
        return out
    out = step('all',bootstrap=True);del out
    for action in plan[:INTERNAL]:
        out = step(action);del out
    loss = None
    for i in range(DIFFUSION):
        remaining = (x==mask_id)&region
        out = step(plan[INTERNAL+i] if i<DIFFUSION-1 else 'all')
        if training:
            # Only currently masked answer positions contribute. Prompt excluded.
            term = torch.nn.functional.cross_entropy(out.logits[remaining].float(),batch['labels'][remaining]) / DIFFUSION
            loss = term if loss is None else loss+term
        with torch.no_grad():
            proposed,reveal = transfer(logits=out.logits.detach(),temperature=0,
                remasking='low_confidence',mask_index=remaining,x=x,
                num_transfer_tokens=torch.full((x.shape[0],),CANVAS//DIFFUSION,device=x.device,dtype=torch.long),
                threshold=None,factor=None,suppress_token_ids=None)
        assert not (reveal & ~remaining).any() and (reveal.sum(-1)==CANVAS//DIFFUSION).all()
        x = torch.where(reveal,batch['labels'] if training else proposed,x)
        del out
    assert calls==1+INTERNAL+DIFFUSION and not (x[region]==mask_id).any()
    if training:
        assert torch.isfinite(loss), 'Nonfinite loss'
    return loss,x,state,cond


def check():
    import torch
    from typing import NamedTuple
    from candidate_trajectory.selective_memory import slot_mask
    class Output(NamedTuple):
        logits:torch.Tensor
        new_state:torch.Tensor
        past_cond:torch.Tensor
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.tensor(.1))
        def forward(self,input_ids,state=None,past_cond=None,**kwargs):
            state=torch.ones(1,64,2)*self.w if state is None else state*(1+self.w)+self.w
            cond=state.mean().reshape(1,1)
            # Last vocabulary item is MASK; the toy predicts an ordinary token.
            scores=torch.cat((torch.arange(19).float(),torch.tensor([-1.]))) *(cond+self.w)
            return Output(scores.unsqueeze(1).expand(1,input_ids.shape[1],20),state,cond)
    def transfer(logits,mask_index,**kwargs):
        reveal=torch.zeros_like(mask_index)
        reveal[0,mask_index[0].nonzero().flatten()[:2]]=True
        return logits.argmax(-1),reveal
    row={'input_ids':[1,2,3,4], 'labels':[-100,-100,3,4]}
    batch=collate([row],19,0)
    plan=['A','B','C','D','all','A']
    audit=[];toy=Toy()
    loss,_,_,_=trajectory(toy,batch,plan,19,True,transfer,audit)
    loss.backward();assert toy.w.grad is not None and torch.isfinite(toy.w.grad)
    assert len(audit)==8 and all(torch.equal(a['tokens'],audit[0]['tokens']) for a in audit[:5])
    assert [int((a['tokens'][:,-CANVAS:]==19).sum()) for a in audit]==[8,8,8,8,8,6,4,2]
    for a in audit:
        if a['previous'] is not None:
            frozen=(~slot_mask(a['previous'],a['action'])).expand_as(a['previous'])
            assert torch.equal(a['accepted'][frozen],a['previous'][frozen])
    # The inference path must never access labels or reference metadata.
    _,tokens,_,_=trajectory(toy,{k:v for k,v in batch.items() if k!='labels'},plan,19,False,transfer)
    assert tokens.shape==batch['input_ids'].shape
    counts=Counter(a for plan in schedules() for a in plan)
    assert max(counts.values())-min(counts.values())<=1
    assert schedules()==schedules()
    return {'status':'passed','forwards_per_example':8,'gradient_through_selective_updates':True,
        'unselected_slots_exact':True,'masks_before_each_forward':[8,8,8,8,8,6,4,2],
        'inference_requires_no_labels':True,'balanced_consumed_actions':dict(counts)}


def load_model(config,checkpoint,training):
    from dllm.utils.models import get_model,get_tokenizer
    args=SimpleNamespace(model_name_or_path=str(checkpoint),backbone_name_or_path=config['backbone']['model_name_or_path'],
        freeze_backbone=True,dtype='bfloat16',attn_implementation=None)
    model=get_model(args,is_recurrent=True,training=training)
    t=get_tokenizer(args);t.pad_token=t.eos_token
    return model,t


def train(directory):
    torch,sft=dependencies()
    from datasets import load_from_disk
    from transformers import TrainerCallback,set_seed
    from dllm.core.trainers.recurrent_mdlm import RecurrentMDLMTrainer
    from safetensors.torch import load_file,save_file
    cfg=json.loads((directory/'configuration.json').read_text())
    for rel,digest in cfg['code_sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(DATA/'manifest.json')==cfg['data_manifest_sha256']
    manifest=json.loads((DATA/'manifest.json').read_text())
    # Read only training artifacts. Development/test are not opened here.
    for rel in ('processed/train.jsonl','token_arrays/train.jsonl'):
        assert sha(DATA/rel)==manifest['files'][rel]
    model,t=load_model(cfg,cfg['initial_checkpoint'],True)
    assert all(not p.requires_grad for p in model.model.parameters())
    assert all(n.split('.')[0] in MODULES for n,p in model.named_parameters() if p.requires_grad)
    initial=module_snapshot(model);before=hash_parameters(model.model)
    write(directory/'backbone_before_sha256.json',before)
    rows=[json.loads(s) for s in (DATA/'token_arrays/train.jsonl').read_text().splitlines()]
    from datasets import Dataset
    ds=Dataset.from_list(rows)
    fn=lambda rows:collate(rows,t.mask_token_id,t.eos_token_id)
    args=sft.TrainingArguments(output_dir=str(directory/'checkpoints'),max_steps=STEPS,
        per_device_train_batch_size=1,gradient_accumulation_steps=1,learning_rate=2e-5,
        lr_scheduler_type='constant',logging_steps=1,eval_strategy='no',
        save_strategy='steps',save_steps=32,save_total_limit=4,report_to='none',bf16=True,
        seed=42,data_seed=42,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        dataloader_num_workers=0,disable_tqdm=True,remove_unused_columns=False,group_by_length=False)
    args.backbone_name_or_path=cfg['backbone']['model_name_or_path']
    plans=schedules()
    class Trainer(RecurrentMDLMTrainer):
        def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
            plan=plans[self.state.global_step]
            loss,_,_,_=trajectory(model,inputs,plan,t.mask_token_id,True)
            return (loss,{'loss':loss}) if return_outputs else loss
    class Audit(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model=None,**kw):
            report=grad_report(model)
            append(directory/'gradients.jsonl',{'step':state.global_step+1,'modules':report,
                'consumed_update_actions':plans[state.global_step]})
        def on_log(self,args,state,control,logs=None,**kw):
            append(directory/'metrics.jsonl',{'step':state.global_step,**(logs or {})})
            write(directory/'status.json',{'stage':'training','step':state.global_step,'total_steps':STEPS})
    trainer=Trainer(model=model,processing_class=t,args=args,train_dataset=ds,data_collator=fn,callbacks=[Audit()])
    trainer.save_model(str(directory/'initial-checkpoint'))
    stored=load_file(str(directory/'initial-checkpoint/recurrent_modules.safetensors'))
    original=load_file(str(Path(cfg['initial_checkpoint'])/'recurrent_modules.safetensors'))
    assert stored.keys()==original.keys() and all(torch.equal(stored[k],original[k]) for k in stored)
    del stored,original
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train();torch.cuda.reset_peak_memory_stats()
    longest=max(range(len(rows)),key=lambda i:len(rows[i]['input_ids']))
    batch={k:v.to('cuda') for k,v in fn([rows[longest]]).items()}
    audit=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss,_,_,_=trajectory(model,batch,['A','B','C','D','all','A'],t.mask_token_id,True,audit=audit)
    loss.backward()
    from candidate_trajectory.selective_memory import slot_mask
    assert len(audit)==8
    assert [int((a['tokens'][:,-CANVAS:]==t.mask_token_id).sum()) for a in audit]==[8,8,8,8,8,6,4,2]
    for a in audit:
        if a['previous'] is not None:
            frozen=(~slot_mask(a['previous'],a['action'])).expand_as(a['previous'])
            assert torch.equal(a['accepted'][frozen],a['previous'][frozen])
    grads=grad_report(model)
    assert all(grads[m]['nonzero_gradient_tensors']>0 for m in MODULES)
    write(directory/'preflight.json',{'status':'passed','forwards':len(audit),
        'loss':float(loss.detach()),'gradients':grads,'memory':memory(torch),
        'slots_preserved_exactly':True,'targets_hidden_during_internal_updates':True,
        'initial_weights_equal_source_checkpoint':True})
    assert all(torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters() if n in initial)
    model.zero_grad(set_to_none=True);del loss,batch,audit;gc.collect();torch.cuda.empty_cache();set_seed(42)
    write(directory/'status.json',{'stage':'training','step':0,'total_steps':STEPS,'preflight':'passed'})
    trainer.train();trainer.save_state()
    assert hash_parameters(model.model)==before,'Backbone changed'
    gradients=[json.loads(s) for s in (directory/'gradients.jsonl').read_text().splitlines()]
    assert len(gradients)==STEPS
    assert [g['consumed_update_actions'] for g in gradients]==plans
    assert all(any(g['modules'][m]['nonzero_gradient_tensors']>0 for g in gradients) for m in MODULES)
    changes={m:sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters()
        if n in initial and n.startswith(m+'.')) for m in MODULES}
    assert all(changes.values())
    checkpoint=directory/f'checkpoints/checkpoint-{STEPS}'
    stored=load_file(str(checkpoint/'recurrent_modules.safetensors'))
    live={f'{m}.{n}':v.detach().cpu() for m in MODULES for n,v in getattr(model,m).state_dict().items()}
    assert stored.keys()==live.keys() and all(torch.equal(stored[k],live[k]) for k in live)
    model.eval()
    batch={k:v.to('cuda') for k,v in fn([rows[0]]).items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        # Teacher-forced training replay only: not a generated candidate response.
        loss,tokens,state,cond=trajectory(model,batch,plans[0],t.mask_token_id,True)
    save_file({'loss':loss.detach().cpu().reshape(1),'state':state.contiguous().cpu(),
        'conditioning':cond.contiguous().cpu()},str(directory/'reload_expected.safetensors'))
    write(directory/'training_verification.json',{'status':'passed','optimizer_steps':STEPS,
        'backbone_unchanged':True,'changed_tensors':changes,'saved_weights_exact':True,
        'action_counts':dict(Counter(a for plan in plans for a in plan)),
        'all_modules_received_nonzero_gradients':True,'candidate_generation_performed':False,
        'accuracy_evaluation_performed':False})
    write(directory/'status.json',{'stage':'training_complete_reload_pending','step':STEPS})


def reload(directory):
    torch,_=dependencies()
    from safetensors.torch import load_file
    cfg=json.loads((directory/'configuration.json').read_text())
    model,t=load_model(cfg,directory/f'checkpoints/checkpoint-{STEPS}',False);model.eval()
    row=json.loads((DATA/'token_arrays/train.jsonl').read_text().splitlines()[0])
    batch={k:v.to('cuda') for k,v in collate([row],t.mask_token_id,t.eos_token_id).items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        loss,_,state,cond=trajectory(model,batch,schedules()[0],t.mask_token_id,True)
    expected=load_file(str(directory/'reload_expected.safetensors'))
    assert torch.equal(expected['loss'],loss.cpu().reshape(1))
    assert torch.equal(expected['state'],state.cpu()) and torch.equal(expected['conditioning'],cond.cpu())
    write(directory/'reload_verification.json',{'status':'passed','fresh_process':True,'bitwise_equal':True})
    write(directory/'status.json',{'stage':'complete','step':STEPS,
        'next':'Report training verification, then wait for user before candidate evaluation.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','launch','train','reload'])
    p.add_argument('--directory',type=Path)
    a=p.parse_args()
    if a.command=='check':print(json.dumps(check(),indent=2));return
    if a.command=='launch':
        report=check()
        source=Path(json.loads((ROOT/'result/reasoning-pilot/latest.json').read_text())['directory'])
        assert json.loads((source/'status.json').read_text())['stage']=='complete'
        directory=ROOT/'result/selective-memory-training'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory.mkdir(parents=True,exist_ok=False)
        code=[Path(__file__),ROOT/'scripts/reasoning_pilot.py',ROOT/'scripts/phase3_pilot.py',
            ROOT/'scripts/candidate_trajectory/selective_memory.py']
        write(directory/'configuration.json',{'seed':42,'steps':STEPS,'learning_rate':2e-5,
            'initial_checkpoint':str(source/'checkpoints/checkpoint-128'),
            'initialization':'Continue arithmetic checkpoint; separate output directory; optimizer reset.',
            'backbone':json.loads((ROOT/'setup/backbone.json').read_text()),
            'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code},
            'data_manifest_sha256':sha(DATA/'manifest.json'),'precision':'BF16','physical_gpu':1,
            'bootstrap_forwards':1,'internal_selective_updates':INTERNAL,'diffusion_rounds':DIFFUSION,
            'answer_canvas':CANVAS,'actions':list(ACTIONS),'planned_consumed_action_counts':dict(Counter(a for plan in schedules() for a in plan)),
            'conditioning_policy':'Native recomputation; never reset or independently masked.',
            'training_policy':'Balanced answer-independent action schedule. Final-answer denoising CE only; GT revealed after prediction at model-confidence positions.',
            'limitations':['No guarantee of useful reasoning or semantic diversity.',
                'Decoder objective differs from previous simultaneous readout; not a matched performance comparison.',
                'Teacher-forced training versus model-predicted inference tokens remains a distribution difference.'],
            'development_or_test_used':False,'automatic_candidate_evaluation':False})
        write(directory/'cpu_verification.json',report)
        write(directory/'status.json',{'stage':'loading'})
        write(ROOT/'result/selective-memory-training/latest.json',{'directory':str(directory)})
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1'
        with (directory/'worker.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'train','--directory',str(directory)],
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=ROOT)
        write(directory/'process.json',{'pid':proc.pid});print(directory,flush=True);return
    try:
        if a.command=='train':
            train(a.directory)
            os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),'reload','--directory',str(a.directory)])
        else:reload(a.directory)
    except Exception:
        import traceback
        write(a.directory/'status.json',{'stage':'failed','traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
