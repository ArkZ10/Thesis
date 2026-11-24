"""Phase 3 only: audited 20-step training pilot and fresh-process reload.

Usage: python scripts/phase3_pilot.py all
No free generation, answer grading, or full training is performed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('time_conditioner','mixer','updater','injector')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');temp.replace(path)

def append(path,obj):
    with path.open('a') as f:
        f.write(json.dumps(obj,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dependencies():
    import torch
    from transformers import set_seed
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('pilot_upstream_sft',ROOT/'vendor/MetaState/examples/recurrent_llada/sft.py')
    sft=importlib.util.module_from_spec(spec);sys.modules[spec.name]=sft;spec.loader.exec_module(sft)
    return torch,sft

def hash_parameters(model):
    result={}
    for name,p in model.named_parameters():
        # Hash every byte, including BF16 tensors; one parameter copied at a time.
        raw=p.detach().cpu().contiguous().view(__import__('torch').uint8).numpy()
        result[name]=hashlib.sha256(memoryview(raw)).hexdigest()
    return result

def memory(torch):
    torch.cuda.synchronize()
    return {'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'free_bytes':torch.cuda.mem_get_info()[0],'total_bytes':torch.cuda.mem_get_info()[1]}

def module_snapshot(model):
    return {f'{m}.{name}':p.detach().cpu().clone() for m in MODULES for name,p in getattr(model,m).named_parameters()}

def grad_report(model):
    import torch
    results={}
    for module in MODULES:
        sq=0.;present=0;nonzero=0;missing=0
        for p in getattr(model,module).parameters():
            if p.grad is None:
                missing+=1;continue
            g=p.grad.detach().float()
            if not torch.isfinite(g).all():raise RuntimeError(f'Nonfinite gradient: {module}')
            present+=1;nonzero+=int(torch.count_nonzero(g)>0);sq+=float(g.square().sum())
        results[module]={'gradient_tensors':present,'nonzero_gradient_tensors':nonzero,'missing_gradient_tensors':missing,'l2_norm':sq**0.5}
    if any(p.grad is not None for p in model.model.parameters()):raise RuntimeError('Frozen backbone received gradients')
    return results

def controlled_forward(model,tokenizer,run):
    import torch
    from datasets import load_from_disk
    from transformers import DataCollatorForSeq2Seq
    row=load_from_disk(str(ROOT/'data/tokenized/sciq-belief-tiny'))['test'][0]
    batch=DataCollatorForSeq2Seq(tokenizer,padding=True,label_pad_token_id=-100,return_tensors='pt')([row])
    batch={k:v.to('cuda') for k,v in batch.items()}
    inputs=batch['input_ids'].clone();inputs[batch['labels']!=-100]=tokenizer.mask_token_id
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        first=model(input_ids=inputs,attention_mask=batch['attention_mask'],state=None,t=torch.ones(1,device='cuda'),use_cache=False,return_dict=True)
        second=model(input_ids=inputs,attention_mask=batch['attention_mask'],state=first.new_state,past_cond=first.past_cond,t=torch.full((1,),0.5,device='cuda'),use_cache=False,return_dict=True)
    result={'logits':second.logits[:,-1].detach().cpu().contiguous(),'state':second.new_state.detach().cpu().contiguous()}
    assert all(torch.isfinite(v).all() for v in result.values())
    return result

def train(run):
    torch,sft=dependencies()
    from datasets import load_from_disk
    from transformers import DataCollatorForSeq2Seq, TrainerCallback, set_seed
    from dllm.utils.models import get_model,get_tokenizer
    from dllm.core.trainers.recurrent_mdlm import RecurrentMDLMTrainer
    cfg=json.loads((run/'configuration.json').read_text())
    # Verify every frozen source/data file before using it.
    for rel,expected in cfg['preserved_file_sha256'].items():
        assert sha(ROOT/rel)==expected,f'Frozen input changed: {rel}'
    args=sft.ModelArguments(model_name_or_path=cfg['backbone_path'],backbone_name_or_path=cfg['backbone_path'],freeze_backbone=True,dtype='bfloat16',num_memory_slots=64,state_hidden_size=1024,time_cond_dim=1024,mixer_dim=768,injector_dim=768,mixer_slot_attn=True,unroll_steps=4,state_norm_weight=1e-4,slot_diversity_weight=0,debug_stats=True)
    sft.prepare_recurrent_model_args(args)
    model=get_model(args,is_recurrent=True,training=True)
    tokenizer=get_tokenizer(args);tokenizer.pad_token=tokenizer.eos_token
    assert all(not p.requires_grad for p in model.model.parameters())
    trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.split('.')[0] in MODULES for n in trainable)
    assert all(p.requires_grad for m in MODULES for p in getattr(model,m).parameters())
    write(run/'trainable_parameters.json',{'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(trainable.values()),'by_name':trainable,'backbone_frozen':True})
    initial=module_snapshot(model)
    before=hash_parameters(model.model);write(run/'backbone_before_sha256.json',before)
    dataset=load_from_disk(str(ROOT/'data/tokenized/sciq-belief-tiny'))
    collator=DataCollatorForSeq2Seq(tokenizer,padding=True,label_pad_token_id=-100,return_tensors='pt')
    training_args=sft.TrainingArguments(
        output_dir=str(run/'checkpoints'),max_steps=20,num_train_epochs=1,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=1,
        learning_rate=2e-5,logging_steps=1,eval_strategy='steps',eval_steps=10,eval_on_start=True,save_strategy='steps',save_steps=10,save_total_limit=2,report_to='none',
        bf16=True,seed=42,data_seed=42,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},dense_reveal_mix=0.75,reveal_position_policy='random',state_norm_threshold=1.0,
        dataloader_num_workers=0,disable_tqdm=True,group_by_length=False)
    training_args.backbone_name_or_path=cfg['backbone_path']
    class AuditedTrainer(RecurrentMDLMTrainer):
        def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
            loss=super().compute_loss(model,inputs,return_outputs=return_outputs,**kwargs)
            value=loss[0] if return_outputs else loss
            if not torch.isfinite(value).all():raise RuntimeError('Nonfinite loss')
            return loss
    class Audit(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            append(run/'gradients.jsonl',{'optimizer_step':state.global_step+1,'modules':grad_report(model)})
        def on_log(self,args,state,control,logs=None,**kwargs):
            append(run/'metrics.jsonl',{'global_step':state.global_step,**(logs or {})})
    trainer=AuditedTrainer(model=model,processing_class=tokenizer,args=training_args,train_dataset=dataset['train'],eval_dataset=dataset['test'],data_collator=collator,callbacks=[Audit()])
    # Fit check at the longest actual example in the 5,000-record training pilot.
    full=load_from_disk(str(ROOT/'data/tokenized/sciq-belief-pilot'))['train']
    longest=max(range(len(full)),key=lambda i:len(full[i]['input_ids']))
    batch=collator([full[longest]]);batch={k:v.to('cuda') for k,v in batch.items()}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train();torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=trainer.compute_loss(model,batch)
    loss.backward()
    profile={'loss':float(loss.detach()),'sequence_tokens':len(full[longest]['input_ids']),'gradients':grad_report(model),'memory':memory(torch),'note':'Full forward/backward, no optimizer update. Longest current training example; not a 512-token stress test.'}
    write(run/'memory_preflight.json',profile)
    model.zero_grad(set_to_none=True);del loss,batch;__import__('gc').collect();torch.cuda.empty_cache()
    assert all(torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters() if n in initial),'Preflight changed weights'
    set_seed(42);torch.cuda.reset_peak_memory_stats()
    output=trainer.train();trainer.save_state()
    # Trainer already saves checkpoint-20 through its lightweight checkpoint code.
    checkpoint=run/'checkpoints/checkpoint-20'
    assert checkpoint.joinpath('recurrent_modules.safetensors').is_file()
    after=hash_parameters(model.model);write(run/'backbone_after_sha256.json',after)
    assert before==after,'Backbone bytes changed'
    final=module_snapshot(model)
    changes={}
    for module in MODULES:
        names=[n for n in initial if n.startswith(module+'.')]
        changed=[n for n in names if not torch.equal(initial[n],final[n])]
        changes[module]={'tensor_count':len(names),'changed_tensor_count':len(changed),'changed_names':changed,'max_absolute_change':max(float((final[n].float()-initial[n].float()).abs().max()) for n in names)}
        assert changed,f'No parameter changes in {module}'
    grads=[json.loads(s) for s in (run/'gradients.jsonl').read_text().splitlines()]
    assert len(grads)==20
    assert all(any(r['modules'][m]['nonzero_gradient_tensors']>0 for r in grads) for m in MODULES),'A module never received nonzero gradients'
    expected=controlled_forward(model,tokenizer,run)
    repeated=controlled_forward(model,tokenizer,run)
    assert all(torch.equal(expected[k],repeated[k]) for k in expected),'Controlled forward not repeatable'
    from safetensors.torch import save_file,load_file
    save_file(expected,str(run/'reload_expected.safetensors'))
    saved=load_file(str(checkpoint/'recurrent_modules.safetensors'))
    final_state={f'{m}.{n}':v.detach().cpu() for m in MODULES for n,v in getattr(model,m).state_dict().items()}
    assert set(saved)==set(final_state)
    assert all(torch.equal(saved[k],final_state[k]) for k in final_state),'Saved memory state differs'
    write(run/'training_verification.json',{'status':'passed','optimizer_steps':trainer.state.global_step,'backbone_unchanged':True,'all_backbone_parameter_hashes_checked':len(before),'modules':changes,'finite_losses_and_gradients':True,'each_module_received_nonzero_gradient':True,'training_metrics':output.metrics,'training_peak_memory':memory(torch),'checkpoint':str(checkpoint),'checkpoint_weights_match':True,'controlled_forward_repeatable':True,'free_generation_performed':False})
    print('PHASE3_TRAINING_VERIFICATION_PASSED',flush=True)

def reload(run):
    torch,_=dependencies()
    from dllm.utils.models import get_model,get_tokenizer
    from types import SimpleNamespace
    from safetensors.torch import load_file
    checkpoint=run/'checkpoints/checkpoint-20'
    cfg=json.loads((run/'configuration.json').read_text())
    args=SimpleNamespace(model_name_or_path=str(checkpoint),backbone_name_or_path=cfg['backbone_path'],dtype='bfloat16',attn_implementation=None,freeze_backbone=True)
    model=get_model(args,is_recurrent=True,training=False)
    tokenizer=get_tokenizer(args);tokenizer.pad_token=tokenizer.eos_token
    expected=load_file(str(run/'reload_expected.safetensors'))
    actual=controlled_forward(model,tokenizer,run)
    diff={k:float((expected[k].float()-actual[k].float()).abs().max()) for k in expected}
    assert all(torch.equal(expected[k],actual[k]) for k in expected),f'Reloaded forward differs: {diff}'
    before=json.loads((run/'backbone_before_sha256.json').read_text())
    assert hash_parameters(model.model)==before,'Reloaded backbone differs'
    saved=load_file(str(checkpoint/'recurrent_modules.safetensors'))
    live={f'{m}.{n}':v.detach().cpu() for m in MODULES for n,v in getattr(model,m).state_dict().items()}
    assert set(saved)==set(live) and all(torch.equal(saved[k],live[k]) for k in live)
    write(run/'reload_verification.json',{'status':'passed','fresh_process':True,'memory_weights_exact':True,'backbone_exact':True,'controlled_forward_max_abs_difference':diff,'free_generation_performed':False})
    print('PHASE3_FRESH_RELOAD_PASSED',flush=True)

def all_stages():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT/'result/phase3'/stamp;run.mkdir(parents=True,exist_ok=False)
    source=json.loads((ROOT/'setup/manifest.json').read_text())
    validated=json.loads((ROOT/'data/manifests/validation.json').read_text())
    assert validated['status']=='passed'
    preserved={}
    for rel,value in source['source']['file_sha256'].items():preserved['vendor/MetaState/'+rel]=value
    for rel,value in validated['file_sha256'].items():preserved['data/'+rel]=value
    preserved['scripts/phase3_pilot.py']=sha(Path(__file__))
    cfg={'created_at_utc':datetime.now(timezone.utc).isoformat(),'phase':3,'scope':'Training plumbing pilot only; stop before phase 4','physical_gpu':1,'cuda_visible_devices':'1','seed':42,'optimizer_steps':20,'batch_size':1,'gradient_accumulation_steps':1,'precision':'bfloat16','gradient_checkpointing':True,'unroll_steps':4,'memory_slots':64,'state_hidden_size':1024,'mixer_dim':768,'injector_dim':768,'time_cond_dim':1024,'mixer_slot_attn':True,'learning_rate':2e-5,'dataset':'data/tokenized/sciq-belief-tiny','training_examples':40,'development_examples':20,'backbone_path':json.loads((ROOT/'setup/backbone.json').read_text())['model_name_or_path'],'model_revision':json.loads((ROOT/'setup/backbone.json').read_text())['revision'],'source_commit':source['source']['commit'],'preserved_file_sha256':preserved}
    write(run/'configuration.json',cfg)
    write(ROOT/'result/phase3/latest.json',{'directory':str(run)})
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1'
    print('Pilot directory:',run,flush=True)
    try:
        for stage in ('train','reload'):
            with (run/f'{stage}.log').open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__)),stage,'--directory',str(run)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        write(run/'completion.json',{'status':'passed','phase':3,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'next_phase_requires_user_confirmation':True})
        print('PHASE 3 PASSED. Await user approval before phase 4.',flush=True)
    except Exception:
        write(run/'completion.json',{'status':'failed','phase':3,'error':traceback.format_exc()})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['all','train','reload'])
    parser.add_argument('--directory',type=Path)
    a=parser.parse_args()
    if a.stage=='all':all_stages()
    else:
        assert a.directory
        {'train':train,'reload':reload}[a.stage](a.directory)
