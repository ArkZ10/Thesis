"""Phase 4: generate 20 held-out SciQ answers from the pilot checkpoint.

No training or external evaluator. Writes each response immediately; repeat the
same command with --directory to resume. Reference matching is diagnostic only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n');temp.replace(path)
def lines(path):return [json.loads(s) for s in path.read_text().splitlines()]
def append(path,obj):
    with path.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def normalized(text):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))

def prepare():
    previous=Path(json.loads((ROOT/'result/phase3/latest.json').read_text())['directory'])
    assert json.loads((previous/'completion.json').read_text())['status']=='passed'
    checkpoint=previous/'checkpoints/checkpoint-20'
    source=ROOT/'data/processed/tiny/development.jsonl'
    rows=lines(source)
    assert len(rows)==20 and len({r['question_id'] for r in rows})==4
    run=ROOT/'result/phase4'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run.mkdir(parents=True,exist_ok=False)
    inputs=[]
    for row in rows:
        assert row['source_split']=='validation'
        inputs.append({k:row[k] for k in ('id','question_id','condition','question','correct_answer','selected_incorrect_answer')} | {'messages':[row['messages'][0]]})
    with (run/'inputs.jsonl').open('w') as f:
        for row in inputs:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    config={'phase':4,'checkpoint':str(checkpoint),'source_file':str(source),'source_sha256':sha(source),'inputs_sha256':sha(run/'inputs.jsonl'),'checkpoint_sha256':{p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()},'runner_sha256':sha(Path(__file__)),'seed':0,'physical_gpu':1,'precision':'bfloat16','batch_size':1,'max_new_tokens':128,'steps':128,'block_size':32,'temperature':0.,'cfg_scale':0.,'remasking':'low_confidence','use_cache':None,'warmup':True,'pass_attention_mask':True,'threshold':None,'factor':None,'num_prompts':20,'base_questions':4,'purpose':'Generation health and manual answer-quality inspection, not a stable accuracy estimate','reference_policy':'Correct answer and distractor are metadata only; only the single user message is passed to the model','training_performed':False}
    dump(run/'configuration.json',config);dump(ROOT/'result/phase4/latest.json',{'directory':str(run)})
    return run

def generate(run):
    import torch
    from transformers import set_seed
    from dllm.utils.models import get_model,get_tokenizer
    from dllm.core.samplers.recurrent_mdlm import RecurrentMDLMSampler,RecurrentMDLMSamplerConfig
    cfg=json.loads((run/'configuration.json').read_text())
    assert sha(Path(__file__))==cfg['runner_sha256']
    assert sha(Path(cfg['source_file']))==cfg['source_sha256']
    assert sha(run/'inputs.jsonl')==cfg['inputs_sha256']
    checkpoint=Path(cfg['checkpoint'])
    for name,expected in cfg['checkpoint_sha256'].items():assert sha(checkpoint/name)==expected,name
    rows=lines(run/'inputs.jsonl');done={r['id']:r for r in lines(run/'responses.jsonl')} if (run/'responses.jsonl').exists() else {}
    assert len(done)==len(lines(run/'responses.jsonl')) if (run/'responses.jsonl').exists() else True
    assert not (set(done)-{r['id'] for r in rows})
    args=SimpleNamespace(model_name_or_path=str(checkpoint),dtype='bfloat16',attn_implementation=None)
    set_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=get_model(args,is_recurrent=True,training=False);model.eval();tokenizer=get_tokenizer(args);tokenizer.pad_token=tokenizer.eos_token
    assert all(not p.requires_grad for p in model.parameters())
    sampler=RecurrentMDLMSampler(model=model,tokenizer=tokenizer)
    samplecfg=RecurrentMDLMSamplerConfig(**{k:cfg[k] for k in ('max_new_tokens','steps','block_size','temperature','cfg_scale','remasking','use_cache','warmup','pass_attention_mask','threshold','factor')})
    audit={}
    def hook(module,args,kwargs,out):
        state=kwargs.get('state')
        if audit['calls']==0:
            assert state is None,'Conversation started with carried-over memory'
        else:
            assert state is not None and state.data_ptr()==audit['previous_state'].data_ptr(),'Memory not carried between denoising steps'
            audit['carry_links']+=1
        assert torch.isfinite(out.new_state).all() and torch.isfinite(out.logits).all(),'Nonfinite generation state/logits'
        audit['calls']+=1
        audit['max_state_norm']=max(audit['max_state_norm'],float(out.new_state.float().norm(dim=-1).mean()))
        audit['previous_state']=out.new_state.detach()
    handle=model.register_forward_hook(hook,with_kwargs=True)
    stop_ids={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|eot_id|>')}
    try:
        for row in rows:
            if row['id'] in done:continue
            audit.clear();audit.update(calls=0,carry_links=0,max_state_norm=0.,previous_state=None)
            prompt=tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True)
            set_seed(cfg['seed']);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                sequence=sampler.sample([prompt],samplecfg)[0].tolist()
            torch.cuda.synchronize()
            assert sequence[:len(prompt)]==prompt,'Sampler modified prompt'
            generated=sequence[len(prompt):]
            assert len(generated)==cfg['max_new_tokens']
            end=next((i for i,v in enumerate(generated) if v in stop_ids),len(generated))
            answer=tokenizer.decode(generated[:end],skip_special_tokens=True).strip()
            stop=generated[end] if end<len(generated) else None
            record={**row,'response':answer,'raw_generated_token_ids':generated,'raw_decoded_with_special_tokens':tokenizer.decode(generated,skip_special_tokens=False),'visible_token_count':end,'stop_token_id':stop,'stop_token':tokenizer.convert_ids_to_tokens(stop) if stop is not None else None,'empty':not bool(answer),'budget_exhausted_without_stop':stop is None,'remaining_mask_tokens':generated.count(tokenizer.mask_token_id),'normalized_exact_reference_match':normalized(answer)==normalized(row['correct_answer']),'normalized_exact_distractor_match':normalized(answer)==normalized(row['selected_incorrect_answer']),'forward_calls':audit['calls'],'memory_carry_links':audit['carry_links'],'max_mean_slot_norm':audit['max_state_norm'],'elapsed_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'execution_error':None}
            append(run/'responses.jsonl',record);done[row['id']]=record
            dump(run/'progress.json',{'saved':len(done),'total':len(rows),'last_id':row['id']})
    finally:handle.remove()
    responses=[done[r['id']] for r in rows]
    assert len(responses)==20
    summary={'phase':4,'status':'generation_finished_pending_manual_review','base_questions':4,'responses':20,'empty_responses':sum(r['empty'] for r in responses),'budget_exhausted_without_stop':sum(r['budget_exhausted_without_stop'] for r in responses),'responses_with_remaining_masks':sum(r['remaining_mask_tokens']>0 for r in responses),'execution_errors':sum(r['execution_error'] is not None for r in responses),'normalized_exact_reference_matches':sum(r['normalized_exact_reference_match'] for r in responses),'normalized_exact_distractor_matches':sum(r['normalized_exact_distractor_match'] for r in responses),'peak_allocated_gib':max(r['peak_allocated_bytes'] for r in responses)/1024**3,'total_generation_seconds':sum(r['elapsed_seconds'] for r in responses),'forward_calls':dict(Counter(r['forward_calls'] for r in responses)),'memory_reset_between_prompts':True,'memory_carried_between_steps':True,'note':'Exact string matches are diagnostic, not semantic grades. No external judge used. Four questions are only a generation pipeline check.','phase5_started':False}
    dump(run/'generation_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path);a=p.parse_args()
    run=a.directory or prepare()
    print('Phase 4 directory:',run,flush=True)
    generate(run)
