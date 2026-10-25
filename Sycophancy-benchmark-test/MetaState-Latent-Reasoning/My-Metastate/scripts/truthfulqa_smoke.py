"""Evaluate the phase-3 prototype on 100 TruthfulQA smoke inputs.

Commands: prepare, run, worker --rank 0|1, grade, report, status.
Uses saved baseline/discrete results read-only. No training, full dataset, or
SciQ evaluation. Background workflow produces comparisons automatically.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import phase3_pilot as p3
from phase5 import credential,parse_fact

ROOT=p3.ROOT
SMOKE=Path('/root/Thesis/Sycophancy-benchmark-test/LLaDA/TruthfulQASyco/smoketest')
DATA=Path('/root/Thesis/Sycophancy-benchmark-test/TruthfulQA-sycophancy/data/processed/smoke')
PREVIOUS={'baseline':SMOKE/'runs/20261018T041005_967364Z_regrade','discrete_reasoning':SMOKE/'reasoning/result/20261019T102414_955986Z'}
CONDITIONS=('supports_correct','doubts_correct','supports_incorrect','doubts_incorrect','neutral')

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def prepare():
    phase3=Path(read(ROOT/'result/phase3/latest.json')['directory'])
    assert read(phase3/'completion.json')['status']=='passed'
    checkpoint=phase3/'checkpoints/checkpoint-20'
    instruction=read(ROOT/'data/manifests/policy.json')['instruction']
    inputs=sorted(rows(DATA/'belief_prompts.jsonl')+rows(DATA/'neutral_prompts.jsonl'),key=lambda r:r['id'])
    assert len(inputs)==100 and len({r['id'] for r in inputs})==100 and len({r['question_id'] for r in inputs})==20
    for r in inputs:
        assert r['correct_answers'] and len(r['messages'])==1 and r['messages'][0]['role']=='user'
        r['base_messages']=r['messages'];r['messages']=[{'role':'user','content':r['base_messages'][0]['content']+'\n\n'+instruction}]
    for path in PREVIOUS.values():
        saved=rows(path/'results.jsonl');assert {r['input']['id'] for r in saved}=={r['id'] for r in inputs}
        index={r['input']['id']:r['input'] for r in saved}
        assert all(index[r['id']]['question']==r['question'] and index[r['id']]['correct_answers']==r['correct_answers'] for r in inputs)
    run=ROOT/'result/truthfulqa-smoke'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');run.mkdir(parents=True,exist_ok=False)
    with (run/'inputs.jsonl').open('w') as f:
        for r in inputs:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    (run/'judge_prompt.txt').write_bytes((PREVIOUS['baseline']/'judge_prompt.txt').read_bytes())
    assert (run/'judge_prompt.txt').read_bytes()==(PREVIOUS['discrete_reasoning']/'judge_prompt.txt').read_bytes()
    frozen={str(p):p3.sha(p) for p in [Path(__file__),ROOT/'scripts/phase5.py',ROOT/'scripts/phase3_pilot.py',DATA/'belief_prompts.jsonl',DATA/'neutral_prompts.jsonl',ROOT/'data/manifests/policy.json']}
    for path in checkpoint.iterdir():
        if path.is_file():frozen[str(path)]=p3.sha(path)
    for path in PREVIOUS.values():
        for name in ('results.jsonl','judge_prompt.txt','generation_config.json','metrics.json'):frozen[str(path/name)]=p3.sha(path/name)
    cfg={'checkpoint':str(checkpoint),'training':'Phase 3: 20 optimizer steps on 40 SciQ examples from eight questions; no additional training','input_sha256':p3.sha(run/'inputs.jsonl'),'judge_prompt_sha256':p3.sha(run/'judge_prompt.txt'),'instruction':instruction,'generation':{'seed':0,'precision':'bfloat16','batch_size':1,'max_new_tokens':128,'steps':128,'block_size':32,'temperature':0.,'cfg_scale':0.,'remasking':'low_confidence','use_cache':None,'warmup':True,'pass_attention_mask':True},'judge':{'requested_model':'gpt-4o-2024-08-06','required_returned_model':'gpt-4o-2024-08-06','temperature':0,'max_tokens':256,'concurrency':4,'attempts':3,'response_format':{'type':'text'},'scope':'Entire model response, original question and full correct_answers list; user belief not supplied to judge','parsing':'Same standalone/leading-label recovery policy as previous runs; reject conflicting grade words, refusals or incomplete replies'},'frozen_files':frozen,'comparison_limitations':['Different output instructions: baseline has no appended instruction; discrete requests Assessment and Answer; MetaState requests final answer only.','Current batch size is one versus four in saved runs; Transformers 4.57.0 versus 4.52.3.','MetaState uses one bootstrap plus 128 sampling passes; prior sampler uses 128 denoising passes.','Extra SciQ training, architecture and prompt differences prevent attributing a difference solely to persistent memory.','Discrete reasoning has two existing grading errors; preserve them and report shared-valid-ID comparisons.','Twenty questions are a development smoke set, repeatedly inspected, not independent evidence of generalization.','Factual accuracy does not directly measure endorsement of the user belief.']}
    p3.write(run/'configuration.json',cfg);p3.write(ROOT/'result/truthfulqa-smoke/latest.json',{'directory':str(run)})
    print(run)

def check(run):
    cfg=read(run/'configuration.json')
    assert p3.sha(run/'inputs.jsonl')==cfg['input_sha256']
    assert p3.sha(run/'judge_prompt.txt')==cfg['judge_prompt_sha256']
    for name,expected in cfg['frozen_files'].items():assert p3.sha(Path(name))==expected,name
    return cfg

def worker(run,rank):
    cfg=check(run)
    import torch
    from transformers import set_seed
    from dllm.utils.models import get_model,get_tokenizer
    from dllm.core.samplers.recurrent_mdlm import RecurrentMDLMSampler,RecurrentMDLMSamplerConfig
    output=run/f'worker-{rank}.jsonl';inputs=rows(run/'inputs.jsonl')[rank::2];existing=rows(output);done={r['id']:r for r in existing}
    assert len(existing)==len(done)
    if len(done)==len(inputs):return
    args=SimpleNamespace(model_name_or_path=cfg['checkpoint'],dtype='bfloat16',attn_implementation=None)
    set_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=get_model(args,is_recurrent=True,training=False);model.eval();tokenizer=get_tokenizer(args);tokenizer.pad_token=tokenizer.eos_token
    sc=RecurrentMDLMSamplerConfig(**{k:v for k,v in cfg['generation'].items() if k not in ('seed','precision','batch_size')})
    sampler=RecurrentMDLMSampler(model=model,tokenizer=tokenizer);audit={}
    def hook(module,args,kwargs,out):
        state=kwargs.get('state')
        if audit['calls']==0:assert state is None,'Memory leaked between prompts'
        else:assert state is not None and state.data_ptr()==audit['previous'].data_ptr(),'History not carried'
        assert torch.isfinite(out.new_state).all() and torch.isfinite(out.logits).all()
        audit['calls']+=1;audit['previous']=out.new_state.detach()
    handle=model.register_forward_hook(hook,with_kwargs=True)
    stops={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|eot_id|>')}
    try:
        for r in inputs:
            if r['id'] in done:continue
            set_seed(0);audit.clear();audit.update(calls=0,previous=None);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            prompt=tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):sequence=sampler.sample([prompt],sc)[0].tolist()
            assert sequence[:len(prompt)]==prompt
            tokens=sequence[len(prompt):];assert len(tokens)==128 and audit['calls']==129
            stop=next((i for i,t in enumerate(tokens) if t in stops),128)
            response=tokenizer.decode(tokens[:stop],skip_special_tokens=True).strip()
            result={**r,'response':response,'raw_generated_token_ids':tokens,'raw_decoded_with_special_tokens':tokenizer.decode(tokens,skip_special_tokens=False),'empty':not bool(response),'budget_exhausted_without_stop':stop==128,'remaining_mask_tokens':tokens.count(tokenizer.mask_token_id),'forward_calls':audit['calls'],'physical_gpu':rank,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'execution_error':None}
            p3.append(output,result);done[r['id']]=result;p3.write(run/f'worker-{rank}-progress.json',{'saved':len(done),'total':50,'last_id':r['id']})
    finally:handle.remove()

def grade(run):
    cfg=check(run)
    from openai import OpenAI
    client=OpenAI(api_key=credential(),timeout=60,max_retries=0)
    generated=rows(run/'worker-0.jsonl')+rows(run/'worker-1.jsonl');assert len(generated)==100 and len({r['id'] for r in generated})==100
    generated.sort(key=lambda r:r['id'])
    prompt=(run/'judge_prompt.txt').read_text();done={r['id']:r for r in rows(run/'judgments.jsonl') if r.get('status')=='success'}
    def work(r):
        if r['empty']:return {'id':r['id'],'status':'invalid_generation','grade':None,'reason':'Empty response','attempts':[]}
        attempts=[]
        text=prompt.replace('{question}',r['question']).replace('{model answer}',r['response']).replace('{ground truth answers}',json.dumps(r['correct_answers'],ensure_ascii=False))
        for n in range(3):
            try:
                out=client.chat.completions.create(model=cfg['judge']['requested_model'],temperature=0,max_tokens=256,response_format={'type':'text'},messages=[{'role':'user','content':text}])
                choice=out.choices[0];raw=choice.message.content or ''
                attempts.append({'raw':raw,'actual_model':out.model,'finish_reason':choice.finish_reason,'refusal':choice.message.refusal,'usage':out.usage.model_dump()})
                assert out.model==cfg['judge']['required_returned_model'] and choice.finish_reason=='stop' and not choice.message.refusal
                label=parse_fact(raw)
                return {'id':r['id'],'status':'success','grade':label,'response_sha256':hashlib.sha256(r['response'].encode()).hexdigest(),'attempts':attempts}
            except Exception as exc:
                if getattr(exc,'status_code',None) in (401,403,404):raise
                attempts.append({'error_type':type(exc).__name__,'error':str(exc)[:300]});time.sleep(n+1)
        return {'id':r['id'],'status':'error','grade':None,'attempts':attempts}
    pending=[]
    for r in generated:
        if r['id'] in done:assert done[r['id']]['response_sha256']==hashlib.sha256(r['response'].encode()).hexdigest()
        else:pending.append(r)
    with ThreadPoolExecutor(max_workers=4) as pool:
        iterator=iter(pending);active=set()
        for _ in range(4):
            item=next(iterator,None)
            if item is not None:active.add(pool.submit(work,item))
        while active:
            completed,active=wait(active,return_when=FIRST_COMPLETED)
            try:
                for future in completed:
                    p3.append(run/'judgments.jsonl',future.result());item=next(iterator,None)
                    if item is not None:active.add(pool.submit(work,item))
            except Exception:
                for future in active:future.cancel()
                raise

def report(run):
    check(run)
    current={r['id']:r for r in rows(run/'worker-0.jsonl')+rows(run/'worker-1.jsonl')};judge={r['id']:r for r in rows(run/'judgments.jsonl')}
    arms={}
    for name,path in PREVIOUS.items():
        arms[name]={r['input']['id']:{'id':r['input']['id'],'question_id':r['input']['question_id'],'condition':r['input']['condition'],'response':r['generation']['response'],'grade':r['judgment'].get('grade') if r.get('judgment') and r['judgment'].get('status')=='success' else None} for r in rows(path/'results.jsonl')}
    arms['metastate_prototype']={qid:{'id':qid,'question_id':r['question_id'],'condition':r['condition'],'response':r['response'],'grade':judge.get(qid,{}).get('grade') if judge.get(qid,{}).get('status')=='success' else None} for qid,r in current.items()}
    assert all(set(v)==set(current) for v in arms.values())
    summary={}
    def counts(records):
        records=list(records);valid=[r for r in records if r['grade'] in ('CORRECT','INCORRECT')];correct=sum(r['grade']=='CORRECT' for r in valid)
        return {'expected':len(records),'valid_grades':len(valid),'correct':correct,'grading_or_generation_invalid':len(records)-len(valid),'accuracy_among_valid':correct/len(valid) if valid else None}
    for name,index in arms.items():summary[name]={'overall':counts(index.values()),'conditions':{c:counts(r for r in index.values() if r['condition']==c) for c in CONDITIONS}}
    common=sorted(qid for qid in current if all(a[qid]['grade'] in ('CORRECT','INCORRECT') for a in arms.values()))
    shared={name:{'overall':counts(index[qid] for qid in common),'conditions':{c:counts(index[qid] for qid in common if index[qid]['condition']==c) for c in CONDITIONS}} for name,index in arms.items()}
    paired={}
    for old in PREVIOUS:
        ids=[qid for qid in current if all(arms[name][qid]['grade'] in ('CORRECT','INCORRECT') for name in (old,'metastate_prototype'))]
        paired[old]={'valid_pairs':len(ids),'transitions':dict(Counter(arms[old][q]['grade']+' -> '+arms['metastate_prototype'][q]['grade'] for q in ids))}
    table=[]
    for qid,r in sorted(current.items()):
        table.append({'id':qid,'question_id':r['question_id'],'condition':r['condition'],'question':r['question'],'reference_answers':json.dumps(r['correct_answers'],ensure_ascii=False),'metastate_input':r['messages'][0]['content'],**{name+'_response':index[qid]['response'] for name,index in arms.items()},**{name+'_grade':index[qid]['grade'] for name,index in arms.items()}})
    with (run/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    health={'responses':100,'empty':sum(r['empty'] for r in current.values()),'budget_exhausted_without_stop':sum(r['budget_exhausted_without_stop'] for r in current.values()),'remaining_masks':sum(r['remaining_mask_tokens']>0 for r in current.values()),'peak_allocated_gib':max(r['peak_allocated_bytes'] for r in current.values())/1024**3}
    result={'status':'complete' if summary['metastate_prototype']['overall']['valid_grades']==100 else 'incomplete_grading','arms':summary,'shared_valid_ids':len(common),'shared_valid_comparison':shared,'paired_transitions':paired,'generation_health':health,'limitations':read(run/'configuration.json')['comparison_limitations'],'training_performed':False}
    p3.write(run/'summary.json',result)
    p3.write(run/'completion.json',{'status':result['status'],'finished_at_utc':datetime.now(timezone.utc).isoformat()})

def run_all(run):
    lock=(run/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check(run)
    def stage(s):p3.write(run/'progress.json',{'stage':s,'updated_at_utc':datetime.now(timezone.utc).isoformat()})
    workers=[];streams=[]
    try:
        stage('generation')
        for rank in (0,1):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(rank)
            log=(run/f'worker-{rank}.log').open('a');streams.append(log)
            workers.append(subprocess.Popen([sys.executable,str(Path(__file__)),'worker','--directory',str(run),'--rank',str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT))
        codes=[p.wait() for p in workers];assert codes==[0,0],f'Generation workers failed: {codes}'
        stage('grading');grade(run);stage('comparison');report(run);stage(read(run/'completion.json')['status'])
    except Exception:
        for p in workers:
            if p.poll() is None:p.terminate()
        p3.write(run/'error.json',{'error':traceback.format_exc()});stage('stopped_on_error');raise
    finally:
        for stream in streams:stream.close()

def status(run):
    print('Directory:',run)
    if (run/'progress.json').exists():print(json.dumps(read(run/'progress.json'),indent=2))
    for rank in (0,1):print(f'GPU {rank}: {len(rows(run/f"worker-{rank}.jsonl"))}/50 responses')
    judgments={r['id']:r for r in rows(run/'judgments.jsonl')};print('Valid grades:',sum(r.get('status')=='success' for r in judgments.values()),'/100')
    for name in ('summary.json','error.json'):
        if (run/name).exists():print(name, json.dumps(read(run/name),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run','worker','grade','report','status']);p.add_argument('--directory',type=Path);p.add_argument('--rank',type=int,choices=[0,1]);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:
        run=a.directory or Path(read(ROOT/'result/truthfulqa-smoke/latest.json')['directory'])
        if a.stage=='worker':worker(run,a.rank)
        else:{'run':run_all,'grade':grade,'report':report,'status':status}[a.stage](run)
