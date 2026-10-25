"""Regrade saved smoke outputs consistently; no generation or training."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import traceback
import rerun_assessment as judge

C=judge.common
SMOKE=judge.ROOT.parent
NEW_DISCRETE=Path(C.prepared.read_json(judge.ROOT/'result/latest_rerun.json')['result_dir'])
SOURCES={
 'baseline':SMOKE/'runs/20261018T041005_967364Z_regrade/responses.jsonl',
 'discrete_reasoning':NEW_DISCRETE/'responses.jsonl',
 'answer_only_control':SMOKE/'latent-reasoning/result/20261019T111844_710802Z/control/responses.jsonl',
 'repaired_latent_workspace':SMOKE/'latent-reasoning/repair/result/verification_recurrent_before/responses.jsonl',
 'metastate_prototype':judge.META,
}

def read(p):return C.prepared.read_json(p)
def rows(p):return C.prepared.read_jsonl(p) if p.exists() else []
def response_rows(name):
 p=SOURCES[name]
 return [r for filename in ('worker-0.jsonl','worker-1.jsonl') for r in rows(p/filename)] if name=='metastate_prototype' else rows(p)

def prepare():
 run=judge.ROOT/'result'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_compare_all');run.mkdir()
 source_rows=judge.saved_rows(judge.SOURCE);expected={r['id'] for r in source_rows}
 (run/'judge_prompt.txt').write_bytes((NEW_DISCRETE/'judge_prompt.txt').read_bytes())
 frozen={str(Path(__file__)):C.digest(Path(__file__).read_bytes()),str(Path(judge.__file__)):C.digest(Path(judge.__file__).read_bytes())}
 (run/'inputs.jsonl').write_bytes(C.prepared.jsonl_bytes(source_rows))
 for name,p in SOURCES.items():
  records=response_rows(name)
  assert len(records)==100 and {r['id'] for r in records}==expected,name
  files=[p/'worker-0.jsonl',p/'worker-1.jsonl'] if name=='metastate_prototype' else [p]
  for file in files:frozen[str(file)]=C.digest(file.read_bytes())
  target=run/name;target.mkdir();(target/'responses.jsonl').write_bytes(C.prepared.jsonl_bytes(records))
 frozen[str(NEW_DISCRETE/'judgments.jsonl')]=C.digest((NEW_DISCRETE/'judgments.jsonl').read_bytes())
 reused=C.latest(NEW_DISCRETE/'judgments.jsonl')
 assert len(reused)==100 and all(r['status']=='success' and r['response_format']==judge.FORMAT for r in reused.values())
 current={r['id']:r for r in rows(run/'discrete_reasoning/responses.jsonl')}
 for qid,r in reused.items():
  assert r['response_sha256']==C.digest(current[qid]['response'].encode())
  assert r['rubric_sha256']==C.digest((run/'judge_prompt.txt').read_bytes())
  assert r['grade']==judge.parse(r['attempt_details'][-1]['raw_judge_response'],r['attempt_details'][-1]['finish_reason'],r['attempt_details'][-1]['refusal'])
 (run/'discrete_reasoning/judgments.jsonl').write_bytes(C.prepared.jsonl_bytes(list(reused.values())))
 manifest={'status':'prepared','expected_records_per_arm':100,'arms':list(SOURCES),'sources':{k:str(v) for k,v in SOURCES.items()},'source_artifact_sha256':frozen,'judge_prompt_sha256':C.digest((run/'judge_prompt.txt').read_bytes()),'judge_model':C.SNAPSHOT,'temperature':0,'max_tokens':256,'response_format':judge.FORMAT,'reused_grades':{'discrete_reasoning':100},'generation_performed':False,'training_performed':False,'limitations':['Consistent grading does not remove differences in prompts, batch sizes, model execution, or SciQ training.','These 20 repeatedly inspected questions are not an independent generalization test.','Factual accuracy is not a direct measure of user-belief endorsement.','No trained memory-reset or untrained MetaState comparison was completed; memory contribution is not isolated.']}
 C.save(run/'manifest.json',manifest);C.save(judge.ROOT/'result/latest_comparison.json',{'result_dir':str(run)})
 print(run)

def verify(run):
 m=read(run/'manifest.json')
 for name,value in m['source_artifact_sha256'].items():assert C.digest(Path(name).read_bytes())==value,name
 assert C.digest((run/'judge_prompt.txt').read_bytes())==m['judge_prompt_sha256']
 return m

def grade(run):
 import fcntl
 lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m=verify(run);index={r['id']:r for r in rows(run/'inputs.jsonl')};client=judge.client()
 pending=[]
 for name in SOURCES:
  responses=rows(run/name/'responses.jsonl');done=C.latest(run/name/'judgments.jsonl')
  for response in responses:
   previous=done.get(response['id'],{})
   if previous.get('status')=='success':
    assert previous['response_sha256']==C.digest(response['response'].encode())
   else:pending.append((name,response))
 C.save(run/'progress.json',{'stage':'grading','new_judgments_expected':len(pending)})
 def work(item):
  name,response=item
  return name,judge.judge_one(client,index[response['id']],response,run,m)
 with ThreadPoolExecutor(max_workers=4) as pool:
  iterator=iter(pending);active=set()
  for _ in range(4):
   item=next(iterator,None)
   if item is not None:active.add(pool.submit(work,item))
  saved=0
  while active:
   finished,active=wait(active,return_when=FIRST_COMPLETED)
   try:
    for future in finished:
     name,result=future.result()
     with (run/name/'judgments.jsonl').open('a') as f:C.append(f,result)
     saved+=1
     C.save(run/'progress.json',{'stage':'grading','new_judgments_saved':saved,'new_judgments_expected':len(pending)})
     item=next(iterator,None)
     if item is not None:active.add(pool.submit(work,item))
   except Exception:
    for future in active:future.cancel()
    raise
 report(run)

def report(run):
 import numpy as np
 m=verify(run);inputs={r['id']:r for r in rows(run/'inputs.jsonl')};metrics={};combined={};table=[]
 for name in SOURCES:
  responses={r['id']:r for r in rows(run/name/'responses.jsonl')};grades=C.latest(run/name/'judgments.jsonl');values={}
  for qid,row in inputs.items():
   j=grades.get(qid,{})
   values[qid]=j['grade'] if j.get('status')=='success' else None
   table.append({'method':name,'id':qid,'question_id':row['question_id'],'condition':row['condition'],'question':row['question'],'reference_answers':json.dumps(row['correct_answers'],ensure_ascii=False),'response':responses[qid]['response'],'grade':values[qid]})
  valid=sum(v is not None for v in values.values());correct=sum(v=='CORRECT' for v in values.values())
  metrics[name]={'correct':correct,'valid_grades':valid,'expected':100,'accuracy':correct/valid if valid else None,'invalid':100-valid,'conditions':{c:{'correct':sum(values[q]=='CORRECT' for q,r in inputs.items() if r['condition']==c),'valid':sum(values[q] is not None for q,r in inputs.items() if r['condition']==c),'expected':20} for c in C.CONDITIONS}}
  combined[name]=values
 shared=[qid for qid in inputs if all(v[qid] is not None for v in combined.values())]
 paired={}
 for a,b in [('metastate_prototype','baseline'),('metastate_prototype','discrete_reasoning'),('metastate_prototype','answer_only_control'),('metastate_prototype','repaired_latent_workspace'),('repaired_latent_workspace','answer_only_control')]:
  ids=[q for q in inputs if combined[a][q] is not None and combined[b][q] is not None]
  transitions=Counter(combined[b][q]+' -> '+combined[a][q] for q in ids)
  grouped={}
  for q in ids:
   base=inputs[q]['question_id'];grouped.setdefault(base,[]).append(int(combined[a][q]=='CORRECT')-int(combined[b][q]=='CORRECT'))
  if grouped:
   deltas=np.array([sum(v)/len(v) for _,v in sorted(grouped.items())]);rng=np.random.default_rng(42);boot=deltas[rng.integers(0,len(deltas),size=(5000,len(deltas)))].mean(axis=1)
   interval=np.quantile(boot,[.025,.975]).tolist()
  else:interval=None
  paired[a+' minus '+b]={'valid_pairs':len(ids),'transitions':dict(transitions),'accuracy_difference_points':100*(sum(combined[a][q]=='CORRECT' for q in ids)-sum(combined[b][q]=='CORRECT' for q in ids))/len(ids) if ids else None,'question_cluster_bootstrap_95_difference_points':[100*x for x in interval] if interval else None}
 result={'status':'complete' if len(shared)==100 else 'incomplete_grading','shared_valid_prompt_ids':len(shared),'judge_model':C.SNAPSHOT,'response_format':'strict_json_schema','metrics':metrics,'paired_comparisons':paired,'limitations':m['limitations'],'generation_performed':False,'training_performed':False}
 C.save(run/'summary.json',result)
 with (run/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 with (run/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['method','correct','valid_grades','accuracy']);w.writeheader();w.writerows({'method':name,**{k:values[k] for k in ('correct','valid_grades','accuracy')}} for name,values in metrics.items())
 C.save(run/'progress.json',{'stage':result['status'],'valid_grades':{name:v['valid_grades'] for name,v in metrics.items()}})
 print(json.dumps(result,indent=2))

def status(run):
 print('Result directory:',run)
 for name in SOURCES:
  grades=C.latest(run/name/'judgments.jsonl');print(name,sum(v.get('status')=='success' for v in grades.values()),'/100')
 for file in ('summary.json','error.json'):
  if (run/file).exists():print(file,json.dumps(read(run/file),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','grade','report','status']);p.add_argument('--result-dir',type=Path);a=p.parse_args()
 if a.command=='prepare':prepare()
 else:
  run=a.result_dir or Path(read(judge.ROOT/'result/latest_comparison.json')['result_dir'])
  try:{'grade':grade,'report':report,'status':status}[a.command](run)
  except Exception:
   C.save(run/'error.json',{'error':traceback.format_exc()});raise
