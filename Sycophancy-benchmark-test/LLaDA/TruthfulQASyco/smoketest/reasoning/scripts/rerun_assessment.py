"""Fresh discrete-reasoning smoke trial with schema-constrained factual grades.

Preserves prior trials, exact Assessment instruction and generation configuration.
Only the judge output transport changes: strict {"grade": enum} replaces prose
parsing. Refusals, incomplete responses and API errors remain explicit failures.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import csv
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import run_assessment as original

common=original.common
ROOT=original.ROOT
SOURCE=ROOT/'result/20261019T102414_955986Z'
META=Path('/root/Thesis/Sycophancy-benchmark-test/MetaState-Latent-Reasoning/My-Metastate/result/truthfulqa-smoke/20261020T165133Z')
SCHEMA={'type':'object','properties':{'grade':{'type':'string','enum':['CORRECT','INCORRECT']}},'required':['grade'],'additionalProperties':False}
FORMAT={'type':'json_schema','json_schema':{'name':'factual_answer_grade','strict':True,'schema':SCHEMA}}


def parse(raw,finish='stop',refusal=None):
    if finish!='stop' or refusal:raise ValueError('Incomplete or refused judge reply')
    value=json.loads(raw)
    if not isinstance(value,dict) or set(value)!= {'grade'} or value['grade'] not in ('CORRECT','INCORRECT'):
        raise ValueError('Invalid constrained grade object')
    return value['grade']


def saved_rows(directory):
    return sorted([r for name in ('belief_prompts.jsonl','neutral_prompts.jsonl') for r in common.prepared.read_jsonl(directory/'inputs'/name)],key=lambda r:(r['source_row_index'],common.CONDITIONS.index(r['condition'])))


def prepare():
    original_rows,source,prompt,files,parsing,smoke,dataset=original.validate()
    previous_rows=saved_rows(SOURCE)
    assert original_rows==previous_rows,'Original prepared inputs changed'
    old=common.prepared.read_json(SOURCE/'manifest.json')
    import importlib.metadata
    assert {k:importlib.metadata.version(k) for k in old['generation_identity']['packages']}==old['generation_identity']['packages'],'Use original diffuguard environment'
    directory=ROOT/'result'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_rerun')
    directory.mkdir(parents=True,exist_ok=False);(directory/'inputs').mkdir()
    for name in ('belief_prompts.jsonl','neutral_prompts.jsonl','dataset_manifest.json'):(directory/'inputs'/name).write_bytes((SOURCE/'inputs'/name).read_bytes())
    for name in ('assessment_instruction.txt','generation_config.json','judge_prompt.txt'):(directory/name).write_bytes((SOURCE/name).read_bytes())
    assert (directory/'assessment_instruction.txt').read_text()==original.INSTRUCTION+'\n'
    judge=copy.deepcopy(old['judge_config']);judge.update(requested_model=common.SNAPSHOT,response_format=FORMAT,output_transport='Strict JSON grade enum; rubric text unchanged')
    identity=copy.deepcopy(old['generation_identity']);identity['runner_source_sha256']=common.digest(Path(__file__).read_bytes())
    frozen={str(p):common.digest(p.read_bytes()) for p in SOURCE.rglob('*') if p.is_file()}
    frozen.update({str(Path(__file__)):common.digest(Path(__file__).read_bytes()),str(Path(original.__file__)):common.digest(Path(original.__file__).read_bytes())})
    frozen.update(common.runtime_sources(files))
    for path in (original.BASELINE/'results.jsonl',META/'comparison.csv',META/'summary.json',META/'judgments.jsonl'):
        frozen[str(path)]=common.digest(path.read_bytes())
    manifest={'run_id':directory.name,'status':'prepared','created_at_utc':common.now(),'expected_records':100,'expected_questions':20,'generation_identity':identity,'generation_config_sha256':common.identity_hash(identity),'chat_template':old['chat_template'],'judge_config':judge,'judge_prompt_sha256':common.digest(prompt),'source_trial':str(SOURCE),'source_artifact_sha256':frozen,'parsing_policy':{'type':'strict_json_schema','schema':SCHEMA,'reject_refusals':True,'require_finish_reason':'stop'},'grading_scope':'Entire Assessment and Answer, with original question and all correct_answers; beliefs not supplied to judge','new_generations':100,'reused_generations':0,'training_performed':False,'limitations':['Judge rubric and returned model unchanged; constrained output format is a recorded protocol change that may affect grades.','Original baseline and MetaState grades remain unchanged in historical comparisons.','Twenty repeatedly used questions are a pipeline/development sample, not independent generalization evidence.']}
    common.save(directory/'manifest.json',manifest);common.save(directory/'judge_config.json',judge);common.save(directory/'judge_schema.json',SCHEMA)
    common.save(ROOT/'result/latest_rerun.json',{'result_dir':str(directory),'run_id':directory.name})
    print(directory,flush=True)
    return directory


def load(directory):
    manifest=common.prepared.read_json(directory/'manifest.json')
    for name,expected in manifest['source_artifact_sha256'].items():
        if common.digest(Path(name).read_bytes())!=expected:raise ValueError('Frozen source changed: '+name)
    if (directory/'assessment_instruction.txt').read_text()!=original.INSTRUCTION+'\n':raise ValueError('Instruction changed')
    for name in ('belief_prompts.jsonl','neutral_prompts.jsonl','dataset_manifest.json'):
        if (directory/'inputs'/name).read_bytes()!=(SOURCE/'inputs'/name).read_bytes():raise ValueError('Input snapshot changed')
    assert common.prepared.read_json(directory/'judge_config.json')==manifest['judge_config']
    assert common.digest((directory/'judge_prompt.txt').read_bytes())==manifest['judge_prompt_sha256']
    assert common.prepared.read_json(directory/'generation_config.json')==manifest['generation_identity']['config']
    return saved_rows(directory),manifest


def judge_one(client,row,response,directory,manifest):
    prompt=(directory/'judge_prompt.txt').read_text()
    text=prompt.replace('{question}',row['question']).replace('{model answer}',response['response']).replace('{ground truth answers}',json.dumps(row['correct_answers'],ensure_ascii=False))
    attempts=[]
    for n in range(3):
        try:
            result=client.chat.completions.create(model=common.SNAPSHOT,temperature=0,max_tokens=256,response_format=FORMAT,messages=[{'role':'user','content':text}])
            choice=result.choices[0];raw=choice.message.content or ''
            attempts.append({'attempt':n+1,'raw_judge_response':raw,'returned_model':result.model,'finish_reason':choice.finish_reason,'refusal':choice.message.refusal,'usage':result.usage.model_dump()})
            if result.model!=common.SNAPSHOT:raise ValueError('Unexpected judge snapshot')
            label=parse(raw,choice.finish_reason,choice.message.refusal)
            return {'id':row['id'],'question_id':row['question_id'],'condition':row['condition'],'status':'success','grade':label,'attempt_details':attempts,'response_sha256':common.digest(response['response'].encode()),'rubric_sha256':manifest['judge_prompt_sha256'],'response_format':FORMAT,'saved_at_utc':common.now()}
        except Exception as exc:
            if getattr(exc,'status_code',None) in (401,403,404):raise
            attempts.append({'error_type':type(exc).__name__,'error':str(exc)[:400]});time.sleep(n+1)
    return {'id':row['id'],'question_id':row['question_id'],'condition':row['condition'],'status':'error','grade':None,'attempt_details':attempts,'saved_at_utc':common.now()}


def client():
    from openai import OpenAI
    return OpenAI(api_key=common.recover_reference()[4].credential(),timeout=60,max_retries=0)


def preflight(directory):
    inputs,manifest=load(directory);judge=client();index={r['id']:r for r in inputs}
    responses=common.latest(SOURCE/'responses.jsonl');prior=common.latest(SOURCE/'judgments.jsonl')
    failed=[qid for qid,j in prior.items() if j.get('status')!='success']
    assert len(failed)==2
    results=[]
    with (directory/'grading_fix_regression.jsonl').open('w') as stream:
        for qid in failed:
            result=judge_one(judge,index[qid],responses[qid],directory,manifest);common.append(stream,result);results.append(result)
    if not all(r['status']=='success' for r in results):raise RuntimeError('Structured grading regression failed; no generation launched')
    repaired={**prior,**{r['id']:r for r in results}}
    good=[r for r in repaired.values() if r.get('status')=='success']
    common.save(directory/'old_trial_repaired_metrics.json',{'source_trial':str(SOURCE),'original_trial_modified':False,'correct':sum(r['grade']=='CORRECT' for r in good),'valid_grades':len(good),'previous_valid_grades':98,'newly_graded_ids':failed,'note':'Retains the original 98 judgments and grades only the two former format failures using constrained JSON; mixed transport protocols are recorded.'})
    common.save(directory/'grading_preflight.json',{'status':'passed','previous_failures_validly_graded':len(results),'model':common.SNAPSHOT})
    print('Two original failures now have valid grades; originals unchanged.',flush=True)


def grade(directory,inputs,manifest):
    api=client();responses=common.latest(directory/'responses.jsonl');previous=common.latest(directory/'judgments.jsonl')
    assert len(responses)==100
    pending=[r for r in inputs if previous.get(r['id'],{}).get('status')!='success']
    with ThreadPoolExecutor(max_workers=4) as pool,(directory/'judgments.jsonl').open('a') as stream:
        iterator=iter(pending);active=set()
        def submit(row):
            response=responses[row['id']]
            if response.get('status')!='success' or not response.get('response'):raise RuntimeError('Generation failure: '+row['id'])
            return pool.submit(judge_one,api,row,response,directory,manifest)
        for _ in range(4):
            row=next(iterator,None)
            if row is not None:active.add(submit(row))
        while active:
            completed,active=wait(active,return_when=FIRST_COMPLETED)
            try:
                for future in completed:
                    common.append(stream,future.result());row=next(iterator,None)
                    if row is not None:active.add(submit(row))
            except Exception:
                for future in active:future.cancel()
                raise


def report(directory,inputs,manifest):
    responses=common.latest(directory/'responses.jsonl');judgments=common.latest(directory/'judgments.jsonl');old_responses=common.latest(SOURCE/'responses.jsonl')
    prior={r['input']['id']:r for r in common.prepared.read_jsonl(original.BASELINE/'results.jsonl')}
    meta={r['id']:r for filename in ('worker-0.jsonl','worker-1.jsonl') for r in common.prepared.read_jsonl(META/filename)};meta_j=common.latest(META/'judgments.jsonl')
    def valid(qid):return judgments.get(qid,{}).get('status')=='success'
    counts=[]
    for condition in common.CONDITIONS:
        group=[r for r in inputs if r['condition']==condition];graded=[r for r in group if valid(r['id'])]
        counts.append({'condition':condition,'expected':20,'valid_grades':len(graded),'correct':sum(judgments[r['id']]['grade']=='CORRECT' for r in graded),'accuracy':sum(judgments[r['id']]['grade']=='CORRECT' for r in graded)/len(graded) if graded else None})
    format_checks={r['id']:original.format_check(responses.get(r['id'],{}).get('response','')) for r in inputs}
    correct=sum(c['correct'] for c in counts);total=sum(c['valid_grades'] for c in counts)
    summary={'status':'complete' if len(responses)==100 and total==100 else 'incomplete','new_generations':len(responses),'correct':correct,'valid_grades':total,'accuracy':correct/total if total else None,'condition_results':counts,'format_compliant':sum(c['compliant'] for c in format_checks.values()),'empty_responses':sum(not r.get('response') for r in responses.values()),'possibly_truncated':sum(bool(r.get('possibly_truncated')) for r in responses.values()),'byte_identical_responses_to_original_trial':sum(responses.get(r['id'],{}).get('response')==old_responses[r['id']]['response'] for r in inputs),'same_output_tokens_as_original_trial':sum(responses.get(r['id'],{}).get('output_token_ids')==old_responses[r['id']]['output_token_ids'] for r in inputs),'original_trial_repaired':common.prepared.read_json(directory/'old_trial_repaired_metrics.json'),'historical_comparison':{'baseline':{'correct':47,'graded':100},'metastate_prototype':{'correct':56,'graded':100},'original_discrete':{'correct':37,'graded':98}},'limitations':manifest['limitations']}
    with (directory/'results.jsonl').open('w') as stream,(directory/'comparison.csv').open('w',newline='') as csvstream:
        writer=csv.DictWriter(csvstream,fieldnames=['id','condition','question','input_message','reference_answers','discrete_response','discrete_grade','old_discrete_response','baseline_response','baseline_grade','metastate_response','metastate_grade','format_compliant','grading_error']);writer.writeheader()
        for r in inputs:
            q=r['id'];generation=responses.get(q,{});judgment=judgments.get(q,{})
            common.append(stream,{'input':r,'generation':generation,'judgment':judgment,'format_check':format_checks[q]})
            writer.writerow({'id':q,'condition':r['condition'],'question':r['question'],'input_message':r['messages'][0]['content'],'reference_answers':json.dumps(r['correct_answers'],ensure_ascii=False),'discrete_response':generation.get('response'),'discrete_grade':judgment.get('grade') if valid(q) else None,'old_discrete_response':old_responses[q]['response'],'baseline_response':prior[q]['generation']['response'],'baseline_grade':prior[q]['judgment']['grade'],'metastate_response':meta[q]['response'],'metastate_grade':meta_j[q]['grade'],'format_compliant':format_checks[q]['compliant'],'grading_error':None if valid(q) else judgment.get('attempt_details')})
    common.save(directory/'metrics.json',summary);common.save(directory/'format_checks.json',format_checks)
    common.progress(directory,manifest,summary['status'],len(responses),total)
    return summary


def run(directory):
    inputs,manifest=load(directory)
    with common.locked(directory):
        failure=None
        try:
            assert common.prepared.read_json(directory/'grading_preflight.json')['status']=='passed'
            common.generate(directory,inputs,manifest)
            import torch
            gc.collect();torch.cuda.empty_cache()
            common.update(directory,manifest,status='grading')
            grade(directory,inputs,manifest)
        except BaseException as exc:
            failure=exc
            common.save(directory/'error.json',{'error':traceback.format_exc()})
        finally:
            metrics=report(directory,inputs,manifest);common.update(directory,manifest,status='failed' if failure else metrics['status'])
        if failure:raise failure
        print(json.dumps({'status':metrics['status'],'correct':metrics['correct'],'valid_grades':metrics['valid_grades']}),flush=True)


def status(directory):
    print('Result directory:',directory)
    for name in ('progress.json','metrics.json','error.json'):
        path=directory/name
        if path.exists():print(name,json.dumps(common.prepared.read_json(path),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','preflight','run','status']);p.add_argument('--result-dir',type=Path);a=p.parse_args()
    if a.command=='prepare':prepare()
    else:
        directory=a.result_dir or Path(common.prepared.read_json(ROOT/'result/latest_rerun.json')['result_dir'])
        {'preflight':preflight,'run':run,'status':status}[a.command](directory)
