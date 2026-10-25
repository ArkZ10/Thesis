"""Freeze a justified variant and verify the reused 100-input development set."""
import argparse,copy,json,csv,sys
from pathlib import Path
import torch
import audit
from diagnostic_engine import Engine,VARIANTS
C=audit.common


def prepare(variant):
    directory=audit.OUT/('verification_'+variant)
    directory.mkdir(exist_ok=False)
    source=C.prepared.read_json(audit.OLD/'control/manifest.json')
    prompt=(audit.OLD/'control/judge_prompt.txt').read_bytes()
    assert C.digest(prompt)==source['judge_prompt_sha256']==C.EXPECTED_PROMPT_SHA256
    identity=copy.deepcopy(source['generation_identity'])
    sources={str(p):C.digest(p.read_bytes()) for p in [Path(__file__).resolve(),audit.ROOT/'scripts/audit.py',audit.ROOT/'scripts/diagnostic_engine.py',audit.ROOT.parent/'scripts/workspace.py',audit.ROOT.parent/'scripts/run_latent.py']}
    sources.update(C.runtime_sources(C.recover_reference()[2]))
    identity.update(arm=variant,workspace_variant=VARIANTS[variant],repair_sources=sources,
        workspace_positions=0 if variant=='control' else 4,feedback_rounds=VARIANTS[variant]['rounds'],
        implementation_sha256=sources,runner_source_sha256=C.digest(Path(__file__).read_bytes()),
        sampler_source_sha256=C.digest((audit.ROOT/'scripts/diagnostic_engine.py').read_bytes()))
    m={'run_id':directory.name,'status':'prepared','created_at_utc':C.now(),'expected_records':100,
        'generation_identity':identity,'generation_config_sha256':C.identity_hash(identity),
        'judge_config':source['judge_config'],'judge_prompt_sha256':C.digest(prompt),'parsing_policy':source['parsing_policy'],
        'source_sha256':sources,'input_sha256':C.digest((audit.OLD/'inputs.jsonl').read_bytes()),
        'variant':variant,'scope':'Development verification: same 100 smoke inputs reused after inspecting failures; not independent generalization evidence.',
        'selection':C.prepared.read_json(audit.OUT/'selection.json')}
    C.prepared.write_atomic(directory/'inputs.jsonl',(audit.OLD/'inputs.jsonl').read_bytes())
    C.prepared.write_atomic(directory/'judge_prompt.txt',prompt)
    C.save(directory/'judge_config.json',m['judge_config'])
    C.save(directory/'generation_config.json',identity['config'])
    C.save(directory/'manifest.json',m)
    return directory


def load(directory):
    m=C.prepared.read_json(directory/'manifest.json')
    for name,sha in m['source_sha256'].items():
        assert C.digest(Path(name).read_bytes())==sha, f'Frozen source changed: {name}'
    assert C.digest((directory/'inputs.jsonl').read_bytes())==m['input_sha256']
    assert C.digest((directory/'judge_prompt.txt').read_bytes())==m['judge_prompt_sha256']==C.EXPECTED_PROMPT_SHA256
    assert C.identity_hash(m['generation_identity'])==m['generation_config_sha256']
    assert C.prepared.read_json(directory/'judge_config.json')==m['judge_config']
    return C.prepared.read_jsonl(directory/'inputs.jsonl'),m


def summary(directory,rows,m):
    responses=C.latest(directory/'responses.jsonl');judgments=C.latest(directory/'judgments.jsonl')
    metrics=C.aggregate(rows,responses,judgments,m)
    correct=sum(x['correct_responses'] for x in metrics['condition_results'])
    attempted=sum(r['id'] in responses for r in rows)
    metrics.update(attempted=attempted,correct_all_attempts_numerator=correct,correct_all_attempts_denominator=100,
        correct_per_all_100=correct/100,correct_per_graded_nonempty=correct/metrics['successfully_graded_responses'] if metrics['successfully_graded_responses'] else None,
        scope=m['scope'])
    C.save(directory/'metrics.json',metrics)
    C.progress(directory,m,m['status'],metrics['successful_generations'],metrics['successfully_graded_responses'],attempted=attempted)
    combined=[{'id':r['id'],'condition':r['condition'],'question':r['question'],'input_message':r['messages'][0]['content'],
        'reference_answers':r['correct_answers'],'response':responses.get(r['id'],{}).get('response'),
        'grade':judgments.get(r['id'],{}).get('grade') if C.good_judgment(judgments.get(r['id'],{}),r,responses.get(r['id'],{}),m) else None,
        'empty':r['id'] in responses and not responses[r['id']].get('response'),
        'first_end_token_index':responses.get(r['id'],{}).get('first_end_token_index')} for r in rows]
    C.prepared.write_atomic(directory/'results.jsonl',C.prepared.jsonl_bytes(combined))
    with (directory/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(combined[0]));writer.writeheader()
        writer.writerows([{**r,'reference_answers':json.dumps(r['reference_answers'],ensure_ascii=False)} for r in combined])
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','all','grade','report'])
    p.add_argument('--variant',choices=list(VARIANTS))
    p.add_argument('--directory',type=Path)
    a=p.parse_args()
    directory=prepare(a.variant) if a.command=='prepare' else a.directory
    if directory is None: p.error('--directory required')
    rows,m=load(directory)
    with C.locked(directory):
        try:
            if a.command=='all':
                audit.smoke.credential()
                previous=C.latest(directory/'responses.jsonl',repair=True)
                pending=[]
                for r in rows:
                    saved=previous.get(r['id'])
                    if saved:
                        assert saved['input_sha256']==C.identity_hash(r) and saved['generation_config_sha256']==m['generation_config_sha256']
                        assert len(saved['output_token_ids'])==128
                    else:pending.append(r)
                if pending:
                    C.update(directory,m,status='generating')
                    engine=Engine(m['generation_identity']['config'],m['generation_identity']['device'],'recurrent',directory)
                    try:
                        encoded=lambda r:len(audit.smoke.encode_messages(engine.tokenizer,r,engine.mask_id,engine.config))
                        pending.sort(key=lambda r:(encoded(r),r['id']))
                        with (directory/'responses.jsonl').open('a') as stream:
                            for i,row in enumerate(pending):
                                record=engine.run_row(row,m['variant'])
                                record.update(generation_config_sha256=m['generation_config_sha256'],seed=0,batch_size=1,
                                    status='success' if record['response'] else 'error',error=None if record['response'] else 'Empty generated answer')
                                C.append(stream,record)
                                print(json.dumps({'attempted':len(previous)+i+1,'expected':100,'nonempty':bool(record['response']),'id':row['id']}),flush=True)
                                summary(directory,rows,m)
                    finally:
                        engines=[engine];del engine;audit.old.release_engines(engines)
            if a.command in ['all','grade']:
                for attempt in range(3):
                    C.grade(directory,rows,m)
                    metrics=summary(directory,rows,m)
                    if metrics['successfully_graded_responses']==metrics['successful_generations']:break
            metrics=summary(directory,rows,m)
            if a.command in ['all','grade']:
                status='finished' if metrics['attempted']==100 and metrics['successfully_graded_responses']==metrics['successful_generations'] else 'incomplete'
                C.update(directory,m,status=status,completed_at_utc=C.now())
                summary(directory,rows,m)
            print(json.dumps({'directory':str(directory),'status':m['status'],'attempted':metrics['attempted'],'nonempty':metrics['successful_generations'],'graded':metrics['successfully_graded_responses'],'correct_per_all_100':metrics['correct_per_all_100']},indent=2))
        except BaseException as error:
            C.update(directory,m,status='failed',error_type=type(error).__name__)
            summary(directory,rows,m)
            raise

if __name__=='__main__':main()
