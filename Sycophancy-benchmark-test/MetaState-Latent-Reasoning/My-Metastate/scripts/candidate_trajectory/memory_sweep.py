"""Tiny configurable paired memory sweep; child processes release GPU memory between scales."""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .events import Events, atomic_json, read_events, replay
from .engine import load_snapshot
from .runner import rows, sha
import torch


def aggregate(run_directories, directory):
    table=[]; responses=[]; reports=[]; invariant_checks={}
    parents={};controls={};noises={}
    for source in run_directories:
        source=Path(source);r=json.loads((source/'report.json').read_text());reports.append(r);epsilon=r['epsilon']
        assert r['upstream_verification_passed'] and r['event_replay_matches_all_final_outputs']
        for qi,prompt in enumerate(r['prompts']):
            pid=prompt['prompt_id'];parent=load_snapshot(source/'snapshots'/f'p{qi}-parent.pt','cpu')
            if pid in parents:
                for k in ('tokens','memory','conditioning'):assert torch.equal(parent[k],parents[pid][k]),('Starting parent changed across scales',pid,k)
            else:parents[pid]=parent
            control=prompt['outcomes'][0]
            if pid in controls:assert control['raw_response']==controls[pid],('Control changed across scales',pid)
            else:controls[pid]=control['raw_response']
            for ci,seed in enumerate((None,11,22,33)):
                initial=load_snapshot(source/'snapshots'/f'p{qi}-memory-c{ci}-initial.pt','cpu')
                assert torch.equal(initial['tokens'],parent['tokens']) and torch.equal(initial['conditioning'],parent['conditioning'])
                if ci:
                    direction_key=(pid,ci)
                    # Same requested FP32 direction seed at every scale; BF16 rounding differs.
                    state=initial['memory_intervention_metadata'];assert state['perturbation_seed']==seed
                    if direction_key in noises:assert noises[direction_key]==seed
                    noises[direction_key]=seed
            changed=[p for p in prompt['outcomes'] if p['candidate_id']>0]
            first=[p for p in prompt['first_pass_comparisons'] if p['candidate_id']>0]
            table.append({'epsilon':epsilon,'prefix_steps':r.get('prefix_steps',8),'prompt_id':pid,'question':prompt['question'],'changed_final_responses':sum(p['final_answer_changed_vs_control'] for p in changed),'perturbed_candidates':len(changed),'distinct_answers_including_control':len(set(p['raw_response'] for p in prompt['outcomes'])),'first_pass_changed_greedy_candidates':sum(p['greedy_token_changes_all_masked_positions']>0 for p in first),'mean_first_pass_total_variation':sum(p['mean_total_variation'] for p in first)/len(first),'empty':sum(p['empty'] for p in prompt['outcomes']),'invalid':sum(p['invalid'] for p in prompt['outcomes']),'residual_mask_outputs':sum(p['residual_mask_tokens']>0 for p in prompt['outcomes']),'source_directory':str(source)})
        for record in rows(source/'results.jsonl'):responses.append({**record,'sweep_epsilon':epsilon,'source_directory':str(source)})
    invariant_checks={'identical_parent_tokens_memory_conditioning_across_scales':True,'identical_control_answers_across_scales':True,'identical_direction_seeds_across_scales':True,'all_child_sampler_checks_passed':True,'all_child_event_replays_passed':True}
    by_scale=[]
    for epsilon in sorted({r['epsilon'] for r in table}):
        entries=[r for r in table if r['epsilon']==epsilon]
        by_scale.append({'epsilon':epsilon,'questions':len(entries),'changed_final_responses':sum(r['changed_final_responses'] for r in entries),'perturbed_candidates':sum(r['perturbed_candidates'] for r in entries),'questions_with_changed_response':sum(r['changed_final_responses']>0 for r in entries),'first_pass_changed_greedy_candidates':sum(r['first_pass_changed_greedy_candidates'] for r in entries),'empty':sum(r['empty'] for r in entries),'invalid':sum(r['invalid'] for r in entries),'residual_mask_outputs':sum(r['residual_mask_outputs'] for r in entries)})
    report={'experiment':'paired memory-only perturbation scale sweep','temperature':0,'prefix_steps':reports[0].get('prefix_steps',8),'scales':by_scale,'per_question':table,'verification':invariant_checks,'completed_responses_including_repeated_controls':len(responses),'actual_backbone_calls_including_all_verification':sum(r['actual_backbone_calls_including_verification'] for r in reports),'semantic_correctness':'NOT_GRADED','false_belief_endorsement':'NOT_GRADED','usefulness_for_search':'UNASSESSED: literal wording diversity is not a meaningful-alternative or correctness measure. No evaluator/selector trained or called.','limitations':['Three development questions: debugging pilot, not a generalizable estimate.','Random off-training memory perturbations may degrade answers or introduce distribution shift.','Same directions across scales; BF16 rounding means actual directions are approximately, not exactly, scaled copies.','One deterministic completion per memory; no MCTS or reward-based selection.']}
    atomic_json(directory/'report.json',report)
    for filename,records in (('comparison.csv',table),('responses.csv',responses)):
        keys=sorted(set().union(*(r.keys() for r in records)))
        with (directory/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
            for r in records:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    with (directory/'responses.jsonl').open('w') as f:
        for r in responses:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return report


def main(root):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prompts',type=int,default=3);parser.add_argument('--scales',type=float,nargs='+',default=[.01,.03,.05]);parser.add_argument('--prefix-steps',type=int,choices=[0,8],default=8);args=parser.parse_args()
    if not 1<=args.prompts<=3:raise ValueError('Tiny sweep supports 1..3 development questions')
    if len(args.scales)>3 or len(set(args.scales))!=len(args.scales) or any(not 0<x<=.3 for x in args.scales):raise ValueError('Use 1..3 distinct scales in (0,0.3]')
    directory=root/'result/memory-interventions/sweeps'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');directory.mkdir(parents=True,exist_ok=False)
    events=Events(directory);scales=args.scales;run_dirs=[];total_calls=0
    config={'scales':scales,'prefix_steps':args.prefix_steps,'prompts':args.prompts,'temperature':0,'same_noise_seeds':[11,22,33],'control':'unchanged memory at each scale','execution':'sequential','no_training':True,'no_paid_grading':True}
    atomic_json(directory/'config.json',config)
    atomic_json(directory/'manifest.json',{'code_sha256':{str(p.relative_to(root)):sha(p) for p in (root/'scripts/candidate_trajectory').glob('*.py')},'child_runs':[]})
    atomic_json(root/'result/memory-interventions/sweeps/latest.json',{'directory':str(directory)})
    atomic_json(root/'result/candidate-trajectories/latest.json',{'directory':str(directory),'experiment':'memory-only scale sweep'})
    events.emit('run_metadata',run_id=directory.name,configuration=config,experiment='MEMORY-ONLY SCALE SWEEP: sensitivity, not proven reasoning',cancel_support=False)
    print('SWEEP_DIRECTORY='+str(directory),flush=True)
    child=None
    try:
        for epsilon in scales:
            pointer=root/'result/memory-interventions/latest.json';old=json.loads(pointer.read_text())['directory'] if pointer.exists() else None
            logpath=directory/f'scale-{epsilon:.2f}.log';source=None;seen=0
            events.emit('scale_queued',sweep_epsilon=epsilon,status='pending')
            with logpath.open('w') as log:
                child=subprocess.Popen([sys.executable,str(root/'scripts/memory_intervention.py'),'--epsilon',str(epsilon),'--prompts',str(args.prompts),'--prefix-steps',str(args.prefix_steps)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
                while True:
                    if source is None and pointer.exists():
                        current=json.loads(pointer.read_text())['directory']
                        if current!=old:source=Path(current)
                    if source is not None:
                        source_events=read_events(source/'events.jsonl')
                        for record in source_events[seen:]:
                            fields=dict(record);kind=fields.pop('event');fields['source_sequence']=fields.pop('sequence');fields['sweep_epsilon']=epsilon;fields['source_run_directory']=str(source)
                            if fields.get('trajectory_id'):fields['trajectory_id']=f'eps{epsilon:.2f}::'+fields['trajectory_id']
                            if 'actual_backbone_calls' in fields:fields['actual_backbone_calls']+=total_calls
                            if kind=='run_metadata':kind='scale_metadata'
                            elif kind=='run_completed':kind='scale_completed'
                            events.emit(kind,**fields)
                        seen=len(source_events)
                    code=child.poll()
                    if code is not None:
                        if source is not None and seen<len(read_events(source/'events.jsonl')):continue
                        break
                    time.sleep(.25)
            if code!=0 or source is None:raise RuntimeError(f'Scale {epsilon} failed; inspect {logpath}')
            r=json.loads((source/'report.json').read_text());total_calls+=r['actual_backbone_calls_including_verification'];run_dirs.append(str(source))
            atomic_json(directory/'child-runs.json',run_dirs)
            print(f'SCALE_COMPLETED epsilon={epsilon} responses={r["n_completed"]} directory={source}',flush=True)
        report=aggregate(run_dirs,directory)
        replay(read_events(events.path))
        events.emit('run_completed',status='completed',actual_backbone_calls=total_calls,completed_responses=report['completed_responses_including_repeated_controls'])
        atomic_json(root/'result/candidate-trajectories/latest.json',{'directory':str(directory),'experiment':'memory-only scale sweep'})
        print(json.dumps({'directory':str(directory),'scales':report['scales'],'backbone_calls':total_calls},indent=2),flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
        events.emit('run_failed',error=repr(exc));raise
