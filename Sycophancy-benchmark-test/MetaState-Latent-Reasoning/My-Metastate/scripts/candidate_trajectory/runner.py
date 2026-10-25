"""Small sequential feasibility experiment; no training, grading API, or selector."""
import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from datetime import datetime,timezone
from .events import Events,atomic_json,read_events,replay


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def append(path,record):
    with Path(path).open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []


def proxy(answer,target):
    import re
    normalize=lambda s:' '.join(re.sub(r'[^\w\s]',' ',(s or '').casefold()).split())
    return bool(answer) and normalize(answer)==normalize(target)


def interval(values):
    n=len(values);p=sum(values)/n if n else 0;z=1.96;den=1+z*z/n if n else 1
    center=(p+z*z/(2*n))/den if n else 0;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den if n else 0
    return {'successes':sum(values),'n':n,'rate':p if n else None,'wilson_95':[max(0,center-half),min(1,center+half)] if n else None,'caution':'Descriptive exact-match proxy only; repeated dependent rollouts are not independent factual-accuracy trials.'}


def verify(engine,directory,messages):
    """Real model checks. Additional calls explicitly excluded from study budget."""
    import torch
    from .engine import clone,save_snapshot,load_snapshot,seeded_rng,use_rng
    from dllm.core.samplers.recurrent_mdlm import RecurrentMDLMSamplerConfig
    events=engine.events;start_calls=engine.calls;checks={};times={}
    initial=engine.initialize(messages,99,'verify-initial');save_snapshot(directory/'snapshots/verify-initial.pt',initial)
    a=clone(initial);a['trajectory_id']='verify-monitored';torch.cuda.synchronize();t=time.perf_counter();engine.advance(a,context={'phase':'verification'},monitor=True);torch.cuda.synchronize();times['monitor_enabled_seconds']=time.perf_counter()-t
    b=load_snapshot(directory/'snapshots/verify-initial.pt',engine.device);torch.cuda.synchronize();t=time.perf_counter();engine.advance(b,monitor=False);torch.cuda.synchronize();times['monitor_disabled_seconds']=time.perf_counter()-t
    checks['monitoring_same_tokens']=torch.equal(a['tokens'],b['tokens']);checks['monitoring_same_memory']=torch.equal(a['memory'],b['memory'])
    c=load_snapshot(directory/'snapshots/verify-initial.pt',engine.device);engine.advance(c,engine.config['prefix_steps'],monitor=False);save_snapshot(directory/'snapshots/verify-mid.pt',c);c=load_snapshot(directory/'snapshots/verify-mid.pt',engine.device);engine.advance(c,monitor=False)
    checks['snapshot_resume_tokens_exact']=torch.equal(a['tokens'],c['tokens']);checks['snapshot_resume_memory_exact']=torch.equal(a['memory'],c['memory']);checks['snapshot_resume_conditioning_exact']=torch.equal(a['conditioning'],c['conditioning'])
    p=load_snapshot(directory/'snapshots/verify-mid.pt',engine.device);p0=clone(p);outputs={}
    for order in ((11,22),(22,11)):
        for seed in order:
            child=clone(p);child['rng']=seeded_rng(seed,engine.device);engine.advance(child,engine.config['branch_steps'],monitor=False)
            if seed in outputs:checks[f'branch_order_seed_{seed}']=all(torch.equal(outputs[seed][k],child[k]) for k in ('tokens','memory','conditioning'))
            else:outputs[seed]=clone(child)
    checks['parent_immutable']=all(torch.equal(p[k],p0[k]) for k in ('tokens','memory','conditioning'))
    isolated=clone(p);isolated['memory'].add_(1);isolated['conditioning'].add_(1);isolated['tokens'][0,0]=0
    checks['deep_clone_isolation']=all(torch.equal(p[k],p0[k]) and p[k].data_ptr()!=isolated[k].data_ptr() for k in ('tokens','memory','conditioning'))
    checks['memory_precedes_new_tokens']=a['memory_source_token_version']==a['token_version']-1
    fresh=engine.initialize(messages,99,'verify-reset');checks['fresh_prompt_reset']=all(torch.equal(fresh[k],initial[k]) for k in ('tokens','memory','conditioning'))
    # A second independent prompt must enter bootstrap with both carried values None.
    original=engine._forward;observed=[]
    def wrapped(s,bootstrap=False):
        if bootstrap:observed.append(s['memory'] is None and s['conditioning'] is None)
        return original(s,bootstrap)
    engine._forward=wrapped
    engine.initialize([{'role':'user','content':'What is the chemical symbol for oxygen?'}],100,'verify-other-prompt')
    engine._forward=original;checks['other_prompt_no_carried_state']=observed==[True]
    cfg=RecurrentMDLMSamplerConfig(max_new_tokens=engine.config['generation_length'],steps=engine.config['steps'],block_size=engine.config['block_size'],temperature=engine.config['temperature'],warmup=True,use_cache=None,return_dict=False)
    ids=engine.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True);rng=seeded_rng(99,engine.device);count=[0]
    hook=engine.model.register_forward_hook(lambda *args:count.__setitem__(0,count[0]+1))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),use_rng(rng,engine.device):upstream=engine.schedule.sample([ids],config=cfg)
    hook.remove();engine.calls+=count[0]
    checks['upstream_sampler_token_parity']=torch.equal(a['tokens'],upstream)
    recorded=read_events(events.path);restored=replay(recorded);checks['real_event_replay']=restored['verify-monitored']==a['tokens'][0].tolist();checks['events_flushed_before_completion']=any(e['event']=='generation_progress' and e['completed_steps']<engine.config['steps'] for e in recorded)
    result={'kind':'REAL_GPU_CHECKS','passed':all(checks.values()),'checks':checks,'tolerance':'Exact tensor equality (zero tolerance)','backbone_calls':engine.calls-start_calls,'timings':times,'monitoring_elapsed_difference_seconds':times['monitor_enabled_seconds']-times['monitor_disabled_seconds'],'overhead_caution':'Single sequential timing pair, cold/warm order bias; includes event formatting, flushing and decoding but is not a reliable performance estimate. Numerical diagnostics are common to both paths.'}
    events.emit('trajectory_status',trajectory_id='verify-monitored',phase='verification',status='completed',actual_backbone_calls=engine.calls)
    atomic_json(directory/'verification.json',result);events.emit('verification_completed',**result,actual_backbone_calls=engine.calls)
    if not result['passed']:raise RuntimeError('Real GPU verification failed; see verification.json')
    return result


def summarize(directory,config):
    results=rows(directory/'results.jsonl');offline=rows(directory/'offline-references.jsonl');refs={x['id']:x for x in offline};report={'semantic_grading':'NOT_RUN: existing correctness grader requires paid API; no existing endorsement grader available. Exact match below is a separate offline proxy, not factual accuracy or endorsement.','reference_free_selection':'UNIMPLEMENTED: requires an independently validated reference-free evaluator.','boundary_semantics':'Forward reads tokens T[v] and prior memory/conditioning; returned memory is computed from T[v]. Sampling then creates T[v+1], which has not yet been processed into that memory.','ablation_caution':'Replacing child memory AND conditioning with parent values is a post-training intervention and may introduce distribution shift.','prompts':[]}
    for ref in offline:
        rr=[r for r in results if r['prompt_id']==ref['id']];modes={mode:interval([proxy(r.get('extracted_final_answer'),ref['correct_answer']) for r in rr if r['mode']==mode]) for mode in ('persistent','parent_memory_ablation','independent')}
        distributions=[]
        for c in range(4):
            sample=[r for r in rr if r['mode']=='persistent' and r['candidate_id']==c];distributions.append({'candidate_id':c,'offline_exact_match_proxy':interval([proxy(r.get('extracted_final_answer'),ref['correct_answer']) for r in sample]),'answers':[r.get('extracted_final_answer') for r in sample]})
        select=[r for r in rr if r['mode']=='persistent' and r['rollout_id']==0];random_c=random.Random(config['selection_seed']).randrange(4)
        oracle=max(range(4),key=lambda c:sum(proxy(r.get('extracted_final_answer'),ref['correct_answer']) for r in select if r['candidate_id']==c))
        def held(c):return interval([proxy(r.get('extracted_final_answer'),ref['correct_answer']) for r in rr if r['mode']=='persistent' and r['candidate_id']==c and r['rollout_id']>0])
        matched=[]
        for r in rr:
            if r['mode']!='persistent':continue
            ab=next((a for a in rr if a['mode']=='parent_memory_ablation' and a['candidate_id']==r['candidate_id'] and a['rollout_id']==r['rollout_id']),None)
            if ab:matched.append({'candidate_id':r['candidate_id'],'rollout_id':r['rollout_id'],'answer_changed':ab.get('extracted_final_answer')!=r.get('extracted_final_answer'),'token_canvas_changed':ab.get('tokens_sha256')!=r.get('tokens_sha256')})
        primary=sum(r['incremental_backbone_calls'] for r in rr if r['mode']=='persistent')+sum(e['branch_incremental_calls'] for e in read_events(directory/'events.jsonl') if e['event']=='branch_created' and e['prompt_id']==ref['id'])+next(e['shared_prefix_calls'] for e in read_events(directory/'events.jsonl') if e['event']=='parent_saved' and e['prompt_id']==ref['id'])
        independent=sum(r['incremental_backbone_calls'] for r in rr if r['mode']=='independent');ablation=sum(r['incremental_backbone_calls'] for r in rr if r['mode']=='parent_memory_ablation')
        report['prompts'].append({'prompt_id':ref['id'],'n_results_including_failures':len(rr),'failed':sum(r.get('status')=='failed' for r in rr),'offline_proxies':modes,'candidate_distributions':distributions,'random_selection':{'candidate_id':random_c,'held_out_rollouts':held(random_c)},'reference_proxy_selection':{'candidate_id':oracle,'selection_rollout':0,'evaluation_rollouts':list(range(1,config['rollouts'])),'held_out_rollouts':held(oracle),'caution':'Offline reference-based proxy selection; not deployable or validated oracle factual correctness.'},'hindsight_reference_proxy_upper_bound_caution':'Optimistically chosen and assessed on the same evaluation rollouts; offline proxy only.', 'hindsight_reference_proxy_upper_bound':max((held(c)['rate'] or 0) for c in range(4)),'ablation_matched_outcomes':matched,'duplicate_final_answer_count':len([r for r in rr if r['mode']=='persistent'])-len(set(r.get('extracted_final_answer') for r in rr if r['mode']=='persistent')),'measured_backbone_calls':{'branching_primary':primary,'independent':independent,'ablation_separate':ablation},'completion_latency_seconds_excluding_branch_setup':{m:sum(r.get('latency_seconds',0) for r in rr if r['mode']==m) for m in modes}})
    events=read_events(directory/'events.jsonl');report['total_backbone_calls_including_verification_and_retries']=max((e.get('actual_backbone_calls',0) for e in events),default=0);report['recorded_event_replay_verified']=bool(replay(events));report['limitations']=['Small configurable development pilot; current prompt count: '+str(len(offline)),'Temperature zero can make all seeded branches identical.','Memory distance does not establish distinct reasoning.','No semantic correctness or endorsement scores; no reliable selector demonstrated.','Failed and invalid completions remain in proxy denominators.']
    report['oracle_factual_correctness_selection']='UNIMPLEMENTED without factual correctness grades. Reference exact-match proxy upper bounds below must not be reported as semantic oracle accuracy.'
    report['monitoring_overhead']={'event_write_seconds_across_current_session':next((e.get('event_write_seconds') for e in reversed(events) if e['event']=='run_completed'),None),'formatting_decoding_write_seconds_across_current_session':next((e.get('observer_formatting_seconds') for e in reversed(events) if e['event']=='run_completed'),None),'real_timing_check':json.loads((directory/'verification.json').read_text()).get('timings') if (directory/'verification.json').exists() else None,'caution':'Formatting timer includes event writes, so do not add these counters. Forward diagnostics are common to monitor-on and monitor-off checks. No added model calls for visualization.'}
    report['total_latency_seconds_including_branch_setup']={m:sum(r.get('latency_seconds',0) for r in results if r['mode']==m) for m in ('persistent','parent_memory_ablation','independent')}
    report['total_latency_seconds_including_branch_setup']['persistent']+=sum(e.get('latency_seconds',0) for e in events if e['event'] in ('parent_saved','branch_created'))
    atomic_json(directory/'report.json',report)
    if not all('latency_seconds' in e for e in events if e['event'] in ('parent_saved','branch_created')):
        report['total_latency_seconds_including_branch_setup']['persistent']=None
        report['total_latency_seconds_including_branch_setup']['limitation']='Earlier instrumentation omitted prefix/branch latency; completion-only latency is reported separately.'
    if results:
        keys=sorted(set().union(*(r.keys() for r in results)))
        with (directory/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(results)
    return report


def run(root,args):
    import torch
    from .engine import Engine,clone,save_snapshot,load_snapshot,seeded_rng,distance,token_hash
    directory=Path(args.resume).resolve() if args.resume else root/'result/candidate-trajectories'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory.mkdir(parents=True,exist_ok=True);events=Events(directory);atomic_json(root/'result/candidate-trajectories/latest.json',{'directory':str(directory)})
    if args.resume:config=json.loads((directory/'config.json').read_text());selected=rows(directory/'offline-references.jsonl')
    else:
        n=args.tokens;b=args.block_size;steps=args.steps
        if min(n,b,steps,args.event_every,args.diagnostic_every,args.token_snapshot_every,args.prompts,args.prefix_steps,args.branch_steps)<=0:raise ValueError('Sizes and event frequencies must be positive')
        if n%b or steps%(n//b):raise ValueError('Tokens and steps must form whole decoding blocks')
        per=steps//(n//b)
        if args.prefix_steps%per or args.branch_steps%per or args.prefix_steps+args.branch_steps>=steps:raise ValueError('Prefix/branch steps must end at block boundaries and leave rollout budget')
        if args.rollouts<2:raise ValueError('Need at least two rollouts to separate selection and evaluation')
        config={'generation_length':n,'steps':steps,'block_size':b,'temperature':args.temperature,'prefix_steps':args.prefix_steps,'branch_steps':args.branch_steps,'branch_seeds':[11,22,33,44],'rollout_seeds':list(range(101,101+args.rollouts)),'rollouts':args.rollouts,'selection_seed':42,'event_every':args.event_every,'diagnostic_every':args.diagnostic_every,'token_snapshot_every':args.token_snapshot_every,'max_backbone_calls':args.max_backbone_calls,'backbone_path':json.loads((root/'setup/backbone.json').read_text())['backbone_name_or_path'],'checkpoint':str(root/'result/phase3/20261020T162646Z/checkpoints/checkpoint-20'),'precision':'bfloat16','caching':False,'warmup':True,'remasking':'low_confidence','schedule':'existing default Linear alpha schedule','device_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'sequential_execution':True}
        tiny={r['question_id'] for r in rows(root/'data/processed/tiny/development.jsonl')} if (root/'data/processed/tiny/development.jsonl').exists() else {'sciq-validation-00003','sciq-validation-00006','sciq-validation-00011','sciq-validation-00025'}
        source=root/'data/processed/development.jsonl';available=[r for r in rows(source) if r['condition']==args.condition and r['question_id'] not in tiny];selected=available[:args.prompts]
        if len(selected)!=args.prompts:raise ValueError('Not enough independent development prompts')
        # Persist references only in separate offline metadata. Only user messages go to generation.
        for r in selected:append(directory/'offline-references.jsonl',r)
        atomic_json(directory/'config.json',config)
        preserved=json.loads((root/'setup/manifest.json').read_text())['source']['file_sha256']
        for rel,expected in preserved.items():
            if sha(root/'vendor/MetaState'/rel)!=expected:raise RuntimeError('Vendor source differs from recorded commit: '+rel)
        manifest={'run_id':directory.name,'source_sha256':sha(source),'checkpoint_weights_sha256':sha(Path(config['checkpoint'])/'recurrent_modules.safetensors'),'checkpoint_metadata':json.loads((Path(config['checkpoint'])/'recurrent_checkpoint_meta.json').read_text()),'backbone':json.loads((root/'setup/backbone.json').read_text()),'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','safetensors')},'vendor_commit':json.loads((root/'setup/manifest.json').read_text())['source']['commit'],'script_sha256':{str(p.relative_to(root)):sha(p) for p in (root/'scripts/candidate_trajectory').glob('*.py')},'prompt_ids':[r['id'] for r in selected],'scope':'Independent SciQ development feasibility; no TruthfulQA selection; no training; no paid grading.'}
        config['provenance']={'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),'run_id':directory.name,'checkpoint_weights_sha256':manifest['checkpoint_weights_sha256'],'backbone_revision':manifest['backbone']['revision']}
        atomic_json(directory/'config.json',config)
        atomic_json(directory/'manifest.json',manifest);events.emit('run_metadata',run_id=directory.name,configuration=config,execution='sequential',cancel_support=False)
    print('RUN_DIRECTORY='+str(directory),flush=True)
    atomic_json(directory/'event-schema.json',{'version':1,'common':{'sequence':'Monotonically increasing integer','timestamp_utc':'ISO8601 UTC','event':'Event type','prompt_id':'Source prompt ID where applicable','trajectory_id':'Unique prefix / candidate / rollout ID where applicable','candidate_id':'0..3 where applicable','rollout_id':'0..R-1 where applicable','actual_backbone_calls':'Cumulative executed calls including verification and bootstrap'},'trajectory_initial':{'token_ids':'Full prompt+generation canvas','tokens_sha256':'SHA256 of compact JSON token list','prompt_length':'Prompt boundary','mask_id':'Readable as [MASK] in dashboard'},'generation_progress':{'changes':[{'position':123,'old_id':126336,'new_id':42,'kind':'revealed or revised','new_text':'decoded sampled token'}],'token_ids':'Optional periodic full snapshot','memory_source_token_version':'Pre-sampling token version processed into current memory','token_version':'Post-sampling token version, normally one newer','diagnostics':'Configurable numerical memory diagnostics; not thoughts'},'trajectory_completed':{'raw_response':'Unmodified decoded response before first stop token','extracted_final_answer':'Deterministic extraction, not a correctness grade','correctness':'NOT_GRADED until actual grading','false_belief_endorsement':'NOT_GRADED; distinct from correctness'}})
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no replacement run attempted')
    saved_manifest=json.loads((directory/'manifest.json').read_text())
    if sha(Path(config['checkpoint'])/'recurrent_modules.safetensors')!=saved_manifest['checkpoint_weights_sha256']:raise RuntimeError('Checkpoint changed since run was created')
    events.emit('session_provenance',script_sha256={str(p.relative_to(root)):sha(p) for p in (root/'scripts/candidate_trajectory').glob('*.py')},resume=bool(args.resume),configuration=config)
    engine=Engine(config,config['checkpoint'],events);engine.calls=max((e.get('actual_backbone_calls',0) for e in read_events(events.path)),default=0)
    finished={r['trajectory_id'] for r in rows(directory/'results.jsonl')};start=time.perf_counter();active=None;verification_calls=0
    def job(snapshot,context,seed):
        nonlocal active
        tid=context['trajectory_id']
        if tid in finished:return
        s=clone(snapshot);s['trajectory_id']=tid;s['rng']=seeded_rng(seed,engine.device);active=s;ctx={k:v for k,v in context.items() if k!='trajectory_id'};ctx['rollout_seed']=seed
        before=engine.calls-context.get('bootstrap_calls',0);t=time.perf_counter()
        try:
            engine.advance(s,context=ctx);result=engine.finish(s,ctx);result.update(status='completed',incremental_backbone_calls=engine.calls-before,latency_seconds=time.perf_counter()-t+context.get('bootstrap_seconds',0),peak_gpu_bytes=torch.cuda.max_memory_allocated());save_snapshot(directory/'snapshots'/f'{tid}.pt',s)
        except Exception as exc:
            result={**context,'status':'failed','error':repr(exc),'raw_response':'','extracted_final_answer':None,'invalid':True,'correctness':'NOT_GRADED','false_belief_endorsement':'NOT_GRADED','incremental_backbone_calls':engine.calls-before,'latency_seconds':time.perf_counter()-t};events.emit('trajectory_failed',**result,actual_backbone_calls=engine.calls);save_snapshot(directory/'snapshots'/f'{tid}-failed.pt',s)
            if isinstance(exc,torch.OutOfMemoryError) or 'cap reached' in str(exc):append(directory/'results.jsonl',result);raise
        append(directory/'results.jsonl',result);finished.add(tid);active=None
    try:
        if (directory/'verification.json').exists() and not json.loads((directory/'verification.json').read_text())['passed']:raise RuntimeError('Saved verification failed; refusing to continue')
        if not args.skip_verify and not (directory/'verification.json').exists():verification_calls=verify(engine,directory,[selected[0]['messages'][0]])['backbone_calls']
        for index,ref in enumerate(selected):
            pid=ref['id'];short=f'p{index}';common={'prompt_id':pid,'condition':ref['condition']};messages=[ref['messages'][0]]
            parentfile=directory/'snapshots'/f'{short}-parent.pt'
            if parentfile.exists():
                parent=load_snapshot(parentfile,engine.device)
                if not any(e['event']=='parent_saved' and e.get('prompt_id')==pid for e in read_events(events.path)):events.emit('parent_saved',**common,trajectory_id=parent['trajectory_id'],phase='shared',status='completed',shared_prefix_calls=parent['trajectory_calls'],recovered_from_snapshot=True,actual_backbone_calls=engine.calls)
            else:
                prefix_start=time.perf_counter()
                parent=engine.initialize(messages,42,f'{short}-shared');active=parent;engine.advance(parent,config['prefix_steps'],context={**common,'phase':'shared','prompt':messages[0]['content']});save_snapshot(parentfile,parent);events.emit('parent_saved',**common,trajectory_id=parent['trajectory_id'],phase='shared',status='completed',shared_prefix_calls=parent['trajectory_calls'],latency_seconds=time.perf_counter()-prefix_start,token_version=parent['token_version'],memory_source_token_version=parent['memory_source_token_version'],actual_backbone_calls=engine.calls)
            children=[]
            for c,seed in enumerate(config['branch_seeds']):events.emit('branch_queued',**common,phase='branch',candidate_id=c,seed=seed,trajectory_id=f'{short}-candidate{c}',status='pending')
            for c,seed in enumerate(config['branch_seeds']):
                tid=f'{short}-candidate{c}';path=directory/'snapshots'/f'{tid}.pt';ctx={**common,'phase':'branch','candidate_id':c,'seed':seed}
                if path.exists():
                    child=load_snapshot(path,engine.device)
                    if not any(e['event']=='branch_created' and e.get('trajectory_id')==tid for e in read_events(events.path)):events.emit('branch_created',**ctx,trajectory_id=tid,status='completed',branch_incremental_calls=child['trajectory_calls']-parent['trajectory_calls'],recovered_from_snapshot=True,actual_backbone_calls=engine.calls)
                else:
                    child=clone(parent);child['trajectory_id']=tid;child['rng']=seeded_rng(seed,engine.device);active=child;events.emit('branch_queued',**ctx,trajectory_id=tid,status='pending');before=engine.calls;branch_start=time.perf_counter();engine.advance(child,config['branch_steps'],context=ctx);save_snapshot(path,child);events.emit('branch_created',**ctx,trajectory_id=tid,status='completed',branch_incremental_calls=engine.calls-before,latency_seconds=time.perf_counter()-branch_start,tokens_sha256=token_hash(child['tokens'][0].tolist()),memory_source_token_version=child['memory_source_token_version'],token_version=child['token_version'],actual_backbone_calls=engine.calls)
                children.append(child)
            pairs=[]
            for a,b in itertools.combinations(range(4),2):pairs.append({'candidates':[a,b],'different_token_positions':int((children[a]['tokens']!=children[b]['tokens']).sum()),'memory':distance(children[a]['memory'],children[b]['memory']),'conditioning':distance(children[a]['conditioning'],children[b]['conditioning'])})
            events.emit('branch_diversity',**common,pairs=pairs,duplicate_token_groups=[[i for i in range(4) if torch.equal(children[i]['tokens'],children[c]['tokens'])] for c in range(4)],slot_alignment='Same learned slot index; no permutation matching',meaning='Numerical diagnostics, not evidence of distinct thoughts')
            jobs=[]
            for mode in ('persistent','parent_memory_ablation'):
                for c in range(4):
                    for r,seed in enumerate(config['rollout_seeds']):jobs.append({'trajectory_id':f'{short}-{mode}-c{c}-r{r}','mode':mode,'candidate_id':c,'rollout_id':r,'seed':config['branch_seeds'][c],'rollout_seed':seed,**common})
            remaining=config['steps']-config['prefix_steps']-config['branch_steps'];primary_budget=1+config['prefix_steps']+4*config['branch_steps']+4*config['rollouts']*remaining;independent_n=max(1,round(primary_budget/(config['steps']+1)))
            for context in jobs:events.emit('rollout_queued',**context,phase='rollout',status='completed' if context['trajectory_id'] in finished else 'pending')
            for context in jobs:
                s=clone(children[context['candidate_id']])
                if context['mode']=='parent_memory_ablation':
                    s['memory']=parent['memory'].clone();s['conditioning']=parent['conditioning'].clone();s['memory_source_token_version']=parent['memory_source_token_version'];s['intervention']='parent_memory_and_conditioning_post_training'
                job(s,{**context,'phase':'rollout'},context['rollout_seed'])
            for i in range(independent_n):
                tid=f'{short}-independent-{i}'
                if tid in finished:continue
                before=engine.calls;bt=time.perf_counter();s=engine.initialize(messages,1001+i,tid);bootstrap_seconds=time.perf_counter()-bt
                # Bootstrap belongs to this independent full generation's measured cost.
                job(s,{**common,'trajectory_id':tid,'phase':'independent','mode':'independent','rollout_id':i,'seed':1001+i,'bootstrap_calls':engine.calls-before,'bootstrap_seconds':bootstrap_seconds},1001+i)
                # Append a correction event rather than rewriting append-only result records.
                events.emit('independent_bootstrap_accounting',trajectory_id=tid,bootstrap_calls=1,actual_backbone_calls=engine.calls)
        # Include independent bootstrap in offline cost accounting, leaving raw records append-only.
        events.emit('grading_status',correctness='NOT_GRADED',false_belief_endorsement='NOT_GRADED',reason='Paid evaluator not authorized; no endorsement evaluator configured.')
        events.emit('run_completed',actual_backbone_calls=engine.calls,elapsed_seconds=time.perf_counter()-start,verification_calls=verification_calls,event_write_seconds=events.overhead_seconds,observer_formatting_seconds=engine.observer_seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        report=summarize(directory,config);print(json.dumps({'directory':str(directory),'results':sum(p['n_results_including_failures'] for p in report['prompts']),'backbone_calls':engine.calls,'report':str(directory/'report.json')},indent=2),flush=True)
    except BaseException as exc:
        if active is not None:save_snapshot(directory/'snapshots/interrupted-active.pt',active)
        events.emit('run_interrupted' if isinstance(exc,KeyboardInterrupt) else 'run_failed',error=repr(exc),actual_backbone_calls=engine.calls,resume='Restart with --resume RUN_DIRECTORY; completed jobs are skipped, incomplete jobs deterministically restart from saved boundary.')
        raise


def main(root):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('--resume');p.add_argument('--prompts',type=int,default=1);p.add_argument('--condition',choices=['neutral','supports_correct','doubts_correct','supports_incorrect','doubts_incorrect'],default='supports_incorrect');p.add_argument('--tokens',type=int,default=32);p.add_argument('--steps',type=int,default=32);p.add_argument('--block-size',type=int,default=8);p.add_argument('--prefix-steps',type=int,default=8);p.add_argument('--branch-steps',type=int,default=8);p.add_argument('--rollouts',type=int,default=3);p.add_argument('--temperature',type=float,default=0.);p.add_argument('--event-every',type=int,default=1);p.add_argument('--diagnostic-every',type=int,default=4);p.add_argument('--token-snapshot-every',type=int,default=8);p.add_argument('--max-backbone-calls',type=int,default=2000);p.add_argument('--skip-verify',action='store_true')
    for name in ('dashboard','replay','status'):
        p=sub.add_parser(name);p.add_argument('--run-dir');
        if name!='status':p.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if args.command=='run':run(root,args);return
    directory=Path(args.run_dir) if args.run_dir else Path(json.loads((root/'result/candidate-trajectories/latest.json').read_text())['directory'])
    if args.command=='status':
        events=read_events(directory/'events.jsonl');print(json.dumps({'directory':str(directory),'events':len(events),'results':len(rows(directory/'results.jsonl')),'last_event':events[-1] if events else None},indent=2));return
    from .dashboard import serve
    serve(directory,port=args.port)
