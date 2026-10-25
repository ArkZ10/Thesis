"""Repeated Gaussian noise in carried MetaState memory: a small controlled adaptation."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import torch
from .engine import Engine, clone, load_snapshot, save_snapshot, seeded_rng, token_hash
from .events import Events, atomic_json, read_events, replay
from .memory_intervention import perturb_memory, distribution_metrics
from .runner import append, rows, sha, verify


def gaussian_update(s):
    """Private CPU generator; Gaussian coordinates are NOT normalized to fixed norm."""
    plan=s['noise_plan']; memory=s['memory']; generator=torch.Generator(device='cpu')
    generator.set_state(s['agn_rng_state'].cpu())
    noise=torch.randn(memory.shape,generator=generator,dtype=torch.float32)*plan['sigma']
    s['agn_rng_state']=generator.get_state().clone()
    changed=(memory.float()+noise.to(memory.device)).to(memory.dtype)
    delta=changed.float()-memory.float()
    norm=float(delta.norm());base_norm=float(memory.float().norm())
    if not torch.isfinite(changed).all():raise ValueError('Nonfinite noisy memory')
    s['memory']=changed;s['noise_injections']+=1
    stats={'injection_index':s['noise_injections'],'sigma_per_coordinate':plan['sigma'],'reference_rms':plan['reference_rms'],'actual_delta_frobenius':norm,'actual_relative_delta':norm/base_norm if base_norm else None,'realized_gaussian_norm_before_bf16':float(noise.norm()),'actual_changed_elements':int((changed!=memory).sum()),'post_noise_memory_norm':float(changed.float().norm()),'token_version':s['token_version'],'memory_source_token_version':s['memory_source_token_version']}
    s['noise_energy_realized']+=norm*norm
    return stats


def setup_state(parent,group,seed,trajectory_id,config):
    s=clone(parent);s['trajectory_id']=trajectory_id;s['configuration']=clone(config)
    s['rng']=seeded_rng(seed,parent['tokens'].device)
    s['agn_rng_state']=torch.Generator().manual_seed(seed).get_state()
    s['noise_injections']=0;s['noise_energy_realized']=0.
    rms=float(parent['memory'].float().square().mean().sqrt())
    s['noise_plan']={**group,'reference_rms':rms,'sigma':group.get('alpha',0)*rms*(math.sqrt(config['steps']) if group['kind']=='gaussian_once_energy_matched' else 1.)}
    s['intervention']=group['kind'];initial_stats=None
    if group['kind'] in ('repeated_gaussian','gaussian_once_energy_matched'):
        initial_stats=gaussian_update(s)
    elif group['kind']=='legacy_fixed_norm_once':
        previous=s['memory'];s['memory']=perturb_memory(previous,.05,seed)
        norm=float((s['memory'].float()-previous.float()).norm());s['noise_energy_realized']=norm*norm;s['noise_injections']=1
        initial_stats={'injection_index':1,'actual_delta_frobenius':norm,'actual_relative_delta':norm/float(previous.float().norm()),'actual_changed_elements':int((s['memory']!=previous).sum()),'method':'Legacy normalized Gaussian direction, fixed 5% relative norm; not iid AGN'}
    return s,initial_stats


def cpu_check():
    from .memory_intervention import cpu_check as previous_checks
    previous_checks()
    p={'memory':torch.linspace(-1,1,4096).reshape(1,4,1024).to(torch.bfloat16),'tokens':torch.tensor([[1]]),'token_version':0,'memory_source_token_version':0}
    plan={'sigma':.01,'reference_rms':1.}
    def sample(seed):
        return {**clone(p),'noise_plan':plan.copy(),'agn_rng_state':torch.Generator().manual_seed(seed).get_state(),'noise_injections':0,'noise_energy_realized':0.}
    a=sample(11);b=sample(11);outer=torch.get_rng_state().clone();gaussian_update(a);gaussian_update(b)
    assert torch.equal(a['memory'],b['memory']) and torch.equal(torch.get_rng_state(),outer)
    mid=clone(a);gaussian_update(a);gaussian_update(mid);assert torch.equal(a['memory'],mid['memory'])
    c=sample(22);gaussian_update(c);assert not torch.equal(c['memory'],b['memory'])
    return {'kind':'SYNTHETIC_CPU_CHECKS','passed':True,'checks':['private Gaussian RNG reproducible','global token RNG untouched','noise RNG snapshot continuation exact','different seeds produce different memories','BF16 memory finite']}


def run(root,args):
    checks=cpu_check()
    if args.check_only:print(json.dumps(checks,indent=2));return
    if not 1<=args.prompts<=3:raise ValueError('Tiny pilot supports one to three questions')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    directory=root/'result/repeated-agn'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');directory.mkdir(parents=True,exist_ok=False)
    events=Events(directory)
    groups=[{'name':'no_noise','kind':'none','temperature':0.,'alpha':0.}, {'name':'token_sampling_T0.3','kind':'token_sampling','temperature':.3,'alpha':0.}, {'name':'legacy_once_5percent','kind':'legacy_fixed_norm_once','temperature':0.,'alpha':0.}]
    for alpha in (.01,.03):
        groups += [{'name':f'repeated_AGN_{alpha}','kind':'repeated_gaussian','temperature':0.,'alpha':alpha},{'name':f'energy_matched_once_{alpha}','kind':'gaussian_once_energy_matched','temperature':0.,'alpha':alpha}]
    checkpoint=root/'result/phase3/20261020T162646Z/checkpoints/checkpoint-20'
    setup=json.loads((root/'setup/manifest.json').read_text())
    for rel,expected in setup['source']['file_sha256'].items():
        if sha(root/'vendor/MetaState'/rel)!=expected:raise RuntimeError('Vendor source changed: '+rel)
    config={'generation_length':32,'steps':32,'block_size':8,'temperature':0.,'prefix_steps':0,'branch_steps':8,'event_every':1,'diagnostic_every':4,'token_snapshot_every':8,'max_backbone_calls':3600,'backbone_path':json.loads((root/'setup/backbone.json').read_text())['backbone_name_or_path'],'checkpoint':str(checkpoint),'groups':groups,'candidate_seeds':[11,22,33,44],'precision':'bfloat16','caching':False,'reference_rms':'Fixed RMS of each prompt bootstrap memory, common to all groups','noise_timing':'Initial injection after bootstrap; repeated injections after forwards 1..31. Each affects the NEXT forward, not current logits. No unused injection after final forward.','energy_matching':'Expected squared perturbation sum matches: K*D*(alpha*RMS)^2 vs D*(sqrt(K)*alpha*RMS)^2. This does not match dynamical effects or realized BF16 energy.','agn_rng':'Private CPU torch.Generator state carried in each snapshot; independent of token-sampling RNG','scope':'LatentTTS-inspired adaptation to MetaState diffusion memory, not a reproduction; no dropout, training, grading or selector'}
    config['provenance']={'run_id':directory.name,'checkpoint_weights_sha256':sha(checkpoint/'recurrent_modules.safetensors'),'backbone_revision':json.loads((root/'setup/backbone.json').read_text())['revision']}
    manifest={'checkpoint_metadata':json.loads((checkpoint/'recurrent_checkpoint_meta.json').read_text()),'backbone':json.loads((root/'setup/backbone.json').read_text()),'vendor_commit':setup['source']['commit'],'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','safetensors')},'code_sha256':{str(p.relative_to(root)):sha(p) for p in (root/'scripts/candidate_trajectory').glob('*.py')},'paper':'https://arxiv.org/html/2510.07745v4','implementation_reference':'https://github.com/ModalityDance/LatentTTS/blob/main/src/generation_mixin.py'}
    source=root/'data/processed/development.jsonl';tiny={r['question_id'] for r in rows(root/'data/processed/tiny/development.jsonl')}
    selected=[r for r in rows(source) if r['condition']=='supports_incorrect' and r['question_id'] not in tiny][:args.prompts]
    if len(selected)!=args.prompts:raise ValueError('Missing development questions')
    manifest.update(source_sha256=sha(source),prompt_ids=[r['id'] for r in selected]);atomic_json(directory/'manifest.json',manifest);atomic_json(directory/'config.json',config);atomic_json(directory/'synthetic-verification.json',checks)
    atomic_json(root/'result/repeated-agn/latest.json',{'directory':str(directory)});atomic_json(root/'result/candidate-trajectories/latest.json',{'directory':str(directory),'experiment':'repeated AGN adaptation'})
    for ref in selected:append(directory/'offline-references.jsonl',ref)
    events.emit('run_metadata',run_id=directory.name,configuration=config,experiment='REPEATED GAUSSIAN MEMORY NOISE ADAPTATION',cancel_support=False)
    print('RUN_DIRECTORY='+str(directory),flush=True)
    engine=Engine(config,str(checkpoint),events);normal_forward=engine._forward;results=[];all_checks=[];active=None;start=time.perf_counter()
    try:
        native_verification=verify(engine,directory,[selected[0]['messages'][0]])
        for qi,ref in enumerate(selected):
            engine.config['temperature']=0.;parent=engine.initialize([ref['messages'][0]],42,f'p{qi}-bootstrap');active=parent;original_parent=clone(parent);save_snapshot(directory/'snapshots'/f'p{qi}-parent.pt',parent)
            baseline_trace={};baseline_probe=None;baseline_final=None;baseline_final_states={}
            def advance(s,ctx,steps=None,monitor=True,capture_baseline=False):
                nonlocal baseline_probe
                def instrument(state,bootstrap=False):
                    nonlocal baseline_probe
                    step=state['completed_steps'];input_memory=state['memory'].clone();input_tokens=state['tokens'].clone()
                    if capture_baseline:baseline_trace[step]={'memory':input_memory.clone(),'tokens':input_tokens.clone()}
                    logits,diag=normal_forward(state,bootstrap)
                    if step==0:
                        masked=parent['generation_mask'][0];probe=logits[0,masked,:].detach().float()
                        if capture_baseline:baseline_probe=probe.clone()
                        eligible=(torch.arange(parent['tokens'].shape[1],device=engine.device)<parent['prompt_length']+8)[masked]
                        state['first_pass_metrics']=distribution_metrics(baseline_probe,probe,eligible)
                    raw_norm=diag['memory_norm'];noise=None
                    if state['noise_plan']['kind']=='repeated_gaussian' and step<engine.config['steps']-1:
                        noise=gaussian_update(state)
                        if monitor:events.emit('memory_noise',**ctx,trajectory_id=state['trajectory_id'],timing='after update; affects next forward',**noise,actual_backbone_calls=engine.calls)
                    reference=baseline_trace[step]
                    drift=float((input_memory.float()-reference['memory'].float()).norm()/reference['memory'].float().norm().clamp_min(1e-12))
                    token_diff=int((input_tokens!=reference['tokens']).sum())
                    state.setdefault('drift_trace',[]).append({'step':step,'input_memory_relative_frobenius_vs_control':drift,'input_token_differences_vs_control':token_diff,'noise':noise})
                    diag.update(raw_model_memory_norm=raw_norm,post_intervention_memory_norm=float(state['memory'].float().norm()),input_memory_drift_vs_control=drift,input_token_differences_vs_control=token_diff,drift_interpretation='Joint trajectory drift once tokens differ; not isolated latent reasoning',noise_injections=state['noise_injections'])
                    return logits,diag
                engine._forward=instrument
                try:engine.advance(s,steps=steps,context=ctx,monitor=monitor)
                finally:engine._forward=normal_forward
                return s
            for gi,group in enumerate(groups):
                engine.config['temperature']=group['temperature'];shared_ctx={'prompt_id':ref['id'],'condition':ref['condition'],'sweep_epsilon':group['name'],'mode':group['name']}
                view=clone(parent);view['trajectory_id']=f'p{qi}-g{gi}-shared';engine.start_event(view,{**shared_ctx,'phase':'shared','prompt':ref['messages'][0]['content']});events.emit('trajectory_status',**shared_ctx,phase='shared',trajectory_id=view['trajectory_id'],status='completed')
                for ci,seed in enumerate(config['candidate_seeds']):events.emit('rollout_queued',**shared_ctx,phase='rollout',candidate_id=ci,rollout_id=0,seed=seed,trajectory_id=f'p{qi}-g{gi}-c{ci}',status='pending')
                for ci,seed in enumerate(config['candidate_seeds']):
                    tid=f'p{qi}-g{gi}-c{ci}';ctx={**shared_ctx,'phase':'rollout','candidate_id':ci,'rollout_id':0,'seed':seed,'alpha':group['alpha'],'temperature':group['temperature']}
                    s,initial_noise=setup_state(parent,group,seed,tid,engine.config);active=s
                    assert torch.equal(s['tokens'],parent['tokens']) and torch.equal(s['conditioning'],parent['conditioning'])
                    save_snapshot(directory/'snapshots'/f'{tid}-initial.pt',s)
                    if initial_noise:events.emit('memory_noise',**ctx,trajectory_id=tid,timing='after bootstrap; before first decode',**initial_noise)
                    before=engine.calls;t=time.perf_counter();advance(s,ctx,capture_baseline=gi==0 and ci==0)
                    out=engine.finish(s,ctx);out.update(status='completed',group=group['name'],incremental_backbone_calls=engine.calls-before,latency_seconds=time.perf_counter()-t,first_pass_metrics=s['first_pass_metrics'],noise_injections=s['noise_injections'],realized_noise_energy=s['noise_energy_realized'],expected_noise_energy=(.05**2*float(parent['memory'].float().square().sum()) if group['kind']=='legacy_fixed_norm_once' else (32 if group['kind']=='repeated_gaussian' else 1)*parent['memory'].numel()*s['noise_plan']['sigma']**2),drift_trace=s.get('drift_trace',[]))
                    if gi==0 and ci==0:baseline_final=out;baseline_final_states={k:clone(s[k]) for k in ('tokens','memory','conditioning')}
                    out['changed_response_vs_control']=out['extracted_final_answer']!=baseline_final['extracted_final_answer']
                    out['changed_canvas_vs_control']=out['tokens_sha256']!=baseline_final['tokens_sha256']
                    expected_injections=32 if group['kind']=='repeated_gaussian' else 1 if group['kind'] in ('gaussian_once_energy_matched','legacy_fixed_norm_once') else 0
                    assert s['noise_injections']==expected_injections,(tid,s['noise_injections'])
                    save_snapshot(directory/'snapshots'/f'{tid}.pt',s);append(directory/'results.jsonl',out);results.append(out)
                    assert all(torch.equal(parent[k],original_parent[k]) for k in ('tokens','memory','conditioning'))
                    if gi==0:assert all(torch.equal(s[k],baseline_final_states[k]) for k in ('tokens','memory','conditioning'))
                    if qi==0 and group['name']=='repeated_AGN_0.01' and ci==0:
                        reference=clone(s);initial=load_snapshot(directory/'snapshots'/f'{tid}-initial.pt',engine.device);initial['trajectory_id']='verify-agn-resume';before_check=engine.calls
                        advance(initial,ctx,steps=8,monitor=False);save_snapshot(directory/'snapshots/verify-agn-mid.pt',initial);resumed=load_snapshot(directory/'snapshots/verify-agn-mid.pt',engine.device);advance(resumed,ctx,monitor=False)
                        equal=all(torch.equal(resumed[k],reference[k]) for k in ('tokens','memory','conditioning','agn_rng_state'))
                        equal=equal and resumed['noise_energy_realized']==reference['noise_energy_realized'] and resumed['noise_injections']==32
                        assert equal,'Repeated AGN snapshot/resume mismatch'
                        replayed=load_snapshot(directory/'snapshots'/f'{tid}-initial.pt',engine.device);replayed['trajectory_id']='verify-agn-repeat';advance(replayed,ctx,monitor=False)
                        assert all(torch.equal(replayed[k],reference[k]) for k in ('tokens','memory','conditioning','agn_rng_state'))
                        all_checks.append({'repeated_AGN_snapshot_resume_exact':True,'monitor_on_off_and_repeat_exact':True,'noise_rng_state_restored':True,'extra_backbone_calls':engine.calls-before_check})
                        events.emit('agn_verification_completed',checks=all_checks[-1],actual_backbone_calls=engine.calls)
                group_records=[r for r in results if r['prompt_id']==ref['id'] and r['group']==group['name']]
                events.emit('branch_diversity',**shared_ctx,summary={'distinct_answers':len(set(r['raw_response'] for r in group_records)),'changed_vs_control':sum(r['changed_response_vs_control'] for r in group_records),'responses':[{k:r[k] for k in ('candidate_id','raw_response','changed_response_vs_control','noise_injections','realized_noise_energy','expected_noise_energy')} for r in group_records]},warning='Wording diversity is not correctness, coverage or evaluator success.')
            del baseline_trace,baseline_probe
        engine.config['temperature']=0.
        states=replay(read_events(events.path));assert all(token_hash(states[r['trajectory_id']])==r['tokens_sha256'] for r in results)
        summary=[]
        for group in groups:
            rr=[r for r in results if r['group']==group['name']]
            summary.append({'group':group['name'],'responses':len(rr),'changed_vs_control':sum(r['changed_response_vs_control'] for r in rr),'sum_distinct_answers_across_questions':sum(len(set(r['raw_response'] for r in rr if r['prompt_id']==ref['id'])) for ref in selected),'empty':sum(r['empty'] for r in rr),'invalid':sum(r['invalid'] for r in rr),'residual_masks':sum(r['residual_mask_tokens']>0 for r in rr),'no_stop_before_budget':sum(r['budget_exhausted_without_stop'] for r in rr),'backbone_calls':sum(r['incremental_backbone_calls'] for r in rr),'latency_seconds':sum(r['latency_seconds'] for r in rr)})
        report={'experiment':'Repeated Gaussian memory noise adaptation','summary':summary,'responses':len(results),'verification':all_checks,'native_sampler_verification':native_verification['passed'],'event_replay_matches_all_final_states':True,'actual_backbone_calls':engine.calls,'shared_bootstrap_calls':len(selected),'primary_calls_per_group':4*32*len(selected),'extra_verification_calls':native_verification['backbone_calls']+sum(r['extra_backbone_calls'] for r in all_checks),'elapsed_seconds':time.perf_counter()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'correctness':'NOT_GRADED','coverage_any_correct':'NOT_GRADED','reference_free_selector':'NOT_IMPLEMENTED','limitations':['Three repeatedly inspected development questions, a debugging pilot.','Fixed bootstrap RMS makes expected cumulative squared-noise energy comparable, not dynamical effects or realized BF16 changes.','Memory drift is joint with tokens once decoding diverges.','No pretrained LatentRM transfer, dropout, MCTS or reward-model training.','Legacy fixed-norm 5% baseline is not iid AGN.']}
        atomic_json(directory/'report.json',report)
        for name,rr in (('comparison.csv',summary),('results.csv',results)):
            keys=sorted(set().union(*(r.keys() for r in rr)))
            with (directory/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
                for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
        events.emit('run_completed',status='completed',actual_backbone_calls=engine.calls,responses=len(results))
        print(json.dumps({'directory':str(directory),'summary':summary,'backbone_calls':engine.calls},indent=2),flush=True)
    except BaseException as exc:
        engine._forward=normal_forward
        if active is not None:save_snapshot(directory/'snapshots/interrupted-active.pt',active)
        events.emit('run_failed',error=repr(exc),actual_backbone_calls=engine.calls);raise


def main(root):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prompts',type=int,default=3);parser.add_argument('--check-only',action='store_true');run(root,parser.parse_args())
