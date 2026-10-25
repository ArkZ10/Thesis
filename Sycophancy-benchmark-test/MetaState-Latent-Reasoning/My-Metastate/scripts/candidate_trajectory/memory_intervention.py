"""Controlled memory-only intervention; no training, selector, or paid grading."""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import time
import torch
from .engine import Engine, clone, distance, save_snapshot, token_hash
from .events import Events, atomic_json, read_events, replay
from .runner import append, rows, sha, verify


def perturb_memory(memory, epsilon, seed):
    """FP32 isotropic direction on CPU, scaled globally, then cast back to memory dtype."""
    if epsilon < 0 or not torch.isfinite(memory).all():
        raise ValueError('Epsilon must be nonnegative and input memory finite')
    if epsilon == 0:
        return memory.clone()
    base = memory.detach().cpu().float()
    norm = base.double().norm()
    if norm == 0:
        raise ValueError('Cannot define a relative perturbation around zero-norm memory')
    direction = torch.randn(base.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    change = direction * (float(norm) * epsilon / float(direction.double().norm()))
    result = (base + change).to(device=memory.device, dtype=memory.dtype)
    if not torch.isfinite(result).all() or torch.equal(result, memory):
        raise ValueError('Perturbation is nonfinite or vanished after dtype conversion')
    return result


def distribution_metrics(control, changed, eligible):
    """First-pass masked-position logits only; no sampling, grading, or extra forward."""
    a = control.float(); b = changed.float()
    la = a.log_softmax(-1); lb = b.log_softmax(-1)
    pa = la.exp(); pb = lb.exp()
    argmax_diff = a.argmax(-1) != b.argmax(-1)
    return {
        'masked_positions': a.shape[0],
        'eligible_positions_this_block': int(eligible.sum()),
        'greedy_token_changes_all_masked_positions': int(argmax_diff.sum()),
        'greedy_token_changes_eligible_positions': int(argmax_diff[eligible].sum()),
        'mean_kl_control_to_candidate_nats': float((pa * (la - lb)).sum(-1).mean()),
        'mean_total_variation': float((pa - pb).abs().sum(-1).mean() * .5),
        'mean_absolute_logit_change': float((a-b).abs().mean()),
        'max_absolute_logit_change': float((a-b).abs().max()),
        'max_absolute_probability_change': float((pa-pb).abs().max()),
    }


def cpu_check():
    base = torch.linspace(-1, 1, 8192).reshape(1, 8, 1024).to(torch.bfloat16)
    saved = base.clone()
    assert torch.equal(perturb_memory(base, 0, 11), base)
    a = perturb_memory(base, .01, 11); b = perturb_memory(base, .01, 22)
    assert torch.equal(a, perturb_memory(base, .01, 11))
    assert not torch.equal(a, b) and torch.equal(base, saved)
    ratio = float((a.float()-base.float()).norm()/base.float().norm())
    assert .008 < ratio < .013
    x = torch.tensor([[1.,2.,3.],[3.,2.,1.]])
    diag = distribution_metrics(x, x.clone(), torch.tensor([True,False]))
    assert diag['mean_total_variation'] == 0 and diag['greedy_token_changes_all_masked_positions'] == 0
    return {'kind':'SYNTHETIC_CPU_CHECKS', 'passed':True, 'checks':['exact unchanged control','seeded reproducible perturbations','distinct directions','parent immutability','BF16 perturbation retained','identity distribution comparison'], 'actual_relative_bf16_perturbation':ratio}


def run(root, args):
    synthetic = cpu_check()
    if args.check_only:
        print(json.dumps(synthetic, indent=2)); return
    if not 0 < args.epsilon <= .3:
        raise ValueError('This bounded stress test requires 0 < epsilon <= 0.3')
    if args.prefix_steps not in (0,8):
        raise ValueError('Branch boundary must be bootstrap-only (0) or after 8 decoding steps')
    if args.prompts < 1 or args.prompts > 3:
        raise ValueError('Tiny pilot supports 1 to 3 independent development questions')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no substitute model or training attempted')
    directory = root/'result/memory-interventions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory.mkdir(parents=True, exist_ok=False)
    events = Events(directory)
    checkpoint = root/'result/phase3/20261020T162646Z/checkpoints/checkpoint-20'
    setup = json.loads((root/'setup/manifest.json').read_text())
    for rel, expected in setup['source']['file_sha256'].items():
        if sha(root/'vendor/MetaState'/rel) != expected:
            raise RuntimeError('Vendor source differs from recorded commit: '+rel)
    config = {
        'generation_length':32, 'steps':32, 'block_size':8, 'temperature':0.,
        'prefix_steps':args.prefix_steps, 'branch_steps':8, 'event_every':1, 'diagnostic_every':4,
        'token_snapshot_every':8, 'max_backbone_calls':800, 'epsilon':args.epsilon,
        'candidate_seeds':[None,11,22,33], 'decoding_seed':42, 'precision':'bfloat16',
        'caching':False, 'warmup':True, 'remasking':'low_confidence',
        'backbone_path':json.loads((root/'setup/backbone.json').read_text())['backbone_name_or_path'],
        'checkpoint':str(checkpoint), 'device_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'intervention':'memory_only; identical tokens, carried conditioning, RNG, masks and decoding metadata',
        'perturbation_formula':'M_i = cast_BF16(M_FP32 + epsilon * ||M||F * Z_i/||Z_i||F)',
        'boundary_semantics':'After bootstrap, memory has processed the all-masked token canvas at version 0. After each decoding step, memory precedes the newly sampled token version by one. The perturbation is an external intervention.',
    }
    source = root/'data/processed/development.jsonl'
    tiny = {r['question_id'] for r in rows(root/'data/processed/tiny/development.jsonl')}
    selected = [r for r in rows(source) if r['condition']=='supports_incorrect' and r['question_id'] not in tiny][:args.prompts]
    if len(selected) != args.prompts:
        raise ValueError('Not enough independent development questions')
    manifest = {
        'run_id':directory.name, 'experiment':'memory-only sensitivity intervention',
        'source_sha256':sha(source), 'source_prompt_ids':[r['id'] for r in selected],
        'backbone':json.loads((root/'setup/backbone.json').read_text()),
        'checkpoint_metadata':json.loads((checkpoint/'recurrent_checkpoint_meta.json').read_text()),
        'checkpoint_weights_sha256':sha(checkpoint/'recurrent_modules.safetensors'),
        'vendor_commit':setup['source']['commit'],
        'code_sha256':{str(p.relative_to(root)):sha(p) for p in (root/'scripts/candidate_trajectory').glob('*.py')},
        'packages': {name:__import__('importlib.metadata', fromlist=['version']).version(name) for name in ('torch','transformers','numpy','safetensors')},
        'scope':'No training, paid grading, reference-based generation or candidate selection',
    }
    config['provenance']={'run_id':directory.name,'checkpoint_weights_sha256':manifest['checkpoint_weights_sha256'],'backbone_revision':manifest['backbone']['revision']}
    atomic_json(directory/'config.json',config); atomic_json(directory/'manifest.json',manifest)
    atomic_json(directory/'synthetic-verification.json',synthetic)
    atomic_json(root/'result/memory-interventions/latest.json',{'directory':str(directory)})
    # Allow the existing dashboard's default command to discover this explicitly labelled experiment.
    atomic_json(root/'result/candidate-trajectories/latest.json',{'directory':str(directory),'experiment':'memory-only intervention'})
    for ref in selected: append(directory/'offline-references.jsonl',ref)
    events.emit('run_metadata',run_id=directory.name,configuration=config,execution='sequential',cancel_support=False,experiment='MEMORY-ONLY INTERVENTION; numerical sensitivity, not proven reasoning')
    print('RUN_DIRECTORY='+str(directory),flush=True)
    engine = Engine(config,str(checkpoint),events)
    active = None; original_forward = engine._forward; results=[]; findings=[]; check_results=[]
    start = time.perf_counter()
    try:
        verification = verify(engine,directory,[selected[0]['messages'][0]])
        for qi, ref in enumerate(selected):
            common={'prompt_id':ref['id'],'condition':ref['condition']}
            parent = engine.initialize([ref['messages'][0]],42,f'p{qi}-shared'); active=parent
            engine.advance(parent,args.prefix_steps,context={**common,'phase':'shared','prompt':ref['messages'][0]['content']})
            save_snapshot(directory/'snapshots'/f'p{qi}-parent.pt',parent)
            events.emit('trajectory_status',**common,phase='shared',trajectory_id=parent['trajectory_id'],status='completed',actual_backbone_calls=engine.calls)
            saved_parent=clone(parent)
            children=[]; interventions=[]
            for ci, seed in enumerate(config['candidate_seeds']):
                child=clone(parent);child['trajectory_id']=f'p{qi}-memory-c{ci}-r0'
                child['memory']=perturb_memory(parent['memory'],0 if ci==0 else args.epsilon,seed or 0)
                child['intervention']='unchanged_control' if ci==0 else 'isotropic_memory_only_perturbation'
                child['memory_intervention_metadata']={'epsilon':0 if ci==0 else args.epsilon,'perturbation_seed':seed,'naturally_updated_memory_source_version':parent['memory_source_token_version'],'externally_perturbed':ci!=0}
                assert all(torch.equal(child[k],parent[k]) for k in ('tokens','conditioning','generation_mask','prompt_mask'))
                assert torch.equal(child['rng']['torch_cuda'],parent['rng']['torch_cuda'])
                assert child['memory'].data_ptr()!=parent['memory'].data_ptr()
                actual_delta=float((child['memory'].float()-parent['memory'].float()).norm()/parent['memory'].float().norm())
                interventions.append({'candidate_id':ci,'requested_relative_norm':0 if ci==0 else args.epsilon,'actual_relative_norm_after_bf16':actual_delta,'changed_memory_elements':int((child['memory']!=parent['memory']).sum()),'tokens_identical':True,'conditioning_identical':True,'decoding_rng_identical':True})
                children.append(child);save_snapshot(directory/'snapshots'/f'p{qi}-memory-c{ci}-initial.pt',child)
                events.emit('branch_queued',**common,phase='memory_intervention',candidate_id=ci,trajectory_id=child['trajectory_id'],seed=seed,status='pending',intervention=interventions[-1])
            pairs=[{'candidates':[a,b],'different_token_positions':0,'memory':distance(children[a]['memory'],children[b]['memory']),'conditioning':distance(children[a]['conditioning'],children[b]['conditioning'])} for a,b in itertools.combinations(range(4),2)]
            events.emit('branch_diversity',**common,scope='INITIAL: identical tokens and conditioning; only memory changed',pairs=pairs)
            positions=(parent['tokens'][0]==engine.mask).logical_and(parent['generation_mask'][0])
            end=parent['prompt_length']+(parent['block']+1)*config['block_size']
            eligible=(torch.arange(parent['tokens'].shape[1],device=engine.device)<end)[positions]
            control_logits=None; prompt_findings=[]; control_final=None
            for ci, child_initial in enumerate(children):
                child=clone(child_initial);active=child;captured=[]
                ctx={**common,'phase':'memory_intervention','candidate_id':ci,'rollout_id':0,'mode':'unchanged_control' if ci==0 else 'memory_perturbed','seed':config['candidate_seeds'][ci],'decoding_seed':42,'epsilon':0 if ci==0 else args.epsilon}
                def instrument(s,bootstrap=False):
                    nonlocal control_logits
                    logits,diag=original_forward(s,bootstrap)
                    if not captured:
                        # Called after the forward but before sampling. Input tokens were identical.
                        probe=logits[0,positions,:].detach().float()
                        if ci==0:control_logits=probe.clone()
                        metrics=distribution_metrics(control_logits,probe,eligible)
                        metrics.update(candidate_id=ci,comparison='first forward before any token sampling',input_token_version=parent['token_version'],input_tokens_sha256=token_hash(parent['tokens'][0].tolist()),input_conditioning_identical=True,first_forward_input_memory_only_changed=ci!=0)
                        captured.append(metrics)
                    return logits,diag
                engine._forward=instrument
                before=engine.calls; t=time.perf_counter()
                try:
                    engine.advance(child,context=ctx)
                finally:engine._forward=original_forward
                result=engine.finish(child,ctx);result.update(status='completed',incremental_backbone_calls=engine.calls-before,latency_seconds=time.perf_counter()-t,first_pass_metrics=captured[0],intervention=interventions[ci])
                save_snapshot(directory/'snapshots'/f"{child['trajectory_id']}.pt",child)
                append(directory/'results.jsonl',result);results.append(result)
                prompt_findings.append(captured[0]);events.emit('first_pass_memory_sensitivity',**common,trajectory_id=child['trajectory_id'],candidate_id=ci,metrics=captured[0],status='completed',actual_backbone_calls=engine.calls)
                if ci==0:control_final=clone(child)
                assert all(torch.equal(parent[k],saved_parent[k]) for k in ('tokens','memory','conditioning'))
            engine._forward=original_forward
            # Deterministic control repeat from the saved boundary, independent of observer and branch order.
            repeat=clone(children[0]);before=engine.calls;engine.advance(repeat,monitor=False)
            checks={'prompt_id':ref['id'],'unchanged_control_repeat_tokens_exact':torch.equal(repeat['tokens'],control_final['tokens']),'unchanged_control_repeat_memory_exact':torch.equal(repeat['memory'],control_final['memory']),'unchanged_control_repeat_conditioning_exact':torch.equal(repeat['conditioning'],control_final['conditioning']),'parent_immutable':all(torch.equal(parent[k],saved_parent[k]) for k in ('tokens','memory','conditioning')),'reveal_policy_unchanged':all(r['residual_mask_tokens']==0 for r in results if r['prompt_id']==ref['id']),'repeat_verification_backbone_calls':engine.calls-before}
            assert all(v for k,v in checks.items() if isinstance(v,bool)),checks
            check_results.append(checks)
            control=next(r for r in results if r['prompt_id']==ref['id'] and r['candidate_id']==0)
            outcomes=[{'candidate_id':r['candidate_id'],'final_answer_changed_vs_control':r['extracted_final_answer']!=control['extracted_final_answer'],'final_token_canvas_changed_vs_control':r['tokens_sha256']!=control['tokens_sha256'],'raw_response':r['raw_response'],'empty':r['empty'],'invalid':r['invalid'],'residual_mask_tokens':r['residual_mask_tokens']} for r in results if r['prompt_id']==ref['id']]
            findings.append({'prompt_id':ref['id'],'question':ref['question'],'interventions':interventions,'first_pass_comparisons':prompt_findings,'outcomes':outcomes})
            events.emit('branch_diversity',**common,scope='FIRST-PASS SENSITIVITY AND FINAL OUTCOMES',first_pass=prompt_findings,outcomes=outcomes,warning='Post-training memory intervention; distribution shift possible. Sensitivity is not proof of useful reasoning.')
            del control_logits
        states=replay(read_events(events.path))
        assert all(token_hash(states[r['trajectory_id']])==r['tokens_sha256'] for r in results)
        report={'experiment':'memory-only perturbation sensitivity','epsilon':args.epsilon,'prefix_steps':args.prefix_steps,'temperature':0.,'prompts':findings,'n_completed':len(results),'correctness':'NOT_GRADED','false_belief_endorsement':'NOT_GRADED','verification':check_results,'upstream_verification_passed':verification['passed'],'event_replay_matches_all_final_outputs':True,'actual_backbone_calls_including_verification':engine.calls,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-start,'limitations':['One perturbation scale; isotropic noise is not a learned latent proposal.','Potential distribution shift; no claim of meaningful alternative reasoning.','Deterministic outputs can remain identical despite changed distributions.','No reference-free selector, accuracy grading, training, or MCTS.','Memory is externally changed; its source-version tag describes the natural pre-intervention state.']}
        atomic_json(directory/'report.json',report)
        with (directory/'results.csv').open('w',newline='') as f:
            keys=sorted(set().union(*(r.keys() for r in results)));w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
            for r in results:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
        events.emit('run_completed',experiment='memory-only sensitivity',status='completed',actual_backbone_calls=engine.calls,n_completed=len(results),elapsed_seconds=report['elapsed_seconds'])
        print(json.dumps({'directory':str(directory),'n_completed':len(results),'backbone_calls':engine.calls,'report':str(directory/'report.json')},indent=2),flush=True)
    except BaseException as exc:
        engine._forward=original_forward
        if active is not None:save_snapshot(directory/'snapshots/interrupted-active.pt',active)
        events.emit('run_failed',error=repr(exc),actual_backbone_calls=engine.calls)
        raise


def main(root):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epsilon',type=float,default=.01)
    parser.add_argument('--prompts',type=int,default=1)
    parser.add_argument('--prefix-steps',type=int,choices=[0,8],default=8)
    parser.add_argument('--check-only',action='store_true')
    run(root,parser.parse_args())
