"""One-action branching test on 16 development problems; no training or MCTS.

All candidates share an identical bootstrap. Change only the first internal
memory write (all/A/B/C/D), then use all-slot writes for every remaining pass.
Three internal passes and four diffusion rounds match the training structure.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re

from phase3_pilot import ROOT, write, append, sha, dependencies
from selective_memory_pilot import DATA, CANVAS, INTERNAL, DIFFUSION, load_model, trajectory, check
from candidate_trajectory.engine import clone, save_snapshot, load_snapshot
from candidate_trajectory.selective_memory import ACTIONS, forward_with_memory_action, slot_mask


def choose_rows():
    rows=[json.loads(s) for s in (DATA/'processed/development.jsonl').read_text().splitlines()]
    selected=[];rng=random.Random(42)
    for family in sorted({r['family'] for r in rows}):
        pool=sorted((r for r in rows if r['family']==family),key=lambda r:r['question_id'])
        selected.extend(rng.sample(pool,4))
    assert len(selected)==16 and len({r['question_id'] for r in selected})==16
    return selected


def forward(model,s,action,mask,bootstrap=False):
    import torch
    ratio=((s['tokens']==mask)&s['generation_mask']).sum(-1).float()/(s['generation_mask'].sum(-1).float()+1e-6)
    out=forward_with_memory_action(model,state=s['memory'],action=action,input_ids=s['tokens'],
        attention_mask=torch.ones_like(s['tokens']),past_cond=s['conditioning'],
        t=torch.ones(1,device=s['tokens'].device) if bootstrap else ratio,
        use_cache=False,return_dict=True)
    s['memory']=out.new_state;s['conditioning']=out.past_cond;s['trajectory_calls']+=1
    return out


def continue_branch(model,parent,action,mask):
    import torch
    from dllm.core.samplers.utils import get_transfer_index
    s=clone(parent);trace=[]
    initial=clone(s)
    for i in range(INTERNAL):
        previous=s['memory']
        out=forward(model,s,action if i==0 else 'all',mask)
        if i==0:
            frozen=(~slot_mask(previous,action)).expand_as(previous)
            assert torch.equal(s['memory'][frozen],previous[frozen])
            trace.append({'action':action,'changed_slots':int((s['memory']!=previous).any(-1).sum()),
                'unselected_slots_exact':True,'tokens_unchanged':torch.equal(s['tokens'],parent['tokens'])})
        del out
    for i in range(DIFFUSION):
        out=forward(model,s,'all',mask)
        remaining=(s['tokens']==mask)&s['generation_mask']
        before=s['tokens'].clone()
        proposed,reveal=get_transfer_index(logits=out.logits,temperature=0,remasking='low_confidence',
            mask_index=remaining,x=s['tokens'],num_transfer_tokens=torch.tensor([2],device=s['tokens'].device),
            threshold=None,factor=None,suppress_token_ids=None)
        assert (reveal.sum(-1)==2).all() and not (reveal&~remaining).any()
        s['tokens']=torch.where(reveal,proposed,s['tokens'])
        s['completed_steps']+=1
        trace.append({'diffusion_round':i,'remaining_masks':int(((s['tokens']==mask)&s['generation_mask']).sum()),
            'revealed_positions':reveal[0].nonzero().flatten().tolist()})
        assert torch.equal(before[~s['generation_mask']],s['tokens'][~s['generation_mask']])
        del out
    assert torch.equal(parent['memory'],initial['memory'])
    assert torch.equal(parent['conditioning'],initial['conditioning'])
    assert torch.equal(parent['tokens'],initial['tokens'])
    assert s['trajectory_calls']==8
    return s,trace


def summary(records):
    groups={c:{r['question_id']:r for r in records if r['action']==c} for c in ACTIONS}
    ids=set(groups['all'])
    assert len(records)==80 and len(ids)==16 and all(set(g)==ids for g in groups.values())
    per_action={}
    for a,g in groups.items():
        per_action[a]={'correct':sum(r['correct'] for r in g.values()),'total':16,
            'accuracy_percent':100*sum(r['correct'] for r in g.values())/16,
            'unclassifiable':sum(r['parsed_answer'] is None for r in g.values()),
            'execution_errors':sum(r.get('execution_error') is not None for r in g.values())}
    pairs=[]
    for q in sorted(ids):
        rr=[groups[a][q] for a in ACTIONS]
        numbers={r['parsed_answer'] for r in rr if r['parsed_answer'] is not None}
        pairs.append({'question_id':q,'question':rr[0]['question'],'target_answer':rr[0]['target_answer'],
            'answers':{r['action']:r['raw_response'] for r in rr},
            'correct':{r['action']:r['correct'] for r in rr},'distinct_valid_integer_answers':len(numbers),
            'all_correct':groups['all'][q]['correct'],'any_correct':any(r['correct'] for r in rr)})
    outcome={'per_action':per_action,'questions':16,'responses':80,
        'questions_with_multiple_valid_integer_answers':sum(r['distinct_valid_integer_answers']>1 for r in pairs),
        'all_update_correct':sum(r['all_correct'] for r in pairs),
        'any_candidate_correct':sum(r['any_correct'] for r in pairs),
        'rescued_questions':sum(r['any_correct'] and not r['all_correct'] for r in pairs),
        'some_branch_wrong_when_all_correct':sum(r['all_correct'] and not all(r['correct'].values()) for r in pairs),
        'limitations':['Exploratory development subset; not final held-out evaluation.',
            'Any-correct uses ground truth offline; no deployable selector or MCTS.',
            'Only one early selective write is varied; later writes are all-slot.',
            'Differences in answers do not establish distinct reasoning strategies.']}
    return outcome,pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.check_only:print(json.dumps(check(),indent=2));return
    directory=ROOT/'result/selective-memory-candidates'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True,exist_ok=False)
    write(ROOT/'result/selective-memory-candidates/latest.json',{'directory':str(directory)})
    write(directory/'status.json',{'stage':'loading','completed':0,'total':80})
    try:
        torch,_=dependencies()
        run=Path(json.loads((ROOT/'result/selective-memory-training/latest.json').read_text())['directory'])
        assert json.loads((run/'status.json').read_text())['stage']=='complete'
        checkpoint=run/'checkpoints/checkpoint-128'
        cfg=json.loads((run/'configuration.json').read_text())
        selected=choose_rows()
        write(directory/'configuration.json',{'checkpoint':str(checkpoint),'questions':16,'seed':42,
            'selection':'Four seeded questions per arithmetic family, development only.',
            'question_ids':[r['question_id'] for r in selected],'actions':list(ACTIONS),
            'action_timing':'First internal write after shared bootstrap only; remaining writes all.',
            'forwards_per_candidate_including_shared_bootstrap':8,'shared_bootstrap_calls_per_question':1,
            'incremental_forwards_per_candidate':7,'temperature':0,'precision':'BF16','physical_gpu':1,
            'internal_passes_after_bootstrap':3,'diffusion_steps':4,'canvas':8,
            'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),
                ROOT/'scripts/selective_memory_pilot.py',ROOT/'scripts/candidate_trajectory/selective_memory.py')},
            'development_sha256':sha(DATA/'processed/development.jsonl'),'test_used':False,
            'conditioning':'Identical inherited conditioning; native recomputation in each branch.'})
        write(directory/'offline-references.json',selected)
        model,t=load_model(cfg,checkpoint,False);model.eval()
        assert all(not p.requires_grad for p in model.parameters())
        versions={n:p._version for n,p in model.named_parameters()}
        stops={t.eos_token_id,t.convert_tokens_to_ids('<|eot_id|>')}
        records=[];checks={};verification_calls=0
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for index,row in enumerate(selected):
                # Model input is constructed from user messages only.
                prefix=t.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True)
                ids=torch.tensor([prefix+[t.mask_token_id]*CANVAS],device='cuda')
                region=torch.zeros_like(ids,dtype=torch.bool);region[:,-CANVAS:]=True
                parent={'tokens':ids,'memory':None,'conditioning':None,'generation_mask':region,
                    'completed_steps':0,'trajectory_calls':0,'configuration':{'temperature':0,'canvas':8},
                    'rng':{'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state().cpu()}}
                out=forward(model,parent,'all',t.mask_token_id,bootstrap=True);del out
                save_snapshot(directory/'parents'/f'{row["question_id"]}.pt',parent)
                if index==0:
                    loaded=load_snapshot(directory/'parents'/f'{row["question_id"]}.pt','cuda')
                    loaded['generation_mask']=loaded['generation_mask'].to('cuda')
                    assert all(torch.equal(parent[k],loaded[k]) for k in ('tokens','memory','conditioning'))
                    checks['parent_snapshot_roundtrip']=True
                for action in ACTIONS:
                    result={'question_id':row['question_id'],'family':row['family'],'question':row['question'],
                        'action':action,'target_answer':row['target_answer'],'execution_error':None,
                        'raw_response':'','parsed_answer':None,'correct':False}
                    try:
                        child,trace=continue_branch(model,parent,action,t.mask_token_id)
                        if index==0 and action in ('all','A'):
                            batch={'input_ids':ids,'attention_mask':torch.ones_like(ids)}
                            _,native_tokens,native_memory,native_cond=trajectory(model,batch,[action]+['all']*5,t.mask_token_id,False)
                            verification_calls+=8
                            assert torch.equal(child['tokens'],native_tokens)
                            assert torch.equal(child['memory'],native_memory) and torch.equal(child['conditioning'],native_cond)
                            checks['training_inference_path_parity_'+action]=True
                        if index==0 and action=='D':
                            repeated,_=continue_branch(model,parent,'A',t.mask_token_id);verification_calls+=7
                            assert all(torch.equal(repeated[k],first_A[k]) for k in ('tokens','memory','conditioning'))
                            checks['sibling_order_independent']=True
                        if index==0 and action=='A':first_A=clone(child)
                        generated=child['tokens'][0,-CANVAS:].tolist()
                        end=next((i for i,v in enumerate(generated) if v in stops),len(generated))
                        raw=t.decode(generated[:end],skip_special_tokens=False).strip()
                        parsed=int(raw) if re.fullmatch(r'[+-]?\d+',raw) else None
                        result.update(raw_response=raw,parsed_answer=parsed,correct=parsed==int(row['target_answer']),
                            token_ids=generated,has_stop=end<len(generated),trace=trace,
                            residual_masks=generated.count(t.mask_token_id),incremental_backbone_calls=7)
                    except Exception as exc:
                        # Keep runtime failures in the denominator. Verification
                        # assertion failures invalidate the experiment immediately.
                        if isinstance(exc,AssertionError):raise
                        result['execution_error']=repr(exc)
                        raise
                    finally:
                        append(directory/'responses.jsonl',result)
                    records.append(result)
                    write(directory/'status.json',{'stage':'generating','completed':len(records),'total':80})
        assert versions=={n:p._version for n,p in model.named_parameters()}
        report,pairs=summary(records)
        report['primary_backbone_calls']=16*(1+5*7)
        report['verification_backbone_calls']=verification_calls
        report['total_backbone_calls']=report['primary_backbone_calls']+verification_calls
        write(directory/'pairs.json',pairs);write(directory/'summary.json',report)
        checks.update(parent_independent=True,weights_unchanged=True,paired_by_id=True,
            same_compute_each_candidate=True,token_sampling_noise=False)
        write(directory/'verification.json',{'status':'passed','checks':checks})
        write(directory/'status.json',{'stage':'complete','completed':80,'total':80})
        print(json.dumps({'directory':str(directory),**report},indent=2),flush=True)
    except Exception:
        import traceback
        write(directory/'status.json',{'stage':'failed','traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
