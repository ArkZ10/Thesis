"""Stage 4: development-only, deterministic, matched-five-forward ablations."""
import json
import os
from pathlib import Path
import re
import traceback
from datetime import datetime,timezone
from types import SimpleNamespace
from reasoning_pilot import ROOT, DATA, CANVAS, PASSES, STEPS, dependencies, write, append, sha

CONDITIONS=['trained','injection_disabled','reset_memory','reset_memory_and_conditioning','untrained']


def main():
    import torch
    from safetensors.torch import load_file
    run=Path(json.loads((ROOT/'result/reasoning-pilot/latest.json').read_text())['directory'])
    directory=run/'evaluation-development'
    directory.mkdir(exist_ok=False)
    write(directory/'status.json',{'stage':'loading'})
    try:
        assert json.loads((run/'status.json').read_text())['stage']=='complete'
        torch,_=dependencies()
        from dllm.utils.models import get_model,get_tokenizer
        cfg=json.loads((run/'configuration.json').read_text())
        args=SimpleNamespace(model_name_or_path=str(run/f'checkpoints/checkpoint-{STEPS}'),
            backbone_name_or_path=cfg['backbone']['model_name_or_path'],dtype='bfloat16',
            attn_implementation=None,freeze_backbone=True)
        model=get_model(args,is_recurrent=True,training=False);model.eval()
        t=get_tokenizer(args)
        rows=[json.loads(s) for s in (DATA/'processed/development.jsonl').read_text().splitlines()]
        assert len(rows)==64
        write(directory/'configuration.json',{'conditions':CONDITIONS,'split':'development',
            'questions':64,'forwards_per_response':PASSES+1,'canvas':CANVAS,'temperature':0,
            'script_sha256':sha(Path(__file__)),'dataset_sha256':sha(DATA/'processed/development.jsonl'),
            'generation':'Parallel argmax after four internal forwards; stop at first EOT/EOS.',
            'reset_memory':'Replace memory by learned initial state before each nonbootstrap forward; preserve conditioning.',
            'reset_memory_and_conditioning':'Also zero conditioning before each nonbootstrap forward.',
            'injection_disabled':'Compute injector normally but replace its returned embeddings with original embeddings.',
            'test_split_used':False})
        all_results=[]
        stops={t.eos_token_id,t.convert_tokens_to_ids('<|eot_id|>')}
        for condition in CONDITIONS:
            if condition=='untrained':
                weights=load_file(str(run/'initial-checkpoint/recurrent_modules.safetensors'))
                for module in ('time_conditioner','mixer','updater','injector'):
                    prefix=module+'.'
                    getattr(model,module).load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
                del weights
            hook=None
            if condition=='injection_disabled':
                def bypass(module,args,out):
                    return args[0],torch.zeros_like(args[0]),out[2]
                hook=model.injector.register_forward_hook(bypass)
            for row in rows:
                # Construct exclusively from user messages: no labels or target lengths.
                prefix=t.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True)
                ids=torch.tensor([prefix+[t.mask_token_id]*CANVAS],device='cuda')
                state=cond=None
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for step in range(PASSES+1):
                        if step and condition.startswith('reset_memory'):
                            state=model.updater.get_initial_state(1,device=ids.device,dtype=model.get_input_embeddings().weight.dtype)
                            if condition=='reset_memory_and_conditioning':cond=torch.zeros_like(cond)
                        out=model(input_ids=ids,attention_mask=torch.ones_like(ids),state=state,
                                  past_cond=cond,t=torch.ones(1,device='cuda'),use_cache=False,return_dict=True)
                        state,cond=out.new_state,out.past_cond
                        if step==PASSES:
                            tokens=out.logits[0,-CANVAS:].argmax(-1).tolist()
                        del out
                end=next((i for i,x in enumerate(tokens) if x in stops),len(tokens))
                raw=t.decode(tokens[:end],skip_special_tokens=False).strip()
                parsed=int(raw) if re.fullmatch(r'[+-]?\d+',raw) else None
                result={'question_id':row['question_id'],'condition':condition,'question':row['question'],
                        'family':row['family'],'raw_response':raw,'token_ids':tokens,'parsed_answer':parsed,
                        'target_answer':row['target_answer'],'correct':parsed==int(row['target_answer']),
                        'unclassifiable':parsed is None,'has_stop':end<len(tokens),'backbone_calls':PASSES+1}
                append(directory/'responses.jsonl',result);all_results.append(result)
                write(directory/'status.json',{'stage':'evaluating','completed':len(all_results),'total':64*len(CONDITIONS)})
            if hook:hook.remove()
        grouped={c:{r['question_id']:r for r in all_results if r['condition']==c} for c in CONDITIONS}
        assert all(len(g)==64 for g in grouped.values())
        summary={}
        for c,g in grouped.items():
            correct=sum(r['correct'] for r in g.values())
            summary[c]={'correct':correct,'total':64,'accuracy_percent':100*correct/64,
                'unclassifiable':sum(r['unclassifiable'] for r in g.values()),
                'by_family':{f:sum(r['correct'] for r in g.values() if r['family']==f) for f in sorted({r['family'] for r in g.values()})}}
        paired={}
        for c in CONDITIONS[1:]:
            a=grouped['trained'];b=grouped[c]
            paired[c]={'trained_only_correct':sum(a[k]['correct'] and not b[k]['correct'] for k in a),
                       'control_only_correct':sum(b[k]['correct'] and not a[k]['correct'] for k in a),
                       'both_correct':sum(a[k]['correct'] and b[k]['correct'] for k in a),
                       'neither_correct':sum(not a[k]['correct'] and not b[k]['correct'] for k in a),
                       'different_answers':sum(a[k]['parsed_answer']!=b[k]['parsed_answer'] for k in a)}
        write(directory/'summary.json',{'conditions':summary,'paired_vs_trained':paired,
            'total_backbone_calls':len(all_results)*(PASSES+1),'held_out_test_used':False})
        write(directory/'status.json',{'stage':'complete','completed':len(all_results),'total':len(all_results)})
        print(json.dumps(summary,indent=2),flush=True)
    except Exception:
        write(directory/'status.json',{'stage':'failed','traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
