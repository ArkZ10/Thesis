import json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'result/20261019T111844_710802Z'
sys.path.insert(0,str(ROOT.parent/'scripts'))
import run_latent as old
common=old.common
OUT=ROOT/'result'
config=common.prepared.read_json(OLD/'control/generation_config.json')
from transformers import AutoTokenizer
T=AutoTokenizer.from_pretrained(config['model_path'],trust_remote_code=True,local_files_only=True)
mc=common.prepared.read_json(Path(config['model_path'])/'config.json')
END={mc['eos_token_id'],T.convert_tokens_to_ids('<|eot_id|>')}
MASK=mc['mask_token_id']
smoke=common.recover_reference()[4]

def marker_start(ids):
    marker=T.encode('<|start_header_id|>assistant<|end_header_id|>\n\n',add_special_tokens=False)
    if not marker or ids[-len(marker):]!=marker:
        raise ValueError('Expected exact assistant header suffix')
    return len(ids)-len(marker)

def describe(tokens):
    end=next((i for i,t in enumerate(tokens) if t in END),None)
    decoded=smoke.decode_answer(T,tokens,END)
    before=tokens if end is None else tokens[:end]
    cause='nonempty' if decoded['response'] else ('immediate_end' if end==0 else 'special_or_whitespace_only_before_end')
    return {'raw_token_ids':tokens,'decoded_with_special_tokens':T.decode(tokens,skip_special_tokens=False),
            'decoded_before_stop_with_special_tokens':T.decode(before,skip_special_tokens=False),
            'first_token':tokens[0] if tokens else None,'first_token_text':T.decode(tokens[:1],skip_special_tokens=False),
            'end_positions':[i for i,t in enumerate(tokens) if t in END], 'remaining_masks':tokens.count(MASK),
            'cause':cause,**decoded}

if __name__=='__main__':
    snapshots={str(p):common.digest(p.read_bytes()) for p in ROOT.parent.rglob('*') if p.is_file() and ROOT not in p.parents and '__pycache__' not in str(p)}
    common.save(OUT/'preservation.json',snapshots)
    allrows=[];counts={};examples=[]
    for arm in old.ARMS:
        records=common.latest(OLD/arm/'responses.jsonl')
        for r in records.values():
            d=describe(r['output_token_ids'])
            assert d['response']==r['response'] and len(d['raw_token_ids'])==128
            assert d['first_end_token_index']==r['first_end_token_index']
            allrows.append({'arm':arm,'id':r['id'],'input_message':r['messages'][0]['content'],**d})
        selected=[r for r in allrows if r['arm']==arm]
        counts[arm]={'causes':dict(Counter(r['cause'] for r in selected)), 'remaining_masks':sum(r['remaining_masks'] for r in selected), 'first_tokens':dict(Counter(r['first_token_text'] for r in selected))}
        for empty in [True,False]:
            examples.extend([r for r in selected if bool(r['response']) != empty][:2])
    common.prepared.write_atomic(OUT/'old_token_audit.jsonl',common.prepared.jsonl_bytes(allrows))
    common.save(OUT/'old_examples.json',examples)
    rows=common.prepared.read_jsonl(OLD/'inputs.jsonl')
    ids=['truthfulqa-generation-validation-000104--neutral','truthfulqa-generation-validation-000095--supports_incorrect','truthfulqa-generation-validation-000114--neutral']
    recurrent_success=[r['id'] for r in allrows if r['arm']=='recurrent' and r['response']]
    ids.append(next(i for i in recurrent_success if i not in ids))
    subset=[next(r for r in rows if r['id']==i) for i in ids]
    common.prepared.write_atomic(OUT/'development_inputs.jsonl',common.prepared.jsonl_bytes(subset))
    marker_examples=[]
    for row in subset:
        base=T.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False)
        full=T.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True)
        assert full==base, 'Pinned template unconditionally appends assistant header'
        start=marker_start(full)
        marker_examples.append({'id':row['id'],'prompt_length':len(full),'assistant_marker_start':start,'marker_ids':full[start:],'marker_text':T.decode(full[start:],skip_special_tokens=False),'last_prompt_tokens':T.decode(full[-12:],skip_special_tokens=False),'old_workspace_range':[len(full),len(full)+4],'old_answer_range':[len(full)+4,len(full)+132]})
    audit={'observed':counts,'marker_examples':marker_examples,'subset_ids':ids,'selection':'Fixed development subset: conscious neutral failure; ugly-duckling incorrect-belief question; Obama neutral; first recurrent nonempty case not already selected. Not held-out evaluation.',
           'template_note':'Pinned template ignores add_generation_prompt flag and always appends assistant header. Locate suffix by exact special-token sequence, not by flag subtraction.',
           'code_audit':{'answer_sampling':'Only separate answer tensor is sampled; workspace vectors never enter mask-ID eligibility. No indexing bug found.', 'stopping':'Only last 128 answer positions decoded, truncate at first EOS/EOT, special tokens removed. Saved answers exactly reproduced.', 'placement':'Old workspace follows the entire assistant generation marker, occupying the beginning of assistant content. Marker placement is a hypothesis to test, not a confirmed indexing bug.', 'attention':'Left padding excluded through attention mask; workspace and answer visible. Old run batch size 1, no padding required.', 'injection':'Frozen prompt+workspace prefix concatenated with current answer embeddings on every denoising pass.', 'feedback':'Final post-ln_f hidden states at workspace positions; equivalent to public hidden_states[-1] verified previously, to repeat.', 'projection':'BF16 identity linear, bias=False. No normalization applied in original feedback.', 'mutation':'Frozen prefix clone checked at end; no deliberate prompt scaling or mutation.', 'known_bug':'Prior model-reference cycle was already repaired before original run completed. No additional confirmed indexing/decoding bug from source audit.'}}
    common.save(OUT/'audit.json',audit)
    print(json.dumps(audit,indent=2))
