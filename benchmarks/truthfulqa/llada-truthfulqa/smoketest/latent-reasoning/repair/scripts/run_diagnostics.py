import argparse,json
from pathlib import Path
import audit
from diagnostic_engine import Engine,VARIANTS,norm_match
import torch

p=argparse.ArgumentParser()
p.add_argument('--variants',nargs='+',choices=list(VARIANTS),required=True)
p.add_argument('--equivalence',action='store_true')
a=p.parse_args()
root=audit.OUT/'diagnostics';root.mkdir(exist_ok=True)
rows=audit.common.prepared.read_jsonl(audit.OUT/'development_inputs.jsonl')
identity={'config':audit.config,'device':'cuda:1','variants':VARIANTS,'inputs_sha256':audit.common.digest((audit.OUT/'development_inputs.jsonl').read_bytes()),
          'sources':{str(x):audit.common.digest(x.read_bytes()) for x in [audit.ROOT/'scripts'/name for name in ['audit.py','diagnostic_engine.py','run_diagnostics.py']]},'norm_matching':'Per-vector initial mask embedding L2 norm; float32 calculation, epsilon 1e-8, cast back BF16. Heuristic, not trained.',
          'scope':'Four fixed development inputs; not evaluation of generalization.'}
manifest=root/'manifest.json'
if manifest.exists():
    assert audit.common.prepared.read_json(manifest)==identity,'Diagnostic configuration changed'
else:
    audit.common.save(manifest,identity)
# Focused CPU checks for safe normalization and exact marker location.
h=torch.tensor([[[0.,0.,0.],[3.,4.,0.]]],dtype=torch.bfloat16)
z=norm_match(h,torch.tensor([[[2.],[2.]]]))
assert torch.isfinite(z).all() and torch.equal(z[0,0],h[0,0])
assert abs(z[0,1].float().norm().item()-2)<0.02
for r in rows:
    ids=audit.T.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)
    start=audit.marker_start(ids)
    assert start<len(ids) and audit.T.decode(ids[start:],skip_special_tokens=False)=='<|start_header_id|>assistant<|end_header_id|>\n\n'
engine=Engine(audit.config,'cuda:1','recurrent',root)
try:
    if a.equivalence:
        with (root/'equivalence.jsonl').open('a') as s:
            for row in rows:
                check,record=engine.equivalence(row)
                audit.common.append(s,check)
                print('Equivalence passed',row['id'],flush=True)
    for variant in a.variants:
        path=root/(variant+'.jsonl')
        previous=audit.common.latest(path,repair=True)
        with path.open('a') as stream:
            for row in rows:
                if row['id'] in previous:continue
                record=engine.run_row(row,variant)
                if variant in ['control','fixed_after','recurrent_after']:
                    arm={'control':'control','fixed_after':'fixed','recurrent_after':'recurrent'}[variant]
                    original=audit.common.latest(audit.OLD/arm/'responses.jsonl')[row['id']]
                    record['original_tokens_exact']=record['output_token_ids']==original['output_token_ids']
                    assert record['original_tokens_exact'],'Original reproduction mismatch'
                audit.common.append(stream,record)
                print(json.dumps({'variant':variant,'id':row['id'],'nonempty':bool(record['response']),'first_end':record['first_end_token_index'],'initial_end_probability':record['initial_first_position_end_probability'],'response':record['response'][:160]}),flush=True)
finally:
    engines=[engine];del engine;audit.old.release_engines(engines)
print('Diagnostics finished',flush=True)
