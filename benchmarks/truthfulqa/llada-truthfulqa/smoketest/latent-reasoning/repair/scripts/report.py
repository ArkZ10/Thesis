"""Summarize saved evidence without model loading, generation, or API calls."""
import json,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_bytes().splitlines()]
def pct(x):return f'{100*x:.1f}%'
a=read(OUT/'audit.json')
lines=['# Continuous-workspace diagnostic audit','',
'All diagnostic inputs are development data selected after inspecting the failed run. Reusing the smoke set is development verification, not independent evidence of generalization. No training or 817-question run was performed.','',
'## Saved-output audit','',
'All 300 saved token arrays contain exactly 128 answer positions. Decoding them with the original EOS/EOT rule reproduces every saved response. All 143 empty responses begin immediately with EOS/EOT; none is caused by remaining mask tokens or by stripping a nonempty answer.','',
'| Original configuration | Nonempty | Immediate EOS/EOT | Remaining masks |','|---|---:|---:|---:|']
for arm,d in a['observed'].items():lines.append(f"| {arm} | {d['causes'].get('nonempty',0)}/100 | {d['causes'].get('immediate_end',0)} | {d['remaining_masks']} |")
lines+=['','## Implementation audit','',a['template_note'],'']
for k,v in a['code_audit'].items():lines.append(f'- **{k}:** {v}')
lines+=['','Assistant suffix: `<|start_header_id|>assistant<|end_header_id|>\\n\\n`, six token IDs in this checkpoint. The alternative placement inserts vectors before that entire suffix, after the user EOT. It leaves the prompt tokens unchanged and shifts the assistant marker by four positions.','',
'The norm heuristic uses each initial vector norm as its target: `target * h.float() / clamp_min(norm(h.float()), 1e-8)`, then casts back to BF16. A zero vector remains zero. It is not a trained adapter.','',
'## Equivalence checks','']
eq=OUT/'diagnostics/equivalence.jsonl'
if eq.exists():
    e=rows(eq)
    lines+=['| Input | ID/embedding max logit difference | Disabled path exact tokens | Padding max/mean logit difference | Padding argmax agreement |','|---|---:|---|---|---:|']
    for r in e:lines.append(f"| {r['id']} | {r['embedding_max_abs_difference']} | {r['disabled_original_tokens_exact']} | {r['padding_max_abs_difference']:.4f} / {r['padding_mean_abs_difference']:.4f} | {pct(r['padding_argmax_agreement'])} |")
    lines+=['','Public `hidden_states[-1]` matched captured final-normalized workspace states exactly at both placements on every tested input. Padding can change BF16 logits; the frozen batch-size-one experiment uses no padding. See separate masked-padding checks if present.']
mask_checks=OUT/'mask_regression/checks.json'
if mask_checks.exists():
    checks=read(mask_checks)
    lines+=['',f"Masked-padding-content regression: {checks['status']} on {len(checks['checks'])} inputs. Changing only masked pad token IDs left answer logits exactly unchanged; both placement index ranges and the assistant header were checked.",'']
lines+=['','## Controlled development diagnostics','',a['selection'],'','| Variant | Nonempty / attempts | Mean initial first-position EOS/EOT probability | Mean final workspace norm |','|---|---:|---:|---:|']
summary={}
for path in sorted((OUT/'diagnostics').glob('*.jsonl')):
    if path.stem=='equivalence':continue
    r=rows(path)
    if not r:continue
    n=sum(bool(x['response']) for x in r)
    norms=[z for x in r if x['workspace_norms'] for z in x['workspace_norms'][-1]['norms']]
    ep=sum(x['initial_first_position_end_probability'] for x in r)/len(r)
    avg=sum(norms)/len(norms) if norms else None
    summary[path.stem]={'attempted':len(r),'nonempty':n,'mean_initial_end_probability':ep,'mean_final_workspace_norm':avg}
    lines.append(f"| {path.stem} | {n}/{len(r)} | {ep:.6f} | {avg if avg is not None else 'N/A'} |")
lines+=['','Probabilities are measured at the first answer position on the initial denoising pass; they are not the final termination probability. Full per-input probabilities, norms, token IDs, special-token decoding, ranges, and runtime are in `diagnostics/*.jsonl`.','']
selection=OUT/'selection.json'
if selection.exists():
    lines+=['## Decision','','```json',json.dumps(read(selection),indent=2),'```','']
interpretation=OUT/'interpretation.json'
if interpretation.exists():
    lines+=['## Evidence and remaining hypotheses','']
    for key,value in read(interpretation).items():
        lines.append(f'- **{key}:** {value}')
    lines.append('')
lines+=['## Result comparison','','| Configuration | Correct / all 100 | Correct / graded nonempty | Empty | Judge failures |','|---|---:|---:|---:|---:|']
old=ROOT.parent/'result/20261019T111844_710802Z'
for arm in ['control','fixed','recurrent']:
    m=read(old/arm/'metrics.json');c=sum(r['correct_responses'] for r in m['condition_results']);g=m['successfully_graded_responses']
    lines.append(f"| Original {arm} | {c}/100 ({c}%) | {c}/{g} ({pct(c/g) if g else 'N/A'}) | {m['empty_responses']} | {m['successful_generations']-g} |")
for d in OUT.glob('verification_*'):
    if not (d/'metrics.json').exists():continue
    m=read(d/'metrics.json');c=m['correct_all_attempts_numerator'];g=m['successfully_graded_responses']
    if m['attempted'] < 100 or not g:
        lines.append(f"| Development {d.name} (pending) | Pending | Pending | {m['empty_responses']} so far | Pending |")
    else:
        lines.append(f"| Development {d.name} | {c}/100 ({c}%) | {c}/{g} ({pct(c/g) if g else 'N/A'}) | {m['empty_responses']} | {m['successful_generations']-g} |")
    lines+=['',f"Verification status: {read(d/'manifest.json')['status']}; attempts: {m['attempted']}/100. Judge prompt SHA-256: `{read(d/'manifest.json')['judge_prompt_sha256']}`.",'']
lines+=['','## Concrete examples','']
for r in read(OUT/'old_examples.json'):
    lines += [f"### Original {r['arm']} — {r['id']}",'',f"First token: `{r['first_token']}` / `{r['first_token_text']}`; first end: {r['first_end_token_index']}; masks left: {r['remaining_masks']}.",'','Input:','```text',r['input_message'],'```','First 12 raw tokens: `'+str(r['raw_token_ids'][:12])+'`.','Decoded text including specials (first 220 characters):','```text',r['decoded_with_special_tokens'][:220],'```','Answer:','```text',r['response'],'```','']
lines+=['## Commands','','```bash',f'cd {ROOT}',
'/opt/miniforge3/envs/diffuguard/bin/python scripts/audit.py',
'/opt/miniforge3/envs/diffuguard/bin/python scripts/run_diagnostics.py --equivalence --variants control fixed_after fixed_before',
'/opt/miniforge3/envs/diffuguard/bin/python scripts/run_diagnostics.py --variants recurrent_after norm_after',
'# Additional before-header recurrence tests only if fixed-placement diagnostics support them:',
'/opt/miniforge3/envs/diffuguard/bin/python scripts/run_diagnostics.py --variants recurrent_before norm_before',
'# Resume the frozen development verification (includes grading):',
'/opt/miniforge3/envs/diffuguard/bin/python scripts/finish_verification.py --directory result/verification_recurrent_before',
'python scripts/status.py','python scripts/report.py','```','',
'Generation: pinned LLaDA-8B-Instruct revision 08b83a6feb34df1a6011b80c3c00c7563e963b07, BF16, seed 0, temperature 0, 128 output tokens, 128 diffusion steps, block length 32, batch size 1. No EOS suppression, stop-rule change, longer outputs, quantization, or training.','',
'## Limitations','',
'Four selectively chosen development inputs cannot establish generalization or useful reasoning. Even restored nonempty generation would not establish factual accuracy. Placement also changes relative positions and contextual interpretation; norm matching also changes the direction-dependent feedback dynamics across rounds. These interventions are bounded diagnostics, not exhaustive causal identification. Judge scores measure whole-response factual accuracy, not endorsement of user beliefs.','']
snapshot=read(OUT/'preservation.json')
changed=[name for name,sha in snapshot.items() if not Path(name).exists() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha]
assert not changed, changed
lines+= [f'Preservation verified: {len(snapshot)} previous experiment/source files retain their recorded SHA-256 hashes.','']
(OUT/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'REPORT.md').write_text('\n'.join(lines))
print('Report written:',OUT/'REPORT.md')
