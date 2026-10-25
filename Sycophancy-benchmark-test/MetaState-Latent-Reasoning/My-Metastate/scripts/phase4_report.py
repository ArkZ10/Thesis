"""Combine phase-4 generation checks with saved qualitative manual review."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
run=Path(json.loads((ROOT/'result/phase4/latest.json').read_text())['directory'])
rows=[json.loads(s) for s in (run/'responses.jsonl').read_text().splitlines()]
reviews=[json.loads(s) for s in (run/'manual_review.jsonl').read_text().splitlines()]
assert len(rows)==len(reviews)==20
lookup={r['id']:r for r in reviews}
assert len(lookup)==20 and set(lookup)=={r['id'] for r in rows}
config=json.loads((run/'configuration.json').read_text())
for name,sha in config['checkpoint_sha256'].items():
    assert hashlib.sha256((Path(config['checkpoint'])/name).read_bytes()).hexdigest()==sha
assert all(r['forward_calls']==129 and r['memory_carry_links']==128 for r in rows)
summary=json.loads((run/'generation_summary.json').read_text())
summary.update(status='phase4_complete',qualitative_review={'reviewer':'Codex manual inspection against dataset reference answers','scope':'Main answer only; incidental claims not comprehensively fact checked','reference_consistent_main_answers':sum(r['main_answer_assessment']=='REFERENCE_CONSISTENT' for r in reviews),'reference_inconsistent_main_answers':sum(r['main_answer_assessment']=='REFERENCE_INCONSISTENT' for r in reviews),'uncertain_main_answers':sum(r['main_answer_assessment']=='UNCERTAIN' for r in reviews),'false_belief_endorsements_in_supports_incorrect':sum(lookup[r['id']]['selected_incorrect_answer_endorsed'] is True for r in rows if r['condition']=='supports_incorrect'),'supports_incorrect_examples':sum(r['condition']=='supports_incorrect' for r in rows)},verbosity={'heuristic':'More than 60 whitespace-separated words; diagnostic only','responses_over_60_words':sum(len(r['response'].split())>60 for r in rows)},checkpoint_files_unchanged=True,next_phase_requires_user_confirmation=True,phase5_started=False)
with (run/'results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['id','question_id','condition','question','user_message','reference_answer','selected_incorrect_answer','response','main_answer_assessment','selected_incorrect_answer_endorsed','empty','budget_exhausted_without_stop','remaining_mask_tokens','review_reason'])
    writer.writeheader()
    for r in rows:
        review=lookup[r['id']]
        writer.writerow({k:r[k] for k in ('id','question_id','condition','question','selected_incorrect_answer','response','empty','budget_exhausted_without_stop','remaining_mask_tokens')} | {'user_message':r['messages'][0]['content'],'reference_answer':r['correct_answer'],'main_answer_assessment':review['main_answer_assessment'],'selected_incorrect_answer_endorsed':review['selected_incorrect_answer_endorsed'],'review_reason':review['reason']})
(run/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2))
