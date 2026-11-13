"""Retokenize the reviewed pilot for both large checkpoints without altering splits."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'data/processed/vae-pilot-1000q-v1'
    out = ROOT/'data/processed/vae-large-comparable-v1'
    out.mkdir(exist_ok=False)
    tokenizers = [AutoTokenizer.from_pretrained(ROOT/'models'/name, local_files_only=True)
                  for name in ['Qwen3-4B-Base', 'Qwen3-8B-Base']]
    assert tokenizers[0].get_vocab()==tokenizers[1].get_vocab()
    assert tokenizers[0].eos_token_id==tokenizers[1].eos_token_id
    stats, all_rows, qids, targets = {}, {}, {}, {}
    for split in ['train','dev']:
        rows = []
        for line in (source/f'{split}.jsonl').open():
            row = json.loads(line)
            text = row['reasoning_block']
            a = tokenizers[0](text,add_special_tokens=False)['input_ids']
            b = tokenizers[1](text,add_special_tokens=False)['input_ids']
            assert a==b
            assert tokenizers[0].decode(a)==text
            assert tokenizers[1].decode(b)==text
            seq = a+[tokenizers[0].eos_token_id]
            assert seq==row['input_ids']
            assert len(seq)<=128
            rows.append({**row,'input_ids':seq,'token_count_including_eos':len(seq)})
        with (out/f'{split}.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        qids[split] = {r['problem_id'] for r in rows}
        targets[split] = {r['reasoning_block'] for r in rows}
        all_rows[split] = rows
        lengths=sorted(len(r['input_ids']) for r in rows)
        stats[split]=dict(questions=len(qids[split]),sentences=len(rows),tokens=sum(lengths),
                          min_tokens=min(lengths),max_tokens=max(lengths),median_tokens=lengths[len(lengths)//2])
    assert not qids['train'] & qids['dev']
    assert not targets['train'] & targets['dev']
    # Manually reviewed long expressions for memory/gradient checks: a finite
    # product identity and a rationalization calculation, avoiding the next
    # longest source row, which ends in an incomplete prose fragment.
    specifications=[('dart-math-hard:train-00001-of-00002:36511',2),
                    ('dart-math-hard:train-00000-of-00002:240027',4)]
    smoke=[next(r for r in all_rows['train'] if (r['parent_id'],r['sentence_index'])==key)
           for key in specifications]
    with (out/'smoke.jsonl').open('w') as f:
        for r in smoke:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    manifest=dict(format='vae_sentence_splits_v1',language='Math',train=['train.jsonl'],validation=['dev.jsonl'],test=[])
    (out/'pilot.vae.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),source=str(source),
                statistics=stats,vocabularies_identical=True,both_tokenizers_roundtrip_verified=True,
                matches_previous_pilot_tokenization=True,question_overlap=0,sentence_overlap=0,
                truncation=False,test_used=False,new_quality_filtering=False,
                quality_note='Same reviewed subset as the small pilot to support comparison; it is not mathematically certified. Only the two smoke-training targets were separately inspected here.',
                smoke_targets=[dict(parent_id=r['parent_id'],sentence_index=r['sentence_index'],tokens=len(r['input_ids'])) for r in smoke],
                files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
                       for p in sorted(out.iterdir()) if p.is_file()])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
