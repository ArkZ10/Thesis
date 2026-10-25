#!/usr/bin/env python3
"""Read-only progress; no model loading or API calls."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
latest = root/'result/latest.json'
if not latest.exists():
    raise SystemExit('No experiment prepared yet')
directory = Path(json.loads(latest.read_text())['result_dir'])
meta = json.loads((directory/'manifest.json').read_text())
print('Result:', directory)
print('Stage:', meta['status'], '| active arm:', meta.get('active_arm', 'none'))
for arm in ['control','fixed','recurrent']:
    counts = []
    for name in ['responses','judgments']:
        path = directory/arm/(name+'.jsonl')
        rows = {}
        if path.exists():
            lines = path.read_bytes().splitlines(keepends=True)
            for line in lines:
                if line.endswith(b'\n'):
                    record = json.loads(line)
                    rows[record['id']] = record
        counts.append(sum(r.get('status')=='success' for r in rows.values()))
    print(f'{arm}: generated {counts[0]}/100 | graded {counts[1]}/100')
