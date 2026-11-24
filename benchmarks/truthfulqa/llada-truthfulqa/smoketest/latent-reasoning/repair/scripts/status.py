"""Read-only progress; works with the base Python interpreter."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'result'
for path in sorted((root/'diagnostics').glob('*.jsonl')):
    rows=[]
    for line in path.read_bytes().splitlines(keepends=True):
        if line.endswith(b'\n'):rows.append(json.loads(line))
    if path.stem=='equivalence': print('Equivalence checks:',len(rows));continue
    print(path.stem, 'attempted:',len(rows),'nonempty:',sum(bool(r.get('response')) for r in rows))
for directory in root.glob('verification_*'):
    manifest=json.loads((directory/'manifest.json').read_text())
    print(directory.name,'status:',manifest['status'])
    p=directory/'metrics.json'
    if p.exists():
        m=json.loads(p.read_text());print({k:m.get(k) for k in ['attempted','successful_generations','successfully_graded_responses','correct_per_all_100']})
