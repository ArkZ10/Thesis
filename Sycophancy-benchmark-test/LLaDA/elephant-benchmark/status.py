"""Print stage and per-worker progress without touching running journals."""
from datetime import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent
status = root / 'results/run_status.json'
if status.exists():
    print(status.read_text().strip())
for path in sorted((root / 'results').glob('*/*/worker-*/metadata.json')):
    metadata = json.loads(path.read_text())
    progress = path.with_name('progress.json')
    data = json.loads(progress.read_text()) if progress.exists() else {}
    done = metadata.get('completed_rows', data.get('completed', 0))
    print(f'{path.parent.relative_to(root / "results")}: {done}/{metadata["expected_rows"]}')
