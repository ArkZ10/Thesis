"""Read-only progress display. No model loading or API calls."""
from collections import Counter
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent / 'results'


def load(path):
    return json.loads(path.read_text())


def main():
    state_path = ROOT / 'status.json'
    if not state_path.exists():
        print('Run has not started.')
        return
    state = load(state_path)
    print('Run:', ROOT)
    print('Stage:', state.get('stage'), '| Supervisor PID:', state.get('supervisor_pid'))
    for field, label in (('started_at_utc', 'Started'), ('completed_at_utc', 'Completed')):
        if state.get(field):
            stamp = datetime.fromisoformat(state[field]).astimezone(ZoneInfo('Asia/Taipei'))
            print(f'{label}: {stamp:%Y-%m-%d %H:%M:%S %Z}')
    if state.get('error'):
        print('Error:', state['error'])
    manifest_path = ROOT / 'data/manifests/llada_30.json'
    if not manifest_path.exists():
        print('Preparing the selection. See logs/supervisor.log.')
        return
    ids = [r['problem_id'] for r in load(manifest_path)['problems']]
    answers = {}
    for path in (ROOT / 'data/solved/llada_30').glob('**/*.json'):
        item = load(path)
        for attempt in item.get('attempts', []):
            if attempt['model_id'] == 'llada/llada_8b_instruct':
                answers[item['problem_id']] = attempt
    print(f'Generated: {len(answers)}/{len(ids)}')
    for gpu in (0, 1):
        assigned = ids[gpu::2]
        done = sum(problem_id in answers for problem_id in assigned)
        worker = state.get('workers', {}).get(str(gpu), {})
        print(f'GPU {gpu}: {done}/{len(assigned)} saved | worker {worker.get("stage", "pending")} | PID {worker.get("pid", "-")}')
        logfile = ROOT / f'logs/gpu-{gpu}.log'
        if logfile.exists():
            with logfile.open('rb') as stream:
                stream.seek(max(0, logfile.stat().st_size - 4000))
                lines = stream.read().decode(errors='replace').splitlines()
            progress = [line for line in lines if 'LLaDA query' in line]
            if progress:
                print(' ', progress[-1].split(' - ')[-1])
    verdicts = [load(path) for path in (ROOT / 'judgements/llada_30').glob('**/*.json')]
    votes = sum(len(v.get('judgements', [])) for v in verdicts)
    print(f'Judged: {len(verdicts)}/{len(ids)} answers | {votes}/{3 * len(ids)} saved verdicts')
    if verdicts:
        print('Majority verdicts:', dict(Counter(v['majority_vote'] for v in verdicts)))
        print(f'Judge cost (standard token rates): ${sum(v["cost"]["cost"] for v in verdicts):.6f}')
    if (ROOT / 'summary.json').exists():
        print('Final summary:', ROOT / 'summary.json')
    print('Logs:', ROOT / 'logs')


if __name__ == '__main__':
    main()
