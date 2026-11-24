"""Resume the full experiment on two GPUs and score only complete stages."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = Path('/opt/miniforge3/envs/diffuguard/bin/python')
LOGS = ROOT / 'logs'
LOGS.mkdir(exist_ok=True)
ENV = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}


def state(stage, **extra):
    document = {'stage': stage, 'updated_at_utc': datetime.now(timezone.utc).isoformat(), **extra}
    (ROOT / 'results/run_status.json').write_text(json.dumps(document, indent=2) + '\n')
    print(json.dumps(document), flush=True)


def active_inference(task, split, worker):
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if 'llada_elephant.inference' not in args or (path / 'cwd').resolve() != ROOT:
                continue
            values = lambda flag, default=None: args[args.index(flag)+1] if flag in args else default
            if values('--task') == task and values('--split') == split and values('--worker-index', '0') == str(worker):
                return int(path.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return None


def complete(task, split, worker):
    path = ROOT / f'results/{task}/{split}/worker-{worker}/metadata.json'
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return data.get('completed_rows') == data['expected_rows']


def launch(task, split, worker=0, workers=1, datasets=None):
    if complete(task, split, worker):
        return None
    pid = active_inference(task, split, worker)
    if pid:
        print(f'Adopting existing {task}/{split} worker {worker}, PID {pid}', flush=True)
        return pid
    command = [str(PYTHON), '-u', '-m', 'llada_elephant.inference', '--task', task, '--split', split,
               '--device', f'cuda:{worker}', '--worker-index', str(worker), '--workers', str(workers)]
    if datasets:
        command += ['--datasets', *datasets]
    with (LOGS / f'{task}_{split}_worker{worker}.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    return process


def wait_worker(task, split, worker, process):
    while not complete(task, split, worker):
        if isinstance(process, subprocess.Popen):
            if process.poll() is not None:
                raise RuntimeError(f'{task}/{split} worker {worker} exited {process.returncode}; see logs')
        elif process is not None:
            if not active_inference(task, split, worker):
                if complete(task, split, worker):
                    return
                raise RuntimeError(f'Existing {task}/{split} worker {worker} stopped before completion')
        else:
            raise RuntimeError(f'{task}/{split} worker {worker} is incomplete')
        time.sleep(10)


def command(module, args, log_name):
    with (LOGS / log_name).open('a') as stream:
        subprocess.run([str(PYTHON), '-u', '-m', module, *args], cwd=ROOT, env=ENV,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def saved_advice_datasets():
    """Keep an interrupted run's inputs even if new datasets become available."""
    paths = sorted((ROOT / 'results/advice/full').glob('worker-*/metadata.json'))
    if not paths:
        return None
    sources = json.loads(paths[0].read_text())['input_sha256']
    for path in paths[1:]:
        if json.loads(path.read_text())['input_sha256'] != sources:
            raise ValueError('Existing advice workers used different input datasets')
    return [Path(filename).stem for filename in sources]


def main():
    validation = json.loads((ROOT / 'reports/generation_validation.json').read_text())
    if not validation['optimized_vs_reference_single_exact_tokens'] or not validation['batched_sample_matches_reference_label']:
        raise RuntimeError('Generation validation has not passed')
    sample = json.loads((ROOT / 'results/moral/sample/summary.json').read_text())
    if sample['counts']['denominator_pairs'] != 10 or sample['execution_error_responses']:
        raise RuntimeError('Sample moral pipeline did not pass')
    state('full_moral_and_sample_advice')
    moral0 = launch('moral', 'full', 0, 2)
    # The initial sample advice run was started on GPU 1 with worker index 0.
    sample_advice = active_inference('advice', 'sample', 0)
    if not complete('advice', 'sample', 0):
        if not sample_advice:
            raise RuntimeError('Run the sample advice generation before this supervisor')
        wait_worker('advice', 'sample', 0, sample_advice)
    moral1 = launch('moral', 'full', 1, 2)
    command('llada_elephant.judge', ['--split', 'sample'], 'judge_sample.log')
    state('full_moral', sample_advice_scored=True)
    wait_worker('moral', 'full', 0, moral0)
    wait_worker('moral', 'full', 1, moral1)
    command('llada_elephant.scoring', ['--split', 'full'], 'score_moral_full.log')
    datasets = saved_advice_datasets()
    if datasets is None:
        # New runs may include newly available data; resumed runs keep their
        # original input identity so their fingerprints and denominators match.
        with (LOGS / 'fetch_ss.log').open('a') as stream:
            subprocess.run([str(PYTHON), 'fetch_ss.py'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        datasets = ['AITA-YTA', 'OEQ']
        if (ROOT / 'data/full/SS.csv').exists():
            datasets.append('SS')
    ss_status = 'available' if 'SS' in datasets else 'blocked: OSF scheduled maintenance'
    state('full_advice', full_moral_scored=True, datasets=datasets, full_ss=ss_status)
    workers = [launch('advice', 'full', worker, 2, datasets) for worker in (0, 1)]
    for worker, process in enumerate(workers):
        wait_worker('advice', 'full', worker, process)
    state('judging_full_advice', full_moral_scored=True)
    command('llada_elephant.judge', ['--split', 'full'], 'judge_full.log')
    state('available_data_complete', full_ss=ss_status,
          reports=['reports/moral_sample.md', 'reports/moral_full.md', 'reports/advice_sample.md', 'reports/advice_full.md'])


if __name__ == '__main__':
    # Advisory lock prevents two supervisors from launching duplicate GPU workers.
    import fcntl
    with (ROOT / 'results/supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as e:
            state('failed', error=f'{type(e).__name__}: {e}')
            raise
