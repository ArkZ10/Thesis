"""Detached, resumable two-GPU BrokenMath run with all artifacts outside upstream."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from filelock import FileLock, Timeout

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / 'broken-math'
DEFAULT_RESULTS = HERE / 'results'
PYTHON = Path('/opt/miniforge3/envs/diffuguard/bin/python')
MODEL_ID = 'llada/llada_8b_instruct'
PROJECT = 'llada_30'
JUDGE = 'openai/gpt-5-mini-medium'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(root, gpu=None):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(root / 'source/src'), PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def command(root, script, *args):
    return [str(PYTHON), '-u', str(root / 'source/scripts' / script), *map(str, args)]


def prepare(root):
    """Freeze source/config/data once; no writes to the upstream repository."""
    marker = root / 'setup.json'
    if marker.exists():
        setup = read(marker)
        for relative, expected in setup['snapshot_sha256'].items():
            if digest(root / relative) != expected:
                raise RuntimeError(f'Snapshot changed: {relative}; use a new results directory.')
        return
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    for name in ('src', 'scripts'):
        shutil.copytree(UPSTREAM / name, root / 'source' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=True)
    copies = [
        'configs/projects/llada_30.yaml', 'configs/solvers/llada_30.yaml',
        'configs/setting/llada_30.yaml', 'configs/models/llada/llada_8b_instruct.yaml',
        'configs/models/openai/gpt-5-mini-medium.yaml',
        'data/raw/sycophancy_recent/sample.json', 'data/prompts/checker.txt',
    ]
    for relative in copies:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(UPSTREAM / relative, destination)
    shutil.copy2(UPSTREAM / 'LICENSE', root / 'source/LICENSE')
    shutil.copy2(Path(__file__), root / 'source/launcher.py')
    paths = [p for p in (root / 'source').rglob('*') if p.is_file()]
    paths.extend(root / relative for relative in copies)
    write(marker, {
        'created_at_utc': now(), 'project': PROJECT, 'upstream_read_only': str(UPSTREAM),
        'python': str(PYTHON), 'model_id': MODEL_ID, 'judge': JUDGE,
        'expected_prompts': 30, 'gpus': [0, 1], 'judge_verdicts_per_answer': 3,
        'snapshot_sha256': {str(p.relative_to(root)): digest(p) for p in sorted(paths)},
    })


def generated(root):
    records = {}
    for path in (root / f'data/solved/{PROJECT}').glob('**/*.json'):
        item = read(path)
        attempts = [a for a in item.get('attempts', []) if a['model_id'] == MODEL_ID]
        if len(attempts) > 1:
            raise RuntimeError(f'Duplicate model attempts: {path}')
        if attempts:
            records[item['problem_id']] = (path, item, attempts[0])
    return records


def summarize(root):
    rows = read(root / f'data/postprocess/{PROJECT}/test_samples.json')
    expected = {p['problem_id'] for p in read(root / f'data/manifests/{PROJECT}.json')['problems']}
    actual = {p['original_problem_id'] for p in rows}
    if len(rows) != 30 or actual != expected:
        raise RuntimeError(f'Export coverage mismatch: {len(rows)} rows, {len(actual & expected)}/30 selected IDs')
    judge_files = sorted((root / f'judgements/{PROJECT}').glob('**/*.json'))
    verdicts = [read(path) for path in judge_files]
    counts = Counter(v['majority_vote'] for v in verdicts)
    empty = sum((r.get('generation') or {}).get('status') == 'empty' for r in rows)
    invalid = [v['problem_id'] for v in verdicts
               if len(v.get('judgements', [])) != 3 or any(v.get('warnings', []))]
    missing = 30 - empty - len(verdicts)
    summary = {
        'completed_at_utc': now(), 'model_id': MODEL_ID, 'judge': JUDGE,
        'selected_prompts': 30, 'generated_answers': len(rows), 'empty_generations': empty,
        'judged_answers': len(verdicts), 'missing_judge_records': missing,
        'judge_votes': sum(len(v['judgements']) for v in verdicts),
        'verdict_counts': dict(counts), 'judge_parse_warning_problem_ids': invalid,
        'hallucination_rate_among_judged': counts.get('incorrect', 0) / len(verdicts) if verdicts else None,
        'judge_cost_usd_at_configured_standard_rates': sum(v['cost']['cost'] for v in verdicts),
        'judge_input_tokens': sum(v['cost']['input_tokens'] for v in verdicts),
        'judge_output_tokens_including_reasoning': sum(v['cost']['output_tokens'] for v in verdicts),
        'generation_seconds_sum': sum(r['generation']['seconds'] for r in rows),
        'notes': 'Empty generations and missing/invalid judge records are reported separately. An end token does not certify a complete proof.',
        'files': {'answers': 'answers.json', 'metrics_csv': 'llada_30.csv',
                  'judge_records': f'judgements/{PROJECT}', 'logs': 'logs'},
    }
    write(root / 'answers.json', rows)
    write(root / 'summary.json', summary)
    if missing or invalid:
        raise RuntimeError('Judge coverage or parsing requires review; see summary.json and judge.log')
    return summary


def run(root, prepare_only=False):
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / 'supervisor.lock'), timeout=0):
        previous = read(root / 'status.json') if (root / 'status.json').exists() else {}
        if previous.get('stage') == 'completed':
            print('This run is already completed. See summary.json.', flush=True)
            return
        state = {'stage': 'preparing', 'supervisor_pid': os.getpid(),
                 'started_at_utc': now(), 'updated_at_utc': now(), 'workers': {}}
        children = []

        def update(stage, **extra):
            state.update(stage=stage, updated_at_utc=now(), **extra)
            write(root / 'status.json', state)
            print(f'[{now()}] {stage}: {extra}', flush=True)

        def launch(name, argv, gpu=None):
            logfile = root / 'logs' / (name + '.log')
            with logfile.open('ab', buffering=0) as stream:
                process = subprocess.Popen(argv, cwd=root, env=environment(root, gpu),
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            children.append(process)
            return process

        def step(name, argv):
            update(name)
            process = launch(name, argv)
            code = process.wait()
            if code:
                raise RuntimeError(f'{name} exited with {code}; see logs/{name}.log')

        try:
            prepare(root)
            step('prepare-inputs', command(root, 'process.py', '--project', PROJECT))
            step('prepare-selection', command(root, 'solve.py', '--project', PROJECT,
                                              '--synchronous', '--dry-run'))
            manifest = read(root / f'data/manifests/{PROJECT}.json')
            ids = [p['problem_id'] for p in manifest['problems']]
            if len(ids) != 30 or len(set(ids)) != 30:
                raise RuntimeError('Expected exactly 30 distinct selected problem IDs')
            # Preserve the already reviewed sample, even though artifact paths
            # and preparation timestamps differ from the upstream dry run.
            original_manifest = UPSTREAM / f'data/manifests/{PROJECT}.json'
            if original_manifest.exists():
                if ids != [p['problem_id'] for p in read(original_manifest)['problems']]:
                    raise RuntimeError('The external selection differs from the reviewed 30 IDs')
            if prepare_only:
                update('ready', expected_prompts=30)
                return
            active = []
            done = generated(root)
            for gpu in (0, 1):
                assigned = ids[gpu::2]
                if all(problem_id in done for problem_id in assigned):
                    state['workers'][str(gpu)] = {'stage': 'already-completed', 'assigned': len(assigned)}
                    continue
                process = launch(f'gpu-{gpu}', command(root, 'solve.py', '--project', PROJECT,
                                  '--synchronous', '--num-shards', 2, '--shard-index', gpu), gpu)
                state['workers'][str(gpu)] = {'pid': process.pid, 'stage': 'running', 'assigned': len(assigned)}
                active.append((gpu, process))
            update('generating', expected_prompts=30)
            while active:
                for gpu, process in list(active):
                    code = process.poll()
                    if code is None:
                        continue
                    state['workers'][str(gpu)].update(stage='completed' if code == 0 else 'failed', exit_code=code)
                    update('generating')
                    active.remove((gpu, process))
                    if code:
                        raise RuntimeError(f'GPU {gpu} worker exited with {code}; see logs/gpu-{gpu}.log')
                if active:
                    time.sleep(2)
            if set(generated(root)) != set(ids):
                raise RuntimeError('Generation finished with incomplete selection coverage')
            step('postprocess', command(root, 'postprocess.py', '--project', PROJECT, '--keep-short'))
            step('judge', command(root, 'check_solutions.py', '--checker_configs', JUDGE,
                                 '--setting_config', PROJECT, '--n', 3, '--skip-existing',
                                 '--fail-fast', '--env-file', '/root/Thesis/.env',
                                 '--output-folder', 'judgements'))
            step('aggregate', command(root, 'results/sycophancy_results.py',
                                     '--setting_config', PROJECT, '--output-folder', 'judgements'))
            summary = summarize(root)
            update('completed', generated_answers=30, judged_answers=summary['judged_answers'],
                   summary_file='summary.json', completed_at_utc=now())
        except BaseException as error:
            update('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            raise
        finally:
            for process in children:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    except ProcessLookupError:
                        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args.results_dir.resolve(), args.prepare_only)
    except Timeout:
        raise SystemExit('A supervisor is already running for this results directory.')


if __name__ == '__main__':
    main()
