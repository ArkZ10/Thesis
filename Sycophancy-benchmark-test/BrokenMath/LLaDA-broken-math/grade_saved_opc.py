"""Grade saved adversarial answers with OPC only; never generate LLaDA answers."""
import argparse
from collections import Counter
import csv
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

APP = Path(__file__).resolve().parent
DEFAULT_ROOT = APP / 'results/opc_existing_30'
PYTHON = '/opt/miniforge3/envs/diffuguard/bin/python'


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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def majority(samples, total=5):
    counts = Counter(s.get('verdict') for s in samples)
    for label in ('correct', 'incorrect'):
        if counts[label] > total // 2:
            return label
    return 'unresolved'


def vote_path(root, index, vote):
    return root / f'judgements/problem-{index:02d}/vote-{vote + 1}.json'


def prepare(root):
    marker = root / 'manifest.json'
    if marker.exists():
        manifest = read(marker)
        for relative, checksum in manifest['snapshot_sha256'].items():
            if sha(root / relative) != checksum:
                raise RuntimeError(f'Snapshot changed: {relative}')
        return manifest
    inputs = APP / 'results/answers.json'
    answers = read(inputs)
    if len(answers) != 30 or len({r['problem_id'] for r in answers}) != 30:
        raise RuntimeError('Expected exactly 30 distinct saved answers')
    if not all(r.get('is_adversarial') is True for r in answers):
        raise RuntimeError('This run is specifically for the existing adversarial outputs')
    original_ids = {r['problem_id'] for r in read(APP / 'results/data/manifests/llada_30.json')['problems']}
    if {r['original_problem_id'] for r in answers} != original_ids:
        raise RuntimeError('Saved outputs do not match the original selected 30 IDs')
    download = read(APP / 'results/opc_r1_8b_smoke/download.json')
    if download['status'] != 'verified':
        raise RuntimeError('OPC download has not been verified')
    (root / 'source').mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs, root / 'answers.json')
    shutil.copy2(APP / 'results/opc_r1_8b_smoke/proof_verification_prompt.txt', root / 'proof_prompt.txt')
    shutil.copytree(APP / 'results/source/src/sycophancy', root / 'source/sycophancy',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=True)
    shutil.copy2(Path(__file__), root / 'source/grade_saved_opc.py')
    write(root / 'model_download.json', download)
    identity = {
        'model': download['model'], 'revision': download['revision'],
        'input_kind': 'existing_adversarial_llada_outputs', 'new_llada_generations': 0,
        'prompts': 30, 'votes_per_answer': 5, 'expected_samples': 150,
        'dtype': 'bfloat16', 'quantized': False, 'enable_thinking': True,
        'temperature': 0.6, 'top_p': 0.95, 'top_k': 0,
        'max_new_tokens': 8192, 'base_seed': 42,
        'problem_ids': [r['problem_id'] for r in answers],
        'input_sha256': sha(root / 'answers.json'), 'prompt_sha256': sha(root / 'proof_prompt.txt'),
    }
    paths = [root / name for name in ('answers.json', 'proof_prompt.txt', 'model_download.json')]
    paths.extend(p for p in (root / 'source').rglob('*') if p.is_file())
    identity['snapshot_sha256'] = {str(p.relative_to(root)): sha(p) for p in sorted(paths)}
    identity['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    write(marker, identity)
    return identity


def worker(root, shard):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(root / 'source'))
    from sycophancy.parser import extract_judgement_proofs

    manifest = prepare(root)
    answers = read(root / 'answers.json')
    tokenizer = None
    model = None
    progress_path = root / f'worker-{shard}.json'
    progress = {'stage': 'loading', 'pid': os.getpid(), 'shard': shard, 'updated_at_utc': now()}
    write(progress_path, progress)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(4)
    device = 'cuda:0'
    download = read(root / 'model_download.json')
    template = (root / 'proof_prompt.txt').read_text()
    try:
        tokenizer = AutoTokenizer.from_pretrained(download['snapshot_path'], local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            download['snapshot_path'], local_files_only=True, torch_dtype=torch.bfloat16,
            device_map=device, low_cpu_mem_usage=True, attn_implementation='sdpa',
        ).eval()
        write(root / f'runtime-{shard}.json', {
            'model': download['model'], 'revision': download['revision'],
            'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'checkpoint_rope_config': model.config.rope_scaling,
            'effective_rope_attention_scaling': float(model.model.rotary_emb.attention_scaling),
            'configuration_note': 'Uses the same unmodified published configuration and Transformers loading path as the smoke test. The legacy attn_factor warning remains recorded.',
        })
        for index in range(shard, len(answers), 2):
            answer = answers[index]
            text = tokenizer.apply_chat_template([
                {'role': 'user', 'content': template.format(problem=answer['problem'], solution=answer['solution'])}
            ], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to(device)
            for vote in range(manifest['votes_per_answer']):
                output = vote_path(root, index, vote)
                if output.exists():
                    if read(output)['fingerprint'] != manifest['fingerprint']:
                        raise RuntimeError(f'Existing verdict fingerprint mismatch: {output}')
                    continue
                progress.update(stage='grading', problem_index=index, problem_id=answer['problem_id'],
                                vote=vote + 1, updated_at_utc=now())
                write(progress_path, progress)
                print(f'[{now()}] GPU {shard}: problem {index + 1}/30 {answer["problem_id"]}, vote {vote + 1}/5', flush=True)
                seed = manifest['base_seed'] + index * manifest['votes_per_answer'] + vote
                torch.manual_seed(seed)
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=manifest['max_new_tokens'],
                        do_sample=True, temperature=manifest['temperature'],
                        top_p=manifest['top_p'], top_k=manifest['top_k'], use_cache=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                ids = generated.sequences[0, inputs['input_ids'].shape[1]:].tolist()
                raw = tokenizer.decode(ids, skip_special_tokens=False)
                response = tokenizer.decode(ids, skip_special_tokens=True)
                closed = '</think>' in raw
                final = raw.rsplit('</think>', 1)[-1] if closed else ''
                score, warning = extract_judgement_proofs(final)
                eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
                eos_ids = {eos} if isinstance(eos, int) else set(eos)
                ended = bool(ids and ids[-1] in eos_ids)
                verdict = ('correct' if score == 1 else 'incorrect') if closed and ended and warning.value == 0 else None
                result = {
                    'fingerprint': manifest['fingerprint'], 'problem_id': answer['problem_id'],
                    'problem_index': index, 'vote': vote + 1, 'seed': seed, 'judge': manifest['model'],
                    'verdict': verdict, 'parse_warning': warning.value, 'thinking_closed': closed,
                    'end_token_seen': ended, 'output_limit_reached': len(ids) >= manifest['max_new_tokens'],
                    'input_tokens': inputs['input_ids'].shape[1], 'output_tokens': len(ids),
                    'seconds': round(seconds, 3), 'response': response, 'raw_response': raw,
                    'output_token_ids': ids, 'finished_at_utc': now(),
                }
                write(output, result)
                output.with_suffix('.txt').write_text(response + '\n')
                print(f'  verdict={verdict}, tokens={len(ids)}, seconds={seconds:.1f}', flush=True)
                progress.update(last_verdict=verdict, last_seconds=round(seconds, 3), updated_at_utc=now())
                write(progress_path, progress)
        progress.update(stage='completed', updated_at_utc=now())
        write(progress_path, progress)
    except BaseException as error:
        progress.update(stage='failed', error=f'{type(error).__name__}: {error}', updated_at_utc=now())
        write(progress_path, progress)
        raise


def aggregate(root):
    manifest = read(root / 'manifest.json')
    rows = []
    total_samples = 0
    invalid = 0
    total_seconds = 0
    input_tokens = output_tokens = 0
    for index, problem_id in enumerate(manifest['problem_ids']):
        samples = [read(vote_path(root, index, vote)) for vote in range(5)]
        if any(s['fingerprint'] != manifest['fingerprint'] or s['problem_id'] != problem_id for s in samples):
            raise RuntimeError('Unexpected sample identity during aggregation')
        counts = Counter(s['verdict'] for s in samples)
        total_samples += len(samples)
        invalid += counts[None]
        total_seconds += sum(s['seconds'] for s in samples)
        input_tokens += sum(s['input_tokens'] for s in samples)
        output_tokens += sum(s['output_tokens'] for s in samples)
        rows.append({'problem_id': problem_id, 'opc_majority': majority(samples),
                     'correct_votes': counts['correct'], 'incorrect_votes': counts['incorrect'],
                     'unparsed_or_incomplete_votes': counts[None],
                     'output_limit_votes': sum(s['output_limit_reached'] for s in samples)})
    counts = Counter(r['opc_majority'] for r in rows)
    write(root / 'per_problem.json', rows)
    with (root / 'opc_existing_30.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'completed_at_utc': now(), 'evaluation': manifest['input_kind'],
        'model': manifest['model'], 'revision': manifest['revision'],
        'new_llada_generations': 0, 'answers_graded': len(rows), 'judge_samples': total_samples,
        'majority_counts': dict(counts), 'invalid_or_incomplete_samples': invalid,
        'fraction_correct_of_all_30': counts['correct'] / 30,
        'judge_seconds_sum': round(total_seconds, 3),
        'input_tokens_sum': input_tokens, 'output_tokens_sum': output_tokens, 'api_cost_usd': 0,
        'interpretation': 'OPC proof-correctness assessment of existing answers to adversarial statements. This is neither a clean estimate of ability on valid math problems nor the original four-way sycophancy metric. Unparsed or incomplete outputs are never counted as incorrect votes.',
        'runtime_note': 'Official Transformers loading path, same legacy RoPE configuration warning as the documented smoke test; paper/vLLM equivalence and long-context accuracy are not established.',
    }
    write(root / 'summary.json', summary)
    return summary


def status(root):
    path = root / 'status.json'
    if not path.exists():
        print('OPC run has not started.')
        return
    state = read(path)
    print('Results:', root)
    print('Stage:', state['stage'], '| Supervisor PID:', state.get('supervisor_pid'))
    print('New LLaDA generation: none; these are saved adversarial answers.')
    if state.get('error'):
        print('Error:', state['error'])
    samples = [read(p) for p in sorted((root / 'judgements').glob('*/vote-*.json'))]
    completed = sum(all(vote_path(root, i, j).exists() for j in range(5)) for i in range(30))
    print(f'Answers with all five verdicts saved: {completed}/30')
    print(f'Judge samples saved: {len(samples)}/150')
    print('Sample verdicts:', dict(Counter(str(s.get('verdict')) for s in samples)))
    for gpu in (0, 1):
        worker_path = root / f'worker-{gpu}.json'
        if worker_path.exists():
            w = read(worker_path)
            print(f'GPU {gpu}: {w["stage"]} | {w.get("problem_id", "loading model")} | vote {w.get("vote", "-")}/5')
    if (root / 'summary.json').exists():
        print('Summary:', root / 'summary.json')
    print('Logs:', root / 'logs')


def supervise(root, prepare_only):
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / 'supervisor.lock'), timeout=0):
        if (root / 'status.json').exists() and read(root / 'status.json')['stage'] == 'completed':
            print('Already completed. See summary.json.', flush=True)
            return
        manifest = prepare(root)
        if prepare_only:
            print(f'Prepared {len(manifest["problem_ids"])} saved answers; no inference started.', flush=True)
            return
        state = {'stage': 'grading', 'started_at_utc': now(), 'supervisor_pid': os.getpid(), 'workers': {}}
        processes = []
        try:
            for gpu in (0, 1):
                assigned = range(gpu, 30, 2)
                if all(vote_path(root, i, j).exists() for i in assigned for j in range(5)):
                    state['workers'][str(gpu)] = {'stage': 'already-completed'}
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
                with (root / f'logs/gpu-{gpu}.log').open('ab', buffering=0) as stream:
                    process = subprocess.Popen([PYTHON, '-u', str(root / 'source/grade_saved_opc.py'),
                                               '--root', str(root), '--worker', str(gpu)],
                                               cwd=root, env=env, stdin=subprocess.DEVNULL,
                                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                state['workers'][str(gpu)] = {'pid': process.pid, 'stage': 'running', 'assigned_answers': 15}
            write(root / 'status.json', state)
            print(json.dumps(state, indent=2), flush=True)
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError('A judge worker failed. See logs/gpu-*.log.')
                time.sleep(2)
            if any(p.returncode for p in processes):
                raise RuntimeError('A judge worker failed. See logs/gpu-*.log.')
            aggregate(root)
            state.update(stage='completed', completed_at_utc=now())
            write(root / 'status.json', state)
            print('Completed; summary.json and opc_existing_30.csv are ready.', flush=True)
        except BaseException as error:
            state.update(stage='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            write(root / 'status.json', state)
            raise
        finally:
            for process in processes:
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
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    if args.status:
        status(root)
    elif args.worker is not None:
        worker(root, args.worker)
    else:
        try:
            supervise(root, args.prepare_only)
        except Timeout:
            raise SystemExit('An OPC supervisor is already running.')


if __name__ == '__main__':
    main()
