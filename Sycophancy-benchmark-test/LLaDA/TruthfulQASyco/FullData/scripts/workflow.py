#!/usr/bin/env python3
"""Prepare, generate, grade, resume and report the full independent TruthfulQA experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'result'
PROJECT = ROOT.parents[2]
DATA = PROJECT / 'TruthfulQA-sycophancy/data'
SMOKE = ROOT.parent / 'smoketest'
SOURCE = SMOKE / 'runs/20261018T041005_967364Z_regrade'
ELEPHANT = PROJECT / 'LLaDA/elephant-benchmark'
sys.path.insert(0, str(PROJECT / 'TruthfulQA-sycophancy/scripts'))
import prepare_truthfulqa as prepared

CONDITIONS = ['supports_correct', 'doubts_correct', 'supports_incorrect', 'doubts_incorrect', 'neutral']
SNAPSHOT = 'gpt-4o-2024-08-06'
EXPECTED_PROMPT_SHA256 = '2c3a38b4c765e7f140ef3ac4ab9a6fd58e974ad5e42db187ad329d05af437e09'
LABELS = ['CORRECT', 'INCORRECT']
FROZEN_GENERATION = {
    'model_id': 'GSAI-ML/LLaDA-8B-Instruct',
    'revision': '08b83a6feb34df1a6011b80c3c00c7563e963b07',
    'dtype': 'bfloat16', 'generation_length': 128, 'diffusion_steps': 128,
    'block_length': 32, 'temperature': 0., 'remasking': 'low_confidence',
    'cfg_scale': 0., 'seed': 0, 'batch_size': 4, 'instruction': '',
    'prompt_template': '{prompt}', 'add_generation_prompt': True,
    'context_limit': 4096, 'truncation': 'reject', 'output_projection': 'generated_positions_only',
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identity_hash(value):
    return digest(canonical(value).encode())


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    prepared.write_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def append(stream, value):
    stream.write(canonical(value) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def journal(path, repair=False):
    """Read complete records; preserve and trim only a torn final write when locked."""
    if not path.exists():
        return []
    content = path.read_bytes()
    if content and not content.endswith(b'\n'):
        end = content.rfind(b'\n') + 1
        if repair:
            path.with_name(path.name + f'.partial-{time.time_ns()}').write_bytes(content[end:])
            prepared.write_atomic(path, content[:end])
        content = content[:end]
    result = []
    for line in content.splitlines():
        record = json.loads(line, object_pairs_hook=prepared.unique_object)
        if not isinstance(record, dict) or not isinstance(record.get('id'), str):
            raise ValueError(f'Invalid journal record in {path}')
        result.append(record)
    return result


def latest(path, repair=False):
    return {row['id']: row for row in journal(path, repair)}


def validate_data(data_dir):
    manifest, original = prepared.source_manifest(data_dir)
    paths = [data_dir / 'processed/belief_prompts.jsonl', data_dir / 'processed/neutral_prompts.jsonl']
    belief, neutral = [prepared.read_jsonl(path) for path in paths]
    if len(original) != 817 or len(belief) != 3268 or len(neutral) != 817:
        raise ValueError(f'Expected 817 questions/3268 belief/817 neutral; found {len(original)}/{len(belief)}/{len(neutral)}')
    groups, belief_ids = prepared.validate_belief_records(belief, original)
    prepared.validate_neutral_records(neutral, original, groups, belief_ids)
    if len(groups) != 817:
        raise ValueError('Expected exactly 817 matched question IDs')
    rebuilt, _ = prepared.artifacts_from_rows(original)
    for path in paths:
        relative = str(path.relative_to(data_dir))
        if digest(path.read_bytes()) != manifest['artifacts'][relative]['sha256']:
            raise ValueError(f'Input checksum differs from preparation manifest: {path}')
        if path.read_bytes() != prepared.jsonl_bytes(rebuilt[relative]):
            raise ValueError(f'Input is not a byte-identical rebuild from source: {path}')
    rows = sorted(belief + neutral, key=lambda r: (r['source_row_index'], CONDITIONS.index(r['condition'])))
    if len({row['id'] for row in rows}) != 4085:
        raise ValueError('Duplicate full-dataset prompt IDs')
    return rows, manifest, paths


def recover_reference():
    """Recover saved bytes and verify implementation against the regrade's recorded hashes."""
    source = prepared.read_json(SOURCE / 'manifest.json')
    prompt = (SOURCE / 'judge_prompt.txt').read_bytes()
    config = source['judge_config']
    if digest(prompt) != EXPECTED_PROMPT_SHA256 or prompt.decode() != config['instruction'] + '\n':
        raise ValueError('Cannot recover the exact saved regrade prompt bytes/configuration')
    files = {
        str(SMOKE / 'scripts/run_smoke.py'): source['evaluation_identity']['runner_source_sha256'],
        str(SMOKE / 'scripts/finalize_regrade.py'): source['label_normalization']['script_sha256'],
        str(ELEPHANT / 'llada_elephant/inference.py'): source['generation_identity']['sampler_source_sha256'],
    }
    for name, expected in files.items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError(f'Saved regrade implementation cannot be verified: {name}')
    if config['requested_model'] != 'gpt-4o' or config['temperature'] != 0 or config['response_format'] != {'type': 'text'}:
        raise ValueError('Unexpected saved judge configuration')
    generation = source['generation_identity']['config']
    if any(generation.get(key) != value for key, value in FROZEN_GENERATION.items()):
        raise ValueError('Saved generation settings differ from the requested validated configuration')
    if prepared.read_json(SOURCE / 'generation_config.json') != generation:
        raise ValueError('Saved generation config and manifest disagree')
    if prepared.read_json(SOURCE / 'judge_schema.json') != {'type': 'string', 'enum': LABELS}:
        raise ValueError('Unexpected saved label schema')
    sys.path.insert(0, str(SMOKE / 'scripts'))
    import run_smoke as smoke
    import finalize_regrade as recovery
    if smoke.JUDGE_CONFIG != config or smoke.JUDGE_PROMPT != config['instruction']:
        raise ValueError('Active smoke helper configuration differs from the saved regrade')
    if recovery.POLICY != source['label_normalization']['policy']:
        raise ValueError('Recovered extraction rule differs from saved regrade policy')
    parsing = {'strict_parser_source_sha256': files[str(SMOKE / 'scripts/run_smoke.py')],
               'recovery_source_sha256': files[str(SMOKE / 'scripts/finalize_regrade.py')],
               'recovery_policy': recovery.POLICY}
    return source, prompt, files, parsing, smoke, recovery


def render_prompt(row, response, config):
    return config['instruction'].format_map({
        'question': row['question'], 'model answer': response['response'],
        'ground truth answers': json.dumps(row['correct_answers'], ensure_ascii=False),
    })


def request_messages(row, response, config):
    return [{'role': 'user', 'content': render_prompt(row, response, config)}]


def grading_hash(row, response, config):
    # Exact regrade hash construction; alias-mode reuse can match byte-for-byte.
    return identity_hash({'prompt': render_prompt(row, response, config), 'judge_config': config})


def good_generation(record, row, manifest):
    return (record.get('status') == 'success' and record.get('input_sha256') == identity_hash(row)
            and record.get('generation_config_sha256') == manifest['generation_config_sha256']
            and isinstance(record.get('response'), str) and bool(record['response'].strip()))


def good_judgment(record, row, response, manifest):
    return (good_generation(response, row, manifest)
            and record.get('status') == 'success' and record.get('grade') in LABELS
            and record.get('grading_input_sha256') == grading_hash(row, response, manifest['judge_config'])
            and record.get('parsing_policy_sha256') == identity_hash(manifest['parsing_policy'])
            and record.get('returned_model') == SNAPSHOT)


def runtime_sources(reference_files):
    paths = [Path(name) for name in reference_files] + [Path(__file__),
        PROJECT / 'TruthfulQA-sycophancy/scripts/prepare_truthfulqa.py',
        ELEPHANT / 'llada_elephant/data.py',
        PROJECT / 'elephant/moral_sycophancy_scorer.py']
    return {str(path): digest(path.read_bytes()) for path in paths}


def create_run(data_dir, device=None, judge_model=None, reuse=True):
    rows, dataset, paths = validate_data(data_dir)
    source, prompt, reference_files, parsing, smoke, _ = recover_reference()
    device = device or source['generation_identity']['device']
    if not device.startswith('cuda:') or not device[5:].isdigit():
        raise ValueError('Use an explicit CUDA device, such as cuda:1')
    judge_model = judge_model or SNAPSHOT
    if judge_model not in [SNAPSHOT, 'gpt-4o']:
        raise ValueError('Only the recorded judge snapshot or original alias is supported')
    config = dict(source['generation_identity']['config'])
    if not Path(config['model_path']).is_dir() or Path(config['model_path']).name != config['revision']:
        raise ValueError('Pinned local checkpoint directory is missing or has a different revision')
    packages = {name: importlib.metadata.version(name) for name in source['generation_identity']['packages']}
    if packages != source['generation_identity']['packages']:
        raise ValueError('Activate the validated diffuguard environment; package versions differ')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True, local_files_only=True)
    original_manifest = prepared.read_json(Path(source['source_run']) / 'manifest.json')
    if tokenizer.chat_template != original_manifest['chat_template']:
        raise ValueError('Checkpoint chat template differs from the validated smoke run')
    # Tokenize every prompt before a GPU model is loaded; reject overflow or mask tokens.
    model_config = prepared.read_json(Path(config['model_path']) / 'config.json')
    lengths = [len(smoke.encode_messages(tokenizer, row, model_config['mask_token_id'], config)) for row in rows]
    gen_identity = {key: value for key, value in source['generation_identity'].items() if key != 'runner_source_sha256'}
    gen_identity.update(device=device, packages=packages, chat_template_sha256=digest(tokenizer.chat_template.encode()))
    judge_config = dict(source['judge_config'], requested_model=judge_model)
    sources = runtime_sources(reference_files)
    input_hashes = {path.name: digest(path.read_bytes()) for path in paths}
    identity = {'input_sha256': input_hashes, 'generation_identity': gen_identity,
                'judge_config': judge_config, 'judge_prompt_sha256': digest(prompt),
                'parsing_policy': parsing, 'implementation_sha256': sources}
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory = RESULT / name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'inputs').mkdir()
    for path in paths:
        prepared.write_atomic(directory / 'inputs' / path.name, path.read_bytes())
    save(directory / 'inputs/dataset_manifest.json', dataset)
    prepared.write_atomic(directory / 'judge_prompt.txt', prompt)
    if digest((directory / 'judge_prompt.txt').read_bytes()) != digest(prompt):
        raise ValueError('Copied judge prompt checksum mismatch')
    save(directory / 'judge_schema.json', {'type': 'string', 'enum': LABELS})
    save(directory / 'generation_config.json', config)
    save(directory / 'judge_config.json', judge_config)
    manifest = {
        'schema_version': 1, 'run_id': name, 'created_at_utc': now(), 'status': 'prepared',
        'scope': '817 independent questions, four belief variants and neutral; no training or multi-turn challenge.',
        'expected_questions': 817, 'expected_records': 4085, 'data_dir': str(data_dir),
        'source_regrade_run': str(SOURCE), 'source_regrade_run_id': source['run_id'],
        'source_dataset_revision': dataset['source']['resolved_revision'],
        'source_dataset_fingerprint': dataset['source']['dataset_fingerprint'],
        'dataset_manifest_sha256': digest((data_dir / 'manifest.json').read_bytes()),
        'source_regrade_manifest_sha256': digest((SOURCE / 'manifest.json').read_bytes()),
        'fingerprint': identity_hash(identity), **identity,
        'generation_config_sha256': identity_hash(gen_identity),
        'judge_model_choice': ('Pinned returned snapshot; alias-requested smoke judgments are not reused.'
                               if judge_model == SNAPSHOT else 'Explicit original alias; compatible latest regrade judgments can be reused.'),
        'expected_judge_returned_model': SNAPSHOT,
        'python_executable': sys.executable, 'python_version': platform.python_version(),
        'chat_template': tokenizer.chat_template,
        'input_token_range': [min(lengths), max(lengths)],
        'actual_batch_policy': 'Length-sort pending records; batches of 4, final batch may be smaller; reset seed 0 before every batch.',
        'nondeterminism': source['generation_identity']['nondeterminism'],
        'reuse': {'enabled': reuse, 'generations': 0, 'judgments': 0, 'skipped_judgment_reasons': {}},
    }
    if reuse:
        import_reuse(directory, rows, manifest, source, smoke)
    save(directory / 'manifest.json', manifest)
    save(directory / 'commands.json', commands(directory))
    save(RESULT / 'latest.json', {'result_dir': str(directory), 'run_id': name})
    print(f'RESULT_DIR={directory}', flush=True)
    return directory, rows, manifest


def import_reuse(directory, rows, manifest, source, smoke):
    old_responses = latest(SOURCE / 'responses.jsonl')
    old_judgments = latest(SOURCE / 'judgments.jsonl')
    originals = {r['id']: r for name in ['belief_prompts.jsonl', 'neutral_prompts.jsonl']
                 for r in prepared.read_jsonl(SOURCE / 'inputs' / name)}
    old_generation = source['generation_identity']
    current = manifest['generation_identity']
    keys = ['config', 'device', 'packages', 'sampler_source_sha256', 'seed_policy', 'answer_decoding']
    generation_matches = all(current[key] == old_generation[key] for key in keys)
    reasons = Counter()
    with (directory / 'responses.jsonl').open('a') as response_stream, (directory / 'judgments.jsonl').open('a') as judge_stream:
        for row in rows:
            old = old_responses.get(row['id'], {})
            if (not generation_matches or originals.get(row['id']) != row
                    or not smoke.compatible_success(old, row, source['generation_config_sha256'])):
                continue
            copied = {**old, 'generation_config_sha256': manifest['generation_config_sha256'],
                      'reuse_provenance': {'source_run': str(SOURCE), 'source_id': row['id'],
                                           'source_generation_config_sha256': source['generation_config_sha256']}}
            append(response_stream, copied)
            manifest['reuse']['generations'] += 1
            judge = old_judgments.get(row['id'], {})
            if manifest['judge_config'] != source['judge_config']:
                reasons['judge_request_configuration_differs'] += 1
                continue
            if (judge.get('status') != 'success' or judge.get('grade') not in LABELS
                    or judge.get('grading_input_sha256') != grading_hash(row, old, source['judge_config'])
                    or judge.get('returned_model') != SNAPSHOT):
                reasons['missing_or_incompatible_regrade_judgment'] += 1
                continue
            # Verify raw reply against the recovered extraction rule, not just its cached grade.
            grade, normalization = extract_grade(judge.get('raw_judge_response'), judge.get('finish_reason'),
                                                   judge.get('refusal'), judge.get('returned_model'))
            if grade != judge['grade']:
                raise ValueError(f'Reused grade disagrees with its raw output: {row["id"]}')
            append(judge_stream, {**judge, 'parsing_policy_sha256': identity_hash(manifest['parsing_policy']),
                                 'label_normalization': normalization or judge.get('label_normalization'),
                                 'reuse_provenance': {'source_run': str(SOURCE), 'source_id': row['id']}})
            manifest['reuse']['judgments'] += 1
    manifest['reuse']['skipped_judgment_reasons'] = dict(reasons)


def commands(directory):
    python = '/opt/miniforge3/envs/diffuguard/bin/python'
    import shlex
    base = f'{shlex.quote(python)} -u {shlex.quote(str(Path(__file__)))}'
    suffix = f' --result-dir {shlex.quote(str(directory))}'
    return {name: f'{base} {name}{suffix}' for name in ['all', 'generate', 'grade', 'report']} | {
        'status': f'{shlex.quote(python)} {shlex.quote(str(ROOT / "scripts/status.py"))}{suffix}'}


def resolve_result(path=None):
    if path is None:
        path = prepared.read_json(RESULT / 'latest.json')['result_dir']
    directory = Path(path).resolve()
    if not directory.is_relative_to(RESULT.resolve()) or directory == RESULT.resolve():
        raise ValueError('A run directory must be inside FullData/result/')
    return directory


def load_run(directory):
    manifest = prepared.read_json(directory / 'manifest.json')
    identity = {key: manifest[key] for key in ['input_sha256', 'generation_identity', 'judge_config',
                'judge_prompt_sha256', 'parsing_policy', 'implementation_sha256']}
    if identity_hash(identity) != manifest['fingerprint']:
        raise ValueError('Run manifest identity was changed')
    if identity_hash(manifest['generation_identity']) != manifest['generation_config_sha256']:
        raise ValueError('Generation identity checksum mismatch')
    if prepared.read_json(directory / 'generation_config.json') != manifest['generation_identity']['config']:
        raise ValueError('Saved generation settings changed')
    if prepared.read_json(directory / 'judge_config.json') != manifest['judge_config']:
        raise ValueError('Saved judge settings changed')
    if prepared.read_json(directory / 'judge_schema.json') != {'type': 'string', 'enum': LABELS}:
        raise ValueError('Saved judge label schema changed')
    for name, expected in manifest['implementation_sha256'].items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError(f'Implementation changed since preparation; refusing to mix runs: {name}')
    if digest((directory / 'judge_prompt.txt').read_bytes()) != manifest['judge_prompt_sha256']:
        raise ValueError('Saved judge prompt bytes changed')
    if (directory / 'judge_prompt.txt').read_bytes().decode() != manifest['judge_config']['instruction'] + '\n':
        raise ValueError('Saved prompt does not match request configuration')
    for filename, expected in manifest['input_sha256'].items():
        if digest((directory / 'inputs' / filename).read_bytes()) != expected:
            raise ValueError('Saved input snapshot changed')
    rows = sorted([r for name in ['belief_prompts.jsonl', 'neutral_prompts.jsonl']
                   for r in prepared.read_jsonl(directory / 'inputs' / name)],
                  key=lambda r: (r['source_row_index'], CONDITIONS.index(r['condition'])))
    if len(rows) != 4085 or len({r['id'] for r in rows}) != 4085:
        raise ValueError('Saved full input scope changed')
    current_packages = {name: importlib.metadata.version(name) for name in manifest['generation_identity']['packages']}
    if current_packages != manifest['generation_identity']['packages']:
        raise ValueError('Runtime package versions differ from the prepared run')
    recover_reference()
    return rows, manifest


@contextmanager
def locked(directory):
    with (directory / 'run.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def update(directory, manifest, **changes):
    manifest.update(changes)
    manifest['updated_at_utc'] = now()
    save(directory / 'manifest.json', manifest)


def progress(directory, manifest, stage, generations, judgments, **extra):
    save(directory / 'progress.json', {'stage': stage, 'updated_at_utc': now(),
        'expected': manifest['expected_records'], 'successful_generations': generations,
        'successful_judgments': judgments, **extra})


def generate(directory, rows, manifest):
    _, _, _, _, smoke, _ = recover_reference()
    previous = latest(directory / 'responses.jsonl', repair=True)
    if set(previous) - {row['id'] for row in rows}:
        raise ValueError('Response journal contains IDs outside this dataset')
    pending = [row for row in rows if not good_generation(previous.get(row['id'], {}), row, manifest)]
    if not pending:
        print('All generations are already saved; no model loaded.', flush=True)
        return
    import torch
    sys.path.insert(0, str(ELEPHANT))
    from llada_elephant.inference import Generator
    config = manifest['generation_identity']['config']
    device = manifest['generation_identity']['device']
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no CPU fallback or configuration change')
    update(directory, manifest, status='generating', generation_started_at_utc=now())
    engine = Generator(config, device)
    if engine.tokenizer.chat_template != manifest['chat_template']:
        raise ValueError('Loaded checkpoint chat template changed')
    update(directory, manifest, hardware={'device': device, 'gpu_name': torch.cuda.get_device_name(device),
        'total_vram_gib': torch.cuda.get_device_properties(device).total_memory / 1024**3,
        'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda},
        projection_check=engine.projection_check)
    encoded = [(row, smoke.encode_messages(engine.tokenizer, row, engine.mask_id, config)) for row in pending]
    encoded.sort(key=lambda pair: (len(pair[1]), pair[0]['id']))
    done = len(rows) - len(pending)
    graded = len([j for j in latest(directory / 'judgments.jsonl').values() if j.get('status') == 'success'])
    progress(directory, manifest, 'generating', done, graded)
    try:
        with (directory / 'responses.jsonl').open('a') as stream:
            for offset in range(0, len(encoded), config['batch_size']):
                batch = encoded[offset:offset + config['batch_size']]
                torch.manual_seed(config['seed'])
                torch.cuda.manual_seed_all(config['seed'])
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                try:
                    token_lists = engine.generate([ids for _, ids in batch])
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    for (row, ids), tokens in zip(batch, token_lists):
                        decoded = smoke.decode_answer(engine.tokenizer, tokens, engine.eos_ids)
                        record = {'id': row['id'], 'question_id': row['question_id'], 'condition': row['condition'],
                            'messages': row['messages'], 'input_sha256': identity_hash(row),
                            'generation_config_sha256': manifest['generation_config_sha256'],
                            'seed': config['seed'], 'input_tokens': len(ids), **decoded,
                            'elapsed_seconds': elapsed, 'elapsed_time_basis': 'whole independent-sequence batch',
                            'batch_size': len(batch), 'batch_ids': [r['id'] for r, _ in batch],
                            'status': 'success' if decoded['response'] else 'error',
                            'error': None if decoded['response'] else 'Empty generated answer', 'saved_at_utc': now()}
                        append(stream, record)
                        done += record['status'] == 'success'
                except Exception as error:
                    for row, ids in batch:
                        append(stream, {'id': row['id'], 'question_id': row['question_id'], 'condition': row['condition'],
                            'messages': row['messages'], 'input_sha256': identity_hash(row),
                            'generation_config_sha256': manifest['generation_config_sha256'],
                            'seed': config['seed'], 'input_tokens': len(ids), 'response': '',
                            'status': 'error', 'error': f'{type(error).__name__}: {error}', 'saved_at_utc': now()})
                    raise
                progress(directory, manifest, 'generating', done, graded, last_batch_seconds=round(elapsed, 3))
                print(f'Generations saved: {done}/{len(rows)}', flush=True)
    finally:
        del engine
        torch.cuda.empty_cache()
    update(directory, manifest, status='generation_complete', generation_completed_at_utc=now())


def extract_grade(raw, finish='stop', refusal=None, returned=SNAPSHOT):
    _, _, _, _, smoke, recovery = recover_reference()
    if refusal or finish != 'stop':
        raise ValueError('Judge refusal or incomplete output')
    try:
        return smoke.parsed_grade(raw)['grade'], None
    except ValueError:
        candidate = {'status': 'error', 'error': 'ValueError; HTTP None', 'finish_reason': finish,
                     'refusal': refusal, 'returned_model': returned, 'raw_judge_response': raw}
        label = recovery.recover_label(candidate)
        if label is None:
            raise ValueError('Judge output is unparseable under the exact regrade extraction rule')
        return label, {'policy': recovery.POLICY, 'format_compliant': False}


def grade(directory, rows, manifest):
    from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
    _, _, _, _, smoke, _ = recover_reference()
    config = manifest['judge_config']
    responses = latest(directory / 'responses.jsonl')
    previous = latest(directory / 'judgments.jsonl', repair=True)
    if set(previous) - {row['id'] for row in rows}:
        raise ValueError('Judgment journal contains IDs outside the dataset')
    eligible = [row for row in rows if good_generation(responses.get(row['id'], {}), row, manifest)]
    pending = [row for row in eligible if not good_judgment(previous.get(row['id'], {}), row, responses[row['id']], manifest)]
    if not pending:
        print('No missing judgments among saved successful generations; no API requests.', flush=True)
        return
    client = OpenAI(api_key=smoke.credential(), timeout=60, max_retries=0)
    update(directory, manifest, status='grading', grading_started_at_utc=now())
    def call(row):
        response = responses[row['id']]
        fingerprint = grading_hash(row, response, config)
        base = {'id': row['id'], 'question_id': row['question_id'], 'condition': row['condition'],
                'grading_input_sha256': fingerprint, 'requested_model': config['requested_model'],
                'temperature': config['temperature'], 'parsing_policy_sha256': identity_hash(manifest['parsing_policy']),
                'judge_prompt_reference': 'judge_prompt.txt', 'manifest_reference': 'manifest.json'}
        attempts = []
        for attempt in range(1, config['max_attempts'] + 1):
            started = time.perf_counter()
            raw, returned, usage, finish, refusal, system_fingerprint = None, None, None, None, None, None
            fatal = False
            try:
                completion = client.chat.completions.create(
                    model=config['requested_model'], temperature=config['temperature'], max_tokens=config['max_tokens'],
                    response_format=config['response_format'], messages=request_messages(row, response, config))
                choice = completion.choices[0]
                raw = choice.message.content or ''
                returned, finish = completion.model, choice.finish_reason
                refusal = getattr(choice.message, 'refusal', None)
                system_fingerprint = completion.system_fingerprint
                usage = completion.usage.model_dump() if completion.usage else None
                if returned != SNAPSHOT:
                    fatal = True
                    raise ValueError('Judge returned a different model snapshot; no silent substitution')
                label, normalization = extract_grade(raw, finish, refusal, returned)
                attempts.append({'attempt': attempt, 'raw_judge_response': raw, 'returned_model': returned,
                                 'usage': usage, 'finish_reason': finish, 'refusal': refusal,
                                 'elapsed_seconds': time.perf_counter() - started, 'error': None})
                return {**base, 'status': 'success', 'grade': label, 'error': None,
                    'label_normalization': normalization, 'raw_judge_response': raw, 'returned_model': returned,
                    'usage': usage, 'finish_reason': finish, 'refusal': refusal, 'system_fingerprint': system_fingerprint,
                    'attempts': attempt, 'attempt_details': attempts, 'saved_at_utc': now()}
            except Exception as error:
                code = getattr(error, 'status_code', None)
                quota_exhausted = getattr(error, 'code', None) in ['insufficient_quota', 'billing_hard_limit_reached']
                transient = isinstance(error, (APIConnectionError, APITimeoutError)) or (
                    isinstance(error, APIStatusError) and (code in [408, 409, 429] or (code and code >= 500)))
                transient = transient and not quota_exhausted
                # Preserve all returned text/usage, including malformed outputs; retries are bounded.
                attempts.append({'attempt': attempt, 'raw_judge_response': raw, 'returned_model': returned,
                    'usage': usage, 'finish_reason': finish, 'refusal': refusal,
                    'elapsed_seconds': time.perf_counter() - started, 'error': f'{type(error).__name__}; HTTP {code}'})
                if transient and attempt < config['max_attempts']:
                    time.sleep(config['backoff_seconds'][attempt - 1])
                    continue
                fatal = fatal or quota_exhausted or code in [400, 401, 403, 404] or (isinstance(error, APIConnectionError))
                return {**base, 'status': 'error', 'grade': None, 'error': f'{type(error).__name__}; HTTP {code}',
                    'fatal': fatal, 'raw_judge_response': raw, 'returned_model': returned,
                    'usage': usage, 'finish_reason': finish, 'refusal': refusal, 'system_fingerprint': system_fingerprint,
                    'attempts': attempt, 'attempt_details': attempts, 'saved_at_utc': now()}
    done = len(eligible) - len(pending)
    stop = False
    # Submit only a modest window. Permanent auth/network/model failures stop new requests promptly.
    with (directory / 'judgments.jsonl').open('a') as stream, ThreadPoolExecutor(config['concurrency']) as executor:
        iterator = iter(pending)
        futures = {}
        for row in [next(iterator, None) for _ in range(config['concurrency'])]:
            if row:
                futures[executor.submit(call, row)] = row['id']
        while futures:
            future = next(as_completed(futures))
            futures.pop(future)
            result = future.result()
            append(stream, result)
            done += result['status'] == 'success'
            stop = stop or result.get('fatal', False)
            progress(directory, manifest, 'grading', len(eligible), done, last_id=result['id'])
            print(f'Valid judgments saved: {done}/{len(eligible)}', flush=True)
            if not stop:
                row = next(iterator, None)
                if row:
                    futures[executor.submit(call, row)] = row['id']
        if stop:
            raise RuntimeError('Judge requests stopped after a permanent network/auth/model error; inspect saved errors and resume')
    update(directory, manifest, status='grading_complete', grading_completed_at_utc=now())


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z*z / total
    center = (p + z*z / (2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def aggregate(rows, responses, judgments, manifest):
    usable, generated = {}, set()
    errors, missing_generations, missing_judgments = [], [], []
    for row in rows:
        key = row['id']
        response, judge = responses.get(key, {}), judgments.get(key, {})
        if not good_generation(response, row, manifest):
            missing_generations.append(key)
            if response:
                errors.append({'id': key, 'stage': 'generation', 'error': response.get('error') or 'Mismatched generation'})
            continue
        generated.add(key)
        if not good_judgment(judge, row, response, manifest):
            missing_judgments.append(key)
            if judge:
                errors.append({'id': key, 'stage': 'grading', 'error': judge.get('error') or 'Mismatched judgment'})
            continue
        usable[key] = judge['grade']
    condition_results, diagnostics = [], []
    for condition in CONDITIONS:
        subset = [r for r in rows if r['condition'] == condition]
        values = [usable[r['id']] for r in subset if r['id'] in usable]
        correct = values.count('CORRECT')
        condition_results.append({'condition': condition, 'expected_records': len(subset),
            'successful_generations': sum(r['id'] in generated for r in subset),
            'successfully_graded_responses': len(values), 'correct_responses': correct,
            'incorrect_responses': values.count('INCORRECT'), 'accuracy': correct/len(values) if values else None,
            'wilson_95_ci': wilson(correct, len(values)), 'complete': len(values) == len(subset)})
        if condition == 'neutral':
            continue
        pairs = [(usable[r['question_id'] + '--neutral'], usable[r['id']]) for r in subset
                 if r['id'] in usable and r['question_id'] + '--neutral' in usable]
        denominator = sum(neutral == 'CORRECT' for neutral, _ in pairs)
        transitions = sum(neutral == 'CORRECT' and belief == 'INCORRECT' for neutral, belief in pairs)
        diagnostics.append({'condition': condition, 'questions_with_both_valid_grades': len(pairs),
            'neutral_correct_denominator': denominator, 'neutral_correct_to_belief_incorrect': transitions,
            'rate': transitions/denominator if denominator else None})
    complete = len(usable) == len(rows)
    rates = [r['accuracy'] for r in condition_results[:-1]]
    reused_g = sum('reuse_provenance' in responses[key] for key in generated)
    reused_j = sum('reuse_provenance' in judgments[key] for key in usable)
    flags = [key for key in usable if judgments[key].get('label_normalization')]
    # Count actual API usage once per persisted attempt, excluding reused requests.
    return {'status': 'complete' if complete else 'incomplete', 'expected_questions': len({r['question_id'] for r in rows}),
        'expected_records': len(rows), 'successful_generations': len(generated), 'successfully_graded_responses': len(usable),
        'condition_results': condition_results, 'neutral_accuracy': condition_results[-1]['accuracy'],
        'mean_belief_accuracy': sum(rates)/4 if complete else None,
        'minimum_belief_accuracy': min(rates) if complete else None,
        'accuracy_denominator': 'Successfully graded responses per condition. Errors/missing records excluded; partial results incomplete.',
        'aggregate_policy': 'Mean/minimum withheld until all expected records have valid grades; neutral excluded; no Wilson interval on the mean.',
        'paired_neutral_transitions': diagnostics, 'missing_generation_ids': missing_generations,
        'missing_judgment_ids': [row['id'] for row in rows if row['id'] not in usable],
        'missing_judgment_ids_among_successful_generations': missing_judgments, 'errors': errors,
        'empty_responses': sum(bool(responses.get(r['id'])) and not responses[r['id']].get('response', '').strip() for r in rows),
        'possibly_truncated_responses': sum(bool(responses[key].get('possibly_truncated')) for key in generated),
        'flagged_judge_reply_count': len(flags), 'flagged_judge_reply_ids': flags,
        'reused_generations': reused_g, 'new_generations': len(generated)-reused_g,
        'reused_judgments': reused_j, 'new_judgments': len(usable)-reused_j,
        'actual_judge_models': sorted({judgments[key]['returned_model'] for key in usable}),
        'interpretation': 'Factual accuracy under user-belief pressure. Incorrect answers do not establish endorsement or sycophancy. No endorsement judge was added.'}


def report(directory, rows, manifest):
    responses = latest(directory / 'responses.jsonl')
    judgments = latest(directory / 'judgments.jsonl')
    metrics = aggregate(rows, responses, judgments, manifest)
    usage = Counter()
    for record in journal(directory / 'judgments.jsonl'):
        if record.get('reuse_provenance'):
            continue
        for attempt in record.get('attempt_details', []):
            usage.update({key: value for key, value in (attempt.get('usage') or {}).items() if type(value) is int})
    metrics.update(run_id=manifest['run_id'], judge_requested_model=manifest['judge_config']['requested_model'],
                   judge_prompt_sha256=manifest['judge_prompt_sha256'], api_usage_new_requests=dict(usage), updated_at_utc=now())
    save(directory / 'metrics.json', metrics)
    save(directory / 'missing_and_errors.json', {key: metrics[key] for key in [
        'missing_generation_ids', 'missing_judgment_ids', 'missing_judgment_ids_among_successful_generations', 'errors', 'flagged_judge_reply_ids']})
    fields = ['id', 'question_id', 'condition', 'question', 'input_messages', 'reference_answers', 'response', 'grade',
              'generation_status', 'grading_status', 'generation_error', 'grading_error', 'possibly_truncated',
              'flagged_judge_reply', 'raw_judge_reply', 'reused_generation', 'reused_judgment']
    with (directory / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            response, judge = responses.get(row['id'], {}), judgments.get(row['id'], {})
            writer.writerow({'id': row['id'], 'question_id': row['question_id'], 'condition': row['condition'],
                'question': row['question'], 'input_messages': canonical(row['messages']),
                'reference_answers': canonical(row['correct_answers']), 'response': response.get('response'),
                'grade': judge.get('grade') if good_judgment(judge, row, response, manifest) else None,
                'generation_status': response.get('status', 'missing'), 'grading_status': judge.get('status', 'missing'),
                'generation_error': response.get('error'), 'grading_error': judge.get('error'),
                'possibly_truncated': response.get('possibly_truncated'), 'flagged_judge_reply': bool(judge.get('label_normalization')),
                'raw_judge_reply': judge.get('raw_judge_response'), 'reused_generation': bool(response.get('reuse_provenance')),
                'reused_judgment': bool(judge.get('reuse_provenance'))})
    lines = ['# Full TruthfulQA belief-confirmation results', '', f'Status: **{metrics["status"]}**.', '',
        '| Condition | Expected | Generated | Graded | Correct | Accuracy | 95% Wilson CI |',
        '|---|---:|---:|---:|---:|---:|---|']
    def percentage(value):
        return f'{value:.2%}' if value is not None else 'undefined'
    for item in metrics['condition_results']:
        interval = item['wilson_95_ci']
        text = ' – '.join(percentage(value) for value in interval) if interval else 'undefined'
        lines.append(f'| {item["condition"]} | {item["expected_records"]} | {item["successful_generations"]} | {item["successfully_graded_responses"]} | {item["correct_responses"]} | {percentage(item["accuracy"])} | {text} |')
    lines += ['', f'Mean across four belief conditions: {percentage(metrics["mean_belief_accuracy"])}.',
        f'Minimum across four belief conditions: {percentage(metrics["minimum_belief_accuracy"])}.',
        metrics['accuracy_denominator'], metrics['aggregate_policy'], '',
        '## Paired neutral-correct → belief-incorrect', '',
        '| Condition | Both valid | Neutral-correct denominator | Became incorrect | Rate |', '|---|---:|---:|---:|---:|']
    for item in metrics['paired_neutral_transitions']:
        lines.append(f'| {item["condition"]} | {item["questions_with_both_valid_grades"]} | {item["neutral_correct_denominator"]} | {item["neutral_correct_to_belief_incorrect"]} | {percentage(item["rate"])} |')
    lines += ['', '## Configuration and completeness', '',
        f'Judge prompt SHA-256: `{manifest["judge_prompt_sha256"]}`; copied verbatim from `{manifest["source_regrade_run"]}`.',
        f'Judge requested: `{manifest["judge_config"]["requested_model"]}`; actual models: `{metrics["actual_judge_models"]}`. Temperature 0; text output; one user message.',
        manifest['judge_model_choice'],
        f'Checkpoint/generation configuration: `{canonical(manifest["generation_identity"]["config"])}`.',
        f'Device: `{manifest["generation_identity"]["device"]}`. {manifest["actual_batch_policy"]}',
        f'Seed policy: {manifest["generation_identity"]["seed_policy"]}', manifest['nondeterminism'],
        f'Reused/new generations: {metrics["reused_generations"]}/{metrics["new_generations"]}; reused/new judgments: {metrics["reused_judgments"]}/{metrics["new_judgments"]}.',
        f'Empty responses: {metrics["empty_responses"]}; possible truncations: {metrics["possibly_truncated_responses"]}; flagged judge replies: {metrics["flagged_judge_reply_count"]}.',
        f'Missing generations: {len(metrics["missing_generation_ids"])}; missing/failed grades among generated answers: {len(metrics["missing_judgment_ids_among_successful_generations"])}; recorded current errors: {len(metrics["errors"])}.',
        'All missing IDs and current errors are listed in `missing_and_errors.json`; historical attempts and API usage remain in the journals.',
        metrics['interpretation'], '', '## Commands', '', '```json', json.dumps(commands(directory), indent=2), '```', '']
    prepared.write_atomic(directory / 'report.md', '\n'.join(lines).encode())
    save(directory / 'commands.json', commands(directory))
    progress(directory, manifest, manifest['status'], metrics['successful_generations'], metrics['successfully_graded_responses'], metrics_status=metrics['status'])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'prepare', 'all', 'generate', 'grade', 'report'])
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--result-dir', type=Path, help='Existing experiment inside FullData/result/')
    parser.add_argument('--new', action='store_true', help='Create a separate timestamped experiment')
    parser.add_argument('--device', help='CUDA device for a new run; defaults to the saved cuda:1')
    parser.add_argument('--judge-model', choices=[SNAPSHOT, 'gpt-4o'], help='New-run judge; defaults to the pinned snapshot')
    parser.add_argument('--no-reuse', action='store_true', help='Disable importing compatible smoke results in a new run')
    args = parser.parse_args()
    if args.command == 'validate':
        rows, _, _ = validate_data(args.data_dir.resolve())
        recover_reference()
        print(json.dumps({'status': 'passed', 'questions': 817, 'belief_prompts': 3268, 'neutral_prompts': 817,
                          'total': len(rows), 'judge_prompt_sha256': EXPECTED_PROMPT_SHA256}, indent=2))
        return
    if args.new and args.result_dir:
        parser.error('--new and --result-dir are mutually exclusive')
    fresh = args.new or (args.result_dir is None and not (RESULT / 'latest.json').exists())
    if fresh:
        if args.command in ['grade', 'report']:
            parser.error('Prepare a run first, or pass an existing --result-dir')
        directory, rows, manifest = create_run(args.data_dir.resolve(), args.device, args.judge_model, not args.no_reuse)
    else:
        directory = resolve_result(args.result_dir)
        rows, manifest = load_run(directory)
        if args.device and args.device != manifest['generation_identity']['device']:
            parser.error('Device differs from this run; use --new for a different configuration')
        if args.judge_model and args.judge_model != manifest['judge_config']['requested_model']:
            parser.error('Judge differs from this run; use --new for a different configuration')
        if args.no_reuse:
            parser.error('--no-reuse applies only when creating a new experiment')
    print(f'RESULT_DIR={directory}', flush=True)
    with locked(directory):
        failure = None
        try:
            if args.command == 'all':
                recover_reference()[4].credential()  # Validate key availability before expensive model work.
            if args.command in ['all', 'generate']:
                generate(directory, rows, manifest)
            if args.command in ['all', 'grade']:
                grade(directory, rows, manifest)
        except BaseException as error:
            failure = error
            update(directory, manifest, status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   error_type=type(error).__name__)
        finally:
            metrics = report(directory, rows, manifest)
            if not failure and args.command != 'prepare':
                update(directory, manifest, status='complete' if metrics['status'] == 'complete' else 'incomplete')
                progress(directory, manifest, manifest['status'], metrics['successful_generations'], metrics['successfully_graded_responses'])
        print(json.dumps({key: metrics[key] for key in ['status', 'successful_generations', 'successfully_graded_responses',
            'reused_generations', 'new_generations', 'reused_judgments', 'new_judgments']}, indent=2), flush=True)
        if failure:
            raise failure


if __name__ == '__main__':
    main()
