#!/usr/bin/env python3
"""Regrade saved smoke responses in a separate run, without loading LLaDA."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil

import run_smoke as smoke


def source_checksums(source_run):
    return {str(path.relative_to(source_run)): smoke.digest(path.read_bytes())
            for path in sorted(source_run.rglob('*')) if path.is_file()}


def initialize_regrade(source_run, run_dir=None):
    source_run = source_run.resolve()
    source_manifest = json.loads((source_run / 'manifest.json').read_text())
    rows, dataset_manifest, paths = smoke.validate_inputs(Path(source_manifest['data_dir']))
    for path in paths:
        if path.read_bytes() != (source_run / 'inputs' / path.name).read_bytes():
            raise ValueError(f'Source snapshot differs from validated prepared inputs: {path.name}')
    if json.loads((source_run / 'inputs/dataset_manifest.json').read_text()) != dataset_manifest:
        raise ValueError('Source dataset manifest differs from the validated preparation manifest')
    gen_hash = source_manifest['generation_config_sha256']
    if smoke.digest(smoke.canonical(source_manifest['generation_identity']).encode()) != gen_hash:
        raise ValueError('Source generation identity checksum mismatch')
    responses = smoke.latest(source_run / 'responses.jsonl')
    if set(responses) != {row['id'] for row in rows}:
        raise ValueError('Source must contain exactly the 100 smoke response IDs')
    if not all(smoke.compatible_success(responses[row['id']], row, gen_hash) for row in rows):
        raise ValueError('Source must contain 100 successful, input/config-matched responses')
    checksums = source_checksums(source_run)
    identity = {
        'source_run': str(source_run), 'source_run_id': source_manifest['run_id'],
        'source_artifact_sha256': checksums,
        'generation_identity': source_manifest['generation_identity'],
        'generation_config_sha256': gen_hash, 'judge_config': smoke.JUDGE_CONFIG,
        'evaluation_identity': {
            'runner_source_sha256': smoke.digest(Path(smoke.__file__).read_bytes()),
            'regrade_source_sha256': smoke.digest(Path(__file__).read_bytes()),
            'openai_version': importlib.metadata.version('openai'),
        },
    }
    fingerprint = smoke.digest(smoke.canonical(identity).encode())
    if run_dir is not None:
        run_dir = run_dir.resolve()
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        if manifest.get('fingerprint') != fingerprint:
            raise ValueError('Cannot resume: source artifacts, judge settings, code or packages changed')
        if (run_dir / 'responses.jsonl').read_bytes() != (source_run / 'responses.jsonl').read_bytes():
            raise ValueError('Copied responses changed; refusing to grade')
        if (run_dir / 'judge_prompt.txt').read_text() != smoke.JUDGE_PROMPT + '\n':
            raise ValueError('Saved judge prompt changed')
        if json.loads((run_dir / 'judge_schema.json').read_text()) != smoke.JUDGE_SCHEMA:
            raise ValueError('Saved judge label schema changed')
        return run_dir, rows, manifest
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_regrade'
    run_dir = smoke.ROOT / 'runs' / name
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_run / 'inputs', run_dir / 'inputs')
    for filename in ['responses.jsonl', 'generation_config.json']:
        shutil.copyfile(source_run / filename, run_dir / filename)
    (run_dir / 'judge_prompt.txt').write_text(smoke.JUDGE_PROMPT + '\n')
    smoke.save_json(run_dir / 'judge_schema.json', smoke.JUDGE_SCHEMA)
    manifest = {
        'schema_version': 2, 'run_id': name, 'created_at_utc': smoke.timestamp(),
        'status': 'initialized', 'run_type': 'regrade_saved_responses',
        'fingerprint': fingerprint, **identity, 'data_dir': source_manifest['data_dir'],
        'source_dataset_revision': source_manifest['source_dataset_revision'],
        'source_dataset_fingerprint': source_manifest['source_dataset_fingerprint'],
        'smoke_input_sha256': source_manifest['smoke_input_sha256'],
        'expected_records': 100, 'expected_questions': 20,
        'scope': 'Regrade 100 saved smoke responses; no inference or training.',
        'agreement_grading': 'Not measured: incorrect does not imply agreement with the user.',
    }
    smoke.save_json(run_dir / 'manifest.json', manifest)
    print(f'RUN_DIR={run_dir}', flush=True)
    return run_dir, rows, manifest


def compare(run_dir, rows, manifest, metrics):
    source_run = Path(manifest['source_run'])
    before = smoke.latest(source_run / 'judgments.jsonl')
    after = smoke.latest(run_dir / 'judgments.jsonl')
    old_metrics = json.loads((source_run / 'metrics.json').read_text())
    old_conditions = {row['condition']: row for row in old_metrics['condition_results']}
    changes = []
    comparable = 0
    for row in rows:
        old, new = before.get(row['id'], {}), after.get(row['id'], {})
        if (old.get('status') != 'success' or new.get('status') != 'success'
                or old.get('grade') not in smoke.JUDGE_SCHEMA['enum']
                or new.get('grade') not in smoke.JUDGE_SCHEMA['enum']):
            continue
        comparable += 1
        if old['grade'] != new['grade']:
            changes.append({'id': row['id'], 'question_id': row['question_id'],
                            'condition': row['condition'], 'question': row['question'],
                            'old_grade': old['grade'], 'new_grade': new['grade']})
    conditions = [{
        'condition': current['condition'],
        'old_correct': old_conditions[current['condition']]['correct_responses'],
        'old_graded': old_conditions[current['condition']]['successfully_graded_responses'],
        'old_accuracy': old_conditions[current['condition']]['accuracy'],
        'new_correct': current['correct_responses'],
        'new_graded': current['successfully_graded_responses'], 'new_accuracy': current['accuracy'],
    } for current in metrics['condition_results']]
    summary = {
        'source_run_id': manifest['source_run_id'], 'new_run_id': manifest['run_id'],
        'comparable_judgments': comparable, 'changed_judgments': len(changes),
        'unchanged_judgments': comparable - len(changes),
        'incorrect_to_correct': sum(row['new_grade'] == 'CORRECT' for row in changes),
        'correct_to_incorrect': sum(row['new_grade'] == 'INCORRECT' for row in changes),
        'condition_results': conditions, 'changes': changes,
        'limitation': 'Changes reflect the new judge prompt/output protocol and possible API nondeterminism; LLaDA responses are identical.',
    }
    smoke.save_json(run_dir / 'comparison.json', summary)
    with (run_dir / 'changed_grades.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'question_id', 'condition', 'question', 'old_grade', 'new_grade'])
        writer.writeheader()
        writer.writerows(changes)
    lines = ['# Regrading comparison', '',
             f'{comparable} comparable responses; {len(changes)} grades changed.', '',
             '| Condition | Previous correct | New correct | Previous accuracy | New accuracy |',
             '|---|---:|---:|---:|---:|']
    for item in conditions:
        old_rate = f'{item["old_accuracy"]:.1%}' if item['old_accuracy'] is not None else 'undefined'
        new_rate = f'{item["new_accuracy"]:.1%}' if item['new_accuracy'] is not None else 'undefined'
        lines.append(f'| {item["condition"]} | {item["old_correct"]}/{item["old_graded"]} | {item["new_correct"]}/{item["new_graded"]} | {old_rate} | {new_rate} |')
    lines.extend(['', summary['limitation'], ''])
    (run_dir / 'comparison.md').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, help='Existing regrading run to resume')
    args = parser.parse_args()
    smoke.credential()  # Fail before creating a run if no key is configured.
    run_dir, rows, manifest = initialize_regrade(args.source_run, args.run_dir)
    with (run_dir / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            smoke.grade(run_dir, rows, manifest)
            metrics = smoke.report(run_dir, rows, manifest)
            summary = compare(run_dir, rows, manifest, metrics)
            if source_checksums(Path(manifest['source_run'])) != manifest['source_artifact_sha256']:
                raise ValueError('Original run changed during regrading')
            print(json.dumps({key: summary[key] for key in ['comparable_judgments', 'changed_judgments', 'incorrect_to_correct', 'correct_to_incorrect']}), flush=True)
            if metrics['status'] != 'complete':
                raise RuntimeError('Some responses lack valid grades; inspect judgments.jsonl and resume this run')
        except BaseException as error:
            smoke.update_manifest(run_dir, manifest, status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error_type=type(error).__name__)
            raise


if __name__ == '__main__':
    main()
