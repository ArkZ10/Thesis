#!/usr/bin/env python3
"""Recover explicit leading grades from formatting-only errors, retaining raw output."""
import argparse
import fcntl
import json
from pathlib import Path
import re

import regrade_smoke as regrade
import run_smoke as smoke

POLICY = ('Use a standalone CORRECT/INCORRECT on the first nonempty line. '
          'Preserve all additional text. Reject any additional grade label, refusals, '
          'incomplete output, and errors other than label-format validation errors.')


def recover_label(judgment):
    if (judgment.get('status') != 'error' or judgment.get('error') != 'ValueError; HTTP None'
            or judgment.get('finish_reason') != 'stop' or judgment.get('refusal')
            or not judgment.get('returned_model')):
        return None
    raw = judgment.get('raw_judge_response')
    if not isinstance(raw, str) or not raw.strip():
        return None
    lines = raw.strip().splitlines()
    if lines[0].strip() not in smoke.JUDGE_SCHEMA['enum']:
        return None
    if re.search(r'\b(?:CORRECT|INCORRECT|UNCERTAIN)\b', '\n'.join(lines[1:]), re.IGNORECASE):
        return None
    return lines[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir, rows, manifest = regrade.initialize_regrade(args.source_run, args.run_dir)
    with (run_dir / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = smoke.latest(run_dir / 'judgments.jsonl')
        responses = smoke.latest(run_dir / 'responses.jsonl')
        sources = {row['id']: row for row in rows}
        recovered_ids = []
        with (run_dir / 'judgments.jsonl').open('a') as stream:
            for key, judgment in previous.items():
                label = recover_label(judgment)
                if label is None:
                    continue
                if judgment.get('grading_input_sha256') != smoke.grading_hash(sources[key], responses[key]):
                    raise ValueError('Cannot normalize a judgment from a different prompt/input')
                updated = {
                    **judgment, 'status': 'success', 'error': None, 'grade': label,
                    'saved_at_utc': smoke.timestamp(),
                    'label_normalization': {
                        'policy': POLICY, 'format_compliant': False,
                        'original_error': judgment['error'],
                        'source_judgment_saved_at_utc': judgment['saved_at_utc'],
                    },
                }
                smoke.event(stream, updated)
                recovered_ids.append(key)
        all_judgments = smoke.latest(run_dir / 'judgments.jsonl')
        exception_ids = sorted(key for key, row in all_judgments.items() if row.get('label_normalization'))
        smoke.update_manifest(run_dir, manifest, label_normalization={
            'policy': POLICY, 'script_sha256': smoke.digest(Path(__file__).read_bytes()),
            'recovered_ids': exception_ids,
        })
        metrics = smoke.report(run_dir, rows, manifest)
        metrics['judge_output_format_exceptions'] = exception_ids
        metrics['strict_label_responses'] = sum(
            row.get('status') == 'success' and not row.get('label_normalization')
            for row in all_judgments.values())
        smoke.save_json(run_dir / 'metrics.json', metrics)
        with (run_dir / 'report.md').open('a') as stream:
            stream.write('\n## Judge output formatting\n\n')
            stream.write(f'{metrics["strict_label_responses"]} strict label responses; {len(exception_ids)} explicit leading labels recovered from replies with extra explanation. Raw replies and failed attempts are retained.\n\n')
            for key in exception_ids:
                stream.write(f'- `{key}`\n')
        summary = regrade.compare(run_dir, rows, manifest, metrics)
        if regrade.source_checksums(args.source_run.resolve()) != manifest['source_artifact_sha256']:
            raise ValueError('Original source run changed')
        print(json.dumps({'newly_normalized': recovered_ids, 'status': metrics['status'],
                          'changed_judgments': summary['changed_judgments']}), flush=True)
        if metrics['status'] != 'complete':
            raise RuntimeError('Unresolved grading errors remain')


if __name__ == '__main__':
    main()
