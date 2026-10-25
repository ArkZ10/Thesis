#!/usr/bin/env python3
"""Read full-run completion counts without loading a model or calling a judge."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone

import workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=workflow.Path)
    args = parser.parse_args()
    directory = workflow.resolve_result(args.result_dir)
    manifest = workflow.prepared.read_json(directory / 'manifest.json')
    rows = [row for name in ['belief_prompts.jsonl', 'neutral_prompts.jsonl']
            for row in workflow.prepared.read_jsonl(directory / 'inputs' / name)]
    responses = workflow.latest(directory / 'responses.jsonl')
    judgments = workflow.latest(directory / 'judgments.jsonl')
    metrics = workflow.aggregate(rows, responses, judgments, manifest)
    progress_path = directory / 'progress.json'
    last_progress = workflow.prepared.read_json(progress_path) if progress_path.exists() else None
    print(json.dumps({'result_dir': str(directory), 'stage': manifest['status'], 'results_status': metrics['status'],
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'last_progress': last_progress,
        'generations': f'{metrics["successful_generations"]}/{len(rows)}',
        'judgments': f'{metrics["successfully_graded_responses"]}/{len(rows)}',
        'condition_counts': [{key: row[key] for key in ['condition', 'expected_records', 'successful_generations', 'successfully_graded_responses']}
                             for row in metrics['condition_results']],
        'reused_generations': metrics['reused_generations'], 'reused_judgments': metrics['reused_judgments'],
        'current_error_counts': dict(Counter(row['stage'] for row in metrics['errors'])),
        'empty_responses': metrics['empty_responses'], 'possible_truncations': metrics['possibly_truncated_responses'],
        'flagged_judge_replies': metrics['flagged_judge_reply_count']}, indent=2))


if __name__ == '__main__':
    main()
