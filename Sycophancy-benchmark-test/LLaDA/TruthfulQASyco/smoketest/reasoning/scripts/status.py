#!/usr/bin/env python3
"""Read saved assessment experiment progress without starting work."""
import argparse
import json
import run_assessment as experiment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--result-dir', type=experiment.Path)
args = parser.parse_args()
directory = experiment.resolve(args.result_dir)
manifest = experiment.common.prepared.read_json(directory / 'manifest.json')
rows = [row for name in ['belief_prompts.jsonl', 'neutral_prompts.jsonl']
        for row in experiment.common.prepared.read_jsonl(directory / 'inputs' / name)]
responses = experiment.common.latest(directory / 'responses.jsonl')
judgments = experiment.common.latest(directory / 'judgments.jsonl')
metrics = experiment.common.aggregate(rows, responses, judgments, manifest)
checks = [experiment.format_check(responses[row['id']]['response']) for row in rows
          if experiment.common.good_generation(responses.get(row['id'], {}), row, manifest)]
print(json.dumps({'result_dir': str(directory), 'stage': manifest['status'],
    'generations': f'{metrics["successful_generations"]}/100', 'judgments': f'{metrics["successfully_graded_responses"]}/100',
    'format_compliant': sum(c['compliant'] for c in checks), 'possible_truncations': metrics['possibly_truncated_responses'],
    'errors': metrics['errors']}, indent=2))
