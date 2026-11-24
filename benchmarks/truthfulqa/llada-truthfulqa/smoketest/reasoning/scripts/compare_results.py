#!/usr/bin/env python3
"""Add baseline-versus-assessment paired transitions after generation/grading exits."""
import argparse
import csv
import json
from pathlib import Path

import run_assessment as experiment


def paired_comparison(old_metrics, new_metrics):
    original = {r['condition']: r for r in old_metrics['additional_diagnostic']['conditions']}
    result = []
    for current in new_metrics['paired_neutral_transitions']:
        previous = original[current['condition']]
        before = previous['correct_to_incorrect_rate']
        after = current['rate']
        result.append({'condition': current['condition'],
            'baseline_neutral_correct_denominator': previous['neutral_correct_questions'],
            'baseline_became_incorrect': previous['correct_to_incorrect_questions'],
            'baseline_rate': before,
            'assessment_neutral_correct_denominator': current['neutral_correct_denominator'],
            'assessment_became_incorrect': current['neutral_correct_to_belief_incorrect'],
            'assessment_rate': after,
            'rate_difference_points': 100*(after-before) if before is not None and after is not None else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path)
    args = parser.parse_args()
    directory = experiment.resolve(args.result_dir)
    with experiment.common.locked(directory):
        rows, manifest = experiment.load(directory)
        new = experiment.common.prepared.read_json(directory / 'metrics.json')
        old = experiment.common.prepared.read_json(experiment.BASELINE / 'metrics.json')
        paired = paired_comparison(old, new)
        comparison = experiment.common.prepared.read_json(directory / 'comparison.json')
        comparison['paired_neutral_transition_comparison'] = paired
        comparison['diagnostic_denominators'] = ('Each arm uses questions with valid neutral and belief grades, '
            'restricted to answers correct under that arm’s neutral prompt. Denominators may differ across arms.')
        experiment.common.save(directory / 'comparison.json', comparison)
        experiment.common.save(directory / 'paired_comparison.json', paired)
        with (directory / 'paired_comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        marker = '\n## Baseline versus assessment: paired transitions\n'
        text = (directory / 'report.md').read_text().split(marker, 1)[0]
        lines = [marker, '', '| Condition | Baseline became incorrect | Baseline rate | Assessment became incorrect | Assessment rate |',
                 '|---|---:|---:|---:|---:|']
        pct = lambda value: f'{value:.2%}' if value is not None else 'undefined'
        for item in paired:
            lines.append(f'| {item["condition"]} | {item["baseline_became_incorrect"]}/{item["baseline_neutral_correct_denominator"]} | {pct(item["baseline_rate"])} | {item["assessment_became_incorrect"]}/{item["assessment_neutral_correct_denominator"]} | {pct(item["assessment_rate"])} |')
        lines += ['', comparison['diagnostic_denominators'], '']
        experiment.common.prepared.write_atomic(directory / 'report.md', (text + '\n'.join(lines)).encode())
        experiment.common.update(directory, manifest, comparison_postprocessor={
            'source_sha256': experiment.common.digest(Path(__file__).read_bytes()),
            'completed_at_utc': experiment.common.now()})
        print(json.dumps({'status': new['status'], 'paired_comparison': paired}, indent=2))


if __name__ == '__main__':
    main()
