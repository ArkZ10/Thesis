"""Run the repository's exact social rubrics with a separate, pinned evaluator."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

from openai import OpenAI
from .data import ROOT, UPSTREAM, digest, write_csv
from .inference import load_journal, append_record
from .scoring import collect, validate_run_metadata
sys.path.insert(0, str(UPSTREAM))
from sycophancy_scorers import create_prompt
from moral_sycophancy_scorer import wilson_interval


def api_client(config):
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        path = Path(config['credential_file'])
        for line in path.read_text().splitlines():
            if line.strip().startswith('OPENAI_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('\"\'')
    if not key:
        raise ValueError('Missing OPENAI_API_KEY')
    return OpenAI(api_key=key, timeout=60, max_retries=3)


def judge(split, allow_partial=False):
    config = json.loads((ROOT / 'configs/judge.json').read_text())
    generation_metadata = validate_run_metadata('advice', split, allow_partial)
    rows = collect('advice', split)
    from .data import read_csv
    expected_keys = {
        f'{Path(filename).stem}:{row["id"]}'
        for filename in generation_metadata['input_sha256']
        for row in read_csv(ROOT / filename)
    }
    if not allow_partial and {row['key'] for row in rows} != expected_keys:
        raise ValueError('Incomplete advice response coverage')
    rubric_sha = digest((UPSTREAM / 'sycophancy_scorers.py').read_bytes())
    # Persistent content-addressed cache safely reuses identical rubric calls.
    cache_dir = ROOT / 'results/judge_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    journal = cache_dir / 'judgments.jsonl'
    old = load_journal(journal)
    cache = {r['cache_key']: r for r in old if r['score'] in [0, 1] and not r['error']}
    tasks, lookup = {}, {}
    for row in rows:
        for source in ['model', 'human']:
            response = row['response'] if source == 'model' else row.get('human', '')
            if not response or (source == 'model' and row['error']):
                continue
            for metric in config['metrics']:
                rubric = create_prompt({'prompt': row['prompt'], 'response': response}, metric, 'prompt', 'response')
                key = digest(json.dumps({'model': config['model'], 'temperature': config['temperature'], 'max_tokens': config['max_tokens'], 'system': config['system_prompt'], 'rubric': rubric, 'rubric_sha': rubric_sha}, sort_keys=True).encode())
                tasks[key] = {'cache_key': key, 'rubric': rubric, 'metric': metric}
                lookup[(row['key'], source, metric)] = key
    pending = [task for key, task in tasks.items() if key not in cache]
    client = api_client(config)
    def call(task):
        try:
            result = client.chat.completions.create(
                model=config['model'], temperature=config['temperature'], max_tokens=config['max_tokens'],
                messages=[{'role': 'system', 'content': config['system_prompt']}, {'role': 'user', 'content': task['rubric']}],
            )
            text = result.choices[0].message.content or ''
            match = re.fullmatch(r'\s*([01])[.!]?\s*', text)
            return {'cache_key': task['cache_key'], 'metric': task['metric'], 'score': int(match.group(1)) if match else None,
                    'raw_judgment': text, 'error': '' if match else 'Unclassifiable evaluator output',
                    'requested_model': config['model'], 'actual_model': result.model,
                    'system_fingerprint': result.system_fingerprint,
                    'usage': result.usage.model_dump() if result.usage else None,
                    'created_at_utc': datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            # Record diagnostics without copying credentials from provider messages.
            return {'cache_key': task['cache_key'], 'metric': task['metric'], 'score': None,
                    'raw_judgment': '', 'error': f'{type(e).__name__}; HTTP {getattr(e, "status_code", None)}'}
    print(f'Judge {split}: {len(tasks)} unique calls; {len(pending)} uncached', flush=True)
    failures = 0
    with journal.open('a') as stream, ThreadPoolExecutor(config['concurrency']) as executor:
        futures = {executor.submit(call, task): task for task in pending}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            append_record(stream, result)
            cache[result['cache_key']] = result
            failures += bool(result['error'])
            if i % 20 == 0 or i == len(pending):
                print(f'Judge completed {i}/{len(pending)}; errors={failures}', flush=True)
    annotated = []
    for row in rows:
        item = dict(row)
        for source in ['model', 'human']:
            for metric in config['metrics']:
                key = lookup.get((row['key'], source, metric))
                result = cache.get(key, {})
                # Clear upstream human scores unless independently evaluated here.
                item[f'{metric}_{source}'] = result.get('score')
                item[f'{metric}_{source}_error'] = result.get('error', 'No usable response')
        annotated.append(item)
    out = ROOT / f'results/advice/{split}'
    write_csv(out / 'scored_responses.csv', annotated)
    summaries = {}
    for dataset in sorted({r['dataset'] for r in annotated}):
        subset = [r for r in annotated if r['dataset'] == dataset]
        metrics = {}
        for metric in config['metrics']:
            field = f'{metric}_model'
            valid = [r[field] for r in subset if r[field] in [0, 1]]
            successes = sum(valid)
            paired = [(r[field], r[f'{metric}_human']) for r in subset if r[field] in [0, 1] and r[f'{metric}_human'] in [0, 1]]
            entry = {'total_responses': len(subset), 'valid_judgments': len(valid), 'invalid_or_missing_judgments': len(subset) - len(valid),
                     'positive_count': successes, 'raw_rate': successes / len(valid) if valid else None,
                     'raw_rate_ci95_wilson': wilson_interval(successes, len(valid))}
            if paired:
                differences = [a - b for a, b in paired]
                entry.update({'comparison': 'matched human advice, freshly scored with the same evaluator', 'matched_n': len(paired),
                              'model_rate_on_matched': sum(a for a, b in paired) / len(paired),
                              'human_rate_on_matched': sum(b for a, b in paired) / len(paired),
                              'difference_from_human': sum(differences) / len(paired)})
                # Paired bootstrap preserves prompt-level model/human dependence.
                import numpy as np
                rng = np.random.default_rng(0)
                values = np.array(differences)
                boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(2000)])
                entry['difference_ci95_paired_bootstrap'] = np.quantile(boot, [.025, .975]).tolist()
            elif dataset == 'SS':
                entry.update({'comparison': 'specified baseline; no matching human advice', 'baseline': config['baseline_without_human'],
                              'difference_from_baseline': entry['raw_rate'] - config['baseline_without_human'] if valid else None})
            metrics[metric] = entry
        summaries[dataset] = {'generated_rows': len(subset), 'generation_errors': sum(bool(r['error']) for r in subset),
                              'responses_without_end_token': sum(not r.get('contains_end_token', False) for r in subset), 'metrics': metrics}
    report = {'split': split, 'provisional': allow_partial, 'judge_config': config, 'generation_config': generation_metadata['config'], 'rubric_source_sha256': rubric_sha,
              'datasets': summaries, 'unique_judgments': len(tasks), 'uncached_calls_this_run': len(pending), 'new_errors': failures}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = [f'# LLaDA social evaluation: {split}', '',
             f'Evaluator: {config["model"]}; exact current repository rubrics; temperature 0.', '',
             '| Dataset | Metric | Valid / total | Model rate | Reference | Difference |', '|---|---|---:|---:|---|---:|']
    for name, data in summaries.items():
        for metric, item in data['metrics'].items():
            rate = f'{item["raw_rate"]:.2%}' if item['raw_rate'] is not None else 'unavailable'
            delta = item.get('difference_from_human', item.get('difference_from_baseline'))
            delta_text = f'{delta:+.2%}' if delta is not None else 'unavailable'
            reference = f'human {item["human_rate_on_matched"]:.2%}' if 'human_rate_on_matched' in item else ('0.5 baseline' if name == 'SS' else 'unavailable')
            lines.append(f'| {name} | {metric} | {item["valid_judgments"]} / {item["total_responses"]} | {rate} | {reference} | {delta_text} |')
    lines += ['', 'Human comparisons use matched prompts and the same judge/rubric for both sources.',
              'Higher rates mean more validation, more indirectness, or greater framing acceptance.',
              'Outputs have a fixed 128-token window; inspect responses_without_end_token in summary.json for possible truncation.',
              'Sample results check the pipeline; they are not full-dataset estimates.', '']
    (ROOT / f'reports/advice_{split}.md').write_text('\n'.join(lines))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import fcntl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['sample', 'full'], required=True)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    (ROOT / 'results/judge_cache').mkdir(parents=True, exist_ok=True)
    with (ROOT / 'results/judge_cache/judge.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        judge(**vars(args))
