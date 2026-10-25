"""Validate inputs and recover published data without modifying upstream samples."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parents[1] / 'elephant'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_pairs(original, flipped):
    def indexed(rows):
        result = {}
        for row in rows:
            key = row.get('id', '').strip()
            if not key or key in result:
                raise ValueError('Missing or duplicate pair ID')
            if not row.get('prompt', '').strip():
                raise ValueError('Missing story prompt')
            result[key] = row
        return result
    a, b = indexed(original), indexed(flipped)
    if not a or a.keys() != b.keys():
        raise ValueError('Empty or unmatched pair IDs')
    identical = [key for key in a if a[key]['prompt'].strip() == b[key]['prompt'].strip()]
    if identical:
        raise ValueError(f'Identical opposing perspectives: {identical}')
    return [(a[key], b[key]) for key in a]


def history_file(spec, destination):
    data = subprocess.check_output(['git', 'show', spec], cwd=UPSTREAM)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return list(csv.DictReader(io.StringIO(data.decode()))), {
        'source': f'git:{spec}', 'sha256': digest(data), 'bytes': len(data),
    }


def prepare():
    raw = ROOT / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    published = UPSTREAM / 'full_results_from_paper/sycophancy_analyses/moral_sycophancy'
    full = read_csv(published / 'AITA-NTA-FLIP_og_YTANTA.csv')
    check = read_csv(published / 'AITA-NTA-FLIP_flipped_YTANTA.csv')
    # Both fields coexist within each published record: their pairing is explicit.
    assert [(r['original_NTA_post'], r['flipped_post']) for r in full] == [
        (r['original_NTA_post'], r['flipped_post']) for r in check]
    assert len(full) == 1591
    original_samples = read_csv(UPSTREAM / 'sample_datasets/AITA-NTA-OG_sample.csv')
    bad_samples = read_csv(UPSTREAM / 'sample_datasets/AITA-NTA-FLIP_sample.csv')
    known_ids = {r['prompt'].strip(): r['id'] for r in original_samples}
    original, flipped = [], []
    for row in full:
        source_index = row['']
        key = known_ids.get(row['original_NTA_post'].strip(), f'paper-{source_index}')
        for perspective, prompt, target in [
            ('original', row['original_NTA_post'], original),
            ('flipped', row['flipped_post'], flipped),
        ]:
            target.append({'id': key, 'source_pair_index': source_index,
                           'perspective': perspective, 'prompt': prompt})
    validate_pairs(original, flipped)
    by_id = {r['id']: r for r in flipped}
    sample_a = [{**next(x for x in original if x['id'] == row['id'])} for row in original_samples]
    sample_b = [by_id[row['id']] for row in original_samples]
    validate_pairs(sample_a, sample_b)
    for split, a, b in [('sample', sample_a, sample_b), ('full', original, flipped)]:
        write_csv(ROOT / f'data/{split}/AITA-NTA-OG.csv', a)
        write_csv(ROOT / f'data/{split}/AITA-NTA-FLIP.csv', b)
    for dataset, prompt_col, human_col in [('AITA-YTA', 'prompt', 'top_comment'), ('OEQ', 'prompt', 'human'), ('SS', 'sentence', None)]:
        rows = read_csv(UPSTREAM / f'sample_datasets/{dataset}_sample.csv')
        normalized = []
        for i, row in enumerate(rows):
            entry = {'id': f'{dataset.lower()}-{i:05d}', 'perspective': 'advice',
                     'prompt': row[prompt_col]}
            if human_col:
                entry['human'] = row[human_col]
            entry.update({k: v for k, v in row.items() if k.endswith('_human')})
            normalized.append(entry)
        write_csv(ROOT / f'data/sample/{dataset}.csv', normalized)
    aita, aita_source = history_file('f2e0a5b^:datasets/AITA.csv', raw / 'AITA.csv')
    aita = [r for r in aita if r['ytanta'].strip().upper() == 'YTA']
    assert len(aita) == 2000
    write_csv(ROOT / 'data/full/AITA-YTA.csv', [
        {'id': f'aita-yta-{r[""]}', 'perspective': 'advice', 'prompt': r['prompt'], 'human': r['top_comment']}
        for r in aita])
    oeq, oeq_source = history_file('2333275^:datasets/OEQ.csv', raw / 'OEQ.csv')
    assert len(oeq) == 3027
    # Historical score names differ from the current rubrics. Preserve them as
    # historical metadata, and rejudge human advice with the current rubrics.
    write_csv(ROOT / 'data/full/OEQ.csv', [
        {'id': f'oeq-{r[""]}', 'perspective': 'advice', 'prompt': r['prompt'], 'human': r['human'],
         **{f'historical_{k}': v for k, v in r.items() if k.endswith('_human')}} for r in oeq])
    report = {
        'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM, text=True).strip(),
        'supplied_sample_issue': 'All 10 flipped sample prompts equal their original prompts.',
        'identical_supplied_sample_ids': [r['id'] for r, b in zip(original_samples, bad_samples) if r['prompt'] == b['prompt']],
        'sample_recovery': 'Exact original prompt match to published paired records; retain all 10 original Reddit IDs.',
        'full_pair_id_policy': 'Retain known sample Reddit IDs; use paper-<published CSV index> otherwise. These generated IDs are not Reddit IDs.',
        'paired_sources': [{'path': str(p), 'sha256': digest(p.read_bytes())} for p in sorted(published.glob('*.csv'))],
        'sample_pair_count': len(sample_a), 'full_pair_count': len(full),
        'sample_pairs_structurally_valid': True, 'full_pairs_structurally_valid': True,
        'full_aita_yta_source': aita_source, 'full_oeq_source': oeq_source,
        'full_ss_status': 'Unavailable: OSF returned HTTP 503 scheduled maintenance; only official 10-row sample is local.',
        'osf_url': 'https://osf.io/r3dmj/?view_only=37ee66a8020a45c29a38bd704ca61067',
        'prepared_files': {},
    }
    for p in sorted((ROOT / 'data').glob('*/*.csv')):
        report['prepared_files'][str(p.relative_to(ROOT))] = {
            'sha256': digest(p.read_bytes()), 'rows': len(read_csv(p)),
        }
    (ROOT / 'reports/data_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
