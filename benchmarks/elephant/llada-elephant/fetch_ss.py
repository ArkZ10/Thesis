"""Fetch the official full SS dataset when OSF is available; never substitute data."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import zipfile

import requests
from llada_elephant.data import ROOT, digest, write_csv
import csv

URL = 'https://api.osf.io/v2/nodes/r3dmj/files/osfstorage/'
VIEW = '37ee66a8020a45c29a38bd704ca61067'


def fetch():
    target = ROOT / 'data/full/SS.csv'
    if target.exists():
        print(f'Full SS already exists: {target}')
        return True
    queue = [URL]
    archive_url = None
    while queue and not archive_url:
        url = queue.pop(0)
        response = requests.get(url, params={'view_only': VIEW}, timeout=40)
        if response.status_code != 200:
            report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'url': url,
                      'http_status': response.status_code,
                      'scheduled_maintenance': 'scheduled maintenance' in response.text.lower(),
                      'full_ss_available': False}
            (ROOT / 'reports/osf_access.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report))
            return False
        data = response.json()
        for entry in data.get('data', []):
            attr = entry['attributes']
            if attr.get('name') == 'datasets.zip':
                archive_url = entry['links']['download']
                break
            if attr.get('kind') == 'folder':
                related = entry.get('relationships', {}).get('files', {}).get('links', {}).get('related', {})
                if isinstance(related, dict) and related.get('href'):
                    queue.append(related['href'])
        next_url = data.get('links', {}).get('next')
        if next_url:
            queue.append(next_url)
    if not archive_url:
        raise ValueError('Official datasets.zip was not found in OSF listing')
    response = requests.get(archive_url, params={'view_only': VIEW}, timeout=180)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    candidates = [name for name in archive.namelist() if Path(name).name in ['SS.csv', 'ALP.csv'] and '__MACOSX' not in name]
    if len(candidates) != 1:
        raise ValueError(f'Expected one official SS CSV; found {candidates}')
    raw_csv = archive.read(candidates[0])
    rows = list(csv.DictReader(io.StringIO(raw_csv.decode('utf-8-sig'))))
    if len(rows) != 3777:
        raise ValueError(f'Expected 3,777 SS prompts; found {len(rows)}')
    prompt_col = 'sentence' if 'sentence' in rows[0] else 'prompt'
    prepared = [{'id': f'ss-{i:05d}', 'perspective': 'advice', 'prompt': row[prompt_col]} for i, row in enumerate(rows)]
    if any(not row['prompt'].strip() for row in prepared):
        raise ValueError('Empty SS prompt')
    (ROOT / 'data/raw/datasets.zip').write_bytes(response.content)
    (ROOT / 'data/raw/SS.csv').write_bytes(raw_csv)
    write_csv(target, prepared)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'archive_url': archive_url,
              'archive_sha256': digest(response.content), 'ss_csv_sha256': digest(raw_csv),
              'rows': len(rows), 'full_ss_available': True}
    (ROOT / 'reports/osf_access.json').write_text(json.dumps(report, indent=2) + '\n')
    validation_path = ROOT / 'reports/data_validation.json'
    validation = json.loads(validation_path.read_text())
    validation['full_ss_status'] = 'Official 3,777-row file downloaded from OSF.'
    validation['prepared_files']['data/full/SS.csv'] = {'sha256': digest(target.read_bytes()), 'rows': len(rows)}
    validation_path.write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return True


if __name__ == '__main__':
    try:
        available = fetch()
    except requests.RequestException as error:
        print(f'OSF network error: {type(error).__name__}', file=sys.stderr)
        available = False
    sys.exit(0 if available else 2)
