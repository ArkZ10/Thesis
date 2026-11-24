"""Download a pinned official OPC-R1-8B snapshot and verify its weights."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from huggingface_hub import HfApi, snapshot_download

MODEL = 'INSAIT-Institute/OPC-R1-8B'
REVISION = 'f69e30e0c4184ce763f511cdd16f24bcf8e9ab10'
ROOT = Path(__file__).resolve().parent / 'results/opc_r1_8b_smoke'


def write(report):
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / 'download.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temporary, ROOT / 'download.json')


def main():
    started = time.perf_counter()
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    files = [{'name': f.rfilename, 'bytes': f.size,
              'sha256': f.lfs.sha256 if f.lfs else None}
             for f in info.siblings if f.rfilename.endswith(('.json', '.safetensors', '.jinja', '.md'))]
    report = {'model': MODEL, 'revision': info.sha,
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'downloading', 'files': files, 'verified_files': []}
    required = sum(f['bytes'] or 0 for f in files)
    report['download_bytes'] = required
    cache = Path.home() / '.cache/huggingface/hub'
    cache.mkdir(parents=True, exist_ok=True)
    report['cache_directory'] = str(cache)
    # Existing complete or partial cache blobs can be resumed without duplication.
    cached = cache / ('models--' + MODEL.replace('/', '--'))
    existing = sum(p.stat().st_size for p in (cached / 'blobs').glob('*') if p.is_file())
    if shutil.disk_usage(cache).free < max(0, required - existing) + 1024 ** 3:
        raise RuntimeError('Insufficient free disk space for the checkpoint plus a 1 GiB margin')
    write(report)
    try:
        print(f'Downloading {MODEL}@{info.sha}: {required / 1e9:.2f} GB', flush=True)
        path = Path(snapshot_download(MODEL, revision=info.sha, cache_dir=cache,
                    allow_patterns=[f['name'] for f in files], max_workers=2))
        report.update(status='verifying', snapshot_path=str(path))
        write(report)
        for item in files:
            file = path / item['name']
            if file.stat().st_size != item['bytes']:
                raise RuntimeError(f'File size mismatch: {item["name"]}')
            if item['sha256']:
                checksum = hashlib.sha256()
                with file.open('rb') as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        checksum.update(chunk)
                if checksum.hexdigest() != item['sha256']:
                    raise RuntimeError(f'Official SHA-256 mismatch: {item["name"]}')
                report['verified_files'].append(item['name'])
                write(report)
                print(f'Verified {item["name"]}', flush=True)
        report.update(status='verified', seconds=round(time.perf_counter() - started, 3),
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(report)
        print(json.dumps({k: report[k] for k in ('status', 'revision', 'snapshot_path', 'seconds')}, indent=2), flush=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(report)
        raise


if __name__ == '__main__':
    main()
