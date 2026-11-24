"""Record the pilot source/checkpoint without loading model weights."""
import hashlib, json, pathlib, subprocess, tarfile, io
from datetime import datetime, timezone
ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'MetaState'
COMMIT = '307549ef9b62a8a371c5c8e59cfeacfe537630c1'
REVISION = '08b83a6feb34df1a6011b80c3c00c7563e963b07'
CHECKPOINT = pathlib.Path('/root/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots') / REVISION

def run(*args):
    return subprocess.check_output(args, text=True).strip()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert run('git', '-C', str(SOURCE), 'rev-parse', 'HEAD') == COMMIT
assert not run('git', '-C', str(SOURCE), 'status', '--porcelain'), 'Source checkout has changes'
assert CHECKPOINT.is_dir(), 'Pinned local checkpoint missing'
TARGET = ROOT / 'vendor' / 'MetaState'
if not TARGET.exists() or not any(TARGET.iterdir()):
    TARGET.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(['git', '-C', str(SOURCE), 'archive', COMMIT])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(TARGET)  # Trusted, pinned local Git archive
source_hashes = {str(p.relative_to(TARGET)): digest(p) for p in sorted(TARGET.rglob('*')) if p.is_file()}
index = json.loads((CHECKPOINT / 'model.safetensors.index.json').read_text())
shards = sorted(set(index['weight_map'].values()))
for shard in shards:
    assert (CHECKPOINT / shard).is_file(), f'Missing shard: {shard}'
setup = ROOT / 'setup'
setup.mkdir(exist_ok=True)
baseline = setup / 'original-environment.freeze.txt'
if not baseline.exists():
    baseline.write_text(run('/opt/miniforge3/envs/diffuguard/bin/python', '-m', 'pip', 'freeze') + '\n')
manifest = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'part_1_setup',
    'source': {'original': str(SOURCE), 'commit': COMMIT, 'snapshot': str(TARGET), 'file_sha256': source_hashes},
    'model': {'id': 'GSAI-ML/LLaDA-8B-Instruct', 'revision': REVISION, 'local_path': str(CHECKPOINT),
              'files': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p) if p.suffix != '.safetensors' else None,
                                'resolved_path': str(p.resolve())} for p in sorted(CHECKPOINT.iterdir()) if p.is_file()},
              'weight_check': 'Index-listed shards exist; large weight files not rehashed or loaded'},
    'environment': {'prefix': str(ROOT / '.env'), 'cloned_from': '/opt/miniforge3/envs/diffuguard', 'independent_copy': True},
    'scope': {'training_started': False, 'inference_started': False, 'data_prepared': False},
}
(setup / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(setup / 'backbone.json').write_text(json.dumps({'model_name_or_path': str(CHECKPOINT), 'backbone_name_or_path': str(CHECKPOINT), 'freeze_backbone': True, 'model_id': manifest['model']['id'], 'revision': REVISION}, indent=2) + '\n')
print('Frozen source snapshot and checkpoint metadata:', setup)
