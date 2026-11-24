"""Stable problem selection and resume checks shared by local GPU workers."""
import hashlib
import json
import random
from pathlib import Path

from filelock import FileLock


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_files(files, count, seed):
    ordered = sorted(files)
    if count == -1:
        return ordered
    if not 0 <= count <= len(ordered):
        raise ValueError(f'Cannot select {count} problems from {len(ordered)} files')
    return random.Random(seed).sample(ordered, count)


def ensure_manifest(path, project_config, solver_config, files):
    inputs = []
    for file in files:
        item = json.loads(Path(file).read_text())
        inputs.append({'file': str(file), 'problem_id': item['problem_id'], 'sha256': digest(file)})
    if len({item['problem_id'] for item in inputs}) != len(inputs):
        raise ValueError('Selected problems have duplicate IDs')
    model_files = [Path(project_config.model_config_folder) / (name + '.yaml')
                   for name in project_config.model_configs]
    recipe = {
        'project': project_config.model_dump(), 'solver': solver_config.model_dump(),
        'models': {str(file): digest(file) for file in model_files},
        'sources': {name: digest(Path(__file__).with_name(name))
                    for name in ('llada.py', 'api.py', 'solve.py', 'run_manifest.py')},
        'problems': inputs,
    }
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    document = {'fingerprint': fingerprint, **recipe}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.lock'):
        if path.exists():
            if json.loads(path.read_text()).get('fingerprint') != fingerprint:
                raise ValueError(f'Run inputs or settings changed: {path}. Use a new project for a new experiment.')
        else:
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(document, indent=2) + '\n')
            temporary.replace(path)
    return document
