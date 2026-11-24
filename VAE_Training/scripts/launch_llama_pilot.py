"""Launch the current paper-defined LLaMA VAE configuration with RAM/disk guards."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae.runtime import offload_environment

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'manifests/llama_paper_training_launch.json'


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def supervise():
    launch = json.loads(MANIFEST.read_text())
    run = Path(launch['run_directory'])
    env = offload_environment(ROOT)
    cmd = [sys.executable, '-u', 'scripts/run_llama_pilot.py', '--run-dir', str(run)]
    with Path(launch['log']).open('x') as output:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        launch.update(stage='running', pid=process.pid,
                      minimum_available_host_gib=1000., minimum_free_disk_gib=1000.)
        write(MANIFEST, launch)
        while process.poll() is None:
            ram = psutil.virtual_memory().available / 2**30
            disk = shutil.disk_usage(ROOT).free / 2**30
            launch['minimum_available_host_gib'] = min(ram, launch['minimum_available_host_gib'])
            launch['minimum_free_disk_gib'] = min(disk, launch['minimum_free_disk_gib'])
            if ram < 4 or disk < 12:
                launch['guard_stop_reason'] = f'Available RAM {ram:.2f} GiB or free disk {disk:.2f} GiB below guard'
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                break
            write(MANIFEST, launch)
            time.sleep(2)
        launch.update(stage='completed' if process.returncode == 0 else 'failed',
                      returncode=process.returncode, finished_at_utc=datetime.now(timezone.utc).isoformat())
    if process.returncode == 0:
        assert json.loads((run / 'summary.json').read_text())['completed']
        assert (run / 'checkpoints/latest').is_file()
        work = run / 'work'
        assert work.parent == run and work.name == 'work' and not work.is_symlink()
        shutil.rmtree(work)
        launch['private_working_scratch_removed'] = True
    else:
        path = run / 'status.json'
        status = json.loads(path.read_text()) if path.exists() else {}
        status.update(stage='failed', error=status.get('error') or launch.get('guard_stop_reason') or 'Worker exited; see log')
        write(path, status)
    write(MANIFEST, launch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/vae_llama_paper_math.json',
                        help='Current full-data paper configuration')
    parser.add_argument('--check', action='store_true', help='Read readiness without launching')
    args = parser.parse_args()
    custom = json.loads(args.config.read_text())
    if not custom.get('full_corpus_training'):
        raise ValueError('Configuration must explicitly select full-corpus training')
    if args.supervise:
        supervise(); return
    data = Path(custom['data_directory'])
    if args.check:
        ready = (data / 'summary.json').exists()
        disk = shutil.disk_usage(ROOT).free / 2**30
        print(json.dumps({'data_ready': ready, 'free_disk_gib': disk, 'required_free_disk_gib': 180,
                          'disk_ready': disk > 180,
                          'training_started_by_this_command': False,
                          'data_directory': str(data)}, indent=2))
        return
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        if old['stage'] in ['launching', 'running']:
            raise RuntimeError('A LLaMA pilot launch is already active')
    evidence = json.loads((ROOT / 'manifests/llama_execution_optimization.json').read_text())
    assert evidence['two_updates_finite_and_not_skipped']
    assert evidence['frozen_decoder_unchanged']
    summary = json.loads((data / 'summary.json').read_text())
    assert summary['question_overlap'] == 0 and not summary['test_used']
    assert shutil.disk_usage(ROOT).free > 180 * 2**30
    assert psutil.virtual_memory().available > 48 * 2**30
    for device in [0, 1]:
        result = subprocess.run(['nvidia-smi', '-i', str(device), '--query-compute-apps=pid', '--format=csv,noheader'],
                                check=True, capture_output=True, text=True)
        if result.stdout.strip():
            raise RuntimeError(f'GPU {device} is already in use')
    prefix = 'vae-llama-paper-math-'
    name = prefix + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run = ROOT / 'outputs' / name
    run.mkdir()
    settings = custom
    write(run / 'pilot-config.json', settings)
    write(run / 'data-summary.json', summary)
    source_hashes = {}
    source_files = sorted(str(p.relative_to(ROOT)) for folder in ['scripts', 'vae', 'configs']
                          for p in (ROOT / folder).rglob('*')
                          if p.is_file() and p.suffix in {'.py', '.json'} and '__pycache__' not in p.parts)
    for name_ in source_files:
        target = run / 'source' / name_
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name_, target)
        source_hashes[name_] = hashlib.sha256(target.read_bytes()).hexdigest()
    write(run / 'source-hashes.json', source_hashes)
    launch = {'stage': 'launching', 'run_directory': str(run), 'log': str(ROOT / 'logs' / (name + '.log')),
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'epochs': settings['epochs'],
              'train_sentences': summary['train']['sentences'], 'dev_sentences': summary['dev']['sentences'],
              'effective_batch_size': settings['effective_batch_size'], 'pilot_training': False,
              'full_corpus_training_started': True,
              'initialization': 'fresh pretrained LLaMA weights', 'guard_ram_gib': 4, 'guard_disk_gib': 12}
    write(MANIFEST, launch)
    with (ROOT / 'logs' / (name + '-supervisor.log')).open('x') as output:
        command = [sys.executable, '-u', 'scripts/launch_llama_pilot.py', '--supervise']
        if args.config:
            command.extend(['--config', str(args.config.resolve())])
        supervisor = subprocess.Popen(command,
                         cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print(json.dumps({**launch, 'supervisor_pid': supervisor.pid}, indent=2))


if __name__ == '__main__':
    main()
