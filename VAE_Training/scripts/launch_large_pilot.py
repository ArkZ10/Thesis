"""Detach the authorized pilot with a host-memory guard; no interactive polling."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'manifests/large_pilot_launch.json'


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def supervise(directory):
    # The initial parent writes the launch record before this branch starts.
    while not MANIFEST.exists():time.sleep(0.1)
    launch=json.loads(MANIFEST.read_text())
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MAX_JOBS='2',DS_SKIP_CUDA_CHECK='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with Path(launch['log']).open('x') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','scripts.run_large_pilot','--directory',str(directory)],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch.update(pid=process.pid,stage='running');write(MANIFEST,launch)
        lowest=psutil.virtual_memory().available/2**30
        while process.poll() is None:
            free=psutil.virtual_memory().available/2**30;lowest=min(lowest,free)
            if free<2:
                launch['stop_reason']='Available host RAM fell below 2 GiB; memory guard stopped this pilot only.'
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                break
            time.sleep(2)
        launch.update(stage='completed' if process.returncode==0 else 'failed',returncode=process.returncode,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),minimum_available_host_gib=lowest)
        write(MANIFEST,launch)
        if process.returncode and (directory/'status.json').exists():
            state=json.loads((directory/'status.json').read_text())
            state.update(stage='failed',error=state.get('error') or launch.get('stop_reason') or 'Process exited; see log')
            write(directory/'status.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    if args.supervise:supervise(args.directory.resolve());return
    if MANIFEST.exists():
        old=json.loads(MANIFEST.read_text())
        if old['stage'] in ['launching','running']:raise RuntimeError('A larger pilot is already active')
    verification=json.loads((ROOT/'manifests/large_resume_verification.json').read_text())
    if not verification['passed']:raise RuntimeError('Full checkpoint resume verification is required')
    name='vae-large-pilot-1000q-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    directory=(args.directory or ROOT/'outputs'/name).resolve()
    if directory.exists():raise FileExistsError(directory)
    name=directory.name
    launch=dict(directory=str(directory),log=str(ROOT/'logs'/f'{name}.log'),
        launched_at_utc=datetime.now(timezone.utc).isoformat(),stage='launching',epochs=3,
        train_questions=1000,dev_questions=100,train_sentences=7564,dev_sentences=822,
        micro_batch_size=16,batches_per_epoch=473,planned_updates=1419,
        backend='explicit_model_parallel_streamed_cpu_adam',encoder_gpu=1,decoder_gpu=0,
        initialization='fresh pretrained weights',checkpoint_policy='weights each epoch, full optimizer at final epoch')
    write(MANIFEST,launch)
    with (ROOT/'logs'/f'{name}-supervisor.log').open('x') as log:
        guard=subprocess.Popen([sys.executable,'-u','-m','scripts.launch_large_pilot',
            '--supervise','--directory',str(directory)],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    # The supervisor owns subsequent manifest updates, avoiding write races.
    print(json.dumps({**launch,'supervisor_pid':guard.pid},indent=2),flush=True)


if __name__=='__main__':main()
