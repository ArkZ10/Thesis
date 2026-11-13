"""Run only the bounded large-model smoke test and record host memory usage."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='vae-large-smoke-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--backend',choices=['zero3','split'],default='zero3')
    args=parser.parse_args()
    if Path(args.name).name!=args.name:
        raise ValueError('Run name must be a single directory name')
    out=ROOT/'outputs'/args.name
    if out.exists():raise FileExistsError(out)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MAX_JOBS='2',DS_SKIP_CUDA_CHECK='1',
               PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nproc_per_node=2',
             '-m','scripts.run_large_smoke','--directory',str(out)]
    if args.backend=='split':
        command=[str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_large_split_smoke','--directory',str(out)]
    log=ROOT/'logs'/f'{args.name}.log'
    with log.open('x') as f:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),pid=p.pid,command=command,
                    directory=str(out),log=str(log),stage='running',host_memory_floor_gib=2.0)
        manifest=ROOT/'manifests'/f'{args.name}_launch.json'
        manifest.write_text(json.dumps(record,indent=2)+'\n')
        pointer=ROOT/'manifests/large_smoke_latest.json'
        temporary=pointer.with_suffix('.tmp')
        temporary.write_text(json.dumps({'manifest':str(manifest)})+'\n')
        temporary.replace(pointer)
        lowest=psutil.virtual_memory().available/2**30
        while p.poll() is None:
            free=psutil.virtual_memory().available/2**30
            lowest=min(lowest,free)
            if free<2.0:
                record['stop_reason']='Host available memory fell below 2 GiB; terminated this smoke-test process group.'
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(1)
        record.update(stage='completed' if p.returncode==0 else 'failed',returncode=p.returncode,
                      finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      minimum_available_host_gib=lowest)
        manifest.write_text(json.dumps(record,indent=2)+'\n')
        status=out/'status.json'
        if p.returncode and status.exists():
            prior=json.loads(status.read_text())
            prior.update(stage='failed',last_stage=prior['stage'],returncode=p.returncode,
                         error=record.get('stop_reason','See run log for worker exception'))
            status.write_text(json.dumps(prior,indent=2)+'\n')
        print(json.dumps(record,indent=2),flush=True)
        raise SystemExit(p.returncode or 0)


if __name__=='__main__':main()
