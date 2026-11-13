"""Read-only monitoring of the latest large VAE smoke test; no GPU imports."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    if (ROOT/'manifests/large_pilot_launch.json').exists():
        import runpy
        runpy.run_path(str(ROOT/'scripts/large_pilot_status.py'),run_name='__main__')
        return
    try:
        pointer=json.loads((ROOT/'manifests/large_smoke_latest.json').read_text())
        launch=json.loads(Path(pointer['manifest']).read_text())
        run=Path(launch['directory'])
        status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
    except (OSError,json.JSONDecodeError) as error:
        print('Status is not ready:',error)
        return
    print('Run:',run.name)
    print('Mode: bounded smoke test; multi-epoch pilot has not started')
    print('Job:',launch['stage'],'| Current stage:',status.get('stage','starting'))
    if 'update' in status:print(f"Update: {status['update']}/{status['total_updates']}")
    if 'elapsed_seconds' in status:print(f"Elapsed: {status['elapsed_seconds']:.1f} seconds at last stage change")
    if 'stop_reason' in launch:print('Reason:',launch['stop_reason'])
    print('Log:',launch['log'])
    if (run/'summary.json').exists():
        summary=json.loads((run/'summary.json').read_text())
        print('Updates completed:',summary['optimizer_updates'])
        print('Decoder unchanged:',summary['decoder_unchanged'])
        print('Checkpoint saved:',summary['checkpoint_saved'])
        print('Peak allocated GiB by rank:',[round(r['peak_allocated_gib'],2) for r in summary['ranks']])
    print('Ctrl+C closes this monitor without stopping the test.')


if __name__=='__main__':main()
