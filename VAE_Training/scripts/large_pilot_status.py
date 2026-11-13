"""Read-only live epoch progress for the latest larger VAE pilot."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def number(value):return '-' if value in (None,'') else f'{float(value):.4f}'
def percent(value):return '-' if value in (None,'') else f'{float(value):.2%}'


def main():
    try:
        launch=json.loads((ROOT/'manifests/large_pilot_launch.json').read_text())
        run=Path(launch['directory']);status_path=run/'status.json'
        state=json.loads(status_path.read_text()) if status_path.exists() else {}
    except (OSError,json.JSONDecodeError) as error:
        print('Pilot status not ready:',error);return
    print('Run:',run.name)
    print('Job:',launch['stage'],'| Stage:',state.get('stage','starting'),'| PID:',launch.get('pid','starting'))
    print(f"Epoch {state.get('epoch',0)}/{state.get('epochs',3)} | Batch {state.get('batch',0)}/{state.get('batches',473)} | {state.get('percent',0):.1f}%")
    print('Running train loss:',number(state.get('train_loss')),'| Token accuracy:',percent(state.get('train_token_accuracy')))
    if state.get('stage')=='training' and state.get('epoch_training_eta_seconds') is not None:
        print(f"Epoch training ETA: {state['epoch_training_eta_seconds']/60:.1f} minutes (evaluation/checkpoint time excluded)")
    if state.get('evaluation_total',0):
        print(f"Evaluation: {state.get('evaluation_completed',0)}/{state['evaluation_total']} sentences | Exact so far: {state.get('evaluation_exact','-')}")
    if state.get('updated_at_utc'):
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(state['updated_at_utc'])).total_seconds()
        print(f'Last update: {age:.0f} seconds ago')
    if state.get('error'):print('ERROR:',state['error'])
    if launch.get('stop_reason'):print('STOP:',launch['stop_reason'])
    print('\nEpoch metrics: token accuracy uses correct preceding text; exact is free reconstruction.')
    print(f"{'Epoch':>5} {'Train loss':>11} {'Train tok':>10} {'Dev loss':>10} {'Dev tok':>10} {'Train exact*':>12} {'Dev exact':>10} {'Numeric':>10}")
    if (run/'metrics.csv').exists():
        with (run/'metrics.csv').open() as f:
            for r in csv.DictReader(f):
                epoch='base' if r['epoch']=='0' else r['epoch']
                print(f"{epoch:>5} {number(r.get('train_loss')):>11} {percent(r.get('train_token_accuracy')):>10} {number(r.get('val_loss')):>10} {percent(r.get('val_token_accuracy')):>10} {percent(r.get('train_sample_exact_match_rate')):>12} {percent(r.get('val_exact_match_rate')):>10} {percent(r.get('val_numeric_match_rate')):>10}")
    print('\n*Train exact uses a fixed 128-sentence sample. Dev exact uses all 822 dev sentences.')
    print('Log:',launch['log'])
    print('Ctrl+C closes this monitor without stopping training.')


if __name__=='__main__':main()
