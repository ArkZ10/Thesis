"""Read-only epoch progress and accuracy monitor; never starts training."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', action='store_true')
    args = parser.parse_args()
    path = ROOT / ('manifests/llama_paper_training_launch.json' if args.paper else 'manifests/llama_pilot_launch.json')
    if not path.exists():
        if args.paper:
            preparation = ROOT / 'data/processed/llama-paper-dart-full-v1/status.json'
            print('Paper-defined full-data training has not started.')
            if preparation.exists():
                status = json.loads(preparation.read_text())
                print(f"Data preparation: {status['stage']} | "
                      f"{status.get('training_sentences', 0):,}/{status.get('expected_training_sentences', 0):,} "
                      f"sentences ({status.get('progress_percent', 0):.1f}%)")
                if status.get('error'):
                    print('Preparation error:', status['error'])
            return
        print('No LLaMA pilot has been launched.'); return
    launch = json.loads(path.read_text())
    run = Path(launch['run_directory'])
    state = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
    label = 'LLaMA paper-defined training' if args.paper else 'LLaMA pilot'
    print(f"{label}: {launch['stage']} | Phase: {state.get('stage', 'starting')}")
    print(f"Epoch {state.get('epoch', 0)}/{launch['epochs']} | "
          f"{state.get('progress_unit', 'batch')} {state.get('batch', 0)}/{state.get('batches_per_epoch', launch['train_sentences'])} | "
          f"epoch {state.get('epoch_progress_percent', 0):.1f}% | optimizer steps {state.get('global_steps', 0)}")
    if 'train_token_accuracy' in state:
        print(f"Current epoch train token accuracy: {state['train_token_accuracy']:.2%}")
    if state.get('stage', '').startswith('evaluating'):
        print(f"Evaluation: {state.get('evaluation_batch', 0)}/{state.get('evaluation_total', 0)}")
    if 'remaining_training_seconds' in state and launch['stage'] == 'running':
        print(f"Estimated training remaining: {state['remaining_training_seconds'] / 3600:.2f} hours "
              '(evaluation/checkpoints additional; early estimate unstable)')
    print('Epoch | Train token acc | Dev token acc | Dev exact reconstruction | Numeric preservation')
    for f in sorted(run.glob('epoch-*-metrics.json')):
        row = json.loads(f.read_text()); free = row['dev_free_reconstruction']
        print(f"{row['epoch']:5} | {row['train_teacher_forced']['token_accuracy']:14.2%} | "
              f"{row['dev_teacher_forced']['token_accuracy']:12.2%} | "
              f"{free['exact_count']}/{free['total']} ({free['exact_match_rate']:.2%}) | "
              f"{free['numeric_count']}/{free['numeric_total']}")
    if 'error' in state:
        print('Error:', state['error'])
    print('Run:', run)
    print('Log:', launch['log'])


if __name__ == '__main__':
    main()
