"""Bounded small-encoder reconstruction experiments with latent-only evaluation.

--launch starts a detached job. --status only reads progress. --check only
checks local files/resources. Training runs a clean 16-sentence overfit stage;
the held-out pilot runs only after 16/16 exact training reconstructions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CURRENT = ROOT / 'manifests/small_training.json'


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def monitor():
    if not CURRENT.exists():
        print('No small-encoder experiment launched.'); return
    launch = json.loads(CURRENT.read_text())
    run = Path(launch['run_directory'])
    state = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
    print('Phase:', state.get('phase', 'starting'), '| Status:', state.get('stage', 'starting'))
    print(f"Epoch {state.get('epoch', 0)}/{state.get('epochs', '?')} | "
          f"{state.get('examples_seen_this_epoch', 0)}/{state.get('examples_per_epoch', '?')} sentences | "
          f"{state.get('epoch_progress_percent', 0):.1f}% | Updates {state.get('updates', 0)}")
    if 'teacher_forced_token_accuracy' in state:
        print(f"Current train token accuracy: {state['teacher_forced_token_accuracy']:.2%}")
    if 'step_seconds' in state:
        print(f"Latest update: {state['step_seconds']:.2f} seconds")
    for split, metrics in state.get('evaluation', {}).items():
        print(f"{split}: token {metrics['token_accuracy']:.2%}, "
              f"exact {metrics['exact_count']}/{metrics['total']}, "
              f"numeric {metrics['numeric_count']}/{metrics['numeric_total']}")
    print('Quality:', state.get('quality', 'not established'))
    if state.get('error'): print('Error:', state['error'])
    print('Run:', run)
    print('Log:', launch['log'])


def launch(config_path):
    import psutil
    config = json.loads(config_path.read_text())
    for name in ['encoder', 'decoder']:
        assert (ROOT / config[name] / 'config.json').is_file(), config[name]
    assert shutil.disk_usage(ROOT).free > 24 * 2**30, '24 GiB free disk required'
    assert psutil.virtual_memory().available > 16 * 2**30, '16 GiB host RAM required'
    if CURRENT.exists():
        previous = json.loads(CURRENT.read_text())
        pid = previous.get('pid')
        if pid and psutil.pid_exists(pid):
            command = psutil.Process(pid).cmdline()
            if '--worker' in command and str(ROOT / 'scripts/train_small.py') in command:
                raise RuntimeError('A small-model worker is already running')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True)
    if result.stdout.strip(): raise RuntimeError('GPU compute processes already running')
    name = 'vae-small-encoder-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run = ROOT / 'outputs' / name
    run.mkdir()
    write(run / 'config.json', config)
    for folder in ['vae', 'scripts', 'configs']:
        for source in (ROOT / folder).glob('*.py' if folder != 'configs' else '*.json'):
            target = run / 'source' / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    log = ROOT / 'logs' / (name + '.log')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with log.open('x') as output:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/train_small.py'),
                                    '--worker', '--run-dir', str(run)],
                                   cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    record = {'run_directory': str(run), 'log': str(log), 'pid': process.pid,
              'launched_at_utc': datetime.now(timezone.utc).isoformat()}
    write(CURRENT, record)
    print(json.dumps(record, indent=2))


def train(run):
    import numpy as np
    import psutil
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from vae.adaptive_batches import padded_batch
    from vae.efficient_decoder import checkpoint_frozen_decoder, reconstruction_loss, decode_prefix
    from vae.llama_offload import LlamaLatentEncoder
    from vae.pilot_metrics import normalized, reconstruction_values, numeric_strings
    from vae.runtime import decoder_hash

    config = json.loads((run / 'config.json').read_text())
    start = time.monotonic()
    state = {'stage': 'starting', 'phase': 'setup', 'updates': 0, 'test_used': False}
    def status(**values):
        state.update(values, elapsed_seconds=time.monotonic() - start,
                     updated_at_utc=datetime.now(timezone.utc).isoformat())
        write(run / 'status.json', state)
        print(json.dumps(state), flush=True)
    def append(name, value):
        with (run / name).open('a') as stream: stream.write(json.dumps(value) + '\n')
    try:
        torch.set_num_threads(4)
        torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
        random.seed(config['seed']); np.random.seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        encoder_device = f"cuda:{config['encoder_gpu']}"
        decoder_device = f"cuda:{config['decoder_gpu']}"
        tokenizer = AutoTokenizer.from_pretrained(ROOT / config['decoder'], local_files_only=True)
        encoder_tokenizer = AutoTokenizer.from_pretrained(ROOT / config['encoder'], local_files_only=True)
        assert tokenizer.get_vocab() == encoder_tokenizer.get_vocab(), 'Shared tokenizer vocabulary required'
        eos, bos = tokenizer.eos_token_id, tokenizer.bos_token_id
        bos = -1 if bos is None else bos

        def select(split, count):
            buckets = []
            for dataset in ['gsm8k', 'dart-math-hard']:
                chosen, questions = [], {}
                source = ROOT / f'data/processed/math-sentences-v1/sentences/{dataset}.{split}.jsonl'
                with source.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        assert row['split'] == split
                        ids = tokenizer.encode(row['reasoning_block'], add_special_tokens=False) + [eos]
                        if len(ids) > config['max_tokens_including_eos'] or len(ids) < 5: continue
                        pid = row['problem_id']
                        if questions.get(pid, 0) >= 4: continue
                        questions[pid] = questions.get(pid, 0) + 1
                        chosen.append({**row, 'dataset': dataset, 'input_ids': ids})
                        if len(chosen) == count // 2: break
                assert len(chosen) == count // 2
                buckets.append(chosen)
            return [row for pair in zip(*buckets) for row in pair]
        training = select('train', config['train_sentences'])
        dev = select('dev', config['dev_sentences'])
        assert not ({r['problem_id'] for r in training} & {r['problem_id'] for r in dev})
        assert not ({normalized(r['reasoning_block']) for r in training} &
                    {normalized(r['reasoning_block']) for r in dev})
        overfit = [r for r in training if len(r['input_ids']) <= 48 and numeric_strings(r['reasoning_block'])][:16]
        assert len(overfit) == 16
        for name, rows in [('train', training), ('dev', dev), ('overfit', overfit)]:
            with (run / f'{name}.jsonl').open('w') as stream:
                for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        write(run / 'data-summary.json', {'train': len(training), 'dev': len(dev), 'overfit': len(overfit),
              'question_overlap': 0, 'exact_target_overlap': 0, 'test_used': False,
              'length_selection_not_truncation': True,
              'files': {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                        for name in ['train.jsonl', 'dev.jsonl', 'overfit.jsonl']}})
        status(stage='loading_models')
        backbone = AutoModel.from_pretrained(ROOT / config['encoder'], dtype=torch.float32,
                   device_map={'': encoder_device}, attn_implementation='sdpa', local_files_only=True)
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        decoder = AutoModelForCausalLM.from_pretrained(ROOT / config['decoder'], dtype=torch.bfloat16,
                  device_map={'': decoder_device}, attn_implementation='sdpa', local_files_only=True)
        decoder.requires_grad_(False); decoder.eval()
        checkpoint_frozen_decoder(decoder)
        model = LlamaLatentEncoder(backbone, config['num_latents'], config['latent_dim'],
                 decoder_hidden_size=decoder.config.hidden_size, input_substitution_probability=0.,
                 extra_latent_noise_std=0., sample_posterior=False).to(encoder_device)
        write(run / 'model-setup.json', {'encoder_parameters': sum(p.numel() for p in model.parameters()),
              'decoder_parameters': sum(p.numel() for p in decoder.parameters()),
              'encoder_weights_and_optimizer': 'FP32; BF16 autocast for transformer computation',
              'decoder': 'frozen BF16; checkpointed gradient path to encoder',
              'encoder_hidden_size': backbone.config.hidden_size, 'decoder_hidden_size': decoder.config.hidden_size,
              'latent_shape': [config['num_latents'], config['latent_dim']],
              'no_decoder_targets_during_free_generation': True})
        initial_decoder_hash = decoder_hash(decoder)
        interface = [p for name, p in model.named_parameters() if not name.startswith('encoder.')]
        optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': config['backbone_learning_rate']},
            {'params': interface, 'lr': config['interface_learning_rate']}],
            betas=(.9, .95), weight_decay=.01, fused=True)

        def batch(rows): return padded_batch(rows, eos, encoder_device)
        def prefix_for(rows):
            ids, mask = batch(rows)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prefix, kl, _, _ = model(ids, mask, eos, bos)
            return prefix, kl, ids, mask

        @torch.no_grad()
        def evaluate(rows, label, control=None):
            model.eval()
            records, ce, correct, tokens = [], 0., 0, 0
            for offset in range(0, len(rows), config['physical_batch_size']):
                selected = rows[offset:offset + config['physical_batch_size']]
                prefix, _, ids, mask = prefix_for(selected)
                if control == 'zero':
                    prefix = torch.cat((torch.zeros_like(prefix[:, :-1]), prefix[:, -1:]), dim=1)
                elif control == 'shuffle':
                    prefix = torch.cat((prefix[:, :-1].roll(1, dims=0), prefix[:, -1:]), dim=1)
                _, metrics = reconstruction_loss(decoder, prefix, ids, mask)
                ce += metrics['ce_sum']; correct += metrics['correct']; tokens += metrics['tokens']
                generated = decode_prefix(decoder, prefix, tokenizer, config['free_generation_max_tokens'])
                for row, gen in zip(selected, generated):
                    records.append({**gen, 'target': row['reasoning_block'], 'problem_id': row['problem_id'],
                                    'exact_match': normalized(gen['text']) == normalized(row['reasoning_block'])})
            result = {**reconstruction_values(records), 'token_accuracy': correct / tokens,
                      'ce_loss': ce / tokens, 'token_count': tokens}
            with (run / f'{label}-reconstructions.jsonl').open('w') as stream:
                for row in records: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            return result

        def update(rows, kl_weight):
            if psutil.virtual_memory().available < 3 * 2**30 or shutil.disk_usage(ROOT).free < 12 * 2**30:
                raise RuntimeError('Available RAM or disk below training guard')
            step_start = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
            stats = {'ce_sum': 0., 'correct': 0, 'tokens': 0, 'kl': 0.}
            for offset in range(0, len(rows), config['physical_batch_size']):
                selected = rows[offset:offset + config['physical_batch_size']]
                prefix, kl, ids, mask = prefix_for(selected)
                loss, metrics = reconstruction_loss(decoder, prefix, ids, mask)
                objective = (loss + kl_weight * kl.to(loss.device)) * (len(selected) / len(rows))
                if not torch.isfinite(objective): raise RuntimeError('Nonfinite loss')
                objective.backward()
                for key in ['ce_sum', 'correct', 'tokens']: stats[key] += metrics[key]
                stats['kl'] += float(kl.detach()) * len(selected) / len(rows)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            state['updates'] += 1
            return stats, float(norm), time.monotonic() - step_start

        def checkpoint(epoch, phase, best=False):
            # Save one latest state and one best encoder; atomic replacement bounds disk usage.
            path = run / 'latest.pt'; temporary = run / 'latest.pt.tmp'
            torch.save({'encoder': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'config': config, 'epoch': epoch, 'phase': phase, 'updates': state['updates'],
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'python_rng': random.getstate()}, temporary)
            os.replace(temporary, path)
            if best:
                temporary = run / 'best-encoder.pt.tmp'
                torch.save({'encoder': model.state_dict(), 'config': config, 'epoch': epoch, 'phase': phase}, temporary)
                os.replace(temporary, run / 'best-encoder.pt')

        status(stage='evaluating', phase='overfit_clean', epochs=config['overfit_max_updates'],
               examples_per_epoch=16, quality='not established')
        baseline = evaluate(overfit, 'overfit-baseline')
        write(run / 'baseline.json', baseline)
        passed = False
        for step in range(1, config['overfit_max_updates'] + 1):
            stats, norm, seconds = update(overfit, 0.)
            status(stage='training', phase='overfit_clean', epoch=step,
                   examples_seen_this_epoch=16, epoch_progress_percent=100.,
                   teacher_forced_token_accuracy=stats['correct']/stats['tokens'],
                   loss=stats['ce_sum']/stats['tokens'], gradient_norm=norm, step_seconds=seconds)
            if step % config['overfit_evaluate_every'] == 0:
                status(stage='evaluating')
                metrics = evaluate(overfit, 'overfit-latest')
                append('metrics.jsonl', {'phase': 'overfit_clean', 'update': step, 'train': metrics})
                status(evaluation={'overfit_train': metrics})
                if metrics['exact_count'] == len(overfit):
                    passed = True; checkpoint(step, 'overfit_clean', best=True); break
        if not passed:
            checkpoint(step, 'overfit_clean')
            final_hash = decoder_hash(decoder)
            write(run / 'summary.json', {'completed': True, 'overfit_passed': False,
                  'quality_target_met': False, 'next_action': 'Inspect overfit reconstructions before any larger pilot',
                  'frozen_decoder_unchanged': final_hash == initial_decoder_hash, 'test_used': False})
            status(stage='completed', quality='overfit criterion unmet; held-out pilot not started')
            return

        status(phase='heldout_pilot', stage='training', epochs=config['pilot_epochs'], epoch=0,
               examples_per_epoch=len(training), examples_seen_this_epoch=0, epoch_progress_percent=0.,
               quality='16/16 training overfit passed; generalization unverified')
        best_score = (-1., -1., -1.)
        final_metrics = None
        for epoch in range(1, config['pilot_epochs'] + 1):
            # Introduce a small posterior perturbation gradually; keep exact settings in each epoch report.
            model.sample_posterior = True
            model.posterior_noise_scale = config['pilot_posterior_noise_scale'] * min(1., epoch / 4.)
            model.input_substitution_probability = config['pilot_input_substitution_probability']
            model.extra_latent_noise_std = config['pilot_extra_latent_noise_std']
            order = list(training); random.shuffle(order)
            total_correct, total_tokens = 0, 0
            for offset in range(0, len(order), config['effective_batch_size']):
                selected = order[offset:offset + config['effective_batch_size']]
                stats, norm, seconds = update(selected, config['pilot_kl_weight'])
                total_correct += stats['correct']; total_tokens += stats['tokens']
                status(stage='training', epoch=epoch, examples_seen_this_epoch=offset+len(selected),
                       epoch_progress_percent=100*(offset+len(selected))/len(order),
                       teacher_forced_token_accuracy=total_correct/total_tokens,
                       gradient_norm=norm, step_seconds=seconds)
            status(stage='evaluating')
            train_metrics = evaluate(training[:64], f'epoch-{epoch}-train')
            dev_metrics = evaluate(dev, f'epoch-{epoch}-dev')
            controls = {name: evaluate(dev[:16], f'epoch-{epoch}-{name}', control=name)
                        for name in ['zero', 'shuffle']}
            normal_control = evaluate(dev[:16], f'epoch-{epoch}-normal-control')
            row = {'phase':'heldout_pilot', 'epoch':epoch, 'updates':state['updates'],
                   'train':train_metrics,'dev':dev_metrics,'controls':controls,'normal_control':normal_control,
                   'posterior_noise_scale':model.posterior_noise_scale}
            append('metrics.jsonl',row);write(run/f'epoch-{epoch}-metrics.json',row)
            numeric = dev_metrics['numeric_match_rate'] or 0.
            score = (numeric, dev_metrics['exact_match_rate'], dev_metrics['token_accuracy'])
            improved = score > best_score
            if improved: best_score = score
            checkpoint(epoch,'heldout_pilot',best=improved)
            status(evaluation={'train_sample':train_metrics,'dev':dev_metrics},
                   quality='Held-out results available; LLaDA compatibility not established')
            final_metrics = row
        target = (final_metrics['dev']['exact_match_rate'] >= config['dev_exact_target'] and
                  (final_metrics['dev']['numeric_match_rate'] or 0.) >= config['dev_numeric_target'] and
                  final_metrics['dev']['token_accuracy'] >= config['dev_teacher_forced_target'])
        final_hash = decoder_hash(decoder)
        assert final_hash == initial_decoder_hash, 'Frozen decoder unexpectedly changed'
        write(run/'summary.json',{'completed':True,'overfit_passed':True,'quality_target_met':target,
              'best_score':best_score,'final_epoch':final_metrics,'frozen_decoder_unchanged':True,
              'test_used':False,'llada_training_started':False,'elapsed_seconds':time.monotonic()-start,
              'interpretation':'Engineering thresholds only; equation review, stronger-noise testing and latent-use controls required before LLaDA adaptation.'})
        status(stage='completed', quality='Pilot targets met; further validation required' if target else 'Pilot targets unmet; inspect reconstructions and continue debugging')
    except BaseException as error:
        status(stage='failed',error=f'{type(error).__name__}: {error}')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch',action='store_true')
    group.add_argument('--worker',action='store_true')
    group.add_argument('--status',action='store_true')
    group.add_argument('--check',action='store_true')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/vae_small.json')
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.status:monitor()
    elif args.check:
        c=json.loads(args.config.read_text())
        print(json.dumps({'encoder_cached':(ROOT/c['encoder']/'config.json').exists(),
                          'decoder_cached':(ROOT/c['decoder']/'config.json').exists(),
                          'free_disk_gib':shutil.disk_usage(ROOT).free/2**30,
                          'training_started':False},indent=2))
    elif args.launch:launch(args.config)
    else:
        if args.run_dir is None:parser.error('--worker requires --run-dir')
        train(args.run_dir)


if __name__=='__main__':main()
