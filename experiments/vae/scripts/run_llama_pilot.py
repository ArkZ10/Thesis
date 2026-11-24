"""Train the configured LLaMA VAE with epoch metrics and linked checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.deepspeed import HfDeepSpeedConfig, unset_hf_deepspeed_config
from vae.llama_offload import LlamaLatentEncoder
from vae.linked_checkpoint import atomic_json, capture_rng, install_storage_policy, save_linked_checkpoint
from vae.attention import build_mask
from vae.pilot_metrics import reconstruction_values
from vae.pilot_metrics import normalized
from vae.indexed_math_data import IndexedMathData
from vae.adaptive_batches import physical_batches, padded_batch
from vae.efficient_decoder import checkpoint_frozen_decoder, reconstruction_loss, decode_prefix
from vae.runtime import decoder_hash

ROOT = Path(__file__).resolve().parents[1]


def decoder_metrics(decoder, prefix, ids, mask):
    device = decoder.get_input_embeddings().weight.device
    ids, mask = ids.to(device), mask.to(device)
    text = decoder.get_input_embeddings()(ids)
    prefix = prefix.to(device=text.device, dtype=text.dtype)
    embeds = torch.cat((prefix, text), dim=1)
    keys = torch.cat((mask.new_ones(ids.shape[0], prefix.shape[1]), mask), dim=1)
    attention = build_mask('hybrid', keys, 0, prefix.shape[1] - 1, embeds.dtype)
    hidden = decoder.model(inputs_embeds=embeds, attention_mask=attention,
                           use_cache=False, return_dict=True).last_hidden_state
    offset = prefix.shape[1] - 1
    logits = decoder.lm_head(hidden[:, offset:offset + ids.shape[1]])
    labels = ids.masked_fill(mask == 0, -100)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
    with torch.no_grad():
        correct = int(((logits.argmax(-1) == ids) & mask.bool()).sum())
    return ce, correct, int(mask.sum())




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    settings = json.loads((run / 'pilot-config.json').read_text())
    work = run / 'work'
    work.mkdir(exist_ok=False)
    start = time.monotonic()
    full_corpus = settings.get('full_corpus_training', False)
    execution = settings.get('execution', {})
    adaptive = bool(execution.get('adaptive_batches', False))
    encoder_gpu, decoder_gpu = execution.get('encoder_gpu', 1), execution.get('decoder_gpu', 0)
    if {encoder_gpu, decoder_gpu} != {0, 1}:
        raise ValueError('Encoder and decoder must use the two distinct GPUs')
    encoder_device, decoder_device = f'cuda:{encoder_gpu}', f'cuda:{decoder_gpu}'
    state = {'pilot_training': not full_corpus, 'full_corpus_training_started': full_corpus,
             'epochs': settings['epochs'], 'epoch': 0, 'global_steps': 0,
             'encoder_gpu': encoder_gpu, 'decoder_gpu': decoder_gpu}
    def status(stage, **values):
        state.update(stage=stage, **values, elapsed_seconds=time.monotonic() - start,
                     updated_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(run / 'status.json', state)
        if stage != 'training' or values.get('batch', 0) % settings.get('log_every_microbatches', 1) == 0:
            print(json.dumps(state), flush=True)
    try:
        os.environ.update(LOCAL_RANK=str(encoder_gpu), RANK='0', WORLD_SIZE='1')
        torch.cuda.set_device(encoder_gpu)
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        sampler = torch.Generator().manual_seed(12345)
        dist.init_process_group('nccl', init_method=f'file://{work / "rendezvous"}', rank=0, world_size=1)
        data = Path(settings['data_directory'])
        data = data if data.is_absolute() else ROOT / data
        summary = json.loads((data / 'summary.json').read_text())
        for name, digest in summary['files'].items():
            checksum = hashlib.sha256()
            with (data / name).open('rb') as stream:
                for block in iter(lambda: stream.read(16 * 1024**2), b''):
                    checksum.update(block)
            assert checksum.hexdigest() == digest
        if summary.get('format') == 'llama_token_memmap_v1':
            source = Path(summary['training_source_jsonl'])
            h = hashlib.sha256()
            with source.open('rb') as stream:
                for block in iter(lambda: stream.read(16 * 1024**2), b''):
                    h.update(block)
            assert h.hexdigest() == summary['training_source_sha256']
            train = IndexedMathData(data, source)
        else:
            train = [json.loads(line) for line in (data / 'train.jsonl').open()]
        dev = [json.loads(line) for line in (data / 'dev.jsonl').open()]
        accumulation = settings['effective_batch_size']
        padded_training_count = math.ceil(len(train) / accumulation) * accumulation
        steps_per_epoch = padded_training_count // accumulation
        config_path = execution['deepspeed_config']
        config = json.loads((ROOT / config_path).read_text())
        nominal_micro = execution.get('max_physical_examples', 32) if adaptive else 1
        if accumulation % nominal_micro:
            raise ValueError('Nominal microbatch must divide effective batch')
        config.update(train_micro_batch_size_per_gpu=nominal_micro,
                      gradient_accumulation_steps=accumulation // nominal_micro,
                      train_batch_size=accumulation)
        for component in ['offload_param', 'offload_optimizer']:
            config['zero_optimization'][component]['nvme_path'] = str(work / 'swap')
        atomic_json(run / 'deepspeed.json', config)
        status('loading_models', train_sentences=len(train), dev_sentences=len(dev),
               batches_per_epoch=padded_training_count, optimizer_steps_per_epoch=steps_per_epoch,
               final_batch_repeated_examples=padded_training_count - len(train),
               total_optimizer_steps=steps_per_epoch * settings['epochs'])
        hf_config = HfDeepSpeedConfig(config)
        encoder = AutoModel.from_pretrained(ROOT / 'models/Llama-3.1-8B', dtype=torch.bfloat16,
                    attn_implementation='sdpa', local_files_only=True)
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        unset_hf_deepspeed_config(); del hf_config
        module = LlamaLatentEncoder(encoder, num_latents=settings.get('num_latents', 6),
                                   latent_dim=settings.get('latent_dim', 512))
        decoder = AutoModelForCausalLM.from_pretrained(ROOT / 'models/Llama-3.1-8B',
                    dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': decoder_device},
                    local_files_only=True)
        decoder.requires_grad_(False); decoder.eval()
        if execution.get('decoder_checkpointing', False):
            checkpoint_frozen_decoder(decoder)
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Llama-3.1-8B', local_files_only=True)
        status('hashing_frozen_decoder')
        frozen_hash = decoder_hash(decoder)
        body = list(module.encoder.parameters()); body_ids = {id(p) for p in body}
        heads = [p for p in module.parameters() if id(p) not in body_ids]
        status('initializing_optimizer')
        engine, _, _, _ = deepspeed.initialize(model=module,
            model_parameters=[{'params': body}, {'params': heads}], config=config)
        policy = install_storage_policy(engine, run)
        atomic_json(run / 'model-setup.json', {'trainable_parameters': sum(int(p.ds_numel) for p in module.parameters()),
                    'frozen_decoder_sha256': frozen_hash, 'storage_policy': policy,
                    'precision': 'BF16 live weights/gradient accumulation; FP32 Adam masters/moments'})
        def batch(row):
            seq = row['input_ids']
            configured = settings.get('max_tokens', 128)
            if configured is not None and len(seq) > configured:
                raise RuntimeError('Training target exceeds configured length; never truncate silently')
            length = configured or len(seq)
            ids = torch.tensor([seq + [tokenizer.eos_token_id] * (length - len(seq))], device=encoder_device)
            mask = torch.tensor([[1] * len(seq) + [0] * (length - len(seq))], device=encoder_device)
            return ids, mask
        def totals():
            return {'ce_sum': 0., 'kl_sum': 0., 'tokens': 0, 'correct': 0, 'examples': 0}
        def add(t, ce, kl, correct, tokens):
            t['ce_sum'] += float(ce.detach()) * tokens; t['kl_sum'] += float(kl.detach())
            t['tokens'] += tokens; t['correct'] += correct; t['examples'] += 1
        def values(t):
            ce = t['ce_sum'] / max(1, t['tokens']); kl = t['kl_sum'] / max(1, t['examples'])
            return {**t, 'ce_loss': ce, 'kl_loss': kl, 'loss': ce + 1e-5 * kl,
                    'token_accuracy': t['correct'] / max(1, t['tokens'])}
        epoch_metrics = []
        training_seconds = 0.
        for epoch in range(1, settings['epochs'] + 1):
            engine.train()
            order = torch.randperm(len(train), generator=sampler).tolist()
            # Cover every target. The last incomplete batch repeats at most
            # 127 earlier targets; this local choice is recorded in the status.
            order += order[:padded_training_count - len(order)]
            train_totals = totals()
            epoch_start = time.monotonic()
            if adaptive:
                cursor = 0
                for first in range(0, len(order), accumulation):
                    chosen = [train[index] for index in order[first:first + accumulation]]
                    assert len(chosen) == accumulation
                    batches = list(physical_batches(chosen,
                        max_examples=execution.get('max_physical_examples', 32),
                        max_tokens=execution.get('physical_token_budget', 4096)))
                    for physical_index, rows in enumerate(batches):
                        boundary = physical_index + 1 == len(batches)
                        engine.set_gradient_accumulation_boundary(boundary)
                        ids, mask = padded_batch(rows, tokenizer.eos_token_id, encoder_device, settings.get('max_tokens'))
                        prefix, kl, mu, logvar = engine(ids, mask, tokenizer.eos_token_id, tokenizer.bos_token_id)
                        ce, measured = reconstruction_loss(decoder, prefix, ids, mask,
                            chunk_tokens=execution.get('loss_chunk_tokens', 128))
                        loss = (ce.to(encoder_device) + 1e-5 * kl) * (
                            engine.gradient_accumulation_steps() * len(rows) / accumulation)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss')
                        train_totals['ce_sum'] += measured['ce_sum']
                        train_totals['kl_sum'] += float(kl.detach()) * len(rows)
                        train_totals['tokens'] += measured['tokens']
                        train_totals['correct'] += measured['correct']
                        train_totals['examples'] += len(rows)
                        engine.backward(loss)
                        skipped_before = engine.skipped_steps
                        engine.step()
                        if boundary and (engine.skipped_steps != skipped_before or not torch.isfinite(
                                torch.tensor(engine.get_global_grad_norm()))):
                            raise RuntimeError('Nonfinite or skipped adaptive optimizer update')
                        cursor += len(rows)
                        processed = (epoch - 1) * padded_training_count + cursor
                        elapsed = training_seconds + time.monotonic() - epoch_start
                        rate = processed / max(1e-6, elapsed)
                        status('training', epoch=epoch, batch=cursor, progress_unit='examples',
                            physical_batch_examples=len(rows), physical_batch_tokens=int(mask.numel()),
                            batches_per_epoch=padded_training_count,
                            epoch_progress_percent=100 * cursor / padded_training_count,
                            global_steps=engine.global_steps,
                            training_progress_percent=100 * processed / (padded_training_count * settings['epochs']),
                            examples_per_training_second=rate,
                            remaining_training_seconds=(padded_training_count * settings['epochs'] - processed) / rate,
                            eta_excludes_evaluation_and_checkpoints=True,
                            train_token_accuracy=train_totals['correct'] / train_totals['tokens'],
                            latest_ce=float(ce.detach()), latest_kl=float(kl.detach()),
                            posterior_mu_mean=float(mu), posterior_logvar_mean=float(logvar))
                        del prefix, kl, ce, loss, ids, mask
            else:
                for cursor, index in enumerate(order, 1):
                    ids, mask = batch(train[index])
                    prefix, kl, mu, logvar = engine(ids, mask, tokenizer.eos_token_id, tokenizer.bos_token_id)
                    ce, correct, tokens = decoder_metrics(decoder, prefix, ids, mask)
                    loss = ce.to(encoder_device) + 1e-5 * kl
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    add(train_totals, ce, kl, correct, tokens)
                    engine.backward(loss); engine.step()
                    elapsed = time.monotonic() - epoch_start
                    processed = (epoch - 1) * padded_training_count + cursor
                    rate = processed / max(1e-6, training_seconds + elapsed)
                    status('training', epoch=epoch, batch=cursor, batches_per_epoch=padded_training_count,
                        epoch_progress_percent=100 * cursor / padded_training_count, global_steps=engine.global_steps,
                        training_progress_percent=100 * processed / (padded_training_count * settings['epochs']),
                        examples_per_training_second=rate,
                        remaining_training_seconds=(padded_training_count * settings['epochs'] - processed) / rate,
                        eta_excludes_evaluation_and_checkpoints=True,
                        train_token_accuracy=train_totals['correct'] / train_totals['tokens'],
                        latest_ce=float(ce.detach()), latest_kl=float(kl.detach()),
                        posterior_mu_mean=float(mu), posterior_logvar_mean=float(logvar))
                    del prefix, ce, loss, kl, ids, mask
            training_seconds += time.monotonic() - epoch_start
            engine.eval(); dev_totals = totals(); prefixes = []
            with torch.no_grad():
                for index, row in enumerate(dev, 1):
                    ids, mask = batch(row)
                    prefix, kl, _, _ = engine(ids, mask, tokenizer.eos_token_id, tokenizer.bos_token_id)
                    ce, correct, tokens = decoder_metrics(decoder, prefix, ids, mask)
                    add(dev_totals, ce, kl, correct, tokens)
                    prefixes.append(prefix.detach().to(decoder_device))
                    status('evaluating_teacher_forcing', evaluation_batch=index, evaluation_total=len(dev))
                    del ids, mask, prefix, ce, kl
            prefixes = torch.cat(prefixes)
            records = []
            for first in range(0, len(dev), 4):
                decoded = decode_prefix(decoder, prefixes[first:first + 4], tokenizer)
                for row, pred in zip(dev[first:first + 4], decoded):
                    records.append({'parent_id': row['parent_id'], 'problem_id': row['problem_id'],
                        'sentence_index': row['sentence_index'], 'dataset': row['dataset'],
                        'target': row['reasoning_block'], **pred,
                        'exact_match': normalized(pred['text']) == normalized(row['reasoning_block'])})
                status('evaluating_free_reconstruction', evaluation_batch=len(records), evaluation_total=len(dev))
            with (run / f'epoch-{epoch}-dev-reconstructions.jsonl').open('w') as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            # Keep AE token fixed; explicitly control projected decoder-prefix vectors.
            controls = {}
            for mode in ['zero_projected_prefix', 'shuffled_projected_prefix']:
                controlled = prefixes[:16].clone()
                controlled[:, :-1] = 0 if mode.startswith('zero') else controlled[:, :-1].roll(1, dims=0)
                predictions = []
                for first in range(0, 16, 4):
                    predictions.extend(decode_prefix(decoder, controlled[first:first + 4], tokenizer))
                control_records = [{'target': row['reasoning_block'], **pred,
                    'exact_match': normalized(pred['text']) == normalized(row['reasoning_block'])}
                    for row, pred in zip(dev[:16], predictions)]
                controls[mode] = reconstruction_values(control_records)
                atomic_json(run / f'epoch-{epoch}-{mode}.json', control_records)
                status('evaluating_controls', control=mode)
            del prefixes, controlled
            metrics = {'epoch': epoch, 'global_steps': engine.global_steps,
                'train_teacher_forced': values(train_totals), 'dev_teacher_forced': values(dev_totals),
                'dev_free_reconstruction': reconstruction_values(records), 'projected_prefix_controls': controls,
                'normal_control_first_16': reconstruction_values(records[:16]),
                'training_seconds_so_far': training_seconds,
                'frozen_decoder_unchanged': decoder_hash(decoder) == frozen_hash}
            if not metrics['frozen_decoder_unchanged']:
                raise RuntimeError('Frozen decoder changed')
            epoch_metrics.append(metrics)
            atomic_json(run / f'epoch-{epoch}-metrics.json', metrics)
            with (run / 'epoch-metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            engine.train()
            status('saving_checkpoint', epoch=epoch, global_steps=engine.global_steps,
                   dev_token_accuracy=metrics['dev_teacher_forced']['token_accuracy'],
                   dev_exact_count=metrics['dev_free_reconstruction']['exact_count'], dev_total=len(dev))
            checkpoint = save_linked_checkpoint(engine, run / 'checkpoints', f'step_{engine.global_steps:06d}',
                {'rng': capture_rng(sampler), 'completed_epoch': epoch, 'cursor': len(order), 'order': order,
                 'engine_micro_steps': engine.micro_steps, 'settings': settings,
                 'data_summary_sha256': hashlib.sha256((data / 'summary.json').read_bytes()).hexdigest(),
                 'epoch_metrics': epoch_metrics, 'frozen_decoder_sha256': frozen_hash})
            atomic_json(run / 'latest-checkpoint.json', {'path': str(checkpoint), 'epoch': epoch,
                                                       'global_steps': engine.global_steps, 'resumable': True})
            print('EPOCH_RESULT ' + json.dumps(metrics), flush=True)
        atomic_json(run / 'summary.json', {'completed': True, 'epochs': epoch_metrics,
            'initialization': 'fresh pretrained LLaMA-3.1-8B', 'test_used': False,
            'full_corpus_training_started': full_corpus, 'latest_checkpoint': str(checkpoint),
            'storage_policy': policy, 'total_seconds': time.monotonic() - start})
        status('completed', epoch=settings['epochs'], global_steps=engine.global_steps,
               epoch_progress_percent=100., training_progress_percent=100., latest_checkpoint=str(checkpoint))
    except BaseException as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc)[:1500])
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
