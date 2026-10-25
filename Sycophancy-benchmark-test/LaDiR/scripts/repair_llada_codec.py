"""One bounded codec diagnosis and optimization-budget trial on frozen LLaDA.

Keep the original 16 training / 8 development blocks and architecture unchanged.
Compare 16 vs 64 greedy unmasking steps on the original 128-step checkpoint.
Known-target-length decoding is a separately labeled oracle diagnostic only.
Resume the original optimizer and RNG through global step 1024, evaluate, reload,
then stop. No flow-denoiser training or TruthfulQA generation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file

import train_llada_codec as base
from latent_smoke import write, append

ROOT = base.ROOT
ORIGINAL = ROOT / 'result/llada-blocks/codec-tiny/20261022T060328Z'
TOTAL = 1024


def training_order(n):
    rng = random.Random(42)
    out = []
    while len(out) < TOTAL:
        part = list(range(n)); rng.shuffle(part); out.extend(part)
    return out[:TOTAL]


def check():
    base.verify_code(ORIGINAL)
    assert json.loads((ORIGINAL / 'status.json').read_text())['stage'] == 'complete'
    inputs = json.loads((ORIGINAL / 'inputs.json').read_text())
    order = training_order(len(inputs['train']))
    assert order[:128] == base.order(16)
    previous = base.records(ORIGINAL / 'metrics.jsonl')
    assert len(previous) == 128
    assert all(r['id'] == inputs['train'][i]['id'] for r, i in zip(previous, order[:128]))
    assert set(Counter(order).values()) == {64}
    state = torch.load(ORIGINAL / 'checkpoints/checkpoint-128/training_state.pt', map_location='cpu', weights_only=True)
    assert state['step'] == 128 and state['rng'].dtype == torch.uint8
    assert state['optimizer']['param_groups'][0]['lr'] == base.LR
    assert all(int(v['step']) == 128 for v in state['optimizer']['state'].values())
    return {'status': 'passed', 'original_order_prefix_exact': True,
        'optimizer_and_private_rng_available': True, 'total_steps': TOTAL,
        'additional_steps': TOTAL - 128, 'presentations_per_training_block_at_finish': 64,
        'train_blocks': 16, 'development_blocks': 8}


def restore(wrapper, params, path):
    saved = load_file(str(path))
    assert saved.keys() == params.keys()
    result = wrapper.load_state_dict(saved, strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys) == set(wrapper.state_dict()) - set(saved)
    assert all(torch.equal(p.detach().cpu(), saved[n]) for n, p in params.items())


def verify(directory):
    config = json.loads((directory / 'configuration.json').read_text())
    for path, expected in config['source_and_input_sha256'].items():
        assert base.sha(path) == expected, f'Frozen source/input changed: {path}'
    return config


@torch.no_grad()
def decode(wrapper, tokenizer, template, z, steps, length=64):
    if not 1 <= steps <= length:
        raise ValueError('Invalid unmasking schedule')
    mask = wrapper.backbone.config.mask_token_id
    tokens = torch.full((1, length), mask, device=z.device, dtype=torch.long)
    ends = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')}
    stop_reveals = []
    for i in range(steps):
        remaining = tokens[0].eq(mask).nonzero().flatten()
        scores = wrapper.answer_logits(base.state(template, z), tokens, remaining).float()
        assert torch.isfinite(scores).all()
        scores[..., mask] = -torch.inf
        predicted = scores.argmax(-1)
        confidence = scores.softmax(-1).gather(-1, predicted[..., None]).squeeze(-1)
        count = length // steps + (i < length % steps)
        chosen = confidence[0].topk(count).indices
        positions, values = remaining[chosen], predicted[0, chosen]
        tokens[0, positions] = values
        for pos, token in zip(positions.tolist(), values.tolist()):
            if token in ends:
                stop_reveals.append({'iteration': i + 1, 'position': pos, 'token_id': token})
    assert not tokens.eq(mask).any()
    ids = tokens[0].tolist()
    end = next((i for i, token in enumerate(ids) if token in ends), length)
    return {'decoded_text': tokenizer.decode(ids[:end], skip_special_tokens=False).strip(),
        'token_ids': ids, 'stop_position': None if end == length else end,
        'budget_exhausted': end == length, 'stop_token_reveals': stop_reveals}


def evaluate(wrapper, tokenizer, template, rows, cache, directory, label, steps, status, oracle=False):
    summary = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for split, items in rows.items():
            evaluated = []
            for i, row in enumerate(items):
                status(stage=label, split=split, completed=i, total=len(items))
                mu, _ = base.posterior(wrapper, cache[row['id']])
                length = len(row['input_ids']) + 1 if oracle else 64
                output = decode(wrapper, tokenizer, template, mu.float(), min(steps, length), length)
                result = {**row, **output, 'split': split,
                    'exact_match': ' '.join(output['decoded_text'].split()) == ' '.join(row['text'].split()),
                    'oracle_known_target_length': oracle, 'decoder_length': length,
                    'decoder_steps': min(steps, length)}
                append(directory / f'{label}.jsonl', result)
                evaluated.append(result)
            summary[split] = {'exact': sum(r['exact_match'] for r in evaluated), 'total': len(evaluated),
                'budget_exhausted': sum(r['budget_exhausted'] for r in evaluated),
                'by_kind': {kind: {'exact': sum(r['exact_match'] for r in evaluated if r['kind'] == kind),
                    'total': sum(r['kind'] == kind for r in evaluated)} for kind in ['evidence_excerpt', 'answer_statement']}}
    write(directory / f'{label}-summary.json', summary)
    return summary


def diagnostic(wrapper, tokenizer, template, rows, cache, directory):
    original = {r['id']: r for r in base.records(ORIGINAL / 'after-training.jsonl')}
    output = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        native_rms = float(wrapper._embed(template.prompt_ids).float().square().mean().sqrt())
        for split, items in rows.items():
            for row in items:
                mu, lv = base.posterior(wrapper, cache[row['id']])
                projected = wrapper.from_latent(mu)
                normalized = torch.nn.functional.normalize(mu.float(), dim=-1)
                cosine = normalized @ normalized.transpose(-1, -2)
                cosine = cosine[:, ~torch.eye(3, device='cuda:0', dtype=torch.bool)].mean()
                target, valid = base.targets(row, tokenizer, 'cuda:0')
                _, parts = base.loss(wrapper, template, cache[row['id']], target, valid,
                    wrapper.backbone.config.mask_token_id, full_mask=True)
                output.append({'id': row['id'], 'split': split, 'kind': row['kind'],
                    'reference_text_tokens': len(row['input_ids']), 'original_stop_position': original[row['id']]['stop_position'],
                    'all_masked_ce': float(parts['ce']), 'mean_rms': float(mu.float().square().mean().sqrt()),
                    'posterior_noise_rms': float(lv.float().exp().mean().sqrt()),
                    'mean_slot_cosine': float(cosine),
                    'projected_embedding_rms': float(projected.float().square().mean().sqrt()),
                    'native_prompt_embedding_rms': native_rms})
        first = rows['train'][0]
        mu, _ = base.posterior(wrapper, cache[first['id']])
        reproduced = decode(wrapper, tokenizer, template, mu.float(), 16)
        assert reproduced['token_ids'] == original[first['id']]['token_ids'], 'Original decode did not reproduce.'
    for r in output:
        append(directory / 'diagnostic.jsonl', r)
    summary = {'original_16_step_decode_reproduced': True,
        'by_kind': {kind: {
            'mean_all_masked_ce': sum(r['all_masked_ce'] for r in output if r['kind'] == kind) / sum(r['kind'] == kind for r in output),
            'mean_slot_cosine': sum(r['mean_slot_cosine'] for r in output if r['kind'] == kind) / sum(r['kind'] == kind for r in output),
            'mean_projected_to_native_embedding_rms_ratio': sum(r['projected_embedding_rms'] / r['native_prompt_embedding_rms'] for r in output if r['kind'] == kind) / sum(r['kind'] == kind for r in output)
        } for kind in ['evidence_excerpt', 'answer_statement']}}
    write(directory / 'diagnostic-summary.json', summary)
    return summary


def worker(directory):
    verify(directory)
    status = lambda **kw: write(directory / 'status.json', dict(updated_at_utc=datetime.now(timezone.utc).isoformat(), **kw))
    rows = json.loads((ORIGINAL / 'inputs.json').read_text())
    status(stage='loading_original_checkpoint')
    wrapper, tokenizer, template, params = base.load_runtime()
    restore(wrapper, params, ORIGINAL / 'checkpoints/checkpoint-128/codec.safetensors')
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    frozen_versions = {n: p._version for n, p in wrapper.named_parameters() if n not in params}
    cache = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for items in rows.values():
            for row in items:
                ids = torch.tensor([row['input_ids']], device='cuda:0')
                cache[row['id']] = wrapper._run(wrapper._embed(ids), None, torch.arange(ids.shape[1], device=ids.device)).detach()
    audit = diagnostic(wrapper, tokenizer, template, rows, cache, directory)
    slow_before = evaluate(wrapper, tokenizer, template, rows, cache, directory, 'before-128-slow64', 64, status)
    oracle = evaluate(wrapper, tokenizer, template, rows, cache, directory, 'oracle-known-length-128', 64, status, oracle=True)
    previous_state = torch.load(ORIGINAL / 'checkpoints/checkpoint-128/training_state.pt', map_location='cpu', weights_only=True)
    optimizer = torch.optim.AdamW(params.values(), lr=base.LR, weight_decay=0)
    optimizer.load_state_dict(previous_state['optimizer'])
    rng = torch.Generator(device='cuda:0'); rng.set_state(previous_state['rng'])
    started = time.monotonic()
    for offset, index in enumerate(training_order(16)[128:], 1):
        step = 128 + offset
        optimizer.zero_grad(set_to_none=True)
        row = rows['train'][index]
        target, valid = base.targets(row, tokenizer, 'cuda:0')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            objective, parts = base.loss(wrapper, template, cache[row['id']], target, valid,
                wrapper.backbone.config.mask_token_id, rng)
        objective.backward()
        base.gradient_check(wrapper, params)
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True)
        optimizer.step()
        append(directory / 'metrics.jsonl', {'step': step, 'total_steps': TOTAL, 'id': row['id'],
            'loss': float(objective.detach()), 'ce': float(parts['ce'].detach()), 'kl': float(parts['kl'].detach()),
            'masked_target_tokens': parts['masked'], 'valid_target_tokens': valid,
            'grad_norm_before_clipping': float(norm), 'elapsed_training_seconds': time.monotonic() - started})
        status(stage='training_codec_only', step=step, total_steps=TOTAL)
        if step in (512, TOTAL):
            cp = directory / f'checkpoints/checkpoint-{step}'; cp.mkdir(parents=True)
            save_file({n: p.detach().cpu().contiguous() for n, p in params.items()}, str(cp / 'codec.safetensors'))
            torch.save({'step': step, 'optimizer': optimizer.state_dict(), 'rng': rng.get_state().cpu()}, cp / 'training_state.pt')
    assert frozen_versions == {n: p._version for n, p in wrapper.named_parameters() if n not in params}
    changes = {n: float((p.detach().cpu() - initial[n]).abs().max()) for n, p in params.items()}
    assert any(v > 0 for v in changes.values())
    write(directory / 'training_verification.json', {'status': 'passed', 'additional_steps': TOTAL - 128,
        'global_step': TOTAL, 'original_optimizer_and_rng_restored': True,
        'frozen_parameter_versions_unchanged': True, 'parameter_max_changes': changes})
    after16 = evaluate(wrapper, tokenizer, template, rows, cache, directory, 'after-1024-standard16', 16, status)
    after64 = evaluate(wrapper, tokenizer, template, rows, cache, directory, 'after-1024-slow64', 64, status)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        mu, lv = base.posterior(wrapper, cache[rows['train'][0]['id']])
    save_file({'mean': mu.cpu().contiguous(), 'logvar': lv.cpu().contiguous()}, str(directory / 'reload_expected.safetensors'))
    write(directory / 'summary.json', {'before_128_standard16': json.loads((ORIGINAL / 'summary.json').read_text())['after_training'],
        'before_128_slow64': slow_before, 'after_1024_standard16': after16, 'after_1024_slow64': after64,
        'oracle_known_length_at_128_diagnostic_only': oracle, 'diagnostic': audit,
        'reload_verification': 'pending', 'global_training_steps': TOTAL, 'additional_training_steps': TOTAL - 128,
        'architecture_and_training_data_unchanged': True,
        'limitations': ['Oracle uses reference length and is not a valid inference setting.',
            'Longer optimization does not establish generalization, reasoning, or reduced sycophancy.',
            'No flow-denoiser training or TruthfulQA generation.']})
    status(stage='reload_pending')


def finish(directory):
    verify(directory)
    write(directory / 'status.json', {'stage': 'fresh_process_reload'})
    wrapper, tokenizer, template, params = base.load_runtime()
    restore(wrapper, params, directory / f'checkpoints/checkpoint-{TOTAL}/codec.safetensors')
    row = json.loads((ORIGINAL / 'inputs.json').read_text())['train'][0]
    expected = load_file(str(directory / 'reload_expected.safetensors'))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        mu, lv = wrapper.encode_trace(torch.tensor([row['input_ids']], device='cuda:0'))
        assert torch.equal(mu.cpu(), expected['mean']) and torch.equal(lv.cpu(), expected['logvar'])
        output = decode(wrapper, tokenizer, template, mu.float(), 16)
    saved = next(r for r in base.records(directory / 'after-1024-standard16.jsonl') if r['id'] == row['id'])
    assert output['token_ids'] == saved['token_ids']
    write(directory / 'reload_verification.json', {'status': 'passed', 'posterior_bitwise_equal': True,
        'decoded_tokens_identical': True, 'fresh_process': True})
    summary = json.loads((directory / 'summary.json').read_text()); summary['reload_verification'] = 'passed'
    write(directory / 'summary.json', summary)
    write(directory / 'status.json', {'stage': 'complete', 'step': TOTAL,
        'next': 'Report codec diagnosis and repair results before further stages.'})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'launch', 'worker', 'finish'])
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(check(), indent=2)); return
    if args.command == 'launch':
        checks = check()
        directory = ROOT / 'result/llada-blocks/codec-repair' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory.mkdir(parents=True, exist_ok=False)
        paths = [Path(__file__), ROOT / 'scripts/train_llada_codec.py', ROOT / 'scripts/llada_blocks/model.py',
            ROOT / 'scripts/latent_smoke.py', ROOT / 'scripts/checkpoint_runtime.py',
            ORIGINAL / 'inputs.json', ORIGINAL / 'configuration.json',
            ORIGINAL / 'checkpoints/checkpoint-128/codec.safetensors', ORIGINAL / 'checkpoints/checkpoint-128/training_state.pt']
        paths += [Path(p) for p in json.loads((ORIGINAL / 'configuration.json').read_text())['source_sha256']]
        write(directory / 'configuration.json', {'original_run': str(ORIGINAL), 'seed': 42,
            'global_step_start': 128, 'global_step_end': TOTAL, 'additional_steps': TOTAL - 128,
            'learning_rate': base.LR, 'beta_kl': base.BETA, 'train_blocks': 16, 'development_blocks': 8,
            'trainable': ['queries', 'mean', 'logvar', 'from_latent'], 'backbone_and_flow_modules_frozen': True,
            'changes': 'Optimization budget only; compare 16 and 64 unmasking iterations separately.',
            'oracle_diagnostic': 'Known reference text length+EOT, for diagnosis only; excluded from valid inference results.',
            'hyperparameters_selected_using_development': False, 'flow_training': False, 'truthfulqa_generation': False,
            'source_and_input_sha256': {str(p): base.sha(p) for p in paths}})
        write(directory / 'checks.json', checks)
        write(directory / 'status.json', {'stage': 'starting'})
        write(directory.parent / 'latest.json', {'directory': str(directory)})
        (directory / 'source').mkdir()
        shutil.copyfile(__file__, directory / 'source/repair_llada_codec.py')
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (directory / 'worker.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--directory', str(directory)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd=ROOT)
        write(directory / 'process.json', {'pid': proc.pid}); print(directory, flush=True); return
    try:
        if args.command == 'worker':
            worker(args.directory)
            os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), 'finish', '--directory', str(args.directory)])
        else:
            finish(args.directory)
    except Exception:
        import traceback
        write(args.directory / 'status.json', {'stage': 'failed', 'traceback': traceback.format_exc()}); raise


if __name__ == '__main__':
    main()
