"""Tiny LLaDA codec gate: known-text reconstruction, not reasoning generation.

Commands: check, launch, worker --directory PATH, finish --directory PATH.
Only queries, mean, logvar and from_latent train. LLaDA and flow modules freeze.
Fixed 64-token masked decoder, 16 deterministic confidence-unmasking iterations.
No source question, target length, support passage or belief is a decoder input.
The encoder intentionally receives supplied text: this is a reconstruction test.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import inspect
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
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

from latent_smoke import write, append
from llada_blocks.model import BranchState, LLaDABlocks, state_from_messages

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/llada-blocks/sciq-pilot-v1'
REVISION = '08b83a6feb34df1a6011b80c3c00c7563e963b07'
SOURCE = Path('/root/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots') / REVISION
STEPS, LENGTH, DECODE_STEPS, LR, BETA = 128, 64, 16, 3e-4, 1e-5
REQUEST = 'Reconstruct the text represented by the latent block. Output only the reconstructed text.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    with Path(path).open() as src:
        return [json.loads(line) for line in src]


def codec_parameter(name):
    return name == 'queries' or name.startswith(('mean.', 'logvar.', 'from_latent.'))


def configure(wrapper):
    for name, p in wrapper.named_parameters():
        p.requires_grad_(codec_parameter(name))
    wrapper.eval()
    return {n: p for n, p in wrapper.named_parameters() if p.requires_grad}


def order(n):
    rng = random.Random(42)
    indices = []
    while len(indices) < STEPS:
        current = list(range(n)); rng.shuffle(current); indices.extend(current)
    return indices[:STEPS]


def inputs():
    manifest = json.loads((DATA / 'manifests/validation.json').read_text())
    assert manifest['status'] == 'passed' and manifest['byte_identical_jsonl_rebuild']
    result = {}
    for split, expected in [('train', 8), ('development', 4)]:
        tiny = DATA / f'processed/tiny/{split}.jsonl'
        codec = DATA / f'processed/codec/{split}.jsonl'
        for path in [tiny, codec]:
            assert sha(path) == manifest['files_sha256'][str(path.relative_to(DATA))], path
        ids = {r['question_id'] for r in records(tiny)}
        assert len(ids) == expected
        selected = [r for r in records(codec) if r['question_id'] in ids]
        assert len(selected) == expected * 2 and len({r['id'] for r in selected}) == len(selected)
        assert set(Counter(r['question_id'] for r in selected).values()) == {2}
        result[split] = selected
    assert not {r['question_id'] for r in result['train']} & {r['question_id'] for r in result['development']}
    return result


def load_runtime():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone = AutoModel.from_pretrained(str(SOURCE), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'})
    assert sha(inspect.getfile(type(backbone))) == sha(SOURCE / 'modeling_llada.py')
    torch.manual_seed(42)  # New modules have an explicit initialization seed.
    wrapper = LLaDABlocks(backbone, latent_dim=128, slots=3, max_blocks=2).to('cuda:0')
    params = configure(wrapper)
    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), trust_remote_code=True, local_files_only=True)
    tokenizer.get_vocab = lru_cache(maxsize=1)(tokenizer.get_vocab)
    template = state_from_messages(tokenizer, [{'role': 'user', 'content': REQUEST}], device='cuda:0')
    return wrapper, tokenizer, template, params


def posterior(wrapper, hidden):
    # Cache frozen trace features, not learned posterior values. Verify numerical
    # equivalence with encode_trace during preflight before any optimizer step.
    scores = torch.einsum('sh,bth->bst', wrapper.queries.to(hidden.dtype), hidden) / math.sqrt(hidden.shape[-1])
    pooled = torch.einsum('bst,bth->bsh', scores.softmax(-1), hidden)
    return wrapper.mean(pooled), wrapper.logvar(pooled).clamp(-20, 10)


def state(template, z):
    return BranchState(template.prompt_ids, z[:, None], template.assistant_ids)


def targets(row, tokenizer, device, length=LENGTH):
    end = tokenizer.convert_tokens_to_ids('<|eot_id|>')
    values = list(row['input_ids']) + [end]
    if not values or len(values) > length:
        raise ValueError(f"{row['id']}: target exceeds fixed decoder length; refusing truncation")
    full = torch.full((1, length), tokenizer.eos_token_id, device=device, dtype=torch.long)
    full[0, :len(values)] = torch.tensor(values, device=device)
    return full, len(values)


def loss(wrapper, template, hidden, target, valid, mask_id, rng=None, full_mask=False):
    mu, lv = posterior(wrapper, hidden)
    z = mu.float()
    if rng is not None:
        z = z + torch.exp(.5 * lv.float()) * torch.randn(mu.shape, device=mu.device, generator=rng)
    if full_mask:
        k = valid
    else:
        # Half the examples have no visible target tokens. The other half reveal
        # a uniformly selected subset. Padding is always masked and never graded.
        k = valid if bool(torch.randint(2, (), device=target.device, generator=rng)) else math.ceil(valid / 2)
    positions = torch.randperm(valid, device=target.device, generator=rng)[:k] if not full_mask else torch.arange(valid, device=target.device)
    corrupted = torch.full_like(target, mask_id)
    if k < valid:
        corrupted[:, :valid] = target[:, :valid]
        corrupted[:, positions] = mask_id
    scores = wrapper.answer_logits(state(template, z), corrupted, positions)
    ce = F.cross_entropy(scores.float().flatten(0, 1), target[0, positions], reduction='sum') / valid / (k / valid)
    kl = -.5 * (1 + lv.float() - mu.float().square() - lv.float().exp()).mean()
    total = ce + BETA * kl
    assert torch.isfinite(total)
    return total, {'ce': ce, 'kl': kl, 'masked': k, 'valid': valid}


def gradient_check(wrapper, params):
    checks = {}
    for group in ['queries', 'mean', 'logvar', 'from_latent']:
        chosen = [p.grad for n, p in params.items() if n == group or n.startswith(group + '.')]
        assert chosen and all(g is not None and torch.isfinite(g).all() for g in chosen), group
        assert any(bool(torch.count_nonzero(g)) for g in chosen), group
        checks[group] = 'finite_nonzero'
    assert all(p.grad is None for n, p in wrapper.named_parameters() if n not in params)
    return checks


@torch.no_grad()
def decode(wrapper, tokenizer, template, z):
    mask_id = wrapper.backbone.config.mask_token_id
    tokens = torch.full((1, LENGTH), mask_id, device=z.device, dtype=torch.long)
    for i in range(DECODE_STEPS):
        remaining = tokens[0].eq(mask_id).nonzero().flatten()
        scores = wrapper.answer_logits(state(template, z), tokens, remaining).float()
        assert torch.isfinite(scores).all()
        scores[..., mask_id] = -torch.inf
        predictions = scores.argmax(-1)
        confidence = scores.softmax(-1).gather(-1, predictions[..., None]).squeeze(-1)
        count = min(LENGTH // DECODE_STEPS + (i < LENGTH % DECODE_STEPS), remaining.numel())
        chosen = confidence[0].topk(count).indices
        tokens[0, remaining[chosen]] = predictions[0, chosen]
    assert not tokens.eq(mask_id).any()
    ids = tokens[0].tolist()
    ends = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')}
    end = next((i for i, x in enumerate(ids) if x in ends), len(ids))
    return {'decoded_text': tokenizer.decode(ids[:end], skip_special_tokens=False).strip(),
        'token_ids': ids, 'stop_position': None if end == len(ids) else end,
        'budget_exhausted': end == len(ids)}


def evaluate(wrapper, tokenizer, template, rows, cache, directory, label, status):
    summary = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        zero = decode(wrapper, tokenizer, template, torch.zeros((1, 3, 128), device='cuda:0'))
        write(directory / f'{label}-zero-latent-control.json', zero)
        for split, items in rows.items():
            results = []
            for i, row in enumerate(items):
                status(stage=label, split=split, completed=i, total=len(items))
                mu, _ = posterior(wrapper, cache[row['id']])
                target, valid = targets(row, tokenizer, mu.device)
                _, metrics = loss(wrapper, template, cache[row['id']], target, valid,
                    wrapper.backbone.config.mask_token_id, full_mask=True)
                output = decode(wrapper, tokenizer, template, mu.float())
                same = lambda a, b: ' '.join(a.split()) == ' '.join(b.split())
                result = {**row, **output, 'split': split,
                    'exact_match': same(output['decoded_text'], row['text']),
                    'zero_latent_exact_match': same(zero['decoded_text'], row['text']),
                    'all_masked_teacher_ce': float(metrics['ce'])}
                append(directory / f'{label}.jsonl', result)
                results.append(result)
            summary[split] = {'exact': sum(r['exact_match'] for r in results), 'total': len(results),
                'zero_latent_exact': sum(r['zero_latent_exact_match'] for r in results),
                'budget_exhausted': sum(r['budget_exhausted'] for r in results),
                'mean_all_masked_ce': sum(r['all_masked_teacher_ce'] for r in results) / len(results),
                'by_kind': {kind: {'exact': sum(r['exact_match'] for r in results if r['kind'] == kind),
                    'total': sum(r['kind'] == kind for r in results)} for kind in ['evidence_excerpt', 'answer_statement']}}
    write(directory / f'{label}-summary.json', summary)
    return summary


def verify_code(directory):
    config = json.loads((directory / 'configuration.json').read_text())
    for path, digest in config['source_sha256'].items():
        assert sha(path) == digest, f'Run source changed: {path}'
    assert sha(directory / 'inputs.json') == config['inputs_sha256'], 'Run inputs changed.'
    return config


def worker(directory):
    config = verify_code(directory)
    status = lambda **kw: write(directory / 'status.json', dict(updated_at_utc=datetime.now(timezone.utc).isoformat(), **kw))
    rows = json.loads((directory / 'inputs.json').read_text())
    status(stage='loading_pretrained_llada')
    wrapper, tokenizer, template, params = load_runtime()
    frozen_versions = {n: p._version for n, p in wrapper.named_parameters() if n not in params}
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    cache = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for split, items in rows.items():
            for i, row in enumerate(items):
                status(stage='caching_frozen_trace_features', split=split, completed=i, total=len(items))
                ids = torch.tensor([row['input_ids']], device='cuda:0')
                cache[row['id']] = wrapper._run(wrapper._embed(ids), None, torch.arange(ids.shape[1], device=ids.device)).detach()
        row = rows['train'][0]
        cached = posterior(wrapper, cache[row['id']])
        direct = wrapper.encode_trace(torch.tensor([row['input_ids']], device='cuda:0'))
        assert all(torch.equal(a, b) for a, b in zip(cached, direct))
    write(directory / 'trainable_parameters.json', {'count': sum(p.numel() for p in params.values()),
        'by_name': {n: p.numel() for n, p in params.items()}, 'backbone_frozen': True,
        'denoiser_frozen': True, 'master_dtype': 'FP32'})
    rng = torch.Generator(device='cuda:0').manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    preflight_row = max(rows['train'], key=lambda r: len(r['input_ids']))
    target, valid = targets(preflight_row, tokenizer, 'cuda:0')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        objective, _ = loss(wrapper, template, cache[preflight_row['id']], target,
            valid, wrapper.backbone.config.mask_token_id, rng, full_mask=True)
    objective.backward()
    preflight = gradient_check(wrapper, params)
    write(directory / 'preflight.json', {'status': 'passed', 'cache_matches_direct_encoder': True,
        'gradient_groups': preflight, 'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'loss': float(objective.detach()), 'optimizer_steps': 0})
    wrapper.zero_grad(set_to_none=True)
    del objective
    baseline = evaluate(wrapper, tokenizer, template, rows, cache, directory, 'baseline', status)
    rng.manual_seed(42)
    optimizer = torch.optim.AdamW(params.values(), lr=LR, weight_decay=0)
    started = time.monotonic()
    for step, index in enumerate(order(len(rows['train'])), 1):
        optimizer.zero_grad(set_to_none=True)
        row = rows['train'][index]
        target, valid = targets(row, tokenizer, 'cuda:0')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            objective, parts = loss(wrapper, template, cache[row['id']], target, valid,
                wrapper.backbone.config.mask_token_id, rng)
        objective.backward()
        gradient_check(wrapper, params)
        grad_norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True)
        optimizer.step()
        append(directory / 'metrics.jsonl', {'step': step, 'total_steps': STEPS, 'id': row['id'],
            'loss': float(objective.detach()), 'ce': float(parts['ce'].detach()), 'kl': float(parts['kl'].detach()),
            'masked_target_tokens': parts['masked'], 'valid_target_tokens': valid,
            'grad_norm_before_clipping': float(grad_norm), 'elapsed_training_seconds': time.monotonic() - started})
        status(stage='training_codec', step=step, total_steps=STEPS)
        if step in (64, 128):
            checkpoint = directory / f'checkpoints/checkpoint-{step}'
            checkpoint.mkdir(parents=True)
            save_file({n: p.detach().cpu().contiguous() for n, p in params.items()}, str(checkpoint / 'codec.safetensors'))
            torch.save({'step': step, 'optimizer': optimizer.state_dict(), 'rng': rng.get_state().cpu()}, checkpoint / 'training_state.pt')
    assert frozen_versions == {n: p._version for n, p in wrapper.named_parameters() if n not in params}
    changes = {n: float((p.detach().cpu() - initial[n]).abs().max()) for n, p in params.items()}
    assert any(v > 0 for v in changes.values())
    write(directory / 'training_verification.json', {'status': 'passed', 'optimizer_steps': STEPS,
        'frozen_parameter_versions_unchanged': True, 'parameter_max_changes': changes})
    final = evaluate(wrapper, tokenizer, template, rows, cache, directory, 'after-training', status)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        mu, lv = posterior(wrapper, cache[rows['train'][0]['id']])
    save_file({'mean': mu.cpu().contiguous(), 'logvar': lv.cpu().contiguous()}, str(directory / 'reload_expected.safetensors'))
    write(directory / 'summary.json', {'baseline': baseline, 'after_training': final,
        'training_steps': STEPS, 'reload_verification': 'pending',
        'scope': 'Known-text reconstruction only. No next-block flow training, question solving, TruthfulQA evaluation or sycophancy claim.',
        'decoder_length': LENGTH, 'decoder_steps': DECODE_STEPS, 'reference_length_supplied_to_decoder': False})
    status(stage='reload_pending')


def finish(directory):
    verify_code(directory)
    write(directory / 'status.json', {'stage': 'fresh_process_checkpoint_reload'})
    wrapper, tokenizer, template, params = load_runtime()
    saved = load_file(str(directory / 'checkpoints/checkpoint-128/codec.safetensors'))
    assert saved.keys() == params.keys()
    result = wrapper.load_state_dict(saved, strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys) == set(wrapper.state_dict()) - set(saved)
    assert all(torch.equal(p.detach().cpu(), saved[n]) for n, p in params.items())
    row = json.loads((directory / 'inputs.json').read_text())['train'][0]
    expected = load_file(str(directory / 'reload_expected.safetensors'))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        mu, lv = wrapper.encode_trace(torch.tensor([row['input_ids']], device='cuda:0'))
        assert torch.equal(mu.cpu(), expected['mean']) and torch.equal(lv.cpu(), expected['logvar'])
        output = decode(wrapper, tokenizer, template, mu.float())
    original = next(r for r in records(directory / 'after-training.jsonl') if r['id'] == row['id'])
    assert output['token_ids'] == original['token_ids']
    write(directory / 'reload_verification.json', {'status': 'passed', 'fresh_process': True,
        'posterior_bitwise_equal': True, 'decoded_tokens_identical': True})
    summary = json.loads((directory / 'summary.json').read_text())
    summary['reload_verification'] = 'passed'
    write(directory / 'summary.json', summary)
    write(directory / 'status.json', {'stage': 'complete', 'step': STEPS,
        'next': 'Report results before any denoiser training or TruthfulQA generation.'})
    print(json.dumps(summary, indent=2), flush=True)


def check():
    rows = inputs()
    assert set(Counter(order(16)).values()) == {8}
    assert codec_parameter('queries') and codec_parameter('mean.weight')
    assert not codec_parameter('backbone.model.transformer.wte.weight')
    assert not codec_parameter('to_velocity.weight') and not codec_parameter('time.0.weight')
    from verify_llada_blocks import tiny
    torch.set_num_threads(4); torch.manual_seed(42)
    wrapper = LLaDABlocks(tiny(), latent_dim=8, slots=3, max_blocks=2)
    params = configure(wrapper)
    template = BranchState(torch.tensor([[1, 2]]), torch.empty(1, 0, 3, 8), torch.tensor([[3]]))
    target = torch.tensor([[10, 11, 93, 93, 93, 93, 93, 93]])
    hidden = torch.randn((1, 4, 32))
    total, _ = loss(wrapper, template, hidden, target, 3, 95, torch.Generator().manual_seed(42))
    total.backward(); gradient_check(wrapper, params)
    return {'status': 'passed', 'train_blocks': len(rows['train']), 'development_blocks': len(rows['development']),
        'presentations_per_training_block': 8, 'cpu_frozen_backbone_backward': 'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'launch', 'worker', 'finish'])
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(check(), indent=2)); return
    if args.command == 'launch':
        report = check()
        directory = ROOT / 'result/llada-blocks/codec-tiny' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory.mkdir(parents=True, exist_ok=False)
        write(directory / 'inputs.json', inputs())
        write(directory / 'data_checks.json', report)
        write(directory / 'configuration.json', {'model': 'GSAI-ML/LLaDA-8B-Instruct', 'revision': REVISION,
            'seed': 42, 'steps': STEPS, 'learning_rate': LR, 'beta_kl': BETA, 'train_questions': 8,
            'development_questions': 4, 'train_blocks': 16, 'development_blocks': 8,
            'decoder_request': REQUEST, 'decoder_length': LENGTH, 'decoder_steps': DECODE_STEPS,
            'decoder_length_policy': 'Fixed for every example; includes masked suffix positions. No target length supplied.',
            'latent_shape': [1, 3, 128], 'max_blocks': 2, 'dtype': 'BF16 frozen backbone, FP32 codec master parameters',
            'trainable': ['queries', 'mean', 'logvar', 'from_latent'], 'backbone_frozen': True, 'flow_modules_frozen': True,
            'reconstruction_target': 'Supplied block text plus end-of-turn; padding always masked and excluded from loss.',
            'corruption': '50% full valid-target masking, 50% uniformly selected ceil(n/2) valid positions; inverse probability weighting.',
            'posterior': 'Sampled in training, mean in evaluation; KL average beta=1e-5.',
            'development_used_for_optimization': False, 'truthfulqa_used': False,
            'data_validation_sha256': sha(DATA / 'manifests/validation.json'),
            'inputs_sha256': sha(directory / 'inputs.json'),
            'source_sha256': {str(p): sha(p) for p in [Path(__file__), ROOT / 'scripts/llada_blocks/model.py',
                ROOT / 'scripts/latent_smoke.py', ROOT / 'scripts/checkpoint_runtime.py',
                SOURCE / 'modeling_llada.py', SOURCE / 'configuration_llada.py']},
            'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'safetensors', 'tokenizers']}})
        (directory / 'source/llada_blocks').mkdir(parents=True)
        shutil.copyfile(__file__, directory / 'source/train_llada_codec.py')
        shutil.copyfile(ROOT / 'scripts/llada_blocks/model.py', directory / 'source/llada_blocks/model.py')
        write(directory / 'status.json', {'stage': 'starting'})
        write(directory.parent / 'latest.json', {'directory': str(directory)})
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (directory / 'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--directory', str(directory)],
                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
        write(directory / 'process.json', {'pid': process.pid})
        print(directory, flush=True); return
    try:
        if args.command == 'worker':
            worker(args.directory)
            os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), 'finish', '--directory', str(args.directory)])
        else:
            finish(args.directory)
    except Exception:
        import traceback
        write(args.directory / 'status.json', {'stage': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
