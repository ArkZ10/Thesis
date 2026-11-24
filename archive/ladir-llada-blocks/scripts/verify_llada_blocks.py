"""Stage 1: mechanical checks for LLaDA latent blocks, without training.

Default: tiny randomly initialized model using the pinned LLaDA source on CPU.
--pretrained: additionally check the real local 8B checkpoint on one GPU.
Neither mode evaluates reasoning quality or uses TruthfulQA reference answers.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from latent_smoke import write
from llada_blocks.model import BranchState, LLaDABlocks, block_bias, state_from_messages

ROOT = Path(__file__).resolve().parents[1]
REVISION = '08b83a6feb34df1a6011b80c3c00c7563e963b07'
SOURCE = Path('/root/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots') / REVISION


def tiny():
    config = AutoConfig.from_pretrained(str(SOURCE), trust_remote_code=True, local_files_only=True)
    for k, v in dict(d_model=32, n_heads=4, n_kv_heads=4, n_layers=2,
                     mlp_hidden_size=64, embedding_size=96, vocab_size=96,
                     max_sequence_length=128, mask_token_id=95, pad_token_id=94,
                     eos_token_id=93, init_device='cpu').items():
        setattr(config, k, v)
    return AutoModel.from_config(config, trust_remote_code=True, init_params=True)


def run_checks(model, directory, actual=False):
    device = next(model.parameters()).device
    wrapper = LLaDABlocks(model, latent_dim=8 if not actual else 128, slots=3, max_blocks=3).to(device)
    wrapper.eval()
    if actual:
        tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), trust_remote_code=True, local_files_only=True)
        messages = [{'role': 'user', 'content': 'What is 2 + 3?'}]
        prepared = state_from_messages(tokenizer, messages, dim=wrapper.dim, device=device)
        prompt, assistant = prepared.prompt_ids, prepared.assistant_ids
        joined = torch.cat((prompt, assistant), 1)
        expected = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_tensors='pt').to(device)
        assert torch.equal(joined, expected), 'Separate assistant suffix changed tokenization.'
        trace = tokenizer('2 + 3 = 5.', add_special_tokens=False, return_tensors='pt').input_ids.to(device)
        answer = tokenizer('5', add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    else:
        prompt = torch.tensor([[1, 2, 3, 4]], device=device)
        assistant = torch.tensor([[5, 6]], device=device)
        trace = torch.tensor([[7, 8, 9, 10]], device=device)
        answer = torch.tensor([[11, 12, 13]], device=device)
    state = BranchState(prompt, torch.empty((1, 0, 3, wrapper.dim), device=device), assistant)
    versions = {n: p._version for n, p in model.named_parameters()}

    # First establish no-block output equivalence to the native LLaDA forward.
    native_input = torch.cat((prompt, assistant, answer), 1)
    with torch.no_grad():
        native = model(native_input, use_cache=False).logits[:, -answer.shape[1]:]
        positions = torch.arange(native_input.shape[1] - answer.shape[1], native_input.shape[1], device=device)
        sparse = wrapper._run(wrapper._embed(native_input), None, positions, logits=True)
    projection_delta = float((native.float() - sparse.float()).abs().max())
    # Selecting rows changes the GPU GEMM shape. BF16 may round differently;
    # allow approximately one BF16 unit while keeping the CPU check exact.
    if actual:
        torch.testing.assert_close(native.float(), sparse.float(), rtol=.008, atol=.008)
    else:
        assert torch.equal(native, sparse), 'Sparse projection altered native baseline logits.'

    # Backpropagate through codec and frozen LLaDA: no optimizer is constructed.
    mu, logvar = wrapper.encode_trace(trace)
    known = BranchState(prompt, mu[:, None], assistant)
    mask = torch.ones_like(answer, dtype=torch.bool)
    answer_loss = wrapper.masked_answer_loss(known, answer, mask, model.config.mask_token_id, 1.)
    noise = torch.randn_like(mu)
    flow_loss = wrapper.flow_loss(state, mu.detach(), noise, torch.tensor(.5, device=device))
    kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    loss = answer_loss + flow_loss + .001 * kl
    assert torch.isfinite(loss)
    loss.backward()
    gradient_groups = {}
    for name in ['queries', 'mean', 'logvar', 'from_latent', 'to_velocity', 'time', 'block_index']:
        ps = [wrapper.queries] if name == 'queries' else list(getattr(wrapper, name).parameters())
        gs = [p.grad for p in ps if p.grad is not None]
        assert gs and all(torch.isfinite(g).all() for g in gs), name
        assert any(bool(torch.count_nonzero(g)) for g in gs), name
        gradient_groups[name] = 'finite_nonzero'
    assert all(p.grad is None for p in model.parameters())
    wrapper.zero_grad(set_to_none=True)

    with torch.no_grad():
        known = BranchState(prompt, mu.detach()[:, None], assistant)
        # Same fixed tokens, different continuous block: answer predictions respond.
        changed = BranchState(prompt, torch.zeros_like(known.prefix), assistant)
        first = wrapper.answer_logits(known, answer.masked_fill(mask, model.config.mask_token_id))
        second = wrapper.answer_logits(changed, answer.masked_fill(mask, model.config.mask_token_id))
        delta = float((first.float() - second.float()).abs().max())
        assert delta > 0, 'Answer prediction is insensitive to the latent block.'

        # Actual hidden-state checks supplement inspection of the attention matrix.
        context = wrapper._context(known)
        tail = torch.cat((wrapper._embed(assistant), wrapper._embed(answer)), 1)
        bias = block_bias(prompt.shape[1], 1, 3, tail.shape[1], device)
        positions = torch.arange(context.shape[1], device=device)
        a = wrapper._run(torch.cat((context, tail), 1), bias, positions)
        b = wrapper._run(torch.cat((context, -tail), 1), bias, positions)
        assert torch.equal(a, b), 'Answer region leaked into prompt or completed block.'
        q = prompt.shape[1]
        assert not bool(bias[0, 0, :q, q:].eq(0).any())
        assert bool(bias[0, 0, q:q+3, :q+3].eq(0).all())

        parent_prefix = known.prefix.clone()
        original_prompt = known.prompt_ids.clone()
        children = [wrapper.sample_next(known, seed=s, steps=2) for s in [11, 22, 33, 44]]
        assert torch.equal(known.prefix, parent_prefix) and torch.equal(known.prompt_ids, original_prompt)
        for child in children:
            assert child.prefix.shape[1] == 2
            assert torch.equal(child.prefix[:, :1], parent_prefix)
            assert child.prefix.data_ptr() != known.prefix.data_ptr()
            assert torch.equal(child.prompt_ids, original_prompt)
            assert torch.equal(child.assistant_ids, assistant)
        for i in range(4):
            for j in range(i):
                assert not torch.equal(children[i].prefix[:, -1], children[j].prefix[:, -1])
        same = wrapper.sample_next(known, seed=11, steps=2)
        assert torch.equal(same.prefix, children[0].prefix), 'Same-seed sampling is not reproducible.'
        known.save(directory / 'parent-state.safetensors')
        reloaded = BranchState.load(directory / 'parent-state.safetensors', device)
        restored = wrapper.sample_next(reloaded, seed=11, steps=2)
        assert torch.equal(restored.prefix, children[0].prefix), 'Reload changed continuation.'
        # Changes to one child cannot mutate its siblings or saved parent.
        sibling = children[1].prefix.clone()
        children[0].prefix.add_(1)
        assert torch.equal(children[1].prefix, sibling) and torch.equal(known.prefix, parent_prefix)

    assert versions == {n: p._version for n, p in model.named_parameters()}
    assert len(model.model.transformer.ff_out._forward_pre_hooks) == 0
    return {'status': 'passed', 'actual_pretrained_8b': actual,
        'native_logits_match_sparse_projection': True,
        'sparse_projection_max_absolute_error': projection_delta,
        'sparse_projection_bitwise_equal': torch.equal(native, sparse),
        'sparse_projection_tolerance': {'rtol': .008, 'atol': .008} if actual else 'exact',
        'gradient_groups': gradient_groups, 'frozen_backbone_gradient_free_and_unchanged': True,
        'answer_logit_max_change_with_latent_ablation': delta,
        'answer_to_prefix_leakage_blocked': True, 'shared_prefix_preserved': True,
        'four_numerically_distinct_next_blocks': True, 'same_seed_reproducible': True,
        'saved_state_continuation_identical': True, 'siblings_independent': True,
        'adapter_parameter_count': sum(p.numel() for p in wrapper.parameters() if p.requires_grad),
        'optimizer_steps': 0, 'reasoning_quality_tested': False, 'truthfulqa_evaluated': False,
        'limitation': 'Randomly initialized new modules. Numerical branching and gradient flow do not establish useful reasoning.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    directory = ROOT / 'result/llada-blocks/interface' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True, exist_ok=False)
    write(directory.parent / 'latest.json', {'directory': str(directory)})
    write(directory / 'status.json', {'stage': 'checking'})
    write(directory / 'configuration.json', {'model': 'GSAI-ML/LLaDA-8B-Instruct', 'revision': REVISION,
        'stage': 'Architecture-interface checks only', 'published_ladir_weights_used': False,
        'pretrained_check_requested': args.pretrained, 'training_started': False,
        'adaptation': 'Learned trace codec; conditional flow matching for next continuous block; LLaDA masked answer decoder.',
        'layout': 'user chat prefix / completed latent blocks / assistant suffix / masked answer',
        'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'safetensors']},
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), ROOT / 'scripts/llada_blocks/model.py',
            SOURCE / 'modeling_llada.py', SOURCE / 'configuration_llada.py']}})
    try:
        write(directory / 'cpu.json', run_checks(tiny(), directory))
        if args.pretrained:
            if not torch.cuda.is_available():
                raise RuntimeError('Pretrained verification requires one available GPU.')
            write(directory / 'status.json', {'stage': 'loading_pretrained_8b'})
            torch.cuda.set_device(0)
            torch.backends.cuda.matmul.allow_tf32 = False
            model = AutoModel.from_pretrained(str(SOURCE), trust_remote_code=True,
                local_files_only=True, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'})
            gpu_directory = directory / 'pretrained'
            gpu_directory.mkdir()
            write(directory / 'status.json', {'stage': 'checking_pretrained_8b'})
            with torch.autocast('cuda', dtype=torch.bfloat16):
                report = run_checks(model, gpu_directory, actual=True)
            report['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
            write(directory / 'pretrained.json', report)
        write(directory / 'status.json', {'stage': 'complete', 'training_started': False})
        print(directory, flush=True)
    except Exception:
        import traceback
        write(directory / 'status.json', {'stage': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
