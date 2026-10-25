"""Stage-1 verification only: CPU default; --gpu checks real LLaDA forwards.

No optimizer, answer generation, candidate experiment, or accuracy claims.
Reports go to result/selective-memory/<timestamp>/. Existing runs are preserved.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import NamedTuple

import torch

from phase3_pilot import ROOT, MODULES, dependencies, write, sha
from candidate_trajectory.engine import Engine, clone, save_snapshot, load_snapshot
from candidate_trajectory.selective_memory import (
    ACTIONS, slot_mask, select_memory_update, forward_with_memory_action,
    fork_memory_branch,
)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def check_selection(previous, proposed):
    for action in ACTIONS:
        mask = slot_mask(previous, action).expand_as(previous)
        got = select_memory_update(previous, proposed, action)
        assert torch.equal(got[mask], proposed[mask])
        assert torch.equal(got[~mask], previous[~mask])
        assert got.shape == previous.shape and got.dtype == previous.dtype


def cpu_checks():
    torch.manual_seed(42)
    for dtype in (torch.float32, torch.bfloat16):
        previous = torch.randn(2, 64, 1024, dtype=dtype)
        proposed = torch.randn_like(previous)
        before = previous.clone()
        check_selection(previous, proposed)
        assert torch.equal(before, previous)
    # Gradients select the proposed write for selected slots, and the residual
    # previous state for preserved slots. No detach or in-place mutation.
    previous = torch.randn(1, 64, 4, requires_grad=True)
    proposed = torch.randn_like(previous, requires_grad=True)
    select_memory_update(previous, proposed, 'B').sum().backward()
    mask = slot_mask(previous, 'B').expand_as(previous)
    assert torch.equal(proposed.grad, mask.float())
    assert torch.equal(previous.grad, (~mask).float())
    # Exercise the actual vendor GRU updater at 64x1024 with small conditioning.
    source = ROOT/'vendor/MetaState/dllm/core/modules/recurrent.py'
    spec = importlib.util.spec_from_file_location('selective_check_vendor_recurrent', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = module.RecurrentModuleConfig(state_hidden_size=1024, num_memory_slots=64,
                                      time_cond_dim=8, mixer_dim=8, injector_dim=8)
    updater = module.RecurrentUpdater(cfg).eval()
    previous = torch.randn(1, 64, 1024)
    with torch.no_grad():
        proposed = updater(previous, torch.randn_like(previous), torch.randn(1, 8))
    check_selection(previous, proposed)
    class Output(NamedTuple):
        new_state: torch.Tensor
        logits: torch.Tensor
        past_cond: torch.Tensor
    class Fake:
        def __init__(self):
            self.calls = 0
            self.output = Output(proposed, torch.randn(1, 2, 3), torch.randn(1, 8))
        def __call__(self, **kwargs):
            self.calls += 1
            return self.output
    fake = Fake()
    assert forward_with_memory_action(fake, state=previous, action='all') is fake.output
    got = forward_with_memory_action(fake, state=previous, action='C')
    assert got.logits is fake.output.logits and got.past_cond is fake.output.past_cond
    assert got.new_state is not fake.output.new_state
    assert torch.equal(fake.output.new_state, proposed)
    for kwargs in ({'action':'unknown'}, {'action':'A', 'state':None},
                   {'action':'A', 'state':torch.zeros(1, 32, 4)},
                   {'action':'all', 'use_cache':True}, {'return_dict':False}):
        calls = fake.calls
        try:
            forward_with_memory_action(fake, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid action/shape/bootstrap/cache was accepted')
        assert fake.calls == calls
    parent = {'memory':previous, 'conditioning':torch.randn(1, 8),
              'tokens':torch.tensor([[1, 2, 3]]), 'token_version':2,
              'rng':{'torch_cpu':torch.get_rng_state()}, 'completed_steps':2,
              'quotas':torch.tensor([[1, 1]]), 'configuration':{'temperature':0},
              'remaining_decoding_iterations':30, 'trajectory_id':'parent'}
    # Verify the actual sampler Engine reads the opt-in action from a snapshot.
    for action in ACTIONS:
        engine = Engine.__new__(Engine)
        engine.model = fake
        engine.device = torch.device('cpu')
        engine.calls = 0
        engine.max_calls = 10
        engine.config = {'generation_length':8}
        branch = fork_memory_branch(parent, action, 'engine-check-'+action)
        branch['trajectory_calls'] = 0
        logits, _ = engine._forward(branch, bootstrap=True)
        assert torch.equal(branch['memory'],select_memory_update(previous,proposed,action))
        assert torch.equal(branch['conditioning'],fake.output.past_cond)
        assert torch.equal(logits,fake.output.logits)
        assert branch['trajectory_calls'] == 1 and engine.calls == 1
    original = clone(parent)
    a = fork_memory_branch(parent, 'A', 'child-A')
    b = fork_memory_branch(parent, 'B', 'child-B')
    for key in parent:
        if key != 'trajectory_id':
            assert equal_tree(parent[key], a[key]) and equal_tree(parent[key], b[key])
    a['memory'].add_(1); a['conditioning'].zero_(); a['tokens'].zero_()
    a['rng']['torch_cpu'].zero_(); a['configuration']['temperature'] = 1
    assert equal_tree(parent, original)
    for key in parent:
        if key != 'trajectory_id':
            assert equal_tree(b[key], original[key])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'branch.pt'
        save_snapshot(path, b)
        assert equal_tree(b, load_snapshot(path, 'cpu'))
    return {'status':'passed', 'checks':[
        'FP32/BF16 exact selected writes and preserved slots at 64x1024',
        'real vendor updater proposals', 'differentiable selection',
        'all returns native output', 'conditioning/logits are unmodified',
        'invalid actions/bootstrap/shapes/cache fail before model call',
        'complete branch clone isolation including RNG and conditioning',
        'sampler Engine applies opt-in action from continuation state',
        'branch action survives snapshot roundtrip'], 'vendor_updater_sha256':sha(source)}


def gpu_checks(directory):
    torch, _ = dependencies()
    from dllm.utils.models import get_model, get_tokenizer
    run = Path(json.loads((ROOT/'result/reasoning-pilot/latest.json').read_text())['directory'])
    checkpoint = run/'checkpoints/checkpoint-128'
    args = SimpleNamespace(model_name_or_path=str(checkpoint),
        backbone_name_or_path=json.loads((ROOT/'setup/backbone.json').read_text())['model_name_or_path'],
        dtype='bfloat16', attn_implementation=None, freeze_backbone=True)
    model = get_model(args, is_recurrent=True, training=False).eval()
    t = get_tokenizer(args)
    messages = [{'role':'user', 'content':'This input is for a memory-update verification.'}]
    prefix = t.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = torch.tensor([prefix+[t.mask_token_id]*8], device='cuda')
    kwargs = {'input_ids':ids, 'attention_mask':torch.ones_like(ids),
              't':torch.ones(1, device='cuda'), 'use_cache':False, 'return_dict':True}
    versions = {n:p._version for n,p in model.named_parameters()}
    weights = {f'{m}.{n}':v.detach().cpu().clone() for m in MODULES
               for n,v in getattr(model,m).state_dict().items()}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        bootstrap = model(state=None, **kwargs)
        state = bootstrap.new_state.clone()
        cond = bootstrap.past_cond.clone()
        saved_state, saved_cond = state.clone(), cond.clone()
        native = model(state=state, past_cond=cond, **kwargs)
        next_native = model(state=native.new_state, past_cond=native.past_cond, **kwargs)
        actions = {}
        for action in ACTIONS:
            output = forward_with_memory_action(model, state=state, past_cond=cond,
                                                action=action, **kwargs)
            expected = select_memory_update(state, native.new_state, action)
            assert torch.equal(output.new_state, expected)
            assert torch.equal(output.logits, native.logits)
            assert torch.equal(output.past_cond, native.past_cond)
            # Preserve the vendor NamedTuple's positional and attribute API.
            assert torch.equal(output[1], output.new_state)
            following = model(state=output.new_state, past_cond=output.past_cond, **kwargs)
            if action == 'all':
                assert torch.equal(following.logits, next_native.logits)
                assert torch.equal(following.new_state, next_native.new_state)
                assert torch.equal(following.past_cond, next_native.past_cond)
            changed_slots = (output.new_state != state).any(dim=-1).sum().item()
            actions[action] = {'memory_selection_exact':True,
                'current_logits_unchanged':True, 'conditioning_native':True,
                'changed_slots':changed_slots,
                'next_logits_max_abs_delta_vs_all':float((following.logits.float()-next_native.logits.float()).abs().max())}
            del following, output
        assert torch.equal(state, saved_state) and torch.equal(cond, saved_cond)
    assert versions == {n:p._version for n,p in model.named_parameters()}
    assert all(torch.equal(weights[f'{m}.{n}'],v.detach().cpu())
               for m in MODULES for n,v in getattr(model,m).state_dict().items())
    return {'status':'passed', 'checkpoint':str(checkpoint), 'actions':actions,
        'forwards':13, 'all_path_bitwise_native_including_next_forward':True,
        'input_state_and_conditioning_unchanged':True, 'weights_unchanged':True,
        'token_sampling_performed':False, 'training_performed':False,
        'conditioning_policy':'Native recomputation after current backbone; copied unchanged into siblings; subsequent values may diverge.',
        'device':torch.cuda.get_device_name(), 'precision':'BF16'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    directory = ROOT/'result/selective-memory'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory.mkdir(parents=True, exist_ok=False)
    write(directory/'configuration.json', {'actions':list(ACTIONS),
        'groups':{'A':[0,16], 'B':[16,32], 'C':[32,48], 'D':[48,64]},
        'ranges':'zero-based, end exclusive', 'gpu_requested':args.gpu,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (
            Path(__file__), ROOT/'scripts/candidate_trajectory/selective_memory.py',
            ROOT/'scripts/candidate_trajectory/engine.py')}})
    try:
        report = {'cpu':cpu_checks(), 'gpu':gpu_checks(directory) if args.gpu else {'status':'not_run'}}
        report['status'] = 'passed'
        write(directory/'verification.json', report)
        write(ROOT/'result/selective-memory/latest.json', {'directory':str(directory)})
        print(json.dumps({'directory':str(directory), **report}, indent=2), flush=True)
    except Exception:
        import traceback
        write(directory/'verification.json', {'status':'failed', 'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
