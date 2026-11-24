"""Opt-in selective writes to the existing 64-slot memory; no new parameters.

Actions A/B/C/D write slots 0:16/16:32/32:48/48:64; `all` is native.
The action selects the returned memory AFTER the native updater proposes all
slots. Current logits and returned conditioning retain native semantics. The
selected memory is consumed by the NEXT forward. Bootstrap always uses `all`.
Group labels have no assigned computational meanings. No caching supported.
"""
import torch

from .engine import clone

ACTIONS = ('all', 'A', 'B', 'C', 'D')
SLOTS = 64
GROUP_SIZE = 16


def _validate_action(action):
    if not isinstance(action, str) or action not in ACTIONS:
        raise ValueError(f'Expected one of {ACTIONS}; got {action!r}')


def slot_mask(state, action):
    _validate_action(action)
    if not isinstance(state, torch.Tensor) or state.ndim != 3 or state.shape[1] != SLOTS:
        raise ValueError('Expected memory shape [batch, 64, hidden_size]')
    if not state.is_floating_point():
        raise ValueError('Memory must be floating point')
    mask = torch.zeros((1, SLOTS, 1), device=state.device, dtype=torch.bool)
    if action == 'all':
        mask.fill_(True)
    else:
        start = ('A', 'B', 'C', 'D').index(action) * GROUP_SIZE
        mask[:, start:start + GROUP_SIZE] = True
    return mask


def select_memory_update(previous, proposed, action):
    """Differentiable, out-of-place selection; unselected bits are preserved."""
    mask = slot_mask(previous, action)
    if (previous.shape != proposed.shape or previous.dtype != proposed.dtype
            or previous.device != proposed.device):
        raise ValueError('Previous and proposed memory must share shape, dtype and device')
    # Avoid subtraction/addition: roundoff could change unselected BF16 values.
    return proposed if action == 'all' else torch.where(mask, proposed, previous)


def forward_with_memory_action(model, *, state=None, action='all', **kwargs):
    """Run a native forward, selectively accept its proposed memory update.

    Conditioning is recomputed normally, neither reset nor slot-masked. It can
    diverge on subsequent forwards after the selected memory affects injection.
    This helper retains the autograd graph; branch snapshots intentionally do not.
    """
    _validate_action(action)
    if kwargs.get('use_cache', False) or any(kwargs.get(k) is not None for k in (
            'past_key_values', 'past_mixer_hidden', 'past_injector_sa_kv', 'replace_position')):
        raise ValueError('Selective memory adapter supports no-cache forwards only')
    if kwargs.get('return_dict', True) is not True:
        raise ValueError('Selective memory adapter requires return_dict=True')
    if action != 'all':
        if state is None:
            raise ValueError('Bootstrap with action=all before selecting memory groups')
        slot_mask(state, action)  # Validate before doing expensive model work.
    kwargs['return_dict'] = True
    kwargs['use_cache'] = False
    output = model(state=state, **kwargs)
    if action == 'all':
        return output
    accepted = select_memory_update(state, output.new_state, action)
    # Vendor RecurrentLLaDAModelOutput is an immutable NamedTuple.
    return output._replace(new_state=accepted)


def fork_memory_branch(parent, action, trajectory_id):
    """Copy ALL continuation fields, including conditioning, RNG and counters.

    The selected action applies to future memory writes in Engine.advance. No
    token sampling seed is changed. Each call makes an independent inference copy.
    """
    _validate_action(action)
    if parent.get('memory') is None:
        raise ValueError('Branching requires an already bootstrapped continuation')
    slot_mask(parent['memory'], action)
    child = clone(parent)
    child['memory_update_action'] = action
    child['trajectory_id'] = trajectory_id
    return child
