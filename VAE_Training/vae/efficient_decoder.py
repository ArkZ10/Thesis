"""Memory-efficient frozen decoding; preserve per-sentence reconstruction loss."""
import types

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache

from .attention import build_mask


def checkpoint_frozen_decoder(decoder):
    """Checkpoint each frozen layer without switching it out of eval mode."""
    if any(p.requires_grad for p in decoder.parameters()):
        raise ValueError('Expected a frozen decoder')
    if getattr(decoder, '_vae_checkpointed', False):
        return
    for layer in decoder.model.layers:
        original = layer.forward

        def forward(self, *args, _original=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        layer.forward = types.MethodType(forward, layer)
    decoder._vae_checkpointed = True


def reconstruction_loss(decoder, prefix, ids, mask, *, chunk_tokens=128):
    """Mean sentence CE, as in microbatch-one training, with bounded logits.

    Token-weighted CE and token accuracy are reported separately. Checkpointing
    each LM-head/loss chunk avoids retaining vocabulary-sized logits for every
    token until backward. Source targets are used only in teacher forcing.
    """
    device = decoder.get_input_embeddings().weight.device
    ids, mask = ids.to(device), mask.to(device)
    text = decoder.get_input_embeddings()(ids)
    prefix = prefix.to(device=device, dtype=text.dtype)
    embeds = torch.cat((prefix, text), dim=1)
    keys = torch.cat((mask.new_ones(ids.shape[0], prefix.shape[1]), mask), dim=1)
    attention = build_mask('hybrid', keys, 0, prefix.shape[1] - 1, embeds.dtype)
    hidden = decoder.model(inputs_embeds=embeds, attention_mask=attention,
                           use_cache=False, return_dict=True).last_hidden_state
    offset = prefix.shape[1] - 1
    hidden = hidden[:, offset:offset + ids.shape[1]].reshape(-1, hidden.shape[-1])
    labels = ids.masked_fill(mask == 0, -100).reshape(-1)
    lengths = mask.sum(-1).clamp_min(1)
    weights = (mask / lengths[:, None] / ids.shape[0]).reshape(-1)

    def loss_chunk(states, target, weight):
        logits = decoder.lm_head(states)
        losses = F.cross_entropy(logits.float(), target, ignore_index=-100, reduction='none')
        correct = ((logits.argmax(-1) == target) & (target != -100)).sum()
        return (losses * weight).sum(), losses.detach().sum(), correct.detach()

    objective = hidden.new_zeros((), dtype=torch.float32)
    ce_sum = hidden.new_zeros((), dtype=torch.float32)
    correct = mask.new_zeros(())
    for start in range(0, len(labels), chunk_tokens):
        args = (hidden[start:start + chunk_tokens], labels[start:start + chunk_tokens],
                weights[start:start + chunk_tokens])
        values = checkpoint(loss_chunk, *args, use_reentrant=False) if torch.is_grad_enabled() else loss_chunk(*args)
        objective = objective + values[0]
        ce_sum += values[1]
        correct += values[2]
    return objective, {'ce_sum': float(ce_sum), 'correct': int(correct), 'tokens': int(mask.sum())}


@torch.no_grad()
def decode_prefix(decoder, prefix, tokenizer, maximum=160):
    """Only latent prefix enters free generation; targets are never supplied."""
    embedding = decoder.get_input_embeddings().weight
    prefix = prefix.to(device=embedding.device, dtype=embedding.dtype)
    batch, length = prefix.shape[:2]
    mask = torch.ones((batch, length), device=prefix.device, dtype=torch.long)
    cache = DynamicCache(config=decoder.config)
    output = decoder.model(inputs_embeds=prefix,
        attention_mask=build_mask('hybrid', mask, 0, length - 1, prefix.dtype),
        position_ids=torch.arange(length, device=prefix.device)[None].expand(batch, -1),
        past_key_values=cache, use_cache=True, return_dict=True)
    hidden = output.last_hidden_state[:, -1]
    generated = [[] for _ in range(batch)]
    done = torch.zeros(batch, device=prefix.device, dtype=torch.bool)
    for step in range(maximum):
        logits = decoder.lm_head(hidden)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite reconstruction logits')
        token = logits.argmax(-1)
        for index in range(batch):
            if not done[index]:
                generated[index].append(int(token[index]))
        done |= token == tokenizer.eos_token_id
        if done.all() or step + 1 == maximum:
            break
        token = torch.where(done, tokenizer.eos_token_id, token)
        mask = torch.cat((mask, mask.new_ones(batch, 1)), dim=1)
        output = decoder.model(input_ids=token[:, None], attention_mask=mask,
            position_ids=mask.new_full((batch, 1), mask.shape[1] - 1),
            past_key_values=cache, use_cache=True, return_dict=True)
        hidden = output.last_hidden_state[:, -1]
    rows = []
    for tokens in generated:
        ended = tokenizer.eos_token_id in tokens
        content = tokens[:tokens.index(tokenizer.eos_token_id)] if ended else tokens
        rows.append({'text': tokenizer.decode(content, skip_special_tokens=True),
                     'generated_token_ids': tokens, 'eos_reached': ended,
                     'token_limit_reached': not ended and len(tokens) == maximum})
    return rows
