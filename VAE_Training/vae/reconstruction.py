"""Free reconstruction from latent codes only, with hybrid-prefix attention."""
from contextlib import nullcontext
import re

import torch
from transformers.cache_utils import DynamicCache

from .model import build_mask


def autocast_for(model):
    device = model.latents_query.device
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def decode_latents(model, latents, tokenizer, max_new_tokens=64):
    """Decoder gets [projected latents, AE], followed only by generated tokens.

    Source text/labels are intentionally absent from this function's inputs.
    """
    was_training = model.training
    model.eval()
    try:
        with autocast_for(model):
            prefix = model.decoder_prefix(latents)
            batch, length = prefix.shape[:2]
            mask = torch.ones((batch, length), dtype=torch.long, device=prefix.device)
            cache = DynamicCache(config=model.decoder_model.config)
            output = model.decoder_model.model(
                inputs_embeds=prefix,
                attention_mask=build_mask(model.config.decoder_mask_mode, mask, 0, model.num_latents, prefix.dtype),
                position_ids=torch.arange(length, device=prefix.device)[None].expand(batch, -1),
                past_key_values=cache, use_cache=True, return_dict=True,
            )
            hidden = output.last_hidden_state[:, -1]
            generated = [[] for _ in range(batch)]
            done = torch.zeros(batch, dtype=torch.bool, device=prefix.device)
            for step in range(max_new_tokens):
                scores = model.decoder_model.lm_head(hidden).float()
                if not torch.isfinite(scores).all():
                    raise RuntimeError("Nonfinite reconstruction logits")
                token = scores.argmax(-1)
                for i in range(batch):
                    if not done[i]:
                        generated[i].append(int(token[i]))
                done |= token == tokenizer.eos_token_id
                if done.all() or step + 1 == max_new_tokens:
                    break
                token = torch.where(done, tokenizer.eos_token_id, token)
                mask = torch.cat([mask, mask.new_ones(batch, 1)], dim=1)
                output = model.decoder_model.model(
                    input_ids=token[:, None], attention_mask=mask,
                    position_ids=mask.new_full((batch, 1), mask.shape[1] - 1),
                    past_key_values=cache, use_cache=True, return_dict=True,
                )
                hidden = output.last_hidden_state[:, -1]
        rows = []
        for tokens in generated:
            ended = tokenizer.eos_token_id in tokens
            content = tokens[:tokens.index(tokenizer.eos_token_id)] if ended else tokens
            rows.append({"text": tokenizer.decode(content, skip_special_tokens=True), "token_ids": content,
                         "generated_token_ids": tokens, "eos_reached": ended,
                         "token_limit_reached": not ended and len(tokens) >= max_new_tokens})
        return rows
    finally:
        model.train(was_training)


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def reconstruct_rows(model, tokenizer, rows, control="normal", max_new_tokens=64):
    was_training = model.training
    model.eval()
    try:
        sequences = [tokenizer(r["reasoning_block"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id] for r in rows]
        length = max(map(len, sequences))
        ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long, device=model.latents_query.device)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq, device=ids.device)
            mask[i, :len(seq)] = 1
        with autocast_for(model):
            mu, _ = model.encode(ids, mask)
        if control == "zero":
            mu = torch.zeros_like(mu)
        elif control == "shuffled":
            if len(rows) < 2:
                raise ValueError("Shuffled control requires at least two examples")
            mu = mu.roll(1, dims=0)
        elif control != "normal":
            raise ValueError(control)
        decoded = decode_latents(model, mu, tokenizer, max_new_tokens)
        records = []
        for row, pred in zip(rows, decoded):
            records.append({"parent_id": row["parent_id"], "sentence_index": row["sentence_index"],
                            "problem_id": row["problem_id"], "target": row["reasoning_block"],
                            "control": control, **pred,
                            "exact_match": normalized(pred["text"]) == normalized(row["reasoning_block"])})
        return records
    finally:
        model.train(was_training)
