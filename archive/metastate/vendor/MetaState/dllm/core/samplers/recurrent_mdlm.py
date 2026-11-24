"""
Recurrent MDLM Sampler with state carrying across diffusion steps.

References:
- https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
- Simple and Effective Masked Diffusion Language Models: https://arxiv.org/abs/2406.07524
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, SamplerConfig, SamplerOutput
from dllm.core.samplers.utils import (
    RecurrentSamplerMixin,
    add_gumbel_noise,
    get_num_transfer_tokens,
    get_num_transfer_tokens_uniform,
    get_transfer_index,
    trim_injector_sa_kv,
    trim_mixer_hidden,
    trim_past_key_values,
)


@dataclass
class RecurrentMDLMSamplerConfig(SamplerConfig):
    """Configuration for recurrent MDLM sampling."""

    max_new_tokens: int = 128
    max_length: int | None = None
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False
    use_cache: str | None = None
    threshold: float | None = None
    factor: float | None = None
    debug_stats: bool = False
    """Collect and print debug statistics (state norm, updater gate mean)."""
    pass_attention_mask: bool = True
    """Whether sampler forwards its padding mask into model calls."""
    warmup: bool = False
    """Run a bootstrap forward pass (state=None, t=1.0) before the main sampling
    loop to initialize recurrent state, matching training behavior."""


@dataclass
class RecurrentMDLMSampler(RecurrentSamplerMixin, BaseSampler):
    """
    Recurrent MDLM Sampler that carries state across diffusion steps.

    Key recurrent sampling properties:
    1. Passes timestep `t` to the model for time-dependent injection
    2. Maintains and updates recurrent state across all diffusion steps
    3. Timestep is derived from the global masked ratio across all generation positions
    """

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: RecurrentMDLMSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        """
        Generate text using recurrent masked diffusion language modeling.

        Supports three cache modes:
          - use_cache=None: baseline (no cache, full recomputation each step)
          - use_cache="prefix": cache prompt KV, reprocess only suffix
          - use_cache="dual": cache full KV, selectively replace block positions

        In all modes, recurrent state is maintained and updated at every step,
        persisting across block boundaries. Timestep is computed from the global
        masked ratio across all generation positions.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            SamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = RecurrentMDLMSamplerConfig()

        # ----- Pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        debug_stats = kwargs.get("debug_stats", config.debug_stats)
        use_cache = kwargs.get("use_cache", config.use_cache)
        threshold = kwargs.get("threshold", config.threshold)
        factor = kwargs.get("factor", config.factor)
        pass_attention_mask = kwargs.get(
            "pass_attention_mask", config.pass_attention_mask
        )
        warmup = kwargs.get("warmup", config.warmup)
        # Combine suppress lists once; get_transfer_index handles suppression
        # internally (suppressed for token proposal, unsuppressed for confidence).
        _suppress_ids = (
            list(suppress_tokens or []) + list(begin_suppress_tokens or [])
        ) or None

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # Normalize use_cache
        if use_cache == "none":
            use_cache = None
        if use_cache not in (None, "prefix", "dual"):
            raise ValueError(f"Unknown use_cache mode: {use_cache!r}. Expected None, 'prefix', or 'dual'.")

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id

        attention_mask = None
        if pass_attention_mask:
            attention_mask = torch.zeros(
                (B, T), dtype=torch.long, device=self.model.device
            )
            for i, pl in enumerate(prompt_lens):
                valid_end = min(pl + max_new_tokens, T)
                attention_mask[i, :valid_end] = 1

        def _mask_slice(start: int | None = None, end: int | None = None):
            if attention_mask is None:
                return None
            if start is None and end is None:
                return attention_mask
            return attention_mask[:, start:end]

        # Cache modes require equal prompt lengths
        if use_cache is not None:
            if len(set(prompt_lens)) != 1:
                raise ValueError(
                    f"use_cache={use_cache!r} requires equal prompt lengths. "
                    f"Got {prompt_lens}. Set use_cache=None or pad prompts."
                )
            prompt_len = prompt_lens[0]
        else:
            prompt_len = None

        # Debug stats container
        stats_history = {"state_norm": [], "delta_norm": [], "embedding_norm": []} if debug_stats else None

        # ----- Block scheduling over the appended mask tail -----
        if use_cache is not None and max_new_tokens % block_size != 0:
            raise AssertionError(
                f"max_new_tokens ({max_new_tokens}) must be divisible by "
                f"block_size ({block_size}) when using cache mode (use_cache={use_cache!r})"
            )
        num_blocks = (
            math.ceil(max_new_tokens / block_size)
            if use_cache is None
            else max_new_tokens // block_size
        )
        if steps % num_blocks != 0:
            raise AssertionError(
                f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
            )
        steps_per_block = steps // num_blocks
        histories = [x.clone()] if return_dict else None

        # Global generation region for timestep computation (captured before any unmasking)
        gen_region = (x == mask_id)

        state = None
        past_cond = None

        # Debug: overall timing and NFE tracking
        if debug_stats:
            mode_tag = "threshold" if threshold is not None else ("factor" if factor is not None else "quota")
            cache_tag = use_cache or "none"
            print(f"\n{'='*70}")
            print(f"[SAMPLE] cache={cache_tag}, mode={mode_tag}, threshold={threshold}, "
                  f"temperature={temperature}")
            print(f"[SAMPLE] steps={steps}, block_size={block_size}, max_new_tokens={max_new_tokens}, "
                  f"num_blocks={num_blocks}, steps_per_block={steps_per_block}")
            print(f"[SAMPLE] B={B}, T={T}, prompt_lens={prompt_lens}")
            print(f"{'='*70}")
            _gen_start = time.time()
            _total_nfe = 0

        # ===== Optional warmup: bootstrap forward to initialize state =====
        if warmup:
            warmup_out = self.model(
                input_ids=x,
                t=torch.ones(B, device=self.model.device),
                state=None,
                attention_mask=_mask_slice(),
                return_dict=True,
            )
            state = warmup_out.new_state
            past_cond = getattr(warmup_out, "past_cond", None)
            if debug_stats:
                _total_nfe += 1
                self._collect_debug_stats(
                    stats_history, state,
                    getattr(warmup_out, "delta_norm", None),
                    getattr(warmup_out, "embedding_norm", None),
                )
                _s_norm = state.norm(dim=-1).mean().item() if state is not None else 0.0
                print(f"  [WARMUP] t=1.0, state_norm={_s_norm:.4f}")

        for b in range(num_blocks):
            if debug_stats:
                _block_start = time.time()
                _block_nfe = 0
                _block_steps = 0
                _block_transferred_total = 0

            # Build block boundaries and mask index
            if prompt_len is not None:
                # Cache modes: shared boundaries
                s = prompt_len + b * block_size
                e = min(s + block_size, prompt_len + max_new_tokens, T)
                if s >= e:
                    continue
                block_len = e - s
                block_mask_index = torch.zeros(
                    (B, block_size), dtype=torch.bool, device=x.device
                )
                block_mask_index[:, :block_len] = x[:, s:e] == mask_id
            else:
                # No-cache: per-sample boundaries
                block_mask_index = torch.zeros(
                    (B, block_size), dtype=torch.bool, device=x.device
                )
                for j in range(B):
                    start_j = prompt_lens[j] + b * block_size
                    end_j = min(start_j + block_size, prompt_lens[j] + max_new_tokens, T)
                    if start_j < end_j:
                        width = end_j - start_j
                        block_mask_index[j, :width] = x[j, start_j:end_j] == mask_id

            # Decide how many tokens to reveal per step in this block.
            # Threshold/factor mode: no schedule needed — each step
            # decodes all tokens with confidence >= threshold (with
            # force-unmask of at least 1), loop until all unmasked.
            if threshold is not None or factor is not None:
                num_transfer_tokens = None
                # Match Fast-dLLM LLaDA dual-cache semantics: fixed per-block loop count.
                max_steps = steps_per_block if use_cache == "dual" else None
            else:
                num_transfer_tokens = get_num_transfer_tokens_uniform(
                    mask_index=block_mask_index, steps=steps_per_block
                )
                max_steps = num_transfer_tokens.size(1)

            # ===== Mode 1: No cache =====
            if use_cache is None:
                i = 0
                while True:
                    if max_steps is not None and i >= max_steps:
                        break
                    # Build mask_allowed: all masks up to current block end
                    # (includes leftover masks from previous blocks, matching reference)
                    mask_allowed = (x == mask_id)
                    for j in range(B):
                        end_j = min(prompt_lens[j] + (b + 1) * block_size, prompt_lens[j] + max_new_tokens, T)
                        mask_allowed[j, end_j:] = False
                    n_masked_before = mask_allowed.sum().item() if debug_stats else 0
                    if mask_allowed.sum() == 0:
                        break

                    # Compute timestep from global masked ratio
                    t_input = self._compute_timestep_from_mask(x, mask_id, gen_region)

                    if cfg_scale > 0.0:
                        raise NotImplementedError("CFG is not supported for recurrent MDLM sampling.")

                    _step_t0 = time.time() if debug_stats else 0
                    out = self.model(
                        input_ids=x,
                        t=t_input,
                        state=state,
                        attention_mask=_mask_slice(),
                        return_dict=True,
                        past_cond=past_cond,
                    )
                    logits = out.logits
                    state = out.new_state
                    past_cond = getattr(out, "past_cond", None)

                    if debug_stats:
                        _block_nfe += 1
                        self._collect_debug_stats(
                            stats_history, state,
                            getattr(out, "delta_norm", None),
                            getattr(out, "embedding_norm", None),
                        )

                    if right_shift_logits:
                        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                    if threshold is not None or factor is not None:
                        quota = None
                    else:
                        quota = num_transfer_tokens[:, i]
                    x0, transfer_idx = get_transfer_index(
                        logits=logits, temperature=temperature,
                        remasking=remasking, mask_index=mask_allowed, x=x,
                        num_transfer_tokens=quota, threshold=threshold, factor=factor,
                        suppress_token_ids=_suppress_ids,
                    )
                    x = torch.where(transfer_idx, x0, x)

                    if debug_stats:
                        n_transferred = transfer_idx.sum().item()
                        _block_transferred_total += n_transferred
                        _block_steps += 1
                        _step_dt = time.time() - _step_t0
                        n_remaining = (x == mask_id).sum().item()
                        print(f"  [B{b} S{i:3d}] masked={n_masked_before:4d} -> transferred={n_transferred:4d} "
                              f"remaining={n_remaining:4d} t={t_input[0].item():.3f} dt={_step_dt:.3f}s")

                    i += 1
                    if histories is not None:
                        histories.append(x.clone())

                if debug_stats:
                    _block_dt = time.time() - _block_start
                    _total_nfe += _block_nfe
                    n_leftover = (x == mask_id).sum().item()
                    print(f"  [Block {b} DONE] steps={_block_steps}, NFE={_block_nfe}, "
                          f"transferred={_block_transferred_total}, leftover_masks={n_leftover}, "
                          f"time={_block_dt:.2f}s")
                continue

            # ===== Mode 2: Prefix cache =====
            if use_cache == "prefix":
                # Step 0: full forward with cache
                t_input = self._compute_timestep_from_mask(x, mask_id, gen_region)
                _step_t0 = time.time() if debug_stats else 0
                out_full = self.model(
                    input_ids=x,
                    t=t_input,
                    state=state,
                    attention_mask=_mask_slice(),
                    return_dict=True,
                    use_cache=True,
                    past_cond=past_cond,
                )
                logits_full = out_full.logits
                state = out_full.new_state
                past_cond = getattr(out_full, "past_cond", None)
                past_key_values = getattr(out_full, "past_key_values", None)
                past_mixer_hidden = getattr(out_full, "past_mixer_hidden", None)
                past_injector_sa_kv = getattr(out_full, "past_injector_sa_kv", None)

                if debug_stats:
                    _block_nfe += 1
                    self._collect_debug_stats(
                        stats_history, state,
                        getattr(out_full, "delta_norm", None),
                        getattr(out_full, "embedding_norm", None),
                    )

                if right_shift_logits:
                    logits_full = torch.cat([logits_full[:, :1], logits_full[:, :-1]], dim=1)

                # Step 0 token selection: all masks up to current block end
                mask_allowed = (x == mask_id)
                mask_allowed[:, e:] = False
                n_masked_before = mask_allowed.sum().item() if debug_stats else 0
                if mask_allowed.sum() > 0:
                    quota = None if (threshold is not None or factor is not None) else num_transfer_tokens[:, 0]
                    x0, transfer_idx = get_transfer_index(
                        logits=logits_full, temperature=temperature,
                        remasking=remasking, mask_index=mask_allowed, x=x,
                        num_transfer_tokens=quota, threshold=threshold, factor=factor,
                        suppress_token_ids=_suppress_ids,
                    )
                    x = torch.where(transfer_idx, x0, x)
                    if debug_stats:
                        n_transferred = transfer_idx.sum().item()
                        _block_transferred_total += n_transferred
                        _block_steps += 1
                        _step_dt = time.time() - _step_t0
                        n_remaining = (x[:, :e] == mask_id).sum().item()
                        print(f"  [B{b} S  0] masked={n_masked_before:4d} -> transferred={n_transferred:4d} "
                              f"remaining={n_remaining:4d} t={t_input[0].item():.3f} dt={_step_dt:.3f}s (full fwd)")
                    if histories is not None:
                        histories.append(x.clone())

                # Trim caches to prefix
                if past_key_values is None:
                    raise RuntimeError("Model did not return past_key_values with use_cache=True")
                past_key_values = trim_past_key_values(past_key_values, s)
                if past_mixer_hidden is not None:
                    past_mixer_hidden = trim_mixer_hidden(past_mixer_hidden, s)
                if past_injector_sa_kv is not None:
                    past_injector_sa_kv = trim_injector_sa_kv(past_injector_sa_kv, s)

                # Refinement steps on suffix
                i = 1
                while True:
                    if max_steps is not None and i >= max_steps:
                        break
                    if (x[:, s:e] == mask_id).sum() == 0:
                        break
                    x_suffix = x[:, s:]
                    mask_suffix = x_suffix == mask_id
                    if x_suffix.size(1) > block_len:
                        mask_suffix[:, block_len:] = False
                    n_masked_before = mask_suffix.sum().item() if debug_stats else 0
                    if mask_suffix.sum() == 0:
                        break

                    t_input = self._compute_timestep_from_mask(x, mask_id, gen_region)
                    _step_t0 = time.time() if debug_stats else 0

                    out_suf = self.model(
                        input_ids=x[:, s:],
                        t=t_input,
                        state=state,
                        attention_mask=_mask_slice(s, None),
                        past_key_values=past_key_values,
                        past_mixer_hidden=past_mixer_hidden,
                        past_injector_sa_kv=past_injector_sa_kv,
                        use_cache=True,
                        return_dict=True,
                        past_cond=past_cond,
                    )
                    logits_suf = out_suf.logits
                    state = out_suf.new_state
                    past_cond = getattr(out_suf, "past_cond", None)
                    # Don't update prefix caches — prefix is static, suffix recomputed

                    if debug_stats:
                        _block_nfe += 1
                        self._collect_debug_stats(
                            stats_history, state,
                            getattr(out_suf, "delta_norm", None),
                            getattr(out_suf, "embedding_norm", None),
                        )

                    if right_shift_logits:
                        logits_suf = torch.cat([logits_suf[:, :1], logits_suf[:, :-1]], dim=1)

                    quota = None if (threshold is not None or factor is not None) else num_transfer_tokens[:, i]
                    x0_suf, transfer_suf = get_transfer_index(
                        logits=logits_suf, temperature=temperature,
                        remasking=remasking, mask_index=mask_suffix, x=x_suffix,
                        num_transfer_tokens=quota, threshold=threshold, factor=factor,
                        suppress_token_ids=_suppress_ids,
                    )
                    x_suffix_new = torch.where(transfer_suf, x0_suf, x_suffix)
                    x = torch.cat([x[:, :s], x_suffix_new], dim=1)

                    if debug_stats:
                        n_transferred = transfer_suf.sum().item()
                        _block_transferred_total += n_transferred
                        _block_steps += 1
                        _step_dt = time.time() - _step_t0
                        n_remaining = (x[:, s:e] == mask_id).sum().item()
                        print(f"  [B{b} S{i:3d}] masked={n_masked_before:4d} -> transferred={n_transferred:4d} "
                              f"remaining={n_remaining:4d} t={t_input[0].item():.3f} dt={_step_dt:.3f}s (suffix)")

                    i += 1
                    if histories is not None:
                        histories.append(x.clone())

                if debug_stats:
                    _block_dt = time.time() - _block_start
                    _total_nfe += _block_nfe
                    n_leftover = (x == mask_id).sum().item()
                    print(f"  [Block {b} DONE] steps={_block_steps}, NFE={_block_nfe}, "
                          f"transferred={_block_transferred_total}, leftover_masks={n_leftover}, "
                          f"time={_block_dt:.2f}s")
                continue

            # ===== Mode 3: Dual cache =====
            if use_cache == "dual":
                # Step 0: full forward with cache
                t_input = self._compute_timestep_from_mask(x, mask_id, gen_region)
                _step_t0 = time.time() if debug_stats else 0
                out_full = self.model(
                    input_ids=x,
                    t=t_input,
                    state=state,
                    attention_mask=_mask_slice(),
                    return_dict=True,
                    use_cache=True,
                    past_cond=past_cond,
                )
                logits_full = out_full.logits
                state = out_full.new_state
                past_cond = getattr(out_full, "past_cond", None)
                past_key_values = getattr(out_full, "past_key_values", None)
                past_mixer_hidden = getattr(out_full, "past_mixer_hidden", None)
                past_injector_sa_kv = getattr(out_full, "past_injector_sa_kv", None)
                if past_key_values is None:
                    raise RuntimeError("Model did not return past_key_values with use_cache=True")

                if debug_stats:
                    _block_nfe += 1
                    self._collect_debug_stats(
                        stats_history, state,
                        getattr(out_full, "delta_norm", None),
                        getattr(out_full, "embedding_norm", None),
                    )

                if right_shift_logits:
                    logits_full = torch.cat([logits_full[:, :1], logits_full[:, :-1]], dim=1)

                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, s:e] = True

                # Step 0 token selection: all masks up to current block end
                mask_allowed = (x == mask_id)
                mask_allowed[:, e:] = False
                n_masked_before = mask_allowed.sum().item() if debug_stats else 0
                if mask_allowed.sum() > 0:
                    quota = None if (threshold is not None or factor is not None) else num_transfer_tokens[:, 0]
                    x0, transfer_idx = get_transfer_index(
                        logits=logits_full, temperature=temperature,
                        remasking=remasking, mask_index=mask_allowed, x=x,
                        num_transfer_tokens=quota, threshold=threshold, factor=factor,
                        suppress_token_ids=_suppress_ids,
                    )

                    x = torch.where(transfer_idx, x0, x)
                    if debug_stats:
                        n_transferred = transfer_idx.sum().item()
                        _block_transferred_total += n_transferred
                        _block_steps += 1
                        _step_dt = time.time() - _step_t0
                        n_remaining = (x[:, :e] == mask_id).sum().item()
                        print(f"  [B{b} S  0] masked={n_masked_before:4d} -> transferred={n_transferred:4d} "
                              f"remaining={n_remaining:4d} t={t_input[0].item():.3f} dt={_step_dt:.3f}s (full fwd)")
                    if histories is not None:
                        histories.append(x.clone())

                # Refinement steps on block
                i_step = 1
                while True:
                    if max_steps is not None and i_step >= max_steps:
                        break
                    blk = x[:, s:e]
                    mask_blk = blk == mask_id
                    n_masked_before = mask_blk.sum().item() if debug_stats else 0
                    if mask_blk.sum() == 0:
                        break

                    t_input = self._compute_timestep_from_mask(x, mask_id, gen_region)
                    _step_t0 = time.time() if debug_stats else 0

                    out_blk = self.model(
                        input_ids=x[:, s:e],
                        t=t_input,
                        state=state,
                        attention_mask=_mask_slice(s, e),
                        past_key_values=past_key_values,
                        past_mixer_hidden=past_mixer_hidden,
                        past_injector_sa_kv=past_injector_sa_kv,
                        use_cache=True,
                        replace_position=replace_position,
                        return_dict=True,
                        past_cond=past_cond,
                    )
                    logits_blk = out_blk.logits
                    state = out_blk.new_state
                    past_cond = getattr(out_blk, "past_cond", None)
                    # Update caches (model handles replacement via kv_replace_offset)
                    past_mixer_hidden = getattr(out_blk, "past_mixer_hidden", None)
                    past_injector_sa_kv = getattr(out_blk, "past_injector_sa_kv", None)

                    if debug_stats:
                        _block_nfe += 1
                        self._collect_debug_stats(
                            stats_history, state,
                            getattr(out_blk, "delta_norm", None),
                            getattr(out_blk, "embedding_norm", None),
                        )

                    if right_shift_logits:
                        logits_blk = torch.cat([logits_blk[:, :1], logits_blk[:, :-1]], dim=1)

                    quota = None if (threshold is not None or factor is not None) else num_transfer_tokens[:, i_step]
                    x0_blk, transfer_blk = get_transfer_index(
                        logits=logits_blk, temperature=temperature,
                        remasking=remasking, mask_index=mask_blk, x=blk,
                        num_transfer_tokens=quota, threshold=threshold, factor=factor,
                        suppress_token_ids=_suppress_ids,
                    )

                    blk_new = torch.where(transfer_blk, x0_blk, blk)
                    x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

                    if debug_stats:
                        n_transferred = transfer_blk.sum().item()
                        _block_transferred_total += n_transferred
                        _block_steps += 1
                        _step_dt = time.time() - _step_t0
                        n_remaining = (x[:, s:e] == mask_id).sum().item()
                        print(f"  [B{b} S{i_step:3d}] masked={n_masked_before:4d} -> transferred={n_transferred:4d} "
                              f"remaining={n_remaining:4d} t={t_input[0].item():.3f} dt={_step_dt:.3f}s (block)")

                    i_step += 1
                    if histories is not None:
                        histories.append(x.clone())

                if debug_stats:
                    _block_dt = time.time() - _block_start
                    _total_nfe += _block_nfe
                    n_leftover = (x == mask_id).sum().item()
                    print(f"  [Block {b} DONE] steps={_block_steps}, NFE={_block_nfe}, "
                          f"transferred={_block_transferred_total}, leftover_masks={n_leftover}, "
                          f"time={_block_dt:.2f}s")
                continue

        # Print debug stats summary
        if debug_stats:
            _gen_dt = time.time() - _gen_start
            n_final_masks = (x == mask_id).sum().item()
            print(f"\n{'='*70}")
            print(f"[SUMMARY] total_time={_gen_dt:.2f}s, total_NFE={_total_nfe}, "
                  f"final_masks={n_final_masks}, num_blocks={num_blocks}")
            print(f"[SUMMARY] avg_time_per_NFE={_gen_dt/_total_nfe:.3f}s" if _total_nfe > 0 else "")
            print(f"{'='*70}\n")
            self._print_debug_stats_summary(stats_history)

        # ----- Output format -----
        if not return_dict:
            return x
        return SamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: RecurrentMDLMSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs` using recurrent state.

        Args:
            inputs: List of input sequences containing mask tokens to be filled.
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            SamplerOutput with filled sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = RecurrentMDLMSamplerConfig()

        # ----- Pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        debug_stats = kwargs.get("debug_stats", config.debug_stats)

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # Debug stats container
        stats_history = {"state_norm": [], "delta_norm": [], "embedding_norm": []} if debug_stats else None

        # ----- Blockwise schedule over the entire sequence -----
        # Save initial mask positions for per-block timestep computation.
        # In infill, originally-masked positions are the "maskable" region
        # (analogous to labels != -100 in training).
        initial_mask = (x == mask_id)

        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        state = None
        past_cond = None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                # Compute timestep from global masked ratio
                t_input = self._compute_timestep_from_mask(x, mask_id, initial_mask)

                # ----- Forward pass with recurrent state -----
                if cfg_scale > 0.0:
                    raise NotImplementedError("CFG is not supported for recurrent MDLM infilling.")

                out = self.model(
                    input_ids=x,
                    t=t_input,
                    state=state,
                    attention_mask=attention_mask,
                    return_dict=True,
                    past_cond=past_cond,
                )
                logits = out.logits
                state = out.new_state
                past_cond = getattr(out, "past_cond", None)

                # Collect debug stats
                if debug_stats:
                    self._collect_debug_stats(
                        stats_history, state,
                        getattr(out, "delta_norm", None),
                        getattr(out, "embedding_norm", None),
                    )

                # ----- Standard MDLM selection logic -----
                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Confidence for choosing which masks to commit
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the current block only
                for j in range(B):
                    end_j = start + widths[j]
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # Print debug stats summary
        if debug_stats:
            self._print_debug_stats_summary(stats_history, prefix="Infill")

        # ----- Output format -----
        if not return_dict:
            return x
        return SamplerOutput(sequences=x, histories=histories)
