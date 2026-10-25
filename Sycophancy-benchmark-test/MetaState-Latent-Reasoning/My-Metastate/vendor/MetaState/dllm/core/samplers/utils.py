from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.schedulers import BaseAlphaScheduler


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: BaseAlphaScheduler,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Compute the number of tokens to unmask at each diffusion step.

    For each sample, determines how many masked tokens should be revealed
    per step based on the reverse diffusion schedule.

    Args:
        mask_index: Boolean tensor [B, L] indicating masked positions.
        steps: Number of diffusion steps.
        scheduler: Alpha scheduler defining the masking schedule.
        stochastic: If True, sample from a binomial distribution (probabilistic);
            if False, use deterministic rounding of the expected number of tokens.

    Returns:
        Integer tensor [B, steps] with number of tokens to unmask per step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    )
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps - 1, -1, -1), range(steps)):
            s /= steps
            t /= steps
            reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i, 0].to(torch.float64) * reverse_transfer_prob
                num_transfer_tokens[i, j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i, 0].to(torch.float64)
                num_transfer_tokens[i, j] = (
                    torch.distributions.Binomial(n, reverse_transfer_prob)
                    .sample()
                    .to(torch.int64)
                )
            num_transfer_tokens[i, j] = torch.minimum(
                num_transfer_tokens[i, j], mask_num[i, 0]
            )
            mask_num[i, 0] -= num_transfer_tokens[i, j]
            if mask_num[i, 0].item() == 0:
                break
    # Compact: remove zero entries per row and right-pad to max length
    rows = [row[row > 0] for row in num_transfer_tokens]
    max_len = max(r.numel() for r in rows) if rows else 0
    if max_len == 0:
        return torch.zeros(num_transfer_tokens.size(0), 1, device=num_transfer_tokens.device, dtype=torch.int64)
    padded = torch.zeros(len(rows), max_len, dtype=torch.int64, device=num_transfer_tokens.device)
    for i, r in enumerate(rows):
        padded[i, :r.numel()] = r
    return padded


def get_num_transfer_tokens_uniform(
    mask_index: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """
    Fast-dLLM uniform transfer schedule.

    Split each row's masked-token count across ``steps`` by:
      - assigning ``base = total // steps`` to every step
      - distributing the ``remainder`` to the first ``remainder`` steps.
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")

    total = mask_index.sum(dim=1)  # [B]
    base = torch.div(total, steps, rounding_mode="floor")
    remainder = total - base * steps

    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(torch.long).clone()
    cols = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    num_transfer_tokens = num_transfer_tokens + (cols < remainder.unsqueeze(1)).to(torch.long)
    return num_transfer_tokens


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# ---------------------------------------------------------------------------
# Fast-dLLM KV-cache utilities
# Reference: NVLabs/Fast-dLLM
# ---------------------------------------------------------------------------


def trim_past_key_values(
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
    upto: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Keep only KV up to sequence index ``upto`` (exclusive) along seq-len dim.

    Recurrent LLaDA uses ``[B, H, S, D]`` KV tensors, where sequence is dim -2.
    """
    new_pkv = []
    for layer_kv in past_key_values:
        if layer_kv[0].dim() != 4:
            raise ValueError(
                f"Expected LLaDA KV tensors with 4 dims, got {layer_kv[0].dim()}"
            )
        new_layer = tuple(t[:, :, :upto] for t in layer_kv)
        new_pkv.append(new_layer)
    return new_pkv


def trim_injector_sa_kv(
    past_sa_kv: tuple[torch.Tensor, torch.Tensor] | None,
    upto: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Trim Injector self-attn KV cache to first ``upto`` sequence positions.

    Args:
        past_sa_kv: (K, V) tuple each [B, KVH, N, head_dim], or None.
        upto: Keep only the first ``upto`` sequence positions (dim 2).

    Returns:
        Trimmed (K, V) tuple, or None if input is None.
    """
    if past_sa_kv is None:
        return None
    k, v = past_sa_kv
    return (k[:, :, :upto], v[:, :, :upto])


def trim_mixer_hidden(
    past_mixer_hidden: torch.Tensor,
    upto: int,
) -> torch.Tensor:
    """
    Trim Mixer hidden cache to ``[:, :upto, :]``.

    Args:
        past_mixer_hidden: Down-projected hidden states [B, N, mixer_dim].
        upto: Keep only the first ``upto`` sequence positions.

    Returns:
        Trimmed hidden tensor [B, upto, mixer_dim].
    """
    return past_mixer_hidden[:, :upto]


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor | None = None,
    threshold: float | None = None,
    factor: float | None = None,
    suppress_token_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    3-mode confidence-based token selection for Fast-dLLM cache samplers.

    Returns:
        x0:             (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update

    Priority:
      if factor is not None → dynamic schedule
      elif threshold is not None → threshold mode
      else → top-k mode (num_transfer_tokens required)

    Args:
        suppress_token_ids: Token IDs to suppress (e.g. suppress_tokens +
            begin_suppress_tokens).  Suppressed tokens are excluded from
            token proposals (argmax) but do NOT affect confidence
            computation, preventing inflated softmax confidences from
            causing over-aggressive threshold-based token commitment.
    """
    # 1) Propose tokens (greedy / gumbel-max)
    #    Suppress specified tokens only for the argmax (token proposal).
    #    The original logits are kept unmodified for confidence computation.
    if suppress_token_ids:
        suppress_mask = torch.zeros(
            logits.size(-1), dtype=torch.bool, device=logits.device
        )
        for tid in suppress_token_ids:
            suppress_mask[tid] = True
        logits_with_noise = add_gumbel_noise(
            logits.masked_fill(suppress_mask, float("-inf")),
            temperature=temperature,
        )
    else:
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

    # 2) Confidence from UNSUPPRESSED logits (or random)
    # NOTE: Must use float64 for softmax (matches Fast-dLLM reference).
    # float32 on large vocab (128K+) causes exp() overflow → inflated confidence
    # → too many tokens falsely pass threshold → aggressive early commitment.
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        conf = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        conf = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    # Only propose changes on masked positions
    x0 = torch.where(mask_index, x0, x)

    neg = torch.finfo(conf.dtype).min
    confidence = torch.where(
        mask_index,
        conf,
        torch.tensor(neg, device=conf.device, dtype=conf.dtype),
    )  # (B, L)

    # A) Dynamic factor schedule
    if factor is not None:
        B, L = confidence.shape
        values, idx = torch.sort(confidence, dim=1, descending=True)

        ranks = torch.arange(1, L + 1, device=confidence.device, dtype=values.dtype)
        factor_t = torch.tensor(float(factor), device=confidence.device, dtype=values.dtype)
        thr = 1.0 - (factor_t / (ranks + 1.0))
        thr[0] = -1.0

        accept = values >= thr.unsqueeze(0)
        k = accept.sum(dim=1).to(torch.long)

        n_masked = mask_index.sum(dim=1).to(torch.long)
        k = torch.minimum(k, n_masked)

        cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
        select_sorted = cols < k.unsqueeze(1)

        transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
        transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
        transfer_index = transfer_int.bool() & mask_index
        return x0, transfer_index

    # B) Threshold mode
    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)

        has_mask = mask_index.any(dim=1)
        selected = transfer_index.any(dim=1)
        need_force = has_mask & (~selected)
        if need_force.any():
            max_idx = torch.argmax(confidence, dim=1, keepdim=True)
            force = torch.zeros_like(transfer_index).scatter_(1, max_idx, True)
            transfer_index = (transfer_index | force) & mask_index

        return x0, transfer_index

    # C) Top-k (quota) mode
    if num_transfer_tokens is None:
        raise ValueError(
            "num_transfer_tokens must be provided when threshold is None and factor is None"
        )

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)

    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    values, idx = torch.sort(confidence, dim=1, descending=True)
    B, L = confidence.shape

    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    select_sorted = cols < num_transfer_tokens.unsqueeze(1)

    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index

    return x0, transfer_index


# ---------------------------------------------------------------------------
# Soft diffusion utilities
# ---------------------------------------------------------------------------


def compute_soft_embeddings(
    input_ids: torch.LongTensor,
    embed_weight: torch.Tensor,
    mask_token_id: int,
    soft_token: torch.LongTensor,
    soft_prob: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Build embeddings where ``[MASK]`` positions use a convex combination
    of the mask embedding and a *belief embedding* (probability-weighted
    token embeddings from the nucleus set).

    For each masked position (paper convention — α weights belief side):

        belief_emb = Σ p̄(v) · e_v          (over nucleus set)
        soft_emb   = (1 - α) · e_mask + α · belief_emb

    α is per-position ``[B, L]``, computed from full-vocab entropy by
    :func:`extract_soft_candidates`.  High confidence → high α → more
    belief.  Decoded (non-mask) positions keep their original embeddings.

    Args:
        input_ids: Token IDs ``[B, L]``.
        embed_weight: Embedding matrix ``[V, D]``.
        mask_token_id: Integer ID of the mask token.
        soft_token: Nucleus candidate token IDs ``[B, L, K]``.
        soft_prob: Corresponding probabilities ``[B, L, K]``.
        alpha: Per-position mixing coefficient ``[B, L]``.

    Returns:
        Embeddings ``[B, L, D]`` with soft mixtures at mask positions.
    """
    base_emb = F.embedding(input_ids, embed_weight)  # [B, L, D]
    cand_emb = F.embedding(soft_token, embed_weight)  # [B, L, K, D]
    p = soft_prob / soft_prob.sum(-1, keepdim=True).clamp(min=1e-8)  # [B, L, K]
    belief_emb = (p.unsqueeze(-1) * cand_emb).sum(dim=-2)  # [B, L, D]
    mask_emb = embed_weight[mask_token_id]  # [D]
    a = alpha.unsqueeze(-1)  # [B, L, 1]
    soft_emb = (1 - a) * mask_emb + a * belief_emb
    mask = (input_ids == mask_token_id).unsqueeze(-1)  # [B, L, 1]
    out = torch.where(mask, soft_emb, base_emb)
    return out.to(embed_weight.dtype)


def extract_soft_candidates(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    soft_top_p: float = 0.2,
    r_f: float = 0.2,
    max_nucleus: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract nucleus (top-p) candidate tokens and per-position
    entropy-gated mixing coefficient α.

    **Entropy-gated α** (computed on the full-vocab distribution):

        Ĥ = H / log|V|,   α = r_f · (1 - Ĥ)

    High confidence (low entropy) → α ≈ r_f → more belief embedding.
    High uncertainty (high entropy) → α ≈ 0 → more mask embedding.

    **Nucleus filtering** (top-p on top-``max_nucleus`` candidates):
    keeps tokens whose cumulative probability reaches ``soft_top_p``.

    Args:
        logits: Model output logits ``[B, L, V]``.
        mask_index: Boolean mask ``[B, L]`` (True = still masked).
        soft_top_p: Cumulative probability threshold for nucleus.
        r_f: Maximum mixing ratio (default 0.2).
        max_nucleus: Fixed cap on returned K dimension.

    Returns:
        ``(soft_token, soft_prob, soft_alpha)``
        — tokens ``[B, L, K]``, probs ``[B, L, K]``, alpha ``[B, L]``.
    """
    probs = F.softmax(logits, dim=-1)  # [B, L, V]

    # --- Entropy-gated alpha (full-vocab, before any truncation) ---
    log_probs = F.log_softmax(logits, dim=-1)  # numerically stable
    H = -(probs * log_probs).sum(dim=-1)  # [B, L]
    V = probs.size(-1)
    H_hat = H / math.log(V)  # normalized to [0, 1]
    soft_alpha = r_f * (1.0 - H_hat)  # [B, L], in [0, r_f]
    soft_alpha = soft_alpha.masked_fill(~mask_index, 0.0)

    # --- Nucleus (top-p) filtering ---
    K = min(max_nucleus, V)
    top_prob, top_ids = probs.topk(K, dim=-1)  # [B, L, K] sorted desc
    cumulative = top_prob.cumsum(dim=-1)
    nucleus_mask = (cumulative - top_prob) < soft_top_p  # [B, L, K]
    top_prob = top_prob * nucleus_mask.float()
    # Zero out non-mask positions
    zero_mask = ~mask_index.unsqueeze(-1)  # [B, L, 1]
    top_prob = top_prob.masked_fill(zero_mask, 0.0)
    top_ids = top_ids.masked_fill(zero_mask, 0)
    return top_ids, top_prob, soft_alpha


# ---------------------------------------------------------------------------
# Shared mixin for recurrent samplers
# ---------------------------------------------------------------------------


class RecurrentSamplerMixin:
    """Shared timestep-computation and debug-stats methods for recurrent samplers."""

    def _compute_timestep_from_mask(
        self,
        x: torch.Tensor,
        mask_token_id: int,
        maskable_region: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute timestep from masked ratio within a specific region.

        Returns the masked ratio directly as a continuous float in [0, 1].

        Args:
            x: Input tensor [B, T] with token ids
            mask_token_id: Token id for mask token
            maskable_region: [B, T] bool tensor indicating which positions
                to compute the mask ratio over.

        Returns:
            t: Timestep tensor [B] float in [0.0, 1.0]
        """
        mask_index = (x == mask_token_id)  # [B, T]

        # Count masks only within the maskable region
        masked_count = (mask_index & maskable_region).sum(dim=-1).float()  # [B]
        total_count = maskable_region.sum(dim=-1).float()  # [B]

        # Compute masked ratio within the region
        t = masked_count / (total_count + 1e-6)  # [B]
        assert (t >= 0).all() and (t <= 1).all(), (
            f"Sampler timestep out of [0, 1]: min={t.min().item():.4f}, "
            f"max={t.max().item():.4f}"
        )

        return t

    def _collect_debug_stats(
        self,
        stats_history: dict | None,
        state: torch.Tensor,
        delta_norm: torch.Tensor | None = None,
        embedding_norm: torch.Tensor | None = None,
    ) -> None:
        """
        Collect debug statistics from model outputs.

        Args:
            stats_history: Dict to append stats to, or None to skip
            state: Model state tensor [B, M, D_s]
            delta_norm: L2 norm of injector delta [B], or None
            embedding_norm: L2 norm of input embeddings [B], or None
        """
        if stats_history is None:
            return
        stats_history["state_norm"].append(state.norm(p=2, dim=-1).mean().item())
        if delta_norm is not None:
            stats_history["delta_norm"].append(delta_norm.mean().item())
        if embedding_norm is not None:
            stats_history["embedding_norm"].append(embedding_norm.mean().item())

    def _print_debug_stats_summary(
        self, stats_history: dict | None, prefix: str = ""
    ) -> None:
        """
        Print summary of collected debug statistics.

        Args:
            stats_history: Dict with collected stats, or None to skip
            prefix: Prefix for print message (e.g., "Infill")
        """
        if stats_history is None:
            return
        norms = stats_history["state_norm"]
        avg_norm = np.mean(norms) if norms else 0.0
        delta_norms = stats_history.get("delta_norm", [])
        emb_norms = stats_history.get("embedding_norm", [])
        avg_delta_norm = np.mean(delta_norms) if delta_norms else 0.0
        avg_emb_norm = np.mean(emb_norms) if emb_norms else 0.0
        label = f"[Debug{' ' + prefix if prefix else ''}]"
        print(f"{label} Avg state norm: {avg_norm:.4f}")
        if avg_delta_norm > 0 or avg_emb_norm > 0:
            ratio = avg_delta_norm / (avg_emb_norm + 1e-9)
            print(
                f"{label} Avg delta norm: {avg_delta_norm:.4f}, "
                f"Avg emb norm: {avg_emb_norm:.4f}, delta/emb ratio: {ratio:.4f}"
            )
        # State norm trajectory: detect drift/explosion
        if len(norms) >= 2:
            min_n, max_n = np.min(norms), np.max(norms)
            first_q = np.mean(norms[: max(1, len(norms) // 4)])
            last_q = np.mean(norms[-max(1, len(norms) // 4) :])
            trend = (
                "STABLE"
                if max_n < 2 * (min_n + 1e-9)
                else ("GROWING" if last_q > first_q else "SHRINKING")
            )
            print(
                f"{label} State norm trajectory ({len(norms)} steps): "
                f"min={min_n:.4f}, max={max_n:.4f}, trend={trend}"
            )
