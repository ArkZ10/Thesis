"""
Recurrent MDLM Trainer with predict-before-reveal hybrid trajectory loss.

Training Flow:
1. Initialize: Fully mask input, state=None
2. Bootstrap forward (state=None): Initialize state (no loss)
3. Pre-sample full reveal trajectory: Dirichlet K batches covering all maskable tokens
4. K Step Loop (predict-before-reveal, k=0..K-1):
   a. Forward on current masked input → logits + new_state
   b. Hybrid loss: dense CE over still-masked positions + reveal CE over tokens
      selected for teacher forcing at this step
   c. Teacher-forced reveal: unmask tokens per pre-sampled plan
5. Final loss = reduced_main_loss + state_norm_weight * state_norm
   + slot_div_weight * slot_diversity

K+1 forward passes: 1 bootstrap (state init) + K predict-before-reveal steps.
- k=0: full mask loss (cold-start signal)
- k=1..K-1: partial mask loss (progressive denoising)
- Full unmask endpoint: all maskable tokens revealed by end of trajectory

This enables the recurrent state to learn meaningful cross-step information transfer,
addressing the "Information Island" problem in discrete diffusion.

References:
- Simple and Effective Masked Diffusion Language Models: https://arxiv.org/abs/2406.07524
- Large Language Diffusion Models: https://arxiv.org/abs/2502.09992
"""

import dataclasses
import inspect
import json
import logging
import os
import shutil
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file

from dllm.core.trainers.mdlm import MDLMTrainer
from dllm.core.trainers.utils import NLLMetric, OnEvaluateMetricsCallback, PPLMetric
from dllm.utils.configs import TrainingArguments

log = logging.getLogger(__name__)


# Default backbone paths per model type
_DEFAULT_BACKBONE = {
    "recurrent_llada": "GSAI-ML/LLaDA-8B-Instruct",
}


class RecurrentMDLMTrainer(MDLMTrainer):
    """
    Trainer for Recurrent LLaDA with predict-before-reveal hybrid trajectory loss.

    Extends MDLMTrainer with:
    - Pre-sampled monotone reveal trajectory (Dirichlet-concentrated)
    - Predict-before-reveal: forward pass THEN reveal (matches inference order)
    - Hybrid loss: dense masked CE mixed with reveal-token CE
    - State regularization averaged over effective steps
    - Slot diversity regularization (Gram matrix off-diagonal penalty)
    - Hinge state norm: only penalizes when per-slot L2 norm exceeds threshold

    The recurrent modules (TimeConditioner, Mixer, Updater, Injector) are trained
    while the backbone LLaDA model remains frozen. Only ~1-5% of total parameters
    are updated during training.
    """

    @dataclass
    class RecurrentMDLMConfig(TrainingArguments):
        """Config for RecurrentMDLMTrainer. Recurrent params read from model.recurrent_config."""
        loss_norm_type: str = "token"
        right_shift_logits: bool = False
        state_norm_threshold: float = 1.0
        dense_reveal_mix: float = 0.75
        # trajectory_concentration: float = 20.0
        reveal_position_policy: str = "gt_confidence_topk"

    def __init__(self, args: Any, *pargs, **kwargs):
        transformers.Trainer.__init__(self, args=args, *pargs, **kwargs)
        self.loss_norm_type = args.loss_norm_type
        self.right_shift_logits = args.right_shift_logits
        self.meter = OnEvaluateMetricsCallback(
            trainer=self,
            splits=("train", "eval"),
            metrics={"nll": NLLMetric(), "ppl": PPLMetric()},
        )
        self.add_callback(self.meter)

    @staticmethod
    def _get_aux_loss_weights(config) -> tuple[float, float]:
        aux_weights = config.aux_loss_weights or {}
        state_norm_weight = aux_weights.get("state_norm", 0)
        slot_div_weight = aux_weights.get("slot_diversity", 0)
        return state_norm_weight, slot_div_weight

    @staticmethod
    def _compute_ground_truth_confidence(token_ce: torch.Tensor) -> torch.Tensor:
        """Convert per-token CE into ground-truth token probabilities."""
        return torch.exp(-token_ce.float())

    @staticmethod
    def _compute_model_confidence(logits: torch.Tensor) -> torch.Tensor:
        """Max predicted probability per position (model's top-1 confidence)."""
        return logits.float().softmax(dim=-1).max(dim=-1).values

    @staticmethod
    def _select_topk_masked_positions(
        scores: torch.Tensor,
        mask: torch.Tensor,
        quota: torch.Tensor,
    ) -> torch.Tensor:
        """Select the top-k scored positions within each row's masked region."""
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        selected = torch.zeros_like(mask)

        for b in range(mask.size(0)):
            k = min(int(quota[b].item()), int(mask[b].sum().item()))
            if k <= 0:
                continue
            topk = torch.topk(masked_scores[b], k=k, dim=-1).indices
            selected[b, topk] = True

        return selected

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Compute recurrent diffusion loss via predict-before-reveal trajectory.

        Algorithm:
        1. Initialize with fully masked input, state=None
        2. Bootstrap forward (state=None): init state, no loss
        3. Pre-sample full reveal trajectory: Dirichlet K batches over all maskable tokens
        4. For each step k in [0, K-1]:
           a. Forward on current masked input → logits + new_state (predict)
           b. Hybrid loss on masked/revealed positions for this trajectory step
           c. Teacher-forced reveal per pre-sampled plan
        5. Return reduced_main_loss + state_norm + slot_diversity

        K+1 forward passes: 1 bootstrap (state init) + K predict-before-reveal.

        Args:
            model: RecurrentLLaDAModel (possibly DDP wrapped).
            inputs: Dict with input_ids, labels, and optionally attention_mask.
            return_outputs: If True, return (loss, outputs) tuple.

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True.
        """
        # === Input Validation & Preprocessing ===
        assert self.processing_class.padding_side == "right", \
            "Recurrent trainer requires right-padding for proper mask handling"

        inputs = self._preprocess_inputs(inputs)
        input_ids = inputs["input_ids"]  # [B, L] - ground truth tokens
        labels = inputs["labels"]  # [B, L] - same as input_ids, -100 for non-maskable
        recurrent_attention_mask = inputs.get("attention_mask")  # [B, L] or None

        device = input_ids.device
        B, L = input_ids.shape

        # Maskable positions: where labels != -100 (i.e., not padding/prompt)
        maskable_mask = labels != -100  # [B, L] bool
        num_maskable = maskable_mask.sum(-1)  # [B]

        # === Get Recurrent Config (handle DDP wrapper) ===
        unwrapped = self._unwrap_model(model)
        config = unwrapped.recurrent_config
        K = config.unroll_steps
        state_norm_weight, slot_div_weight = self._get_aux_loss_weights(config)

        # Variable-K: sample K_total with linear curriculum
        K_max_target = getattr(config, "unroll_steps_max", None)
        G = K
        warmup_k_frac = getattr(config, "warmup_k_fraction", 0.5)
        K_total = self._sample_k_total(K, K_max_target, warmup_k_frac)
        W = max(0, K_total - G)  # warmup steps (no gradient, no loss)

        concentration = K_total
        dense_reveal_mix = getattr(self.args, "dense_reveal_mix", 0.75)
        if not 0.0 <= dense_reveal_mix <= 1.0:
            raise ValueError(
                f"dense_reveal_mix must be in [0, 1], got {dense_reveal_mix}"
            )

        reveal_position_policy = getattr(self.args, "reveal_position_policy", "random")
        if reveal_position_policy not in {"random", "gt_confidence_topk", "model_confidence_topk"}:
            raise ValueError(
                f"Invalid reveal_position_policy: {reveal_position_policy}"
            )
        # === Initialize: Fully Masked Input ===
        mask_token_id = self.processing_class.mask_token_id
        current_noised = input_ids.clone()
        current_noised[maskable_mask] = mask_token_id

        # Track which positions are still masked (updated each step)
        still_masked = maskable_mask.clone()  # [B, L] bool

        # === Initialize State (Bootstrap, No Loss) ===
        bootstrap_out = model(
            input_ids=current_noised,
            t=torch.ones(B, device=device),  # 1.0 = fully masked
            state=None,
            attention_mask=recurrent_attention_mask,
            return_dict=True,
            use_cache=False,
        )
        state = bootstrap_out.new_state
        past_cond = getattr(bootstrap_out, "past_cond", None)

        # === Pre-sample full reveal trajectory ===
        # Full unmask: reveal all maskable tokens across K steps
        num_to_reveal = num_maskable

        # reveal_counts[b, k] = number of tokens to reveal after step k
        reveal_counts = self._sample_positive_reveal_counts(
            num_to_reveal, K_total, device, concentration
        )  # [B, K_total]
        if reveal_position_policy == "random":
            # reveal_rank[b, l] = rank of position l among maskable positions (random order)
            reveal_rank = self._sample_random_reveal_rank(maskable_mask, device)  # [B, L]
            # cumulative_reveal[b, k] = total tokens revealed after step k
            cumulative_reveal = reveal_counts.cumsum(dim=-1)  # [B, K_total]
        else:
            reveal_rank = None
            cumulative_reveal = None

        # === Accumulators ===
        valid_sample = num_maskable > 0  # [B] bool
        per_sample_main_loss = torch.zeros(B, device=device)
        per_sample_main_weight = torch.zeros(B, device=device)
        per_sample_effective_steps = torch.zeros(B, device=device, dtype=torch.long)
        per_sample_state_reg = torch.zeros(B, device=device)
        per_sample_slot_div = torch.zeros(B, device=device)

        # === Training Diagnostics ===
        debug = config.debug_stats and model.training
        diag: dict | None = {} if debug else None
        if diag is not None:
            diag["K_total"] = K_total
            diag["warmup_steps"] = W

        # === Phase A: Warmup (no gradient, no loss) ===
        outputs = None  # Guard: needed if loop breaks on first iteration
        for k in range(W):
            active = still_masked.any(dim=-1) & valid_sample
            if not active.any():
                break

            pred_count = still_masked.sum(-1).float()
            pred_ratio = pred_count / (num_maskable.float() + 1e-6)
            t = self._ratio_to_t(pred_ratio)

            with torch.no_grad():
                outputs = model(
                    input_ids=current_noised,
                    t=t,
                    state=state,
                    past_cond=past_cond,
                    attention_mask=recurrent_attention_mask,
                    return_dict=True,
                    use_cache=False,
                )
                state = outputs.new_state
                past_cond = getattr(outputs, "past_cond", None)

            # Teacher-forced reveal (advance mask pattern)
            if reveal_position_policy == "random":
                reveal_start = (
                    cumulative_reveal[:, k - 1]
                    if k > 0
                    else torch.zeros(B, device=device, dtype=torch.long)
                )
                reveal_end = cumulative_reveal[:, k]
                to_unmask = (
                    (reveal_rank >= reveal_start.unsqueeze(1))
                    & (reveal_rank < reveal_end.unsqueeze(1))
                    & maskable_mask
                )
            elif reveal_position_policy == "gt_confidence_topk":
                outputs = self._postprocess_outputs(outputs)
                token_ce = F.cross_entropy(
                    outputs.logits.transpose(1, 2), input_ids, reduction="none"
                ).float()
                gt_conf = self._compute_ground_truth_confidence(token_ce).detach()
                to_unmask = self._select_topk_masked_positions(
                    gt_conf, still_masked, reveal_counts[:, k]
                )
            else:  # model_confidence_topk
                outputs = self._postprocess_outputs(outputs)
                model_conf = self._compute_model_confidence(outputs.logits).detach()
                to_unmask = self._select_topk_masked_positions(
                    model_conf, still_masked, reveal_counts[:, k]
                )

            current_noised = torch.where(to_unmask, input_ids, current_noised)
            still_masked = still_masked & ~to_unmask

        # === Cut gradient chain ===
        if W > 0:
            state = state.detach()
            if past_cond is not None:
                past_cond = past_cond.detach()

        # === Phase B: Training (full gradient, loss) ===
        for k in range(W, K_total):
            # 5.1 Check active samples
            active = still_masked.any(dim=-1) & valid_sample  # [B]
            if not active.any():
                break

            # 5.2 pred_mask = still_masked (positions to predict BEFORE reveal)
            pred_mask = still_masked.clone()  # [B, L]

            # 5.3 t_idx from actual masked ratio
            pred_count = pred_mask.sum(-1).float()  # [B]
            pred_ratio = pred_count / (num_maskable.float() + 1e-6)
            assert (pred_ratio >= 0).all() and (pred_ratio <= 1).all(), (
                f"pred_ratio out of [0, 1]: min={pred_ratio.min().item():.4f}, "
                f"max={pred_ratio.max().item():.4f}"
            )
            t = self._ratio_to_t(pred_ratio)  # [B]

            # 5.4 Single forward pass
            outputs = model(
                input_ids=current_noised,
                t=t,
                state=state,
                past_cond=past_cond,
                attention_mask=recurrent_attention_mask,
                return_dict=True,
                use_cache=False,
            )
            state = outputs.new_state
            past_cond = getattr(outputs, "past_cond", None)

            # 5.5 Postprocess
            outputs = self._postprocess_outputs(outputs)
            logits = outputs.logits  # [B, L, V]

            # 5.6 Token CE on all positions
            token_ce = F.cross_entropy(
                logits.transpose(1, 2),  # [B, V, L]
                input_ids,  # [B, L]
                reduction="none",  # [B, L]
            ).float()  # Cast to float32 for numerical stability under bf16

            # 5.7 Determine reveal set
            if reveal_position_policy == "random":
                reveal_start = (
                    cumulative_reveal[:, k - 1]
                    if k > 0
                    else torch.zeros(B, device=device, dtype=torch.long)
                )
                reveal_end = cumulative_reveal[:, k]  # [B]
                # to_unmask: positions whose rank falls in [reveal_start, reveal_end)
                to_unmask = (
                    (reveal_rank >= reveal_start.unsqueeze(1))
                    & (reveal_rank < reveal_end.unsqueeze(1))
                    & maskable_mask
                )  # [B, L]
            elif reveal_position_policy == "gt_confidence_topk":
                gt_conf = self._compute_ground_truth_confidence(token_ce).detach()
                to_unmask = self._select_topk_masked_positions(
                    gt_conf,
                    still_masked,
                    reveal_counts[:, k],
                )
            else:  # model_confidence_topk
                model_conf = self._compute_model_confidence(logits).detach()
                to_unmask = self._select_topk_masked_positions(
                    model_conf,
                    still_masked,
                    reveal_counts[:, k],
                )

            # 5.8 Hybrid loss: dense masked CE + reveal CE with reveal-proportional weight
            pred_mask_float = pred_mask.float()  # [B, L]
            to_unmask_float = to_unmask.float()  # [B, L]

            dense_sum = (token_ce * pred_mask_float).sum(-1)  # [B]
            dense_mean = dense_sum / pred_count.clamp(min=1)  # [B]

            reveal_sum = (token_ce * to_unmask_float).sum(-1)  # [B]
            reveal_count = to_unmask.sum(-1).float().clamp(min=1)  # [B]
            reveal_mean = reveal_sum / reveal_count  # [B]

            step_weight = reveal_counts[:, k].float() / (num_maskable.float() + 1e-6)
            step_loss_per_sample = step_weight * (
                dense_reveal_mix * dense_mean + (1 - dense_reveal_mix) * reveal_mean
            )
            step_loss_per_sample = step_loss_per_sample * active.float()

            # 5.9 Update meter (count each token exactly once, at its reveal step)
            self.meter.update(
                split="train" if model.training else "eval",
                value=(token_ce * to_unmask_float).detach(),
                weight=to_unmask_float.detach(),
            )

            # Accumulate
            per_sample_main_loss = per_sample_main_loss + step_loss_per_sample
            per_sample_main_weight = per_sample_main_weight + (
                step_weight * active.float()
            )
            per_sample_effective_steps = per_sample_effective_steps + active.long()

            # 5.10 State regularization (hinge: only penalize above threshold)
            if state_norm_weight > 0 and state is not None:
                slot_norms = state.float().norm(p=2, dim=-1)  # [B, M]
                excess = (slot_norms - self.args.state_norm_threshold).clamp(min=0)
                state_reg = excess.pow(2).mean(dim=-1)  # [B]
                per_sample_state_reg = per_sample_state_reg + state_reg * active.float()

            # 5.10b Slot diversity regularization
            if slot_div_weight > 0 and state is not None:
                s_norm = F.normalize(state.float(), dim=-1)  # [B, M, D_s]
                gram = s_norm @ s_norm.transpose(-1, -2)  # [B, M, M]
                M = state.size(1)
                eye = torch.eye(M, device=device).unsqueeze(0)  # [1, M, M]
                slot_div = (gram - eye).pow(2).sum(dim=(-2, -1)) / (M * M)  # [B]
                per_sample_slot_div = per_sample_slot_div + slot_div * active.float()

            # 5.11 Diagnostics
            if diag is not None:
                # Ratio-binned dense CE for monitoring denoising quality
                bins = [(0.75, 1.01, "high"), (0.50, 0.75, "mid_high"),
                        (0.25, 0.50, "mid_low"), (0.00, 0.25, "low")]
                for lo, hi, label in bins:
                    in_bin = (pred_ratio >= lo) & (pred_ratio < hi)
                    if in_bin.any():
                        diag[f"step_{k}/dense_ce_{label}"] = (
                            dense_mean[in_bin].mean().item()
                        )
                if active.any():
                    diag[f"step_{k}/reveal_ce"] = reveal_mean[active].mean().item()
                    diag[f"step_{k}/step_weight"] = step_weight[active].mean().item()
                self._collect_step_diagnostics(
                    diag, k, step_loss_per_sample, state, still_masked,
                    outputs, unwrapped, pred_ratio, to_unmask,
                )

            # 5.12 Teacher-forced reveal
            current_noised = torch.where(to_unmask, input_ids, current_noised)
            still_masked = still_masked & ~to_unmask

        # Guard: degenerate batch (no maskable tokens anywhere)
        if outputs is None:
            outputs = model(
                input_ids=current_noised,
                t=torch.zeros(B, device=device),
                state=state,
                past_cond=past_cond,
                attention_mask=recurrent_attention_mask,
                return_dict=True,
                use_cache=False,
            )
            outputs = self._postprocess_outputs(outputs)

        # === 6. Normalize by loss-bearing step-weight sum and reduce ===
        main_weight = per_sample_main_weight.clamp(min=1e-6)
        per_sample_main_loss = per_sample_main_loss / main_weight
        total_loss = self._reduce_trajectory_loss(
            per_sample_main_loss, num_maskable, valid_sample, B
        )

        # === 7. State norm: averaged over effective steps ===
        avg_state_reg = torch.tensor(0.0, device=device)
        if state_norm_weight > 0:
            eff_steps = per_sample_effective_steps.float().clamp(min=1)
            avg_state_reg = (per_sample_state_reg / eff_steps)
            avg_state_reg = avg_state_reg[valid_sample].mean() if valid_sample.any() else torch.tensor(0.0, device=device)
            total_loss = total_loss + state_norm_weight * avg_state_reg

        # === 7b. Slot diversity: averaged over effective steps ===
        avg_slot_div = torch.tensor(0.0, device=device)
        if slot_div_weight > 0:
            eff_steps = per_sample_effective_steps.float().clamp(min=1)
            avg_slot_div = (per_sample_slot_div / eff_steps)
            avg_slot_div = avg_slot_div[valid_sample].mean() if valid_sample.any() else torch.tensor(0.0, device=device)
            total_loss = total_loss + slot_div_weight * avg_slot_div

        # === 8. Log diagnostics ===
        if diag is not None:
            self._log_training_diagnostics(
                diag, unwrapped, state_norm_weight,
                avg_state_reg,
                per_sample_effective_steps,
                slot_div_weight=slot_div_weight,
                avg_slot_div=avg_slot_div,
            )

        return (total_loss, outputs) if return_outputs else total_loss

    # =========================================================================
    # Trajectory Sampling Helpers
    # =========================================================================

    @torch.no_grad()
    def _sample_positive_reveal_counts(
        self,
        num_maskable: torch.Tensor,  # [B]
        num_steps: int,
        device: torch.device,
        concentration: float,
    ) -> torch.Tensor:
        """
        Sample per-step reveal counts for a monotone trajectory.

        Each sample gets K_eff = min(K, N_b) active steps with ≥1 token each.
        Remaining N_b - K_eff tokens are allocated via Dirichlet-Multinomial.

        Args:
            num_maskable: [B] number of maskable tokens per sample.
            num_steps: K, maximum number of steps.
            device: Target device.
            concentration: Dirichlet concentration parameter.

        Returns:
            [B, K] long tensor of reveal counts per step.
        """
        B = num_maskable.size(0)
        counts = torch.zeros(B, num_steps, device=device, dtype=torch.long)

        for b in range(B):
            N_b = num_maskable[b].item()
            if N_b <= 0:
                continue

            K_eff = min(num_steps, N_b)
            # Guarantee ≥1 token per active step
            counts[b, :K_eff] = 1
            remaining = N_b - K_eff

            if remaining > 0 and K_eff > 0:
                # Dirichlet-Multinomial allocation of remaining tokens
                base = torch.ones(K_eff, device=device) / K_eff
                alpha = concentration * base
                dirichlet = torch.distributions.Dirichlet(alpha)
                probs = dirichlet.sample()  # [K_eff]
                extra = torch.multinomial(probs, remaining, replacement=True)  # [remaining]
                counts[b, :K_eff] += torch.bincount(extra, minlength=K_eff)

        return counts

    @torch.no_grad()
    def _sample_random_reveal_rank(
        self,
        maskable_mask: torch.Tensor,  # [B, L]
        device: torch.device,
    ) -> torch.Tensor:
        """
        Assign a random reveal rank to each maskable position.

        Non-maskable positions get a sentinel rank (L, effectively infinite).

        Args:
            maskable_mask: [B, L] bool mask of maskable positions.
            device: Target device.

        Returns:
            [B, L] long tensor of ranks in [0, N_b-1] for maskable, L for non-maskable.
        """
        B, L = maskable_mask.shape
        # Random scores; non-maskable get -inf so they sort last
        scores = torch.rand(B, L, device=device)
        scores[~maskable_mask] = -1.0

        # argsort gives indices that sort scores descending (highest first)
        # We want random ordering among maskable positions
        sorted_indices = scores.argsort(dim=-1, descending=True)  # [B, L]

        # Scatter to get rank
        ranks = torch.zeros_like(sorted_indices)
        ranks.scatter_(1, sorted_indices, torch.arange(L, device=device).unsqueeze(0).expand(B, -1))

        # Sentinel for non-maskable positions
        ranks[~maskable_mask] = L

        return ranks

    @staticmethod
    def _ratio_to_t(masked_ratio: torch.Tensor) -> torch.Tensor:
        """Convert masked ratio to continuous timestep for time conditioning.

        Note:
            Recurrent training uses the actual masked ratio directly.
        """
        return masked_ratio.float()

    def _sample_k_total(
        self,
        K_base: int,
        K_max_target: int | None,
        warmup_k_frac: float,
    ) -> int:
        """Sample K_total for this batch with linear curriculum.

        When disabled (K_max_target is None), returns K_base (fixed K).
        Otherwise ramps effective_K_max from K_base to K_max_target over
        warmup_k_frac of training, sampling K_total ~ Uniform[1, effective_K_max].
        """
        if K_max_target is None or K_max_target <= K_base:
            return K_base  # disabled, fixed K

        progress = self.state.global_step / max(1, self.state.max_steps)
        ramp = min(1.0, progress / warmup_k_frac) if warmup_k_frac > 0 else 1.0
        effective_K_max = K_base + int((K_max_target - K_base) * ramp)

        # K_total ~ Uniform[1, effective_K_max]
        return torch.randint(1, effective_K_max + 1, (1,)).item()

    def _reduce_trajectory_loss(
        self,
        per_sample_loss: torch.Tensor,  # [B]
        num_maskable: torch.Tensor,  # [B]
        valid_sample: torch.Tensor,  # [B] bool
        batch_size: int,
    ) -> torch.Tensor:
        """
        Reduce per-sample trajectory loss to a scalar.

        Args:
            per_sample_loss: [B] accumulated loss per sample.
            num_maskable: [B] number of maskable tokens per sample.
            valid_sample: [B] bool mask of valid samples.
            batch_size: B.

        Returns:
            Scalar loss.
        """
        if self.loss_norm_type == "token":
            # Weight each sample by num_maskable, normalize by total
            weights = num_maskable.float() * valid_sample.float()
            total_weight = weights.sum().clamp(min=1)
            return (per_sample_loss * weights).sum() / total_weight
        elif self.loss_norm_type == "sequence":
            if valid_sample.any():
                return per_sample_loss[valid_sample].mean()
            return torch.tensor(0.0, device=per_sample_loss.device)
        elif self.loss_norm_type == "batch":
            return per_sample_loss.sum() / batch_size
        else:
            raise ValueError(f"Invalid loss_norm_type: {self.loss_norm_type}")

    # =========================================================================
    # Diagnostics
    # =========================================================================

    @torch.no_grad()
    def _collect_step_diagnostics(
        self,
        diag: dict,
        k: int,
        step_loss_per_sample: torch.Tensor,
        state: torch.Tensor | None,
        still_masked: torch.Tensor,
        outputs: Any,
        unwrapped: nn.Module,
        pred_ratio: torch.Tensor,
        to_unmask: torch.Tensor,
    ) -> None:
        """Collect per-step diagnostic metrics for monitoring recurrent module health."""
        prefix = f"step_{k}"
        s_norm = state.norm(p=2, dim=-1).mean().item() if state is not None else 0.0

        diag[f"{prefix}/loss"] = step_loss_per_sample.mean().item()
        diag[f"{prefix}/state_norm"] = s_norm
        diag[f"{prefix}/still_masked_frac"] = still_masked.float().mean().item()
        diag[f"{prefix}/pred_ratio"] = pred_ratio.mean().item()
        diag[f"{prefix}/reveal_count"] = to_unmask.sum(-1).float().mean().item()

        if k > 0:
            prev_norm = diag.get(f"step_{k-1}/state_norm", 0.0)
            if prev_norm > 1e-9:
                diag[f"{prefix}/state_delta_ratio"] = abs(s_norm - prev_norm) / prev_norm

        delta_norm = getattr(outputs, "delta_norm", None)
        if delta_norm is not None:
            diag[f"{prefix}/delta_norm"] = delta_norm.mean().item()
            embedding_norm = getattr(outputs, "embedding_norm", None)
            if embedding_norm is not None:
                diag[f"{prefix}/injection_ratio"] = (
                    delta_norm / (embedding_norm + 1e-9)
                ).mean().item()

        updater = getattr(unwrapped, "updater", None)
        if updater is not None and hasattr(updater, "_debug_z_gate_mean"):
            diag[f"{prefix}/updater_gate_mean"] = updater._debug_z_gate_mean

        cs = getattr(outputs, "past_cond", None)
        if cs is not None:
            diag[f"{prefix}/past_cond_norm"] = cs.detach().norm(p=2, dim=-1).mean().item()

    @torch.no_grad()
    def _log_training_diagnostics(
        self,
        diag: dict,
        unwrapped: nn.Module,
        state_norm_weight: float,
        avg_state_reg: torch.Tensor,
        effective_steps: torch.Tensor,
        slot_div_weight: float = 0.0,
        avg_slot_div: torch.Tensor | None = None,
    ) -> None:
        """
        Log training diagnostics for monitoring recurrent module health.

        Covers:
        - Per-module gradient norms (verifies gradient flow through zero-bridge)
        - State norm trajectory (catches drift/explosion)
        - Per-step loss decomposition (detects early-step dominance)
        - Mixer weight norm (detects frozen mixer)
        - Bridge weight evolution (detects stalled zero-bridge)
        """
        logs = {}

        # --- Per-step metrics ---
        for key, val in diag.items():
            logs[f"recurrent/{key}"] = val

        # Step loss ratio (k=0 vs k=1)
        if "step_0/loss" in diag and "step_1/loss" in diag and diag["step_1/loss"] > 1e-9:
            logs["recurrent/loss_ratio_k0_k1"] = diag["step_0/loss"] / diag["step_1/loss"]

        # Effective steps
        logs["recurrent/effective_steps_mean"] = effective_steps.float().mean().item()

        # --- Module weight norms ---
        mixer = getattr(unwrapped, "mixer", None)
        if mixer is not None:
            mixer_weight = None
            if hasattr(mixer, "projection"):
                mixer_weight = mixer.projection.weight
            elif hasattr(mixer, "o_proj"):
                mixer_weight = mixer.o_proj.weight
            elif hasattr(mixer, "q_proj"):
                mixer_weight = mixer.q_proj.weight
            if mixer_weight is not None:
                logs["recurrent/mixer_weight_norm"] = mixer_weight.data.float().norm().item()

        injector = getattr(unwrapped, "injector", None)
        if injector is not None:
            if hasattr(injector, "up_proj"):
                logs["recurrent/up_proj_weight_norm"] = injector.up_proj.weight.data.float().norm().item()
            if hasattr(injector, "norm_out"):
                logs["recurrent/norm_out_modulation_norm"] = (
                    injector.norm_out.modulation.weight.data.float().norm().item()
                )

        updater = getattr(unwrapped, "updater", None)
        if updater is not None:
            if hasattr(updater, "gate_proj"):
                logs["recurrent/updater_gate_weight_norm"] = updater.gate_proj.weight.data.float().norm().item()
            if hasattr(updater, "cand_proj"):
                logs["recurrent/updater_cand_weight_norm"] = updater.cand_proj.weight.data.float().norm().item()

        time_conditioner = getattr(unwrapped, "time_conditioner", None)
        if time_conditioner is not None and hasattr(time_conditioner, "mlp"):
            logs["recurrent/time_cond_weight_norm"] = time_conditioner.mlp[0].weight.data.float().norm().item()
            if hasattr(time_conditioner, "content_gate"):
                logs["recurrent/content_gate_norm"] = time_conditioner.content_gate.data.float().norm().item()

        # State norm loss component
        if state_norm_weight > 0:
            logs["recurrent/state_norm_loss"] = avg_state_reg.item()

        # Slot diversity loss component
        if slot_div_weight > 0 and avg_slot_div is not None:
            logs["recurrent/slot_diversity_loss"] = avg_slot_div.item()

        # --- Per-module gradient norms ---
        mixer_grad_mod = None
        if mixer is not None:
            if hasattr(mixer, "o_proj"):
                mixer_grad_mod = mixer.o_proj
            elif hasattr(mixer, "q_proj"):
                mixer_grad_mod = mixer.q_proj

        grad_targets = {
            "injector.up_proj": getattr(injector, "up_proj", None),
            "injector.q_proj": getattr(injector, "q_proj", None),
            "mixer.o_proj": mixer_grad_mod,
        }
        if updater is not None and hasattr(updater, "gate_proj"):
            grad_targets["updater.gate_proj"] = updater.gate_proj
        if updater is not None and hasattr(updater, "cand_proj"):
            grad_targets["updater.cand_proj"] = updater.cand_proj

        for name, mod in grad_targets.items():
            if mod is not None and hasattr(mod, "weight") and mod.weight.grad is not None:
                logs[f"recurrent/grad_{name}"] = mod.weight.grad.data.float().norm().item()

        # Log via HF trainer (shows up in wandb/tensorboard)
        try:
            self.log(logs)
        except Exception as e:
            log.warning("Failed to log training diagnostics: %s", e)
            for k, v in logs.items():
                log.debug(f"  {k}: {v:.6f}")

    # =========================================================================
    # Lightweight Checkpoint: Save Only Recurrent Modules (~100-500MB vs ~16GB)
    # =========================================================================
    #
    # Recurrent checkpoints save only the trainable recurrent modules:
    #   - time_conditioner.*, mixer.*, updater.*, injector.*
    #
    # To reconstruct the full model:
    #   1. Load backbone from backbone_name_or_path (stored in metadata)
    #   2. Initialize recurrent modules with recurrent_config
    #   3. Load recurrent weights from recurrent_modules.safetensors
    #
    # This enables efficient checkpoint storage and faster loading.
    # =========================================================================

    RECURRENT_META_FILENAME = "recurrent_checkpoint_meta.json"
    RECURRENT_WEIGHTS_FILENAME = "recurrent_modules.safetensors"

    # Module names to save (order matters for consistent serialization)
    RECURRENT_MODULE_NAMES = (
        "time_conditioner", "mixer", "updater", "injector",
    )

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap DDP/FSDP wrapped model to access attributes."""
        return model.module if hasattr(model, "module") else model

    def _detect_model_type(self, unwrapped: nn.Module) -> str:
        """Detect the supported recurrent model type."""
        config = getattr(unwrapped, "config", None)
        raw_model_type = getattr(config, "model_type", "recurrent_llada") if config else "recurrent_llada"
        if raw_model_type != "recurrent_llada":
            raise ValueError(f"Only recurrent_llada checkpoints are supported, got {raw_model_type!r}")
        return "recurrent_llada"

    def _get_recurrent_modules_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """
        Extract state dict containing only recurrent module weights.

        Args:
            model: RecurrentLLaDAModel (possibly DDP/FSDP wrapped).

        Returns:
            State dict with keys like "time_conditioner.mlp.0.weight", "mixer.q_proj.weight", etc.
            All tensors are moved to CPU for serialization without dtype coercion.
        """
        unwrapped = self._unwrap_model(model)

        state_dict = {}
        for name in self.RECURRENT_MODULE_NAMES:
            module = getattr(unwrapped, name, None)
            if module is not None:
                for param_name, param in module.state_dict().items():
                    state_dict[f"{name}.{param_name}"] = param.cpu()

        return state_dict

    def _get_recurrent_checkpoint_metadata(self, model: nn.Module) -> dict[str, Any]:
        """
        Build metadata for reconstructing full model from lightweight checkpoint.

        The metadata includes:
        - backbone_name_or_path: HuggingFace model ID or local path to backbone
        - recurrent_config: All recurrent module configuration
        - model_type: "recurrent_llada" for auto-loading

        Args:
            model: RecurrentLLaDAModel (possibly DDP/FSDP wrapped).

        Returns:
            Metadata dict for JSON serialization.
        """
        unwrapped = self._unwrap_model(model)
        model_type = self._detect_model_type(unwrapped)
        config = getattr(unwrapped, "config", None)

        # Resolve backbone path (priority: args > config > default)
        backbone_path = getattr(self.args, "backbone_name_or_path", None)
        if backbone_path is None and config is not None:
            backbone_path = getattr(config, "backbone_name_or_path", None)
        if backbone_path is None:
            backbone_path = _DEFAULT_BACKBONE.get(model_type, _DEFAULT_BACKBONE["recurrent_llada"])

        # Serialize recurrent config (only declared dataclass fields)
        recurrent_config = {}
        if hasattr(unwrapped, "recurrent_config"):
            rc = unwrapped.recurrent_config
            for f in dataclasses.fields(rc):
                val = getattr(rc, f.name)
                if isinstance(val, (str, int, float, bool, type(None), dict, list)):
                    recurrent_config[f.name] = val
                else:
                    log.warning(
                        "Skipping non-serializable config field '%s': %s",
                        f.name,
                        type(val).__name__,
                    )

        return {
            "is_lightweight_checkpoint": True,
            "backbone_name_or_path": backbone_path,
            "recurrent_config": recurrent_config,
            "model_type": model_type,
        }

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None):
        """
        Save lightweight checkpoint with only recurrent module weights.

        Saved files:
        - recurrent_modules.safetensors: TimeConditioner, Mixer, Updater, Injector weights
        - recurrent_checkpoint_meta.json: Backbone path and recurrent config
        - config.json: Full model config with auto_map for trust_remote_code
        - modeling_recurrent_llada.py: Model code for trust_remote_code loading
        - recurrent_llada_config.py: Config code for trust_remote_code loading
        - tokenizer files: For inference convenience

        Args:
            output_dir: Checkpoint directory (defaults to self.args.output_dir).
            state_dict: Ignored (we extract recurrent modules directly).
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.model
        unwrapped = self._unwrap_model(model)

        # 1. Save recurrent module weights (~100-500MB)
        recurrent_state_dict = self._get_recurrent_modules_state_dict(model)
        weights_path = os.path.join(output_dir, self.RECURRENT_WEIGHTS_FILENAME)
        save_file(recurrent_state_dict, weights_path)

        # 2. Save reconstruction metadata
        metadata = self._get_recurrent_checkpoint_metadata(model)
        meta_path = os.path.join(output_dir, self.RECURRENT_META_FILENAME)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # 3. Save config.json with backbone path
        if hasattr(unwrapped, "config"):
            unwrapped.config.backbone_name_or_path = metadata["backbone_name_or_path"]
            unwrapped.config.save_pretrained(output_dir)

        # 4. Copy trust_remote_code files for HuggingFace auto-loading
        self._save_trust_files(output_dir)

        # 5. Save tokenizer for inference convenience
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def _save_trust_files(self, save_dir: str) -> None:
        """
        Copy recurrent module source files for trust_remote_code loading.

        This enables loading checkpoints with:
            AutoModel.from_pretrained(path, trust_remote_code=True)

        Updates config.json with auto_map pointing to local Python files.
        """
        try:
            unwrapped = self._unwrap_model(self.model)
            model_type = self._detect_model_type(unwrapped)
            import dllm.pipelines.recurrent_llada.configs.recurrent_llada_config as config_mod
            import dllm.pipelines.recurrent_llada.models.modeling_recurrent_llada as modeling_mod

            config_filename = "recurrent_llada_config.py"
            modeling_filename = "modeling_recurrent_llada.py"
            config_class = "recurrent_llada_config.RecurrentLLaDAConfig"
            model_class = "modeling_recurrent_llada.RecurrentLLaDAModel"

            # Copy source files
            shutil.copy2(
                inspect.getfile(modeling_mod),
                os.path.join(save_dir, modeling_filename),
            )
            shutil.copy2(
                inspect.getfile(config_mod),
                os.path.join(save_dir, config_filename),
            )

            # Update config.json with auto_map
            cfg_path = os.path.join(save_dir, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)

                cfg["model_type"] = model_type
                cfg["auto_map"] = {
                    "AutoConfig": config_class,
                    "AutoModel": model_class,
                    "AutoModelForMaskedLM": model_class,
                }

                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)

        except Exception as e:
            warnings.warn(f"Could not save trust_remote_code files: {e}")

    # =========================================================================
    # Checkpoint Detection Utilities
    # =========================================================================

    @classmethod
    def is_lightweight_checkpoint(cls, checkpoint_path: str) -> bool:
        """
        Check if a checkpoint is a lightweight recurrent checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.

        Returns:
            True if recurrent_checkpoint_meta.json exists.
        """
        meta_path = os.path.join(checkpoint_path, cls.RECURRENT_META_FILENAME)
        return os.path.exists(meta_path)

    @classmethod
    def load_recurrent_checkpoint_metadata(cls, checkpoint_path: str) -> dict[str, Any] | None:
        """
        Load metadata from a lightweight checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.

        Returns:
            Metadata dict if exists, None otherwise.
        """
        meta_path = os.path.join(checkpoint_path, cls.RECURRENT_META_FILENAME)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)
        return None
