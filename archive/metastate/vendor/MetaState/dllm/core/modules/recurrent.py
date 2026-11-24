"""
Shared Recurrent Modules for Memory-Slot Architecture.

This module implements the three-module recurrent pipeline used by
Recurrent LLaDA. The architecture uses fixed-size memory slots [B, M, D_s]
to decouple state from sequence length.

Forward Flow (at diffusion step k):
    input_ids → embeddings
                    ↓
        IF past_cond is not None:
            injector_cond = past_cond                                      (Direct passthrough)
        ELSE:
            injector_cond = zeros                                          (Bootstrap: no content yet)
                    ↓
        IF state is None:
            state = Updater.get_initial_state(B)  (learnable [B, M, D_s])
                    ↓
        Injector(embeddings, state, injector_cond) → embeddings', sa_kv_cache
                    ↓
        Backbone(embeddings') → hidden_states, logits
                    ↓
        Mixer.prepare(hidden[-1]) → content_summary  (NEW content from backbone)
        cond = TimeConditioner(ref, t, content_summary)  (For Mixer + Updater)
        Mixer.process(state, h_proj, cond) → context
                    ↓
        Updater(state, context, cond) → new_state
                    ↓
        Output: cache cond for next step's Injector

Modules:
    - SharedTimeConditioner: Sinusoidal → MLP, called twice per step (Injector + Mixer/Updater)
    - RecurrentMixer: CrossAttn(Q=state[B,M,Ds], KV=hidden[B,N,D]) → context
    - RecurrentUpdater: GRU-style gated update of state using context
    - RecurrentInjector: CrossAttn(Q=embed[B,N,D], KV=state[B,M,Ds]) + zero-bridge

Key Design Choices:
    - AdaRMSNorm (FiLM/AdaLN-Zero) on all norms for per-channel time modulation
    - Single shared sinusoidal time conditioning
    - Fixed memory slots M (default 64), decoupled from sequence length
    - Zero-bridge injection preserved for stable training
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RMSNorm",
    "AdaRMSNorm",
    "SwiGLUMLP",
    "SharedTimeConditioner",
    "RecurrentMixer",
    "RecurrentUpdater",
    "RecurrentInjector",
    "RecurrentModuleConfig",
    "create_recurrent_module_config",
    "compute_kv_replace_offset",
    "_compute_num_heads",
]

log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
SINUSOIDAL_TIME_EMBED_DIM = 256
_TIME_SCALE = 1000.0


# =============================================================================
# Utilities
# =============================================================================


def compute_kv_replace_offset(
    replace_position: torch.Tensor,
) -> int | None:
    """
    Derive the KV-replace offset from a ``replace_position`` boolean mask.

    In dual-cache mode every row must have the same contiguous block of
    ``True`` values.  This function returns the index of the first ``True``
    in row 0 and validates that all other rows agree.

    Args:
        replace_position: Boolean tensor [B, L].

    Returns:
        The offset (int) or ``None`` if no positions are marked.

    Raises:
        ValueError: If rows do not share the same non-empty contiguous block.
    """
    if replace_position.dim() != 2:
        raise ValueError(
            f"replace_position must have shape [B, L], got {tuple(replace_position.shape)}"
        )

    replace_position = replace_position.to(dtype=torch.bool)
    first_row_nz = replace_position[0].nonzero(as_tuple=True)[0]
    if first_row_nz.numel() == 0:
        if replace_position.any():
            raise ValueError(
                "replace_position has mixed empty/non-empty rows; dual-cache "
                "replacement requires all rows to share the same block"
            )
        return None

    offset = first_row_nz[0].item()
    block_len = first_row_nz.numel()
    expected = torch.arange(
        offset,
        offset + block_len,
        device=replace_position.device,
        dtype=first_row_nz.dtype,
    )
    if not torch.equal(first_row_nz, expected):
        raise ValueError(
            "replace_position row 0 must contain a contiguous True block"
        )

    for bi in range(1, replace_position.size(0)):
        nz_bi = replace_position[bi].nonzero(as_tuple=True)[0]
        if nz_bi.numel() == 0:
            raise ValueError(
                "replace_position has mixed empty/non-empty rows; dual-cache "
                "replacement requires all rows to share the same block"
            )
        expected_bi = torch.arange(
            nz_bi[0].item(),
            nz_bi[0].item() + nz_bi.numel(),
            device=replace_position.device,
            dtype=nz_bi.dtype,
        )
        if not torch.equal(nz_bi, expected_bi):
            raise ValueError(
                f"replace_position row {bi} must contain a contiguous True block"
            )
        if nz_bi[0].item() != offset:
            raise ValueError(
                f"kv_replace_offset differs across batch: row 0={offset}, "
                f"row {bi}={nz_bi[0].item()}"
            )
        if nz_bi.numel() != block_len:
            raise ValueError(
                "replace_position block length differs across batch: "
                f"row 0={block_len}, row {bi}={nz_bi.numel()}"
            )

    return offset


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class RecurrentModuleConfig:
    """
    Unified configuration for all recurrent modules.

    This dataclass is used by SharedTimeConditioner, RecurrentMixer,
    RecurrentUpdater, and RecurrentInjector.

    Attributes:
        d_model: Backbone hidden dimension
        state_hidden_size: State slot dimension D_s. Default for mixer_dim and injector_dim when those are None
        num_memory_slots: M — number of fixed latent memory slots
        time_cond_dim: Shared time conditioner output dim
        debug_stats: Enable training/sampling diagnostics
        initializer_range: Standard deviation for truncated normal init
        init_device: Device for parameter initialization
    """
    d_model: int = 4096
    state_hidden_size: int = 1024
    num_memory_slots: int = 64
    time_cond_dim: Optional[int] = None
    unroll_steps: int = 4
    aux_loss_weights: Optional[Dict[str, float]] = None
    debug_stats: bool = False
    initializer_range: float = 0.02
    init_device: Optional[str] = None

    # Mixer-specific
    mixer_dim: Optional[int] = None  # Internal processing dim for Mixer. None → state_hidden_size
    mixer_num_heads: Optional[int] = None  # Auto: max(1, min(8, mixer_dim // 64))
    mixer_use_gqa: bool = True  # GQA for mixer cross-attention (kv_heads = q_heads // 2)
    mixer_mlp_ratio: float = 2.0  # FFN expansion ratio for Mixer's SwiGLU

    mixer_slot_attn: bool = False  # Pre-cross-attn slot self-attention in Mixer (no positional encoding)

    # Updater-specific
    updater_z_gate_bias: float = 0.0  # Init bias for update gate (sigmoid(0)=0.5, neutral)
    updater_r_gate_bias: float = 0.0   # Init bias for reset gate (sigmoid(0)=0.5, neutral)

    # Variable-K / Truncated BPTT
    unroll_steps_max: Optional[int] = None  # Target K_max. None = disabled (fixed K)
    warmup_k_fraction: float = 0.5         # Fraction of training to ramp effective K_max

    # Injector-specific
    injector_dim: Optional[int] = None  # Bottleneck dim for Injector. None → state_hidden_size
    injector_num_heads: Optional[int] = None  # Auto: max(1, min(8, injector_dim // 64))
    injector_use_gqa: bool = True
    injector_mlp_ratio_bottleneck: float = 2.0

    disable_time_conditioning: bool = False  # Zero out time conditioning (AdaRMSNorm → RMSNorm)

    def __post_init__(self):
        if self.d_model is None or self.d_model <= 0:
            raise ValueError(
                f"d_model must be a positive int, got {self.d_model}"
            )
        if self.num_memory_slots is None or self.num_memory_slots < 1:
            raise ValueError(
                f"num_memory_slots must be >= 1, got {self.num_memory_slots}"
            )
        if self.unroll_steps is None or self.unroll_steps < 1:
            raise ValueError(
                f"unroll_steps must be >= 1, got {self.unroll_steps}"
            )
        if self.state_hidden_size is None or self.state_hidden_size <= 0:
            raise ValueError(
                f"state_hidden_size must be a positive int, got {self.state_hidden_size}"
            )
        if self.mixer_dim is not None and self.mixer_dim <= 0:
            raise ValueError(f"mixer_dim must be positive, got {self.mixer_dim}")
        if self.injector_dim is not None and self.injector_dim <= 0:
            raise ValueError(f"injector_dim must be positive, got {self.injector_dim}")
        if self.unroll_steps_max is not None and self.unroll_steps_max < self.unroll_steps:
            raise ValueError(
                f"unroll_steps_max ({self.unroll_steps_max}) must be >= "
                f"unroll_steps ({self.unroll_steps})"
            )
        if not (0.0 < self.warmup_k_fraction <= 1.0):
            raise ValueError(
                f"warmup_k_fraction must be in (0, 1], got {self.warmup_k_fraction}"
            )

    def get_effective_state_hidden_size(self) -> int:
        return self.state_hidden_size

    def get_effective_mixer_dim(self) -> int:
        return self.mixer_dim if self.mixer_dim is not None else self.state_hidden_size

    def get_effective_injector_dim(self) -> int:
        return self.injector_dim if self.injector_dim is not None else self.state_hidden_size

    def get_effective_time_cond_dim(self) -> int:
        if self.time_cond_dim is not None:
            return self.time_cond_dim
        return min(1024, self.state_hidden_size)


def create_recurrent_module_config(config) -> RecurrentModuleConfig:
    """
    Create a RecurrentModuleConfig from a HuggingFace config.

    Works with RecurrentLLaDAConfig.
    """
    state_hidden_size = config.state_hidden_size
    if state_hidden_size is None:
        log.warning(
            "state_hidden_size is None (legacy config?), falling back to 1024"
        )
        state_hidden_size = 1024

    return RecurrentModuleConfig(
        d_model=config.d_model,
        state_hidden_size=state_hidden_size,
        num_memory_slots=config.num_memory_slots,
        time_cond_dim=config.time_cond_dim,
        unroll_steps=config.unroll_steps,
        aux_loss_weights=config.aux_loss_weights,
        debug_stats=config.debug_stats,
        initializer_range=getattr(config, "initializer_range", 0.02),
        init_device=None,
        # Mixer params
        mixer_dim=getattr(config, "mixer_dim", None),
        mixer_num_heads=getattr(config, "mixer_num_heads", None),
        mixer_use_gqa=getattr(config, "mixer_use_gqa", True),
        mixer_mlp_ratio=getattr(config, "mixer_mlp_ratio", 2.0),
        mixer_slot_attn=getattr(config, "mixer_slot_attn", False),
        # Updater params
        updater_z_gate_bias=getattr(config, "updater_z_gate_bias", 0.0),
        updater_r_gate_bias=getattr(config, "updater_r_gate_bias", 0.0),
        # Variable-K / Truncated BPTT
        unroll_steps_max=getattr(config, "unroll_steps_max", None),
        warmup_k_fraction=getattr(config, "warmup_k_fraction", 0.5),
        # Injector params
        injector_dim=getattr(config, "injector_dim", None),
        injector_num_heads=getattr(config, "injector_num_heads", None),
        injector_use_gqa=getattr(config, "injector_use_gqa", True),
        injector_mlp_ratio_bottleneck=getattr(
            config, "injector_mlp_ratio_bottleneck", 2.0
        ),
        disable_time_conditioning=getattr(config, "disable_time_conditioning", False),
    )


# =============================================================================
# Shared Attention Helpers
# =============================================================================

def _compute_num_heads(dim: int, configured_heads: int | None) -> tuple[int, int]:
    """Compute (num_heads, head_dim). Adjusts heads downward until dim is divisible."""
    if configured_heads is not None:
        num_heads = configured_heads
    else:
        num_heads = max(1, min(8, dim // 64))
        while num_heads > 1 and dim % num_heads != 0:
            num_heads -= 1
    head_dim = dim // num_heads
    assert head_dim * num_heads == dim, (
        f"num_heads={num_heads} does not divide dim={dim}"
    )
    return num_heads, head_dim


def _reshape_for_heads(
    x: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Reshape [B, S, num_heads * head_dim] → [B, num_heads, S, head_dim]."""
    B, S, _ = x.shape
    return x.view(B, S, num_heads, head_dim).transpose(1, 2)


def _expand_kv_heads(
    kv: torch.Tensor, num_q_heads: int, num_kv_heads: int
) -> torch.Tensor:
    """Repeat KV heads to match Q heads for GQA."""
    if num_kv_heads == num_q_heads:
        return kv
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    repeat = num_q_heads // num_kv_heads
    return kv.repeat_interleave(repeat, dim=1)


def _build_sinusoidal_pe_table(
    dim: int, max_seq_len: int = 8192, base: float = 10000.0,
) -> torch.Tensor:
    """Build standard Vaswani sinusoidal positional encoding table.

    Returns:
        [max_seq_len, dim] float32 tensor.
    """
    pos = torch.arange(max_seq_len).float().unsqueeze(1)  # [L, 1]
    half = (dim + 1) // 2  # ceil(dim/2) to handle odd dim
    inv_freq = 1.0 / (base ** (torch.arange(half).float() / half))  # [half]
    sinusoid = pos * inv_freq  # [L, half]
    pe = torch.zeros(max_seq_len, dim)
    pe[:, 0::2] = sinusoid[:, : dim - dim // 2].sin()
    pe[:, 1::2] = sinusoid[:, : dim // 2].cos()
    return pe


def _expand_mask_to_length(
    attention_mask: torch.Tensor, target_len: int
) -> torch.Tensor:
    """Expand a 2D attention mask [B, N] to target_len by prepending ones.

    Used when cached KV is longer than the current mask (cached positions
    are treated as valid).  Returns the mask unchanged if already long enough.
    """
    if target_len > attention_mask.size(1):
        pad_len = target_len - attention_mask.size(1)
        return torch.cat(
            [attention_mask.new_ones(attention_mask.size(0), pad_len), attention_mask],
            dim=1,
        )
    return attention_mask


# =============================================================================
# Utility Classes
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class AdaRMSNorm(nn.Module):
    """RMSNorm + FiLM: per-channel modulation from t_cond.

    zero_init=False (default): (1+scale)*x_hat + shift  → identity at init
    zero_init=True:            scale*x_hat + shift       → zero output at init
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-6,
        zero_init: bool = False,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.zero_init = zero_init
        self.weight = nn.Parameter(torch.ones(dim, device=device))
        self.modulation = nn.Linear(cond_dim, 2 * dim, device=device)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, t_cond: torch.Tensor) -> torch.Tensor:
        x_hat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        mod = self.modulation(t_cond)  # [B, 2*dim]
        scale, shift = mod.unsqueeze(1).chunk(2, dim=-1)  # each [B, 1, dim]
        if self.zero_init:
            return scale * x_hat + shift      # zero at init
        return x_hat * (1 + scale) + shift    # identity at init


class SwiGLUMLP(nn.Module):
    """SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network."""

    def __init__(
        self, d_model: int, hidden_ratio: float = 2.0, dropout: float = 0.0, device=None
    ):
        super().__init__()
        hidden = int(d_model * hidden_ratio)
        self.w_gate = nn.Linear(d_model, hidden, bias=False, device=device)
        self.w_up = nn.Linear(d_model, hidden, bias=False, device=device)
        self.w_down = nn.Linear(hidden, d_model, bias=False, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


# =============================================================================
# Shared Time Conditioner
# =============================================================================

class SharedTimeConditioner(nn.Module):
    """
    Sinusoidal embedding → 2-layer MLP. Single instance shared by all modules.

    t: [B] → sinusoidal_embed → [B, 256] → MLP(256 → cond_dim → cond_dim) → [B, cond_dim]

    Each downstream module has its own small linear head to project from
    cond_dim to whatever shape it needs.
    """

    def __init__(
        self,
        cond_dim: int,
        mixer_dim: Optional[int] = None,
        device=None,
        initializer_range: float = 0.02,
        disable_time_conditioning: bool = False,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.disable_time_conditioning = disable_time_conditioning
        self.sinusoidal_dim = SINUSOIDAL_TIME_EMBED_DIM
        self.initializer_range = initializer_range

        # Register sinusoidal frequency table as buffer
        half_dim = self.sinusoidal_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer("time_emb_table", emb)

        # MLP: sinusoidal_dim → cond_dim → cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.sinusoidal_dim, cond_dim, device=device),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, device=device),
        )

        # Content conditioning: zero-gated residual from mixer's content summary
        self.content_enabled = mixer_dim is not None and mixer_dim > 0
        if self.content_enabled:
            self.content_norm = RMSNorm(mixer_dim, device=device)
            self.content_proj = nn.Linear(mixer_dim, cond_dim, device=device)
            self.content_gate = nn.Parameter(torch.zeros(1, cond_dim, device=device))

        self.reset_parameters()

    def reset_parameters(self):
        std = self.initializer_range
        nn.init.trunc_normal_(self.mlp[0].weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.trunc_normal_(self.mlp[2].weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.mlp[2].bias)
        if self.content_enabled:
            nn.init.ones_(self.content_norm.weight)
            nn.init.trunc_normal_(self.content_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(self.content_proj.bias)
            nn.init.zeros_(self.content_gate)

    def _normalize_timestep(
        self, ref: torch.Tensor, t: Union[int, float, torch.Tensor, None]
    ) -> torch.Tensor:
        """Normalize and expand timestep to [B] float tensor in [0, 1]."""
        B = ref.size(0)
        device = ref.device

        if t is None:
            return torch.zeros(B, device=device)

        if not isinstance(t, torch.Tensor):
            t_val = float(t)
            if not (0.0 <= t_val <= 1.0):
                raise ValueError(f"Timestep scalar out of [0, 1]: {t_val}")
            return torch.full((B,), t_val, device=device)

        t = t.to(device=device).float()
        if t.dim() == 0 or (t.size(0) == 1 and B > 1):
            t = t.expand(B)
        if not (t.min() >= 0.0 and t.max() <= 1.0):
            raise ValueError(
                f"Timestep tensor out of [0, 1]: min={t.min().item()}, max={t.max().item()}"
            )
        return t

    def forward(
        self,
        ref: torch.Tensor,
        t: Union[int, torch.Tensor, None],
        content_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute time conditioning vector with optional content residual.

        Args:
            ref: Reference tensor [B, ...] for batch size and device
            t: Timestep [B] or scalar or None
            content_summary: Optional [B, mixer_dim] content summary from Mixer

        Returns:
            Time conditioning [B, cond_dim]
        """
        if self.disable_time_conditioning:
            B = ref.size(0)
            return torch.zeros(B, self.cond_dim, device=ref.device, dtype=ref.dtype)

        t_norm = self._normalize_timestep(ref, t)  # [B], float in [0,1]
        t_scaled = t_norm * _TIME_SCALE  # [B], maps [0,1] → [0,1000]
        # Sinusoidal embedding
        emb = t_scaled[:, None] * self.time_emb_table[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.to(ref.dtype)
        t_cond = self.mlp(emb)

        if self.content_enabled and content_summary is not None:
            content_delta = self.content_gate * self.content_proj(self.content_norm(content_summary.to(ref.dtype)))
            t_cond = t_cond + content_delta

        return t_cond


# =============================================================================
# Mixer Module (Read: backbone → memory)
# =============================================================================

class RecurrentMixer(nn.Module):
    """
    Cross-attention mixer with bottleneck: Q=state[B,M,Ds], KV=hidden[B,N,D] → context[B,M,Ds].

    Pipeline (at diffusion step k):
        IF mixer_slot_attn:
            state = state + slot_sa_o_proj(SlotSA(slot_sa_norm(state)))  # Pre-bottleneck slot communication (D_s space)
        s_b = s_down(state)                        # [B, M, mixer_dim]
        s_b_normed = norm_s(s_b, t_cond)           # [B, M, mixer_dim]  (time-conditioned)
        h_b = norm_h(h_down(hidden), t_cond)       # [B, N, mixer_dim]
        attn_out = o_proj(Attn(Q=q(s_b_normed), KV=kv(h_b)))   # Pure extraction
        ctx_b = attn_out + ffn(norm_ffn(attn_out, t_cond))
        context = up_proj(ctx_b)                   # [B, M, D_s]

    The Mixer extracts information from backbone hidden states into memory slots.
    The Updater's GRU handles the state residual via (1-z)*state + z*candidate.

    Optional slot self-attention (mixer_slot_attn=True) runs before down-projection
    on raw state in D_s space. Uses plain RMSNorm (timestep-independent).
    No positional encoding (slots are order-invariant). Exact zero-init on slot_sa_o_proj.
    """

    def __init__(self, config: RecurrentModuleConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.state_hidden_size = config.get_effective_state_hidden_size()
        self.mixer_dim = config.get_effective_mixer_dim()

        device = config.init_device
        cond_dim = config.get_effective_time_cond_dim()

        # Attention heads — computed from mixer_dim
        self.num_heads, self.head_dim = _compute_num_heads(self.mixer_dim, config.mixer_num_heads)

        # GQA
        if config.mixer_use_gqa:
            self.num_kv_heads = max(1, self.num_heads // 2)
        else:
            self.num_kv_heads = self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        # Down-projections to mixer_dim
        self.s_down = nn.Linear(self.state_hidden_size, self.mixer_dim, bias=False, device=device)
        self.h_down = nn.Linear(self.d_model, self.mixer_dim, bias=False, device=device)

        # Input norms (after down-projection, in mixer_dim space)
        self.norm_s = AdaRMSNorm(self.mixer_dim, cond_dim, device=device)
        self.norm_h = AdaRMSNorm(self.mixer_dim, cond_dim, device=device)

        # Cross-attention projections (all in mixer_dim)
        self.q_proj = nn.Linear(self.mixer_dim, self.mixer_dim, bias=False, device=device)
        self.k_proj = nn.Linear(self.mixer_dim, self.kv_dim, bias=False, device=device)
        self.v_proj = nn.Linear(self.mixer_dim, self.kv_dim, bias=False, device=device)
        self.o_proj = nn.Linear(self.mixer_dim, self.mixer_dim, bias=False, device=device)

        # Pre-norm + FFN block (in mixer_dim)
        self.norm_ffn = AdaRMSNorm(self.mixer_dim, cond_dim, device=device)
        self.ffn = SwiGLUMLP(
            d_model=self.mixer_dim,
            hidden_ratio=config.mixer_mlp_ratio,
            device=device,
        )

        # Up-projection back to D_s (Updater expects D_s)
        self.up_proj = nn.Linear(self.mixer_dim, self.state_hidden_size, bias=False, device=device)

        # Optional slot self-attention (pre-bottleneck, in D_s space)
        self.has_slot_attn = config.mixer_slot_attn
        if self.has_slot_attn:
            # Compute SA heads from state_hidden_size (not mixer_dim)
            self.slot_sa_num_heads, self.slot_sa_head_dim = _compute_num_heads(
                self.state_hidden_size, config.mixer_num_heads
            )
            if config.mixer_use_gqa:
                self.slot_sa_num_kv_heads = max(1, self.slot_sa_num_heads // 2)
            else:
                self.slot_sa_num_kv_heads = self.slot_sa_num_heads
            slot_sa_kv_dim = self.slot_sa_num_kv_heads * self.slot_sa_head_dim

            self.slot_sa_norm = RMSNorm(self.state_hidden_size, device=device)
            self.slot_sa_q_proj = nn.Linear(self.state_hidden_size, self.state_hidden_size, bias=False, device=device)
            self.slot_sa_k_proj = nn.Linear(self.state_hidden_size, slot_sa_kv_dim, bias=False, device=device)
            self.slot_sa_v_proj = nn.Linear(self.state_hidden_size, slot_sa_kv_dim, bias=False, device=device)
            self.slot_sa_o_proj = nn.Linear(self.state_hidden_size, self.state_hidden_size, bias=False, device=device)

        self.reset_parameters()

    def reset_parameters(self):
        std = self.config.initializer_range
        def trunc_normal_(w):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2 * std, b=2 * std)

        # Norms
        self.norm_s.reset_parameters()
        self.norm_h.reset_parameters()
        self.norm_ffn.reset_parameters()

        # Down-projections + attention + up-projection
        for m in [self.s_down, self.h_down, self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.up_proj]:
            trunc_normal_(m.weight)

        # FFN
        trunc_normal_(self.ffn.w_gate.weight)
        trunc_normal_(self.ffn.w_up.weight)
        trunc_normal_(self.ffn.w_down.weight)

        # Slot self-attention init
        if self.has_slot_attn:
            nn.init.ones_(self.slot_sa_norm.weight)
            # Q/K/V: normal init so attention output is non-zero
            for proj in (self.slot_sa_q_proj, self.slot_sa_k_proj, self.slot_sa_v_proj):
                trunc_normal_(proj.weight)
            # O-proj: zero-init so slot SA contribution starts at zero
            nn.init.zeros_(self.slot_sa_o_proj.weight)

    def prepare(
        self,
        hidden_states: torch.Tensor,           # [B, N_new, D]
        past_mixer_hidden: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        kv_replace_offset: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [B, N] 1=valid 0=pad
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Phase 1: down-project, cache merge, content summary. No t_cond dependency.

        Returns:
            h_proj_full: [B, N_full, mixer_dim]
            present_mixer_hidden: [B, N_full, mixer_dim] if use_cache else None
            content_summary: [B, mixer_dim] masked mean of h_proj_full
        """
        dtype = self.q_proj.weight.dtype
        hidden_states = hidden_states.to(dtype)

        h_proj_new = self.h_down(hidden_states)  # [B, N_new, mixer_dim]
        N_new = hidden_states.size(1)

        if past_mixer_hidden is not None:
            past_mixer_hidden = past_mixer_hidden.to(dtype)
            if kv_replace_offset is not None:
                h_proj_full = past_mixer_hidden.clone()
                h_proj_full[:, kv_replace_offset : kv_replace_offset + N_new] = h_proj_new
            else:
                h_proj_full = torch.cat([past_mixer_hidden, h_proj_new], dim=1)
        else:
            h_proj_full = h_proj_new

        present_mixer_hidden = h_proj_full if use_cache else None

        # Content summary: masked mean pooling (before norm_h, timestep-independent)
        if attention_mask is not None:
            pool_mask = _expand_mask_to_length(attention_mask, h_proj_full.size(1))
            pool_mask_f = pool_mask.float().unsqueeze(-1)  # [B, N_full, 1]
            content_summary = (h_proj_full * pool_mask_f).sum(dim=1) / pool_mask_f.sum(dim=1).clamp(min=1.0)
        else:
            content_summary = h_proj_full.mean(dim=1)  # [B, mixer_dim]

        return h_proj_full, present_mixer_hidden, content_summary

    def process(
        self,
        state: torch.Tensor,        # [B, M, D_s]
        h_proj_full: torch.Tensor,   # [B, N_full, mixer_dim]
        t_cond: torch.Tensor,        # [B, cond_dim]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Phase 2: norm, cross-attention, FFN, up-project. Requires t_cond.

        Returns:
            context: [B, M, D_s]
        """
        dtype = self.q_proj.weight.dtype
        state = state.to(dtype)
        t_cond = t_cond.to(dtype)

        # Optional slot self-attention on state (full D_s space, pre-bottleneck)
        if self.has_slot_attn:
            B_val = state.size(0)
            s_n = self.slot_sa_norm(state)
            q_sa = _reshape_for_heads(self.slot_sa_q_proj(s_n), self.slot_sa_num_heads, self.slot_sa_head_dim)
            k_sa = _reshape_for_heads(self.slot_sa_k_proj(s_n), self.slot_sa_num_kv_heads, self.slot_sa_head_dim)
            v_sa = _reshape_for_heads(self.slot_sa_v_proj(s_n), self.slot_sa_num_kv_heads, self.slot_sa_head_dim)
            k_sa_exp = _expand_kv_heads(k_sa, self.slot_sa_num_heads, self.slot_sa_num_kv_heads)
            v_sa_exp = _expand_kv_heads(v_sa, self.slot_sa_num_heads, self.slot_sa_num_kv_heads)
            sa_out = F.scaled_dot_product_attention(q_sa, k_sa_exp, v_sa_exp, dropout_p=0.0, is_causal=False)
            sa_out = sa_out.transpose(1, 2).contiguous().view(B_val, state.size(1), self.state_hidden_size)
            state = state + self.slot_sa_o_proj(sa_out)

        # Down-project state
        s_b = self.s_down(state)  # [B, M, mixer_dim]

        # Time-conditioned norms for cross-attention
        s_b_normed = self.norm_s(s_b, t_cond)  # [B, M, mixer_dim]
        h_b = self.norm_h(h_proj_full, t_cond)  # [B, N, mixer_dim]

        q = _reshape_for_heads(self.q_proj(s_b_normed), self.num_heads, self.head_dim)
        k = _reshape_for_heads(self.k_proj(h_b), self.num_kv_heads, self.head_dim)
        v = _reshape_for_heads(self.v_proj(h_b), self.num_kv_heads, self.head_dim)

        k_exp = _expand_kv_heads(k, self.num_heads, self.num_kv_heads)
        v_exp = _expand_kv_heads(v, self.num_heads, self.num_kv_heads)

        # Build key-padding mask for cross-attention
        kv_mask = None
        if attention_mask is not None:
            full_mask = _expand_mask_to_length(attention_mask, h_proj_full.size(1))
            kv_mask = full_mask.unsqueeze(1).unsqueeze(1).bool()  # [B, 1, 1, N_full]

        out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=kv_mask, dropout_p=0.0, is_causal=False)
        B, _, M, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, M, self.mixer_dim)

        attn_out = self.o_proj(out)
        context_b = attn_out + self.ffn(self.norm_ffn(attn_out, t_cond))
        context = self.up_proj(context_b)  # [B, M, D_s]

        return context

    def forward(
        self,
        state: torch.Tensor,           # [B, M, D_s]
        hidden_states: torch.Tensor,    # [B, N_new, D]
        t_cond: torch.Tensor,           # [B, cond_dim]
        past_mixer_hidden: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        kv_replace_offset: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [B, N] 1=valid 0=pad
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Full pipeline: prepare + process (convenience wrapper).

        Returns:
            (context, present_mixer_hidden, content_summary):
                context: [B, M, D_s]
                present_mixer_hidden: [B, N_full, mixer_dim] if use_cache else None
                content_summary: [B, mixer_dim]
        """
        h_proj_full, present_mixer_hidden, content_summary = self.prepare(
            hidden_states, past_mixer_hidden, use_cache, kv_replace_offset, attention_mask
        )
        context = self.process(state, h_proj_full, t_cond, attention_mask)
        return context, present_mixer_hidden, content_summary


# =============================================================================
# Updater Module (GRU-style state update)
# =============================================================================

class RecurrentUpdater(nn.Module):
    """
    Time-conditioned GRU-style state updater. Replaces Router + Maintainer.

    s_norm = AdaRMSNorm(state, t_cond)                  [B, M, D_s]
    c_norm = AdaRMSNorm(context, t_cond)               [B, M, D_s]
    joint = cat([s_norm, c_norm], dim=-1)              [B, M, 2*D_s]
    z, r = sigmoid(W_gate(joint))                      [B, M, D_s] each
    candidate = tanh(W_cand(cat([r*s_norm, c_norm])))
    new_state = (1-z) * state + z * candidate          [B, M, D_s]

    Default init: z_gate bias = 0.0 (sigmoid = 0.5, neutral).
    """

    def __init__(self, config: RecurrentModuleConfig):
        super().__init__()
        self.config = config
        self.state_hidden_size = config.get_effective_state_hidden_size()
        self.cond_dim = config.get_effective_time_cond_dim()

        device = config.init_device

        self.state_norm = AdaRMSNorm(self.state_hidden_size, self.cond_dim, device=device)
        self.context_norm = AdaRMSNorm(self.state_hidden_size, self.cond_dim, device=device)

        # Gate projection: [s_norm, context_norm] → [z, r]
        self.gate_proj = nn.Linear(
            2 * self.state_hidden_size, 2 * self.state_hidden_size, device=device
        )

        # Candidate projection: [r*s_norm, context_norm] → candidate
        self.cand_proj = nn.Linear(
            2 * self.state_hidden_size, self.state_hidden_size, device=device
        )

        # Learnable per-slot-distinct initial state (overwritten by reset_parameters)
        self.state_init = nn.Parameter(
            torch.empty(1, config.num_memory_slots, self.state_hidden_size)
        )

        # Learnable slot identity embedding: enters GRU norm input for per-slot
        # gate identity, but NOT the GRU residual path (state), preventing
        # accumulation.
        # Design note: Injector's PE enters the residual base because injector
        # output is a per-step delta (not persistent state). Updater's slot_embed
        # enters only the GRU norm input because the GRU output IS persistent
        # state — adding slot_embed there would cause linear drift across steps.
        self.slot_embed = nn.Parameter(
            torch.empty(1, config.num_memory_slots, self.state_hidden_size)
        )

        self.reset_parameters()

    def get_initial_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Return learnable initial state expanded to batch size."""
        state = self.state_init.expand(batch_size, -1, -1)
        if device is not None or dtype is not None:
            state = state.to(device=device, dtype=dtype)
        return state

    def reset_parameters(self):
        std = self.config.initializer_range
        D_s = self.state_hidden_size

        self.state_norm.reset_parameters()
        self.context_norm.reset_parameters()

        # Gate projection: initialize small, with configurable z and r biases
        nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.gate_proj.bias)
        # z_gate: first D_s entries (configurable, default 0.0 → sigmoid = 0.5, neutral)
        nn.init.constant_(self.gate_proj.bias[:D_s], self.config.updater_z_gate_bias)
        # r_gate: second D_s entries (default 0.0 → sigmoid = 0.5, neutral reset)
        nn.init.constant_(self.gate_proj.bias[D_s:], self.config.updater_r_gate_bias)

        # Candidate projection: standard init
        nn.init.trunc_normal_(self.cand_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.cand_proj.bias)

        # Learnable initial state
        nn.init.trunc_normal_(self.state_init.data, mean=0.0, std=std, a=-2*std, b=2*std)

        # Learnable slot identity embedding
        nn.init.trunc_normal_(self.slot_embed.data, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(
        self,
        state: torch.Tensor,       # [B, M, D_s]
        context: torch.Tensor,     # [B, M, D_s]
        t_cond: torch.Tensor,      # [B, cond_dim]
    ) -> torch.Tensor:
        """
        Update state using GRU-style gated dynamics.

        Args:
            state: Previous state [B, M, D_s]
            context: Mixer output [B, M, D_s]
            t_cond: Time conditioning [B, cond_dim]

        Returns:
            Updated state [B, M, D_s]
        """
        dtype = self.gate_proj.weight.dtype
        state = state.to(dtype)
        context = context.to(dtype)
        t_cond = t_cond.to(dtype)

        B, M, D_s = state.shape

        # Slot embedding: enters GRU norm input for per-slot gate identity,
        # but NOT the GRU residual path (state), preventing accumulation.
        state_for_norm = state + self.slot_embed.to(
            dtype=state.dtype, device=state.device
        )

        s_norm = self.state_norm(state_for_norm, t_cond)
        context = self.context_norm(context, t_cond)
        joint = torch.cat([s_norm, context], dim=-1)  # [B, M, 2*D_s]

        # Compute gates
        gate_logits = self.gate_proj(joint)  # [B, M, 2*D_s]
        z_gate, r_gate = gate_logits.chunk(2, dim=-1)                  # each [B, M, D_s]
        z = torch.sigmoid(z_gate)
        r = torch.sigmoid(r_gate)

        # Store gate mean for diagnostics
        if self.config.debug_stats:
            self._debug_z_gate_mean = z.detach().mean().item()

        # Compute candidate
        cand_input = torch.cat([r * s_norm, context], dim=-1)          # [B, M, 2*D_s]
        candidate = torch.tanh(self.cand_proj(cand_input))  # [B, M, D_s]

        # GRU-style update
        new_state = (1 - z) * state + z * candidate

        return new_state


# =============================================================================
# Injector Module (Write: memory → embeddings, ControlNet-style)
# =============================================================================

class RecurrentInjector(nn.Module):
    """
    Cross-attention injector: Q=embed[B,N,D], KV=state[B,M,Ds] + zero-bridge.

    Includes pre-injection self-attention with sinusoidal PE for position
    awareness (solving identical-mask-token problem). Self-attention uses
    plain RMSNorm (not time-conditioned) so K/V depend only on input + position,
    enabling KV caching across different diffusion timesteps.

    Sinusoidal PE is added to x_down *only* for the SA input (Q/K/V computation),
    NOT in the residual path. This gives position awareness inside attention
    without leaking positional signal into cross-attention/FFN/zero-bridge.

    Pipeline:
        embeddings → down_proj → self-attn(RMSNorm, +PE, KV_cache)
                 → norm_x(t_cond) → cross-attn(Q=emb, KV=state) → FFN → zero-bridge → embeddings'

    Cache type: self-attn KV tuple (K, V) each [B, KVH, N, head_dim].
    Note: KV cache stores keys with PE baked in via the input.
    """

    def __init__(self, config: RecurrentModuleConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.state_hidden_size = config.get_effective_state_hidden_size()
        self.cond_dim = config.get_effective_time_cond_dim()

        device = config.init_device

        # Bottleneck — defaults to state_hidden_size, overridable via injector_dim
        self.d_b = config.get_effective_injector_dim()
        assert self.d_b > 0

        # Attention heads
        self.num_heads, self.head_dim = _compute_num_heads(self.d_b, config.injector_num_heads)

        # GQA
        self.use_gqa = config.injector_use_gqa
        if self.use_gqa:
            self.num_kv_heads = max(1, self.num_heads // 2)
        else:
            self.num_kv_heads = self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        # Down projections to bottleneck
        self.x_down = nn.Linear(self.d_model, self.d_b, bias=False, device=device)
        self.s_down = nn.Linear(self.state_hidden_size, self.d_b, bias=False, device=device)

        # Input normalization (after down-projection, in bottleneck space)
        self.norm_x = AdaRMSNorm(self.d_b, self.cond_dim, device=device)
        self.norm_s = AdaRMSNorm(self.d_b, self.cond_dim, device=device)

        # Sinusoidal PE for SA input (position awareness without residual leakage)
        pe_table = _build_sinusoidal_pe_table(self.d_b, max_seq_len=8192)
        self.register_buffer("pe_table", pe_table, persistent=False)
        self.pe_scale = nn.Parameter(torch.ones(1))

        # Self-attention layers (same head config as cross-attention)
        self.norm_sa = RMSNorm(self.d_b, device=device)
        self.sa_q_proj = nn.Linear(self.d_b, self.d_b, bias=False, device=device)
        self.sa_k_proj = nn.Linear(self.d_b, self.kv_dim, bias=False, device=device)
        self.sa_v_proj = nn.Linear(self.d_b, self.kv_dim, bias=False, device=device)
        self.sa_o_proj = nn.Linear(self.d_b, self.d_b, bias=False, device=device)

        # Cross-attention projections (in bottleneck space)
        self.q_proj = nn.Linear(self.d_b, self.d_b, bias=False, device=device)
        self.k_proj = nn.Linear(self.d_b, self.kv_dim, bias=False, device=device)
        self.v_proj = nn.Linear(self.d_b, self.kv_dim, bias=False, device=device)
        self.o_proj = nn.Linear(self.d_b, self.d_b, bias=False, device=device)

        # Pre-norm before FFN
        self.norm_ffn = AdaRMSNorm(self.d_b, self.cond_dim, device=device)

        # FFN in bottleneck space
        self.ffn = SwiGLUMLP(
            d_model=self.d_b,
            hidden_ratio=config.injector_mlp_ratio_bottleneck,
            device=device,
        )

        # Zero-bridge output: AdaRMSNorm(zero_init=True) + normal-init up_proj
        self.norm_out = AdaRMSNorm(self.d_b, self.cond_dim, zero_init=True, device=device)
        self.up_proj = nn.Linear(self.d_b, self.d_model, bias=False, device=device)

        self.reset_parameters()

    def reset_parameters(self):
        std = self.config.initializer_range

        def trunc_normal_(w):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2 * std, b=2 * std)

        # Norms
        self.norm_x.reset_parameters()
        self.norm_s.reset_parameters()
        self.norm_ffn.reset_parameters()
        self.norm_out.reset_parameters()
        # Soft bias on scale so delta ≈ 1e-3 * W_up @ RMSNorm(x_b) at init:
        # near-zero but nonzero, ensuring all parameters receive gradients from step 1.
        with torch.no_grad():
            self.norm_out.modulation.bias.data[: self.d_b] = 1e-3
        # Self-attention layers: Q/K/V normal init so attention output is non-zero;
        # O-proj zero-init so SA contribution starts at zero (exact zero-bridge)
        nn.init.ones_(self.norm_sa.weight)  # RMSNorm has no reset_parameters()
        for proj in (self.sa_q_proj, self.sa_k_proj, self.sa_v_proj):
            trunc_normal_(proj.weight)
        nn.init.zeros_(self.sa_o_proj.weight)

        # Internal projections
        for m in [self.x_down, self.s_down, self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            trunc_normal_(m.weight)

        # FFN
        trunc_normal_(self.ffn.w_gate.weight)
        trunc_normal_(self.ffn.w_up.weight)
        trunc_normal_(self.ffn.w_down.weight)

        # Up projection: NORMAL init (zero-bridge via norm_out's zero-init scale)
        trunc_normal_(self.up_proj.weight)

        # PE scale
        nn.init.ones_(self.pe_scale)

    def forward(
        self,
        embeddings: torch.Tensor,                         # [B, N_new, D]
        state: Optional[torch.Tensor],                     # [B, M, D_s] or None (cache-only)
        t_cond: torch.Tensor,                              # [B, cond_dim]
        past_injector_sa_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        kv_replace_offset: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,    # [B, N] 1=valid 0=pad
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Inject state information into embeddings via self-attn → cross-attn.

        Pipeline:
            embeddings → down_proj → self-attn(RMSNorm, +PE, KV_cache)
                     → norm_x(t_cond) → cross-attn(Q=emb, KV=state) → FFN → zero-bridge → embeddings'

        Sinusoidal PE is added to x_down *only* for the SA input (Q/K/V),
        NOT in the residual path. The residual ``x_b = x_down + sa_out``
        uses the clean ``x_down`` without PE.

        Self-attention uses plain RMSNorm (not time-conditioned) so K/V depend
        only on input embeddings + position, enabling KV caching across
        different diffusion timesteps. KV cache stores keys with PE baked in.

        When ``state`` is None, operates in cache-only mode: computes self-attn
        and caches KV but skips cross-attention/zero-bridge, returning original
        embeddings with zero delta.

        Args:
            embeddings: Input embeddings [B, N_new, D]
            state: Memory state [B, M, D_s] or None for cache-only mode
            t_cond: Time conditioning [B, cond_dim]
            past_injector_sa_kv: Cached self-attn KV tuple (K, V) each
                [B, KVH, N_cached, head_dim] — K has PE baked in
            use_cache: If True, return self-attn KV for caching
            kv_replace_offset: For dual cache — replace at this offset

        Returns:
            (enhanced_embeddings, delta, present_sa_kv):
                enhanced_embeddings: [B, N_new, D]
                delta: [B, N_new, D]
                present_sa_kv: (K, V) tuple each [B, KVH, N_full, hd] if use_cache
        """
        B, N_new, _ = embeddings.shape
        dtype = self.q_proj.weight.dtype

        # 1. Down-project
        x_down = self.x_down(embeddings.to(dtype))  # [B, N_new, d_b]

        # Derive absolute position start from cache
        if past_injector_sa_kv is not None:
            start = (
                kv_replace_offset
                if kv_replace_offset is not None
                else past_injector_sa_kv[0].size(2)
            )
        else:
            start = 0

        # 2. Self-attention with sinusoidal PE and KV cache
        # PE is added only to the SA input, NOT to x_down (no residual leakage)
        if start + N_new > self.pe_table.size(0):
            raise ValueError(
                f"Sequence position {start + N_new} exceeds PE table size "
                f"{self.pe_table.size(0)}. Use a shorter sequence or increase "
                f"max_seq_len in _build_sinusoidal_pe_table."
            )
        x_pe = x_down + self.pe_scale * self.pe_table[start : start + N_new].to(dtype)
        x_sa = self.norm_sa(x_pe)  # plain RMSNorm, NOT time-conditioned
        q_sa = _reshape_for_heads(
            self.sa_q_proj(x_sa), self.num_heads, self.head_dim
        )  # [B, H, N_new, hd]
        k_new = _reshape_for_heads(
            self.sa_k_proj(x_sa), self.num_kv_heads, self.head_dim
        )  # [B, KVH, N_new, hd]
        v_new = _reshape_for_heads(
            self.sa_v_proj(x_sa), self.num_kv_heads, self.head_dim
        )  # [B, KVH, N_new, hd]

        # Merge K, V with cache (K/V have PE baked in via input)
        if past_injector_sa_kv is not None:
            past_k, past_v = past_injector_sa_kv
            past_k = past_k.to(dtype)
            past_v = past_v.to(dtype)
            if kv_replace_offset is not None:
                # Dual cache: replace slice (new K has correct PE via input)
                k_full = past_k.clone()
                k_full[:, :, kv_replace_offset : kv_replace_offset + N_new] = k_new
                v_full = past_v.clone()
                v_full[:, :, kv_replace_offset : kv_replace_offset + N_new] = v_new
            else:
                # Prefix cache: concatenate
                k_full = torch.cat([past_k, k_new], dim=2)
                v_full = torch.cat([past_v, v_new], dim=2)
        else:
            k_full, v_full = k_new, v_new

        present_sa_kv = (k_full, v_full) if use_cache else None

        # GQA expand + attention (Q: N_new, K/V: N_full)
        k_exp = _expand_kv_heads(k_full, self.num_heads, self.num_kv_heads)
        v_exp = _expand_kv_heads(v_full, self.num_heads, self.num_kv_heads)

        # Build key-padding mask for self-attention
        sa_mask = None
        if attention_mask is not None:
            full_sa_mask = _expand_mask_to_length(attention_mask, k_full.size(2))
            sa_mask = full_sa_mask.unsqueeze(1).unsqueeze(1).bool()  # [B, 1, 1, N_kv]

        a_sa = F.scaled_dot_product_attention(
            q_sa, k_exp, v_exp, attn_mask=sa_mask, dropout_p=0.0, is_causal=False
        )
        a_sa = a_sa.transpose(1, 2).contiguous().view(B, N_new, self.d_b)
        x_b = x_down + self.sa_o_proj(a_sa)  # residual in bottleneck

        # 3. Cache-only mode: state is None → return early with no injection
        if state is None:
            zero_delta = torch.zeros_like(embeddings)
            return embeddings, zero_delta, present_sa_kv

        # 3. Time-conditioned norm (only on new positions, AFTER self-attn)
        t_cond = t_cond.to(dtype)
        state = state.to(dtype)
        x_b = self.norm_x(x_b, t_cond)  # [B, N_new, d_b]

        # 4. State KV
        s_b = self.norm_s(self.s_down(state), t_cond)  # [B, M, d_b]

        # 5. Cross-attention: Q from x_b (new positions), KV from state
        q = _reshape_for_heads(
            self.q_proj(x_b), self.num_heads, self.head_dim
        )  # [B, H, N_new, head_dim]
        k = _reshape_for_heads(self.k_proj(s_b), self.num_kv_heads, self.head_dim)
        v = _reshape_for_heads(self.v_proj(s_b), self.num_kv_heads, self.head_dim)

        k = _expand_kv_heads(k, self.num_heads, self.num_kv_heads)
        v = _expand_kv_heads(v, self.num_heads, self.num_kv_heads)

        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        a = a.transpose(1, 2).contiguous().view(B, N_new, self.d_b)
        a = self.o_proj(a)

        # 6. Attn residual (in bottleneck)
        x_b = x_b + a

        # 7. FFN block: pre-norm → FFN → residual
        x_b = x_b + self.ffn(self.norm_ffn(x_b, t_cond))

        # 8. Zero-bridge via AdaRMSNorm(zero_init=True) + normal-init up_proj
        delta = self.up_proj(self.norm_out(x_b, t_cond))  # [B, N_new, D]

        # Zero out delta at padding positions
        if attention_mask is not None:
            delta = delta * attention_mask.unsqueeze(-1).to(delta.dtype)

        return embeddings + delta, delta, present_sa_kv
