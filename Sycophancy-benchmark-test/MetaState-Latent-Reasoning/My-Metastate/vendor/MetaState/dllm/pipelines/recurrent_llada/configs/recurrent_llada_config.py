"""
Recurrent LLaDA Configuration.

This module defines the configuration classes for Recurrent LLaDA, which extends
the base LLaDA architecture with a memory-slot recurrent state mechanism.

Configuration Hierarchy:
------------------------
1. RecurrentModelConfig (dataclass): Internal configuration for the LLaDA backbone.
   Extends ModelConfig with fields needed during backbone initialization.

2. RecurrentLLaDAConfig (HuggingFace config): HF-compatible config that extends
   LLaDAConfig. Used for model serialization and pretrained model loading.

Key Configuration Groups:
-------------------------
1. State Dimensions: Control the size of recurrent state (memory capacity)
2. Memory Slots: Fixed number of latent memory slots (decoupled from sequence length)
3. Training Settings: Control K-step iterative training behavior
4. Auxiliary Losses: Regularization for stable training
"""
from dataclasses import dataclass
from typing import Optional, Dict

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig, ModelConfig


__all__ = ["RecurrentModelConfig", "RecurrentLLaDAConfig"]


@dataclass
class RecurrentModelConfig(ModelConfig):
    """
    Internal configuration for Recurrent LLaDA model components.

    Internal configuration for the LLaDA backbone model.
    Extends the base ModelConfig and is used to initialize the LLaDAModel
    within RecurrentLLaDAModel. Recurrent modules (Mixer, Updater, Injector)
    use RecurrentModuleConfig from dllm.core.modules.recurrent instead.
    """

    # =========================================================================
    # State Dimensions
    # =========================================================================
    state_hidden_size: int = 1024

    # =========================================================================
    # Memory Slots
    # =========================================================================
    num_memory_slots: int = 64
    time_cond_dim: Optional[int] = None

    # =========================================================================
    # Training Settings
    # =========================================================================
    unroll_steps: int = 4

    # =========================================================================
    # Auxiliary Losses
    # =========================================================================
    aux_loss_weights: Optional[Dict[str, float]] = None

    # =========================================================================
    # Debug Settings
    # =========================================================================
    debug_stats: bool = False


class RecurrentLLaDAConfig(LLaDAConfig):
    """
    HuggingFace-compatible configuration for Recurrent LLaDA.

    This config class extends LLaDAConfig (which extends PretrainedConfig) with
    recurrent-specific parameters. It's used for:
    - Saving/loading pretrained models with from_pretrained()
    - Auto-registration with HF's AutoModel/AutoConfig
    - Serialization to config.json
    """

    model_type = "recurrent_llada"

    def __init__(
        self,
        # State dimensions
        state_hidden_size: int = 1024,
        # Memory slots
        num_memory_slots: int = 64,
        time_cond_dim: Optional[int] = None,
        # Training settings
        unroll_steps: int = 4,
        unroll_steps_max: Optional[int] = None,
        warmup_k_fraction: float = 0.5,
        # Auxiliary losses
        aux_loss_weights: Optional[Dict[str, float]] = None,
        # Debug settings
        debug_stats: bool = False,
        # Initialization
        initializer_range: float = 0.02,
        # Mixer module params
        mixer_dim: Optional[int] = None,
        mixer_num_heads: Optional[int] = None,
        mixer_use_gqa: bool = True,
        mixer_mlp_ratio: float = 2.0,
        # Updater module params
        updater_z_gate_bias: float = 0.0,
        updater_r_gate_bias: float = 0.0,
        mixer_slot_attn: bool = False,
        # Injector module params
        injector_dim: Optional[int] = None,
        injector_num_heads: Optional[int] = None,
        injector_use_gqa: bool = True,
        injector_mlp_ratio_bottleneck: float = 2.0,
        disable_time_conditioning: bool = False,
        **kwargs
    ):
        # Set architectures for HF auto-detection
        kwargs.pop("model_type", None)
        kwargs.setdefault("architectures", ["RecurrentLLaDAModel"])

        super().__init__(**kwargs)
        self.model_type = self.__class__.model_type

        # Store recurrent-specific attributes
        self.state_hidden_size = state_hidden_size
        self.num_memory_slots = num_memory_slots
        self.time_cond_dim = time_cond_dim
        self.unroll_steps = unroll_steps
        self.unroll_steps_max = unroll_steps_max
        self.warmup_k_fraction = warmup_k_fraction
        self.aux_loss_weights = aux_loss_weights or {}
        self.debug_stats = debug_stats
        self.initializer_range = initializer_range
        # Mixer params
        self.mixer_dim = mixer_dim
        self.mixer_num_heads = mixer_num_heads
        self.mixer_use_gqa = mixer_use_gqa
        self.mixer_mlp_ratio = mixer_mlp_ratio

        # Updater params
        self.updater_z_gate_bias = updater_z_gate_bias
        self.updater_r_gate_bias = updater_r_gate_bias
        self.mixer_slot_attn = mixer_slot_attn

        # Injector params
        self.injector_dim = injector_dim
        self.injector_num_heads = injector_num_heads
        self.injector_use_gqa = injector_use_gqa
        self.injector_mlp_ratio_bottleneck = injector_mlp_ratio_bottleneck

        self.disable_time_conditioning = disable_time_conditioning

        # Validate parameters
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        if self.state_hidden_size is None or self.state_hidden_size <= 0:
            raise ValueError(
                f"state_hidden_size must be a positive int, got {self.state_hidden_size}"
            )
        if self.unroll_steps < 1:
            raise ValueError(f"unroll_steps must be >= 1, got {self.unroll_steps}")
        if self.unroll_steps_max is not None and self.unroll_steps_max < self.unroll_steps:
            raise ValueError(
                f"unroll_steps_max ({self.unroll_steps_max}) must be >= "
                f"unroll_steps ({self.unroll_steps})"
            )
        if self.num_memory_slots < 1:
            raise ValueError(f"num_memory_slots must be >= 1, got {self.num_memory_slots}")
