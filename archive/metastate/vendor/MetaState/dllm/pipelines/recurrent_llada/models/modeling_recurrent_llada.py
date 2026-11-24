"""
Recurrent LLaDA Model - Memory-Slot Augmented Discrete Diffusion Language Model.

This module implements Recurrent LLaDA with a 3-module, fixed-memory-slot
architecture that addresses the "Information Island" problem in discrete
diffusion models.

Architecture Overview:
----------------------
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
        Mixer.prepare(hidden[-1]) → content_summary                       (NEW content from backbone)
        cond = TimeConditioner(ref, t, content_summary)                    (For Mixer + Updater)
        Mixer.process(state, h_proj, cond) → context
                    ↓
        Updater(state, context, cond) → new_state
                    ↓
        Output: cache cond for next step's Injector

Key Design Decisions:
---------------------
1. Fixed memory slots [B, M, D_s] decoupled from sequence length N
2. Three modules: Mixer, Updater, Injector (Router eliminated)
3. All RMSNorm, single shared sinusoidal time conditioner
4. ControlNet-style zero-bridge injection preserved
5. Shared modules from dllm.core.modules.recurrent
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, List, Union, NamedTuple, Callable
from dataclasses import fields

import torch
import torch.nn as nn

from dllm.pipelines.llada.models.modeling_llada import (
    LLaDAModel,
    LLaDAPreTrainedModel,
    ActivationCheckpointingStrategy,
)
from dllm.pipelines.llada.models.configuration_llada import ModelConfig
from ..configs.recurrent_llada_config import RecurrentLLaDAConfig, RecurrentModelConfig

from dllm.core.modules.recurrent import (
    SharedTimeConditioner,
    RecurrentMixer,
    RecurrentUpdater,
    RecurrentInjector,
    create_recurrent_module_config,
    compute_kv_replace_offset,
)


__all__ = [
    "RecurrentLLaDAModel",
    "RecurrentLLaDAModelOutput",
]


log = logging.getLogger(__name__)



# =============================================================================
# Model Output Type
# =============================================================================

class RecurrentLLaDAModelOutput(NamedTuple):
    """
    Output type for RecurrentLLaDAModel.forward().

    Attributes:
        logits: Predicted token logits [B, N, vocab_size]
        new_state: Updated recurrent state [B, M, state_hidden_size]
        all_hidden_states: Backbone hidden states if output_hidden_states=True
        delta_norm: L2 norm of injector delta [B] (debug)
        embedding_norm: L2 norm of input embeddings [B] (debug)
        past_key_values: KV cache from backbone (for Fast-dLLM cache modes)
        past_mixer_hidden: Mixer cache: down-projected hidden states [B, N, mixer_dim] for cross-step continuity
        past_injector_sa_kv: Injector self-attn KV cache: (K, V) each [B, KVH, N, head_dim]
        past_cond: Conditioning vector [B, cond_dim] from previous step, passed directly to Injector
    """
    logits: torch.FloatTensor
    new_state: torch.FloatTensor
    all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    delta_norm: Optional[torch.FloatTensor] = None
    embedding_norm: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, ...]]] = None
    past_mixer_hidden: Optional[torch.Tensor] = None
    past_injector_sa_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    past_cond: Optional[torch.Tensor] = None  # [B, cond_dim]


# =============================================================================
# Main Model Class
# =============================================================================

class RecurrentLLaDAModel(LLaDAPreTrainedModel):
    """
    Recurrent LLaDA Model - Memory-Slot Augmented Discrete Diffusion.

    Uses fixed memory slots [B, M, D_s] with three shared modules:
    - time_conditioner: Sinusoidal → MLP, shared by all modules
    - mixer: CrossAttn(Q=state, KV=hidden) → context
    - updater: GRU-style gated state update
    - injector: CrossAttn(Q=embed, KV=state) + zero-bridge
    """

    config_class = RecurrentLLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDALlamaBlock"]

    def __init__(
        self,
        config: RecurrentLLaDAConfig,
        model: Optional[LLaDAModel] = None,
        init_params: bool = False,
    ):
        super().__init__(config)

        # Convert HF config to internal module config
        self.recurrent_config = create_recurrent_module_config(config)

        # Backbone setup
        if model is None:
            self.recurrent_config.init_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = LLaDAModel(
                RecurrentModelConfig(
                    **{f.name: getattr(config, f.name, getattr(ModelConfig(), f.name, None))
                       for f in fields(ModelConfig)
                       if f.name != "init_device"},
                    init_device=self.recurrent_config.init_device,
                ),
                init_params=init_params,
            )
        else:
            self.model = model

        # Shared time conditioner
        cond_dim = self.recurrent_config.get_effective_time_cond_dim()
        mixer_dim = self.recurrent_config.get_effective_mixer_dim()
        self.time_conditioner = SharedTimeConditioner(
            cond_dim,
            mixer_dim=mixer_dim,
            initializer_range=self.recurrent_config.initializer_range,
            disable_time_conditioning=self.recurrent_config.disable_time_conditioning,
        )

        # Recurrent modules (from shared library)
        self.mixer = RecurrentMixer(self.recurrent_config)
        self.updater = RecurrentUpdater(self.recurrent_config)
        self.injector = RecurrentInjector(self.recurrent_config)

        # Initialize recurrent module parameters
        self.time_conditioner.reset_parameters()
        self.mixer.reset_parameters()
        self.updater.reset_parameters()
        self.injector.reset_parameters()


    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func: Callable = None
    ):
        self.model._set_gradient_checkpointing(enable, gradient_checkpointing_func)

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.model.set_activation_checkpointing(strategy)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        # Recurrent-specific arguments
        t: Union[int, torch.Tensor, None] = None,
        state: Optional[torch.FloatTensor] = None,
        # Fast-dLLM cache support
        replace_position: Optional[torch.Tensor] = None,
        # Mixer hidden cache (raw, pre-norm)
        past_mixer_hidden: Optional[torch.Tensor] = None,
        # Injector self-attn KV cache: (K, V) each [B, KVH, N, head_dim]
        past_injector_sa_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # Cached conditioning vector from previous step's Mixer/Updater
        past_cond: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, RecurrentLLaDAModelOutput]:
        """
        Forward pass with recurrent memory-slot state.

        Args:
            input_ids: Input token IDs [B, N]
            inputs_embeds: Pre-computed embeddings [B, N, D]
            t: Diffusion timestep [B] or scalar
            state: Previous recurrent state [B, M, D_s] or None
            replace_position: Boolean mask [B, L] for dual cache mode
            past_mixer_hidden: Mixer cache: down-projected hidden [B, N, mixer_dim] from previous step
            past_injector_sa_kv: Injector self-attn KV cache: (K, V) each [B, KVH, N, hd]
            past_cond: Conditioning vector [B, cond_dim] from previous step, passed directly to Injector

        Returns:
            RecurrentLLaDAModelOutput with logits, new_state, etc.
        """
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not supported in RecurrentLLaDA")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # 1. Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        B, N, D = inputs_embeds.shape

        # 2. Compute replace offset (shared by Injector and Mixer)
        kv_replace_offset = (
            compute_kv_replace_offset(replace_position)
            if replace_position is not None
            else None
        )

        # 3. Injector conditioning: use cached cond directly from previous step
        ref_embed = inputs_embeds[:, 0]  # save BEFORE injection modifies inputs_embeds
        if past_cond is not None:
            injector_cond = past_cond
        else:
            injector_cond = inputs_embeds.new_zeros(B, self.time_conditioner.cond_dim)

        # 4a. Initialize state if None, but preserve bootstrap skip-inject semantics
        state_was_none = state is None
        if state_was_none:
            state = self.updater.get_initial_state(
                B, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        if self.recurrent_config.debug_stats:
            embedding_norm = inputs_embeds.detach().norm(p=2, dim=-1).mean(dim=-1)
        else:
            embedding_norm = None

        # 4b. Injector
        injector_state = None if state_was_none else state
        injected_embeds, delta, present_injector_sa_kv = self.injector(
            inputs_embeds, injector_state, injector_cond,
            past_injector_sa_kv=past_injector_sa_kv,
            use_cache=use_cache,
            kv_replace_offset=kv_replace_offset,
            attention_mask=attention_mask,
        )
        if not state_was_none:
            inputs_embeds = injected_embeds
            delta_norm = (
                delta.detach().norm(p=2, dim=-1).mean(dim=-1)
                if self.recurrent_config.debug_stats
                else None
            )
        else:
            delta_norm = (
                torch.zeros(B, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                if self.recurrent_config.debug_stats
                else None
            )

        # 5. Backbone forward pass (attention_mask=None enables Flash Attention)
        backbone_kwargs = dict(
            input_ids=None,
            input_embeddings=inputs_embeds,
            attention_mask=None,
            attention_bias=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            # The recurrent mixer consumes the final hidden state from the backbone.
            # LLaDAModel does not expose last_hidden_state separately, so the wrapper
            # must request hidden states internally even when it does not return them.
            output_hidden_states=True,
        )
        if replace_position is not None:
            backbone_kwargs["replace_position"] = replace_position

        outputs = self.model(**backbone_kwargs)

        logits = outputs.logits
        backbone_past_key_values = getattr(outputs, "attn_key_values", None)
        last_hidden = outputs.hidden_states[-1]  # [B, N, D] post-final-norm

        # 6. Mixer + Updater
        h_proj_full, present_mixer_hidden, content_summary = self.mixer.prepare(
            last_hidden,
            past_mixer_hidden=past_mixer_hidden,
            use_cache=use_cache,
            kv_replace_offset=kv_replace_offset,
            attention_mask=attention_mask,
        )
        cond = self.time_conditioner(ref_embed, t, content_summary=content_summary)
        context = self.mixer.process(state, h_proj_full, cond, attention_mask)

        # 7. Updater: update state
        new_state = self.updater(state, context, cond)  # [B, M, D_s]

        if not return_dict:
            return (logits, new_state)

        return RecurrentLLaDAModelOutput(
            logits=logits,
            new_state=new_state,
            all_hidden_states=outputs.hidden_states if output_hidden_states else None,
            delta_norm=delta_norm,
            embedding_norm=embedding_norm,
            past_key_values=backbone_past_key_values,
            past_mixer_hidden=present_mixer_hidden,
            past_injector_sa_kv=present_injector_sa_kv,
            past_cond=cond,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple]] = None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}
        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def get_input_embeddings(self) -> nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self) -> nn.Module:
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte
