import json
import os

import accelerate
import torch
import transformers
from safetensors.torch import load_file

from dllm.utils.configs import ModelArguments
from dllm.utils.utils import print_main


RECURRENT_META_FILENAME = "recurrent_checkpoint_meta.json"
RECURRENT_WEIGHTS_FILENAME = "recurrent_modules.safetensors"
RECURRENT_MODULE_NAMES = (
    "time_conditioner",
    "mixer",
    "updater",
    "injector",
)

_LLADA_CHAT_TEMPLATE = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""


def is_lightweight_recurrent_checkpoint(path: str) -> bool:
    return os.path.exists(os.path.join(path, RECURRENT_META_FILENAME))


def get_recurrent_checkpoint_model_type(path: str) -> str | None:
    meta_path = os.path.join(path, RECURRENT_META_FILENAME)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return metadata.get("model_type", "recurrent_llada")


def _device_map():
    if (
        not transformers.modeling_utils.is_deepspeed_zero3_enabled()
        and torch.cuda.is_available()
    ):
        return {"": accelerate.PartialState().local_process_index}
    return None


def _base_loading_params(model_args, config=None) -> dict:
    params = {
        "dtype": getattr(model_args, "dtype", "bfloat16"),
        "device_map": _device_map(),
        "attn_implementation": getattr(model_args, "attn_implementation", None),
    }
    if config is not None:
        params["config"] = config
    return {key: value for key, value in params.items() if value is not None}


def _load_llada_backbone(model_name_or_path: str, params: dict):
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM

    backbone_full = LLaDAModelLM.from_pretrained(model_name_or_path, **params)
    return backbone_full.model if hasattr(backbone_full, "model") else backbone_full


def _iter_recurrent_modules(model):
    for name in RECURRENT_MODULE_NAMES:
        module = getattr(model, name, None)
        if module is not None:
            yield module


def _set_module_requires_grad(module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad


def _get_backbone_reference(model) -> tuple[torch.device | None, torch.dtype | None]:
    backbone = getattr(model, "model", None)
    if backbone is None:
        return None, None

    device = getattr(backbone, "device", None)
    dtype = None
    for param in backbone.parameters():
        if device is None:
            device = param.device
        if param.is_floating_point():
            dtype = param.dtype
            break
    return device, dtype


def _move_recurrent_modules(model, dtype: torch.dtype | None):
    device, backbone_dtype = _get_backbone_reference(model)
    if device is None or str(device) == "meta":
        return

    target_dtype = dtype if dtype is not None else backbone_dtype
    for module in _iter_recurrent_modules(model):
        if target_dtype is None:
            module.to(device)
        else:
            module.to(device=device, dtype=target_dtype)


def _configure_recurrent_runtime(model, model_args, training: bool):
    if training:
        if getattr(model_args, "freeze_backbone", False):
            print_main("Freezing Backbone...")
            if model.model is not None:
                for param in model.model.parameters():
                    param.requires_grad = False
        for module in _iter_recurrent_modules(model):
            _set_module_requires_grad(module, True)
        _move_recurrent_modules(model, dtype=None)
        return

    for param in model.parameters():
        param.requires_grad = False
    _move_recurrent_modules(model, dtype=None)


def _freeze_backbone_and_ensure_trainable(model, model_args, freeze_lm_head: bool = False):
    _configure_recurrent_runtime(model, model_args, training=True)


def _move_recurrent_to_backbone_device(model):
    _move_recurrent_modules(model, dtype=None)


def _recurrent_kwargs_from_args(model_args) -> dict:
    recurrent_keys = [
        "state_hidden_size",
        "num_memory_slots",
        "time_cond_dim",
        "unroll_steps",
        "unroll_steps_max",
        "warmup_k_fraction",
        "aux_loss_weights",
        "debug_stats",
        "initializer_range",
        "mixer_dim",
        "mixer_num_heads",
        "mixer_use_gqa",
        "mixer_mlp_ratio",
        "mixer_slot_attn",
        "updater_z_gate_bias",
        "updater_r_gate_bias",
        "injector_dim",
        "injector_num_heads",
        "injector_use_gqa",
        "injector_mlp_ratio_bottleneck",
        "disable_time_conditioning",
    ]
    return {
        key: getattr(model_args, key)
        for key in recurrent_keys
        if hasattr(model_args, key)
    }


def load_lightweight_recurrent_checkpoint(
    checkpoint_path: str,
    model_args,
    params: dict,
    recurrent_type: str = "llada",
    training: bool = False,
) -> "transformers.PreTrainedModel":
    checkpoint_model_type = get_recurrent_checkpoint_model_type(checkpoint_path)
    if checkpoint_model_type not in (None, "recurrent_llada"):
        raise ValueError(
            f"Only recurrent_llada checkpoints are supported, got {checkpoint_model_type!r}"
        )
    if recurrent_type != "llada":
        raise ValueError("Only recurrent_type='llada' is supported.")

    from dllm.pipelines.recurrent_llada.configs.recurrent_llada_config import (
        RecurrentLLaDAConfig,
    )
    from dllm.pipelines.recurrent_llada.models.modeling_recurrent_llada import (
        RecurrentLLaDAModel,
    )

    meta_path = os.path.join(checkpoint_path, RECURRENT_META_FILENAME)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    backbone_path = metadata.get(
        "backbone_name_or_path", "GSAI-ML/LLaDA-8B-Instruct"
    )
    print_main(f"Loading lightweight Recurrent LLaDA checkpoint from {checkpoint_path}")
    print_main(f"  Backbone: {backbone_path}")

    config = RecurrentLLaDAConfig.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    backbone_params = {key: value for key, value in params.items() if key != "config"}
    backbone = _load_llada_backbone(backbone_path, backbone_params)
    model = RecurrentLLaDAModel(config, model=backbone)

    weights_path = os.path.join(checkpoint_path, RECURRENT_WEIGHTS_FILENAME)
    if os.path.exists(weights_path):
        recurrent_state_dict = load_file(weights_path)
        for name in RECURRENT_MODULE_NAMES:
            module = getattr(model, name, None)
            if module is None:
                continue
            prefix = f"{name}."
            module_state = {
                key[len(prefix):]: value
                for key, value in recurrent_state_dict.items()
                if key.startswith(prefix)
            }
            if module_state:
                missing, unexpected = module.load_state_dict(module_state, strict=False)
                if missing:
                    print_main(f"  {name}: missing keys: {missing}")
                if unexpected:
                    print_main(f"  {name}: unexpected keys: {unexpected}")
    else:
        print_main(f"Warning: no recurrent weights found at {weights_path}")

    _configure_recurrent_runtime(model, model_args, training=training)
    return model


def get_model(
    model_args: ModelArguments,
    config: transformers.PretrainedConfig | None = None,
    is_recurrent: bool = False,
    recurrent_type: str = "llada",
    training: bool = False,
) -> transformers.PreTrainedModel:
    model_name_or_path = getattr(model_args, "model_name_or_path")
    params = _base_loading_params(model_args, config=config)

    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.pipelines.recurrent_llada.configs.recurrent_llada_config import (
        RecurrentLLaDAConfig,
    )
    from dllm.pipelines.recurrent_llada.models.modeling_recurrent_llada import (
        RecurrentLLaDAModel,
    )

    if recurrent_type != "llada":
        raise ValueError("Only recurrent_type='llada' is supported.")

    if is_lightweight_recurrent_checkpoint(model_name_or_path):
        return load_lightweight_recurrent_checkpoint(
            model_name_or_path,
            model_args,
            params,
            recurrent_type=recurrent_type,
            training=training,
        )

    try:
        loaded_config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    except Exception:
        loaded_config = None

    model_type = getattr(loaded_config, "model_type", None)
    if model_type == "recurrent_llada":
        print_main(f"Loading full Recurrent LLaDA model from {model_name_or_path}")
        model = RecurrentLLaDAModel.from_pretrained(
            model_name_or_path, trust_remote_code=True, **params
        )
        _configure_recurrent_runtime(model, model_args, training=training)
        return model

    if is_recurrent:
        if model_type not in (None, "llada"):
            raise ValueError(
                f"Only LLaDA backbones can initialize Recurrent LLaDA, got {model_type!r}"
            )
        print_main(f"Initializing Recurrent LLaDA model from {model_name_or_path}")
        base_config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        final_config = RecurrentLLaDAConfig(**base_config.to_dict())
        for key, value in _recurrent_kwargs_from_args(model_args).items():
            setattr(final_config, key, value)
        final_config._validate()

        backbone_params = {key: value for key, value in params.items() if key != "config"}
        backbone = _load_llada_backbone(model_name_or_path, backbone_params)
        model = RecurrentLLaDAModel(final_config, model=backbone)
        _configure_recurrent_runtime(model, model_args, training=training)
        return model

    if model_type not in (None, "llada"):
        raise ValueError(f"Only base LLaDA models are supported, got {model_type!r}")
    return LLaDAModelLM.from_pretrained(model_name_or_path, **params)


def get_tokenizer(model_args) -> transformers.PreTrainedTokenizer:
    model_name_or_path = getattr(model_args, "model_name_or_path")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.eos_token is None and tokenizer.pad_token is None:
        raise ValueError("LLaDA tokenizer must define eos_token or pad_token.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.pad_token

    tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
    tokenizer.eot_token = "<|eot_id|>"
    tokenizer.chat_template = _LLADA_CHAT_TEMPLATE
    return tokenizer
