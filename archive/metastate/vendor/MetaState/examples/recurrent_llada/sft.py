"""
Recurrent LLaDA SFT Training Script.

This script trains the recurrent state mechanism on top of a frozen LLaDA backbone.
Only the recurrent modules (TimeConditioner, Mixer, Updater, Injector) are trained,
which amounts to approximately 1-5% of total model parameters.

Architecture Overview:
----------------------
Recurrent LLaDA addresses the "Information Island" problem in discrete diffusion
by maintaining working memory across denoising steps. The trainable modules are:

1. TimeConditioner: Shared sinusoidal time embedding → MLP
2. Mixer: Extracts backbone information with cross-attention (Q=state, K/V=hidden)
3. Updater: GRU-style gated state update with time conditioning
4. Injector: Injects state into embeddings via ControlNet-style cross-attention

Usage:
------
Use the release launcher for the paper configuration after setting `DATASET_ARGS`
and `OUTPUT_DIR`:

    DATASET_ARGS=/path/to/tulu3[train:20000,test:20] \
    OUTPUT_DIR=/path/to/output_dir \
    bash sft_llada_1_5_768_20000.sh

or call this entry point through Accelerate with:

    accelerate launch \\
        --config_file scripts/accelerate_configs/zero1.yaml \\
        examples/recurrent_llada/sft.py \\
        --output_dir /path/to/output_dir

Lightweight Checkpointing:
--------------------------
The trainer saves only recurrent module weights (~100-500MB) instead of the full
model (~16GB). Loading requires specifying the backbone:

    model = RecurrentLLaDAModel.from_pretrained(
        "path/to/checkpoint",
        trust_remote_code=True,
    )
"""

from dataclasses import dataclass, field
from typing import Optional

import accelerate
import transformers

import dllm
from dllm.core.trainers.recurrent_mdlm import RecurrentMDLMTrainer

logger = dllm.utils.get_default_logger(__name__)
DEFAULT_NEW_RUN_TIME_COND_DIM = 512


def prepare_recurrent_model_args(model_args: "ModelArguments") -> None:
    """Resolve training-entry defaults before model construction."""
    if model_args.time_cond_dim is None:
        model_args.time_cond_dim = DEFAULT_NEW_RUN_TIME_COND_DIM

    aux_loss_weights = {}
    if model_args.state_norm_weight > 0:
        aux_loss_weights["state_norm"] = model_args.state_norm_weight
    if model_args.slot_diversity_weight > 0:
        aux_loss_weights["slot_diversity"] = model_args.slot_diversity_weight
    model_args.aux_loss_weights = aux_loss_weights


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    """
    Model arguments for Recurrent LLaDA training.

    These arguments control the model architecture, including both the frozen
    backbone and the trainable recurrent modules.
    """

    # =========================================================================
    # Backbone Configuration
    # =========================================================================
    model_name_or_path: str = field(
        default="GSAI-ML/LLaDA-8B-Instruct",
        metadata={
            "help": "Path or HuggingFace model ID of the LLaDA or Recurrent LLaDA backbone. "
        }
    )
    backbone_name_or_path: str = field(
        default="GSAI-ML/LLaDA-8B-Instruct",
        metadata={
            "help": "Path to frozen backbone for lightweight checkpoints. "
                    "Typically same as model_name_or_path."
        }
    )
    freeze_backbone: bool = field(
        default=True,
        metadata={
            "help": "Whether to freeze the LLaDA backbone. Should be True for "
                    "standard Recurrent LLaDA training (only recurrent modules trained)."
        }
    )

    # =========================================================================
    # State Dimensions
    # =========================================================================
    # The state carries information across diffusion steps. Larger sizes allow
    # more information to be stored but increase memory and compute costs.
    state_hidden_size: int = field(
        default=1024,
        metadata={
            "help": "Size of the recurrent state vector (D_s). Default for "
                    "mixer_dim and injector_dim when those are None. Recommended: 1024. "
                    "Larger values = more memory capacity."
        }
    )

    # =========================================================================
    # Memory Slots
    # =========================================================================
    num_memory_slots: int = field(
        default=64,
        metadata={
            "help": "Number of fixed latent memory slots M. State shape is "
                    "[B, M, D_s], decoupled from sequence length."
        }
    )

    # =========================================================================
    # Training Settings
    # =========================================================================
    unroll_steps: int = field(
        default=4,
        metadata={
            "help": "Number of diffusion steps K to unroll during training. "
                    "The trainer starts from fully masked input and iteratively "
                    "unmasks over K steps."
        }
    )
    unroll_steps_max: Optional[int] = field(
        default=None,
        metadata={
            "help": "Target K_max for variable-K training. None = disabled "
                    "(fixed K = unroll_steps). When set, K_total is sampled "
                    "from Uniform[1, effective_K_max] per batch."
        }
    )
    warmup_k_fraction: float = field(
        default=0.5,
        metadata={
            "help": "Fraction of training to ramp effective K_max from "
                    "unroll_steps to unroll_steps_max."
        }
    )
    # =========================================================================
    # Auxiliary Loss Weights
    # =========================================================================
    # Auxiliary losses help regularize training.
    state_norm_weight: float = field(
        default=0,
        metadata={
            "help": "Weight for L2 penalty on state norm. Helps prevent state from "
                    "growing unbounded. Set to 0 to disable."
        }
    )
    slot_diversity_weight: float = field(
        default=0,
        metadata={
            "help": "Weight for slot diversity regularization. Set to 0 to disable."
        }
    )

    # =========================================================================
    # Debug Settings
    # =========================================================================
    debug_stats: bool = field(
        default=False,
        metadata={
            "help": "Whether to collect debug statistics (state norm, updater gate, "
                    "injection ratio) during training. Useful for monitoring."
        }
    )

    # =========================================================================
    # Time Conditioner
    # =========================================================================
    time_cond_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Output dimension of SharedTimeConditioner. "
                    "If omitted in this training entrypoint, it is resolved to 512 "
                    "for new runs. Legacy checkpoints with time_cond_dim=null still "
                    "load through the model fallback. Shared by Updater and Injector "
                    "time projections."
        }
    )

    # =========================================================================
    # Initialization
    # =========================================================================
    initializer_range: float = field(
        default=0.02,
        metadata={
            "help": "Standard deviation for truncated normal initialization of "
                    "recurrent module weights."
        }
    )
    # =========================================================================
    # Mixer Module Params
    # =========================================================================
    mixer_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Internal processing dim for Mixer. None → state_hidden_size."
        }
    )
    mixer_num_heads: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of attention heads in Mixer cross-attention. "
                    "If None, auto-computed as max(1, min(8, mixer_dim // 64)). "
                    "Must divide mixer_dim evenly. mixer_dim defaults to state_hidden_size."
        }
    )
    mixer_use_gqa: bool = field(
        default=True,
        metadata={
            "help": "Use Grouped Query Attention in Mixer (kv_heads = q_heads // 2). "
                    "Reduces memory for cross-attention over long sequences."
        }
    )

    # =========================================================================
    # Updater Module Params
    # =========================================================================
    updater_z_gate_bias: float = field(
        default=0.0,
        metadata={
            "help": "Initial bias for Updater's update gate (z). "
                    "Default 0.0 → sigmoid=0.5 → neutral between old state and new candidate. "
                    "More negative = more conservative state updates."
        }
    )
    updater_r_gate_bias: float = field(
        default=0.0,
        metadata={
            "help": "Initial bias for Updater's reset gate (r). "
                    "sigmoid(0.0)=0.5 means neutral reset at init."
        }
    )
    mixer_slot_attn: bool = field(
        default=True,
        metadata={
            "help": "Enable pre-cross-attn slot self-attention in Mixer. "
                    "Provides inter-slot communication before reading from backbone."
        }
    )

    # =========================================================================
    # Injector Module Params
    # =========================================================================
    injector_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Bottleneck dim for Injector. None → state_hidden_size."
        }
    )
    injector_num_heads: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of attention heads in Injector cross-attention. "
                    "If None, auto-computed as max(1, min(8, injector_dim // 64)). "
                    "Must divide injector_dim evenly. injector_dim defaults to state_hidden_size."
        }
    )
    injector_use_gqa: bool = field(
        default=True,
        metadata={
            "help": "Use Grouped Query Attention in Injector (kv_heads = q_heads // 2)."
        }
    )
    injector_mlp_ratio_bottleneck: float = field(
        default=2.0,
        metadata={
            "help": "FFN expansion ratio in Injector bottleneck SwiGLU MLP. "
                    "hidden_dim = bottleneck_dim * ratio."
        }
    )
    mixer_mlp_ratio: float = field(
        default=2.0,
        metadata={
            "help": "FFN expansion ratio for Mixer's SwiGLU MLP. "
                    "hidden_dim = mixer_dim * ratio. mixer_dim defaults to state_hidden_size."
        }
    )

    disable_time_conditioning: bool = field(
        default=False,
        metadata={"help": "Disable time conditioning (all AdaRMSNorm to RMSNorm)."}
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    """
    Data arguments for Recurrent LLaDA training.
    """

    dataset_args: str = field(
        default="allenai/tulu-3-sft-mixture[train:10000,test:1000]",
        metadata={
            "help": "Dataset specification in format: 'dataset_name[splits]'. "
                    "Example: 'allenai/tulu-3-sft-mixture[train:10000,test:1000]'"
        }
    )
    num_proc: int = field(
        default=64,
        metadata={"help": "Number of processes for dataset preprocessing."}
    )
    truncation: str = field(
        default="filter",
        metadata={
            "help": "Truncation strategy: 'right' (keep rightmost tokens), "
            "or 'filter' (discard long sequences)."
        }
    )


@dataclass
class TrainingArguments(RecurrentMDLMTrainer.RecurrentMDLMConfig):
    """
    Training arguments for Recurrent LLaDA.

    Inherits from RecurrentMDLMConfig which provides recurrent diffusion settings.
    """

    output_dir: str = field(
        default="models/Recurrent-LLaDA/default",
        metadata={"help": "Directory for saving checkpoints and logs."}
    )

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    num_train_epochs: float = field(
        default=1,
        metadata={"help": "Total number of training epochs."}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={
            "help": "Peak learning rate for recurrent modules. "
                    "Slightly higher than typical fine-tuning since modules are new."
        }
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during training."}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during evaluation."}
    )

    # =========================================================================
    # Optimization Settings
    # =========================================================================
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler type."}
    )
    warmup_ratio: float = field(
        default=0.05,
        metadata={"help": "Ratio of total steps for linear warmup."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for clipping."}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW beta1 parameter."}
    )
    adam_beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW beta2 parameter."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )

    # =========================================================================
    # Data Settings
    # =========================================================================
    group_by_length: bool = field(
        default=True,
        metadata={"help": "Whether to group samples by length for efficient batching."}
    )

def train():
    """Main training function."""
    # =========================================================================
    # Argument Parsing
    # =========================================================================
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    prepare_recurrent_model_args(model_args)

    # Pass backbone path for lightweight checkpoint saving
    training_args.backbone_name_or_path = model_args.backbone_name_or_path

    # Print configuration
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # =========================================================================
    # Model Setup
    # =========================================================================
    model = dllm.utils.get_model(
        model_args=model_args,
        is_recurrent=True,
        recurrent_type="llada",
        training=True,
    )

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable Parameters: {trainable_params:,} / {all_params:,} "
        f"({trainable_params/all_params:.2%})"
    )

    # =========================================================================
    # Tokenizer Setup
    # =========================================================================
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Always use eos_token as pad_token so padding positions contain valid
    # vocab tokens — backbone sees reasonable hidden states without attention_mask.
    tokenizer.pad_token = tokenizer.eos_token

    # Ensure mask token is set (required for diffusion)
    if tokenizer.mask_token_id is None:
        mdm = "<|mdm_mask|>"
        mdm_id = tokenizer.convert_tokens_to_ids(mdm)
        if mdm_id is not None and mdm_id != tokenizer.unk_token_id:
            tokenizer.mask_token = mdm
            logger.info(f"Set mask_token to '{mdm}' (ID: {tokenizer.mask_token_id})")
        else:
            raise ValueError(
                f"Tokenizer has no mask_token and '{mdm}' is not in vocab. "
                "LLaDA requires a mask token for diffusion."
            )

    # =========================================================================
    # Dataset Setup
    # =========================================================================
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
        )
        # Truncate/filter long sequences
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # =========================================================================
    # Training
    # =========================================================================
    accelerate.PartialState().wait_for_everyone()
    logger.info("Starting training...")

    trainer = RecurrentMDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=-100,
        ),
    )
    trainer.train()

    # =========================================================================
    # Save Final Model
    # =========================================================================
    # The trainer._save method handles lightweight checkpointing:
    # - Saves only recurrent module weights (not backbone)
    # - Saves config.json and trust_remote_code files
    # - Saves tokenizer
    trainer.save_model(training_args.output_dir)
    logger.info(f"Lightweight checkpoint saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
