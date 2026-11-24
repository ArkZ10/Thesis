#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-GSAI-ML/LLaDA-1.5}"
: "${DATASET_ARGS:?Set DATASET_ARGS, e.g. /path/to/tulu3[train:20000,test:20]}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a writable training output directory}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29056}"

accelerate launch \
    --config_file scripts/accelerate_configs/zero1.yaml \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    examples/recurrent_llada/sft.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --backbone_name_or_path "${MODEL_NAME_OR_PATH}" \
    --state_hidden_size 1024 \
    --num_memory_slots 64 \
    --time_cond_dim 1024 \
    --mixer_dim 768 \
    --injector_dim 768 \
    --unroll_steps 4 \
    --reveal_position_policy random \
    --state_norm_weight 1e-4 \
    --slot_diversity_weight 0 \
    --dense_reveal_mix 0.75 \
    --dataset_args "${DATASET_ARGS}" \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 0.1 \
    --save_total_limit 3 \
    --output_dir "${OUTPUT_DIR}"

bash scripts/eval_checkpoint.sh \
    --output_dir "${OUTPUT_DIR}"
