#!/usr/bin/env bash
# Evaluate Recurrent LLaDA checkpoint-* subdirs or one direct checkpoint.

set -euo pipefail

export PYTHONPATH=.:${PYTHONPATH:-}
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

output_dir=""
checkpoint_path=""
num_gpu=2
use_cache="dual"
threshold="0.9"
block_size="32"
max_new_tokens="256"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)
      output_dir="$2"; shift 2 ;;
    --checkpoint_path)
      checkpoint_path="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    --use_cache)
      use_cache="$2"; shift 2 ;;
    --threshold)
      threshold="$2"; shift 2 ;;
    --block_size)
      block_size="$2"; shift 2 ;;
    --max_new_tokens)
      max_new_tokens="$2"; shift 2 ;;
    *)
      echo "Error: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "${checkpoint_path}" ]]; then
  if [[ ! -d "${checkpoint_path}" ]]; then
    echo "Error: checkpoint_path does not exist: ${checkpoint_path}" >&2; exit 1
  fi
  output_dir="$(dirname "${checkpoint_path}")"
  checkpoints=("direct")
  _direct_ckpt_path="${checkpoint_path}"
  echo ">>> Direct checkpoint mode: ${checkpoint_path}"
elif [[ -n "${output_dir}" ]]; then
  if [[ ! -d "${output_dir}" ]]; then
    echo "Error: output_dir does not exist: ${output_dir}" >&2; exit 1
  fi

  checkpoints=()
  for d in "${output_dir}"/checkpoint-*; do
    [[ -d "$d" ]] || continue
    step="${d##*checkpoint-}"
    if [[ "$step" =~ ^[0-9]+$ ]]; then
      checkpoints+=("${step}")
    fi
  done

  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "Error: No checkpoint-* directories found in ${output_dir}" >&2; exit 1
  fi

  IFS=$'\n' checkpoints=($(printf '%s\n' "${checkpoints[@]}" | sort -rn)); unset IFS
  echo ">>> Found ${#checkpoints[@]} checkpoints: ${checkpoints[*]}"

  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "Error: No checkpoints found" >&2; exit 1
  fi

  echo ">>> Will evaluate (descending): ${checkpoints[*]}"
else
  echo "Error: --output_dir or --checkpoint_path is required" >&2; exit 1
fi
echo

detect_model_type() {
  local ckpt_dir="$1"
  local meta="${ckpt_dir}/recurrent_checkpoint_meta.json"
  if [[ ! -f "${meta}" ]]; then
    echo ""
    return
  fi
  python3 -c "import json; print(json.load(open('${meta}'))['model_type'])" 2>/dev/null
}

model_type=""
for step in "${checkpoints[@]}"; do
  if [[ "${step}" == "direct" ]]; then
    model_type=$(detect_model_type "${_direct_ckpt_path}")
  else
    model_type=$(detect_model_type "${output_dir}/checkpoint-${step}")
  fi
  if [[ -n "${model_type}" ]]; then
    break
  fi
done

if [[ "${model_type}" != "recurrent_llada" ]]; then
  echo "Error: expected model_type recurrent_llada, got '${model_type:-missing}'" >&2
  exit 1
fi

EVAL_PY="examples/recurrent_llada/eval.py"
common_args="--model recurrent_llada"
results_dir="results_llada"
csv_file="${results_dir}/aresults_checkpoint.csv"
output_dir_basename="$(basename "${output_dir}")"

mkdir -p "${results_dir}"

echo ">>> Detected model type: ${model_type}"
echo ">>> Eval script : ${EVAL_PY}"
echo ">>> Results dir : ${results_dir}"
echo ">>> CSV target  : ${csv_file}"
echo

run_one_task() {
  local ckpt_path="$1"
  local model_tag="$2"
  local task="$3"
  local fewshot="$4"
  local unsafe="$5"
  local apply_template="$6"

  local steps="${max_new_tokens}"
  local limit=1500
  local cache_tag="nocache"
  if [[ -n "${use_cache}" ]]; then
    cache_tag="${use_cache}"
  fi
  if [[ -n "${threshold}" ]]; then
    cache_tag="${cache_tag}_t${threshold}"
  fi
  if [[ -n "${block_size}" ]]; then
    cache_tag="${cache_tag}_bs${block_size}"
  fi

  local out="${results_dir}/eval_${model_tag}_${task}_limit${limit}_fs${fewshot}_s${steps}_mn${max_new_tokens}_${cache_tag}.out"
  local model_args="pretrained=${ckpt_path},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},cfg=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

  if [[ -n "${use_cache}" ]]; then
    model_args="${model_args},use_cache=${use_cache}"
  fi
  if [[ -n "${threshold}" ]]; then
    model_args="${model_args},threshold=${threshold}"
  fi

  echo "=== [${model_tag}] Task=${task} fewshot=${fewshot} -> ${out}"

  cmd=(accelerate launch --num_processes "${num_gpu}" "${EVAL_PY}"
       --tasks "${task}" --num_fewshot "${fewshot}"
       ${common_args} --limit "${limit}"
       --model_args "${model_args}")

  if [[ "${apply_template}" == "True" ]]; then
    cmd+=(--apply_chat_template)
  fi

  if [[ "${unsafe}" == "True" ]]; then
    cmd+=(--confirm_run_unsafe_code)
  fi

  if ! "${cmd[@]}" > "./${out}" 2>&1; then
    echo "Error: evaluation failed for task=${task}; see ./${out}" >&2
    return 1
  fi

  python scripts/collect_results.py "./${out}" "${csv_file}" \
      --model "${model_tag}" --task "${task}" \
      --fewshot "${fewshot}" --steps "${steps}" --max_new_tokens "${max_new_tokens}" \
      --limit "${limit}" --use_cache "${use_cache:-}" --threshold "${threshold:-}" \
      --block_size "${block_size:-}"
}

total_ckpts=${#checkpoints[@]}
ckpt_idx=0

for step in "${checkpoints[@]}"; do
  ckpt_idx=$((ckpt_idx + 1))
  if [[ "${step}" == "direct" ]]; then
    ckpt_path="${_direct_ckpt_path}"
    model_tag="$(basename "${_direct_ckpt_path}")"
  else
    ckpt_path="${output_dir}/checkpoint-${step}"
    model_tag="${output_dir_basename}_ckpt${step}"
  fi
  model_tag="${model_tag// /_}"

  echo
  echo "============================================================"
  echo ">>> Checkpoint ${ckpt_idx}/${total_ckpts}: checkpoint-${step}"
  echo ">>> Path: ${ckpt_path}"
  echo ">>> Tag:  ${model_tag}"
  echo "============================================================"

  tpl_gsm8k="True"; tpl_humaneval="True"; tpl_mbpp="False"; tpl_minerva="True"

  run_one_task "${ckpt_path}" "${model_tag}" "gsm8k"                5 "False" "${tpl_gsm8k}"
  run_one_task "${ckpt_path}" "${model_tag}" "humaneval_instruct"   0 "True"  "${tpl_humaneval}"
  run_one_task "${ckpt_path}" "${model_tag}" "mbpp"                 3 "True"  "${tpl_mbpp}"
  run_one_task "${ckpt_path}" "${model_tag}" "minerva_math500"      4 "True"  "${tpl_minerva}"
done

echo
echo "============================================================"
echo ">>> All checkpoint evaluations finished."
echo ">>> Results: ${csv_file}"
echo "============================================================"
