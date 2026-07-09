#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-}"
MODEL_ID="${MODEL_ID:-${ECR_GRPO_MODEL_ID:-}}"
BASE_CONFIG="${BASE_CONFIG:-configs/alfworld_gated_smoke.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/alfworld_fair}"
KERNELS="${KERNELS:-grpo recency evidence gated}"
SEEDS="${SEEDS:-7}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-eval_out_of_distribution}"
NUM_TRAIN_TASKS="${NUM_TRAIN_TASKS:-32}"
NUM_EVAL_TASKS="${NUM_EVAL_TASKS:-32}"
MAX_STEPS="${MAX_STEPS:-50}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_EVAL="${CLEAN_EVAL:-1}"

if [[ -z "${ALFWORLD_CONFIG}" ]]; then
  echo "[error] Set ALFWORLD_CONFIG to your ALFWorld base_config.yaml path." >&2
  exit 2
fi
if [[ -z "${MODEL_ID}" ]]; then
  echo "[error] Set MODEL_ID or ECR_GRPO_MODEL_ID to your HF model path." >&2
  exit 2
fi

cmd=(
  python -m ecr_grpo.run_alfworld
  --base-config "${BASE_CONFIG}"
  --output-root "${OUTPUT_ROOT}"
  --alfworld-config "${ALFWORLD_CONFIG}"
  --model-id "${MODEL_ID}"
  --kernels ${KERNELS}
  --seeds ${SEEDS}
  --train-split "${TRAIN_SPLIT}"
  --eval-split "${EVAL_SPLIT}"
  --num-train-tasks "${NUM_TRAIN_TASKS}"
  --num-eval-tasks "${NUM_EVAL_TASKS}"
  --max-steps "${MAX_STEPS}"
)

if [[ "${CLEAN_EVAL}" == "1" ]]; then
  cmd+=(--clean-eval)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

echo "[alfworld] config=${ALFWORLD_CONFIG}"
echo "[alfworld] model=${MODEL_ID}"
echo "[alfworld] kernels=${KERNELS} seeds=${SEEDS}"
echo "[alfworld] train_split=${TRAIN_SPLIT} eval_split=${EVAL_SPLIT}"
echo "[alfworld] train_tasks=${NUM_TRAIN_TASKS} eval_tasks=${NUM_EVAL_TASKS} max_steps=${MAX_STEPS}"
echo "[alfworld] output=${OUTPUT_ROOT} gpu=${GPU_ID} clean_eval=${CLEAN_EVAL} dry_run=${DRY_RUN} overwrite=${OVERWRITE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
