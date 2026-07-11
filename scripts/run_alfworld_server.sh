#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-}"
MODEL_ID="${MODEL_ID:-${ECR_GRPO_MODEL_ID:-}}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
BASE_CONFIG="${BASE_CONFIG:-configs/alfworld_gated_smoke.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/alfworld_fair}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
KERNELS="${KERNELS:-grpo local recency evidence gated}"
SEEDS="${SEEDS:-7}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-eval_out_of_distribution}"
EVAL_SPLITS="${EVAL_SPLITS:-eval_in_distribution eval_out_of_distribution}"
NUM_TRAIN_TASKS="${NUM_TRAIN_TASKS:-32}"
NUM_EVAL_TASKS="${NUM_EVAL_TASKS:-32}"
MAX_STEPS="${MAX_STEPS:-50}"
NUM_UPDATES="${NUM_UPDATES:-}"
TASKS_PER_UPDATE="${TASKS_PER_UPDATE:-}"
GROUP_SIZE="${GROUP_SIZE:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
TRAIN_DELAY_PROB="${TRAIN_DELAY_PROB:-}"
TERMINAL_REWARD_DELAY="${TERMINAL_REWARD_DELAY:-}"
MISSING_REWARD_PROB="${MISSING_REWARD_PROB:-}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_EVAL="${CLEAN_EVAL:-1}"

if [[ "${OVERWRITE}" == "1" && "${RESUME}" == "1" ]]; then
  echo "[error] OVERWRITE=1 and RESUME=1 are mutually exclusive." >&2
  exit 2
fi

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
  --eval-splits ${EVAL_SPLITS}
  --num-train-tasks "${NUM_TRAIN_TASKS}"
  --num-eval-tasks "${NUM_EVAL_TASKS}"
  --max-steps "${MAX_STEPS}"
)

if [[ -n "${SUMMARY_PATH}" ]]; then
  cmd+=(--summary-path "${SUMMARY_PATH}")
fi
if [[ -n "${ADAPTER_PATH}" ]]; then
  cmd+=(--adapter-path "${ADAPTER_PATH}")
fi
if [[ -n "${NUM_UPDATES}" ]]; then
  cmd+=(--num-updates "${NUM_UPDATES}")
fi
if [[ -n "${TASKS_PER_UPDATE}" ]]; then
  cmd+=(--tasks-per-update "${TASKS_PER_UPDATE}")
fi
if [[ -n "${GROUP_SIZE}" ]]; then
  cmd+=(--group-size "${GROUP_SIZE}")
fi
if [[ -n "${EVAL_EVERY}" ]]; then
  cmd+=(--eval-every "${EVAL_EVERY}")
fi
if [[ -n "${TRAIN_DELAY_PROB}" ]]; then
  cmd+=(--train-delay-prob "${TRAIN_DELAY_PROB}")
fi
if [[ -n "${TERMINAL_REWARD_DELAY}" ]]; then
  cmd+=(--terminal-reward-delay "${TERMINAL_REWARD_DELAY}")
fi
if [[ -n "${MISSING_REWARD_PROB}" ]]; then
  cmd+=(--missing-reward-prob "${MISSING_REWARD_PROB}")
fi

if [[ "${CLEAN_EVAL}" == "1" ]]; then
  cmd+=(--clean-eval)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

echo "[alfworld] config=${ALFWORLD_CONFIG}"
echo "[alfworld] model=${MODEL_ID}"
echo "[alfworld] kernels=${KERNELS} seeds=${SEEDS}"
echo "[alfworld] train_split=${TRAIN_SPLIT} eval_split=${EVAL_SPLIT} eval_splits=${EVAL_SPLITS}"
echo "[alfworld] train_tasks=${NUM_TRAIN_TASKS} eval_tasks=${NUM_EVAL_TASKS} max_steps=${MAX_STEPS}"
echo "[alfworld] updates=${NUM_UPDATES:-config} tasks_per_update=${TASKS_PER_UPDATE:-config} group_size=${GROUP_SIZE:-config}"
echo "[alfworld] train_delay=${TRAIN_DELAY_PROB:-config} terminal_delay=${TERMINAL_REWARD_DELAY:-config} missing_reward=${MISSING_REWARD_PROB:-config}"
echo "[alfworld] output=${OUTPUT_ROOT} gpu=${GPU_ID} clean_eval=${CLEAN_EVAL} dry_run=${DRY_RUN} overwrite=${OVERWRITE} resume=${RESUME}"
if [[ -n "${SUMMARY_PATH}" ]]; then
  echo "[alfworld] summary=${SUMMARY_PATH}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
