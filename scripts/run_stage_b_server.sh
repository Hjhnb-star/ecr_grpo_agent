#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS="${SEEDS:-7 13 21 42 100}"
JOBS="${JOBS:-1}"
GPU_IDS="${GPU_IDS:-0 1}"
SETS="${SETS:-fair}"
OVERWRITE="${OVERWRITE:-0}"
KERNELS="${KERNELS:-grpo recency evidence gated}"
export OVERWRITE

echo "[1/6] Regenerate Stage B configs with seeds: ${SEEDS}"
echo "[config] MODEL_ID=${MODEL_ID:-${ECR_GRPO_MODEL_ID:-<config-default>}}"
echo "[config] KERNELS=${KERNELS} GPU_IDS=${GPU_IDS} JOBS=${JOBS} OVERWRITE=${OVERWRITE}"
echo "[config] OPTIMIZER=grpo REWARD_UNIT=trajectory for grpo, step for recency/evidence/gated"
echo "[config] ACTION_SCORE_BATCH_SIZE=${ECR_GRPO_ACTION_SCORE_BATCH_SIZE:-<config-default>} UPDATE_MODE=${ECR_GRPO_UPDATE_SCORE_MODE:-full_distribution} LR=${ECR_GRPO_STAGE_B_LEARNING_RATE:-<config-default>} TRACE=${ECR_GRPO_TRACE_NUM_TASKS:-<config-default>} RANK=${ECR_GRPO_RANK_NUM_TASKS:-<config-default>}"
if [[ " ${SETS} " == *" fair "* || " ${SETS} " == *" ablation "* ]]; then
  python scripts/make_stage_b_fair_configs.py \
    --base-dir configs/hf_lora_stage_b_nonlocal_gated \
    --pattern 'gated_seed*.json' \
    --out-dir configs/hf_lora_stage_b_fair \
    --out-root runs/hf_lora_stage_b_fair \
    --lag 2 \
    --kernels ${KERNELS} \
    --seeds ${SEEDS}
fi

if [[ " ${SETS} " == *" lag1 "* ]]; then
  python scripts/make_stage_b_fair_configs.py \
    --base-dir configs/hf_lora_stage_b_nonlocal_gated \
    --pattern 'gated_seed*.json' \
    --out-dir configs/hf_lora_stage_b_lag1 \
    --out-root runs/hf_lora_stage_b_lag1 \
    --lag 1 \
    --kernels ${KERNELS} \
    --seeds ${SEEDS}
fi

if [[ " ${SETS} " == *" lag3 "* ]]; then
  python scripts/make_stage_b_fair_configs.py \
    --base-dir configs/hf_lora_stage_b_nonlocal_gated \
    --pattern 'gated_seed*.json' \
    --out-dir configs/hf_lora_stage_b_lag3 \
    --out-root runs/hf_lora_stage_b_lag3 \
    --lag 3 \
    --kernels ${KERNELS} \
    --seeds ${SEEDS}
fi

if [[ " ${SETS} " == *" ablation "* ]]; then
  python scripts/make_stage_b_ablation_configs.py \
    --base-dir configs/hf_lora_stage_b_fair/gated \
    --pattern 'gated_seed*.json' \
    --out-dir configs/hf_lora_stage_b_ablation \
    --out-root runs/hf_lora_stage_b_ablation \
    --lag 2
fi

run_config() {
  local cfg="$1"
  local gpu="${2:-0}"
  local out_dir
  local complete
  out_dir="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["output_dir"])' "${cfg}")"
  complete="$(
    python -c '
import csv
import json
import os
import sys

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(cfg.get("training", {}).get("num_updates", 0))
path = os.path.join(cfg["output_dir"], "eval_metrics.csv")
if not os.path.exists(path):
    print("0")
    raise SystemExit
with open(path, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("0")
    raise SystemExit
try:
    last_update = int(float(rows[-1].get("update", -1)))
except ValueError:
    last_update = -1
print("1" if last_update >= expected else "0")
' "${cfg}"
  )"
  if [[ "${OVERWRITE}" != "1" && "${complete}" == "1" ]]; then
    echo "[skip] ${cfg} -> ${out_dir}"
    return 0
  fi
  if [[ "${OVERWRITE}" != "1" && -f "${out_dir}/eval_metrics.csv" ]]; then
    echo "[rerun-partial] ${cfg} -> ${out_dir}"
  fi
  echo "[run][gpu=${gpu}] ${cfg}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m ecr_grpo.trainer --config "${cfg}"
}
export -f run_config

seed_selected() {
  local cfg="$1"
  local seed
  for seed in ${SEEDS}; do
    if [[ "${cfg}" == *"seed${seed}.json" ]]; then
      return 0
    fi
  done
  return 1
}

kernel_selected() {
  local cfg="$1"
  local kernel
  for kernel in ${KERNELS}; do
    if [[ "${cfg}" == *"/${kernel}/"* ]]; then
      return 0
    fi
  done
  return 1
}

run_set() {
  local label="$1"
  local cfg_root="$2"
  local -a gpus
  read -r -a gpus <<< "${GPU_IDS}"
  if [[ "${#gpus[@]}" -eq 0 ]]; then
    echo "[error] GPU_IDS is empty" >&2
    return 1
  fi
  local max_jobs="${JOBS}"
  if [[ "${max_jobs}" -lt 1 ]]; then
    echo "[error] JOBS must be >= 1" >&2
    return 1
  fi
  if [[ "${max_jobs}" -gt "${#gpus[@]}" ]]; then
    max_jobs="${#gpus[@]}"
  fi
  echo "[run-set] ${label}: ${cfg_root}"
  local -a selected_cfgs
  while IFS= read -r -d '' cfg; do
    if ! seed_selected "${cfg}"; then
      continue
    fi
    if [[ "${label}" != "ablation" ]] && ! kernel_selected "${cfg}"; then
      continue
    fi
    selected_cfgs+=("${cfg}")
  done < <(find "${cfg_root}" -name '*.json' -print0 | sort -z)

  if [[ "${#selected_cfgs[@]}" -eq 0 ]]; then
    echo "[run-set] ${label}: no selected configs"
    return 0
  fi

  local -a pids
  local slot
  for ((slot = 0; slot < max_jobs; slot++)); do
    local gpu="${gpus[$slot]}"
    (
      worker_idx="${slot}"
      while [[ "${worker_idx}" -lt "${#selected_cfgs[@]}" ]]; do
        run_config "${selected_cfgs[$worker_idx]}" "${gpu}"
        worker_idx=$((worker_idx + max_jobs))
      done
    ) &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  return "${failed}"
}

RUN_FAILURES=0

if [[ " ${SETS} " == *" fair "* ]]; then
  echo "[2/6] Run fair comparison"
  run_set "fair" "configs/hf_lora_stage_b_fair" || RUN_FAILURES=1
fi

if [[ " ${SETS} " == *" lag1 "* ]]; then
  echo "[3/6] Run lag=1 comparison"
  run_set "lag1" "configs/hf_lora_stage_b_lag1" || RUN_FAILURES=1
fi

if [[ " ${SETS} " == *" lag3 "* ]]; then
  echo "[4/6] Run lag=3 comparison"
  run_set "lag3" "configs/hf_lora_stage_b_lag3" || RUN_FAILURES=1
fi

if [[ " ${SETS} " == *" ablation "* ]]; then
  echo "[5/6] Run gated ablations"
  run_set "ablation" "configs/hf_lora_stage_b_ablation" || RUN_FAILURES=1
fi

echo "[6/6] Summarize"
if [[ " ${SETS} " == *" fair "* ]]; then
  python scripts/summarize_stage_b.py --run-root runs/hf_lora_stage_b_fair --output-dir runs/hf_lora_stage_b_fair
fi
if [[ " ${SETS} " == *" lag1 "* ]]; then
  python scripts/summarize_stage_b.py --run-root runs/hf_lora_stage_b_lag1 --output-dir runs/hf_lora_stage_b_lag1
fi
if [[ " ${SETS} " == *" lag3 "* ]]; then
  python scripts/summarize_stage_b.py --run-root runs/hf_lora_stage_b_lag3 --output-dir runs/hf_lora_stage_b_lag3
fi
if [[ " ${SETS} " == *" ablation "* ]]; then
  python scripts/summarize_stage_b.py --run-root runs/hf_lora_stage_b_ablation --output-dir runs/hf_lora_stage_b_ablation
fi

echo "Done. Key summaries:"
echo "  runs/hf_lora_stage_b_fair/stage_b_summary.csv"
echo "  runs/hf_lora_stage_b_lag1/stage_b_summary.csv"
echo "  runs/hf_lora_stage_b_lag3/stage_b_summary.csv"
echo "  runs/hf_lora_stage_b_ablation/stage_b_summary.csv"

if [[ "${RUN_FAILURES}" != "0" ]]; then
  echo "[error] One or more runs failed. Summaries above include only completed runs." >&2
  exit 1
fi
