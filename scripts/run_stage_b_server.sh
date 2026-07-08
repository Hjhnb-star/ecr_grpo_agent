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
  out_dir="$(python -c "import json; print(json.load(open('${cfg}', encoding='utf-8'))['output_dir'])")"
  if [[ "${OVERWRITE}" != "1" && -f "${out_dir}/eval_metrics.csv" ]]; then
    echo "[skip] ${cfg} -> ${out_dir}"
    return 0
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
  local max_jobs="${JOBS}"
  if [[ "${max_jobs}" -gt "${#gpus[@]}" ]]; then
    max_jobs="${#gpus[@]}"
  fi
  echo "[run-set] ${label}: ${cfg_root}"
  local idx=0
  local active=0
  while IFS= read -r -d '' cfg; do
    if ! seed_selected "${cfg}"; then
      continue
    fi
    if [[ "${label}" != "ablation" ]] && ! kernel_selected "${cfg}"; then
      continue
    fi
    local gpu="${gpus[$((idx % ${#gpus[@]}))]}"
    run_config "${cfg}" "${gpu}" &
    idx=$((idx + 1))
    active=$((active + 1))
    if [[ "${active}" -ge "${max_jobs}" ]]; then
      wait -n
      active=$((active - 1))
    fi
  done < <(find "${cfg_root}" -name '*.json' -print0 | sort -z)
  wait
}

if [[ " ${SETS} " == *" fair "* ]]; then
  echo "[2/6] Run fair comparison"
  run_set "fair" "configs/hf_lora_stage_b_fair"
fi

if [[ " ${SETS} " == *" lag1 "* ]]; then
  echo "[3/6] Run lag=1 comparison"
  run_set "lag1" "configs/hf_lora_stage_b_lag1"
fi

if [[ " ${SETS} " == *" lag3 "* ]]; then
  echo "[4/6] Run lag=3 comparison"
  run_set "lag3" "configs/hf_lora_stage_b_lag3"
fi

if [[ " ${SETS} " == *" ablation "* ]]; then
  echo "[5/6] Run gated ablations"
  run_set "ablation" "configs/hf_lora_stage_b_ablation"
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
