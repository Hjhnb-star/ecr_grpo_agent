#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0 1}"
KERNELS="${KERNELS:-grpo gated residual}"
SEEDS="${SEEDS:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/alfworld_mainline}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a gpu_array <<< "${GPU_IDS}"
if (( ${#gpu_array[@]} < 1 )); then
  echo "[error] GPU_IDS must contain at least one GPU id." >&2
  exit 2
fi

summary_dir="${OUTPUT_ROOT}/_job_summaries"
mkdir -p "${summary_dir}"
active_pids=()
active_labels=()
overall_status=0
job_index=0

wait_batch() {
  local index
  for index in "${!active_pids[@]}"; do
    if ! wait "${active_pids[$index]}"; then
      echo "[failed] ${active_labels[$index]}" >&2
      overall_status=1
    fi
  done
  active_pids=()
  active_labels=()
}

for kernel in ${KERNELS}; do
  for seed in ${SEEDS}; do
    gpu="${gpu_array[$((job_index % ${#gpu_array[@]}))]}"
    label="${kernel}_seed${seed}"
    echo "[launch] ${label} gpu=${gpu}"
    GPU_ID="${gpu}" \
    KERNELS="${kernel}" \
    SEEDS="${seed}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    SUMMARY_PATH="${summary_dir}/${label}.csv" \
      bash scripts/run_alfworld_server.sh &
    active_pids+=("$!")
    active_labels+=("${label}")
    job_index=$((job_index + 1))

    if (( ${#active_pids[@]} >= ${#gpu_array[@]} )); then
      wait_batch
    fi
  done
done

if (( ${#active_pids[@]} > 0 )); then
  wait_batch
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  PYTHONPATH="src:${PYTHONPATH:-}" \
    python scripts/summarize_alfworld.py --run-root "${OUTPUT_ROOT}"
fi

exit "${overall_status}"
