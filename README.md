# ECR-GRPO Agent

This is a lightweight research codebase for:

> **ECR-GRPO: Event-Conditioned Credit Refill GRPO for long-horizon and asynchronous LLM agents.**

The first runnable version focuses on a controlled synthetic async benchmark. It validates
the core algorithmic mechanism before integrating heavier agent benchmarks such as
ALFWorld, ScienceWorld, WebShop, or tool-orchestration tasks.

## What Is Implemented

- Synthetic long-horizon agent environment.
- Async wrapper with delay, timeout, missing reward, interruption events, and optional
  diagnostic metadata stripping for no-oracle credit assignment.
- Pending step buffer.
- Event-conditioned credit refill kernels:
  - trajectory / trajectory-uniform GRPO-compatible baselines
  - uniform
  - recency
  - dependency-aware
  - evidence attribution without oracle step links
  - gated evidence for routing local feedback to recency and non-local feedback to evidence
- GRPO-style group-relative advantages:
  - `advantage_mode="trajectory"` for trajectory-return GRPO compatibility
  - `advantage_mode="step"` for event-conditioned ECR credit updates
- GRPO adapter that converts ECR-refilled rewards into standard GRPO samples:
  - `credit.output="step_reward"` or `"trajectory_reward"` selects the reward unit
  - `optimizer.name="grpo"` and `optimizer.update_impl="standard_grpo"` keep the optimizer fixed
  - `training.optimizer="grpo"` keeps the policy optimizer fixed
  - `training.grpo_reward_unit="step"` uses ECR-refilled step rewards
  - `training.grpo_reward_unit="trajectory"` uses trajectory rewards
- Lightweight tabular text-action policy for smoke experiments.
- Optional HuggingFace causal-LM policy with LoRA, candidate-action scoring, and a compact
  clipped GRPO update. The streaming distribution mode computes the exact candidate-softmax
  first-order gradient while retaining only one candidate graph at a time. The legacy
  selected approximation is rejected by reportable ALFWorld configs.
- Fixed-game ALFWorld adapter plus a manifest/factory adapter for external text benchmarks.
- Train/eval CLI.
- Stage B config generation, server runner, and summary scripts.
- Unit tests.

## Quick Start

```powershell
cd E:\yf\ecr_grpo_agent
$env:PYTHONPATH = "$PWD\src"
python -m ecr_grpo.trainer --config configs\smoke.json
python -m unittest discover tests
```

Or:

```powershell
.\scripts\run_smoke.ps1
```

Outputs are written to `runs/smoke/`.

To compare credit kernels:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m ecr_grpo.run_baselines --config configs\smoke.json --updates 30
```

This writes `runs/baselines/comparison.csv`.

To test the non-oracle attribution path:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m ecr_grpo.trainer --config configs\smoke_evidence_no_oracle.json
```

In this setting the async wrapper removes `related_step_id`, `related_tool`, and
`related_subgoal` before credit assignment. The `evidence` kernel must infer event-to-step
weights from generic signals: event time, action/effect text, observation deltas, and optional
metadata tags. This is the recommended path for arguing that ECR-GRPO is a general credit
assignment algorithm rather than a benchmark-specific rule system. The older `dependency`
kernel is best treated as an oracle/upper-bound baseline when a benchmark exposes exact links.

For HF/LoRA experiments where local next-action learning can dominate, use
`"kernel": "gated_evidence"` to keep ordinary partial rewards sharp while still applying
evidence attribution to `non_local_support` events. The router keeps the same `AsyncEvent`
interface and classifies events from `event_type`, `observation_delta`, and metadata tags.

Important baseline naming:

- `recency` is a local step-credit heuristic baseline, not standard GRPO.
- The trajectory GRPO baseline is `grpo`, implemented as
  `credit.kernel="trajectory_uniform"` plus `training.advantage_mode="trajectory"`.
- ECR/Gated runs still use standard GRPO; they only change reward construction,
  usually `credit.kernel="gated_evidence"`, `credit.output="step_reward"`, and
  `optimizer.update_impl="standard_grpo"`.

## HuggingFace + LoRA Placeholder

Install optional dependencies:

```powershell
pip install -e ".[hf]"
```

Edit `configs\hf_lora_synthetic_placeholder.json` and replace:

```text
REPLACE_WITH_HF_MODEL_ID_OR_LOCAL_PATH
```

Then run:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m ecr_grpo.trainer --config configs\hf_lora_synthetic_placeholder.json
```

HF policies default to discrete candidate-action scoring: each available action is
scored as `log p(action | prompt)`, normalized into an action distribution, sampled
during training, and reused for clipped updates and entropy logging. To temporarily
use the older free-generation path, set `"action_selection": "generate"` under
`policy`. Candidate scoring can be chunked with `"action_score_batch_size"` to reduce
GPU memory peaks.

For two RTX 4090 cards, prefer:

```json
{
  "policy": {
    "action_selection": "score",
    "action_score_batch_size": 2,
    "action_score_normalization": "mean",
    "action_score_calibration": "pmi",
    "use_chat_template": true,
    "temperature": 1.0,
    "update_score_mode": "full_distribution"
  }
}
```

`action_score_normalization="mean"` removes the short-action bias from candidate scoring.
`action_score_calibration="pmi"` subtracts each action's prompt-independent prior score,
which prevents generic or fluent wrong actions from dominating the candidate list.
`use_chat_template=true` uses the tokenizer's instruction-tuned chat format when available.
`update_score_mode="full_distribution"` uses the same normalized candidate-action
distribution for sampling, old logprobs, PPO/GRPO ratios, and entropy logging. This is the
recommended default. `update_score_mode="selected"` is retained only as a lower-memory
approximation: it updates the selected action score directly and should not be reported as
the strict GRPO candidate-distribution objective.

## Stage B HF/LoRA Experiments on Server

The fair Stage B comparison keeps the optimizer fixed as GRPO and varies only reward
construction:

- `grpo`: GRPO fed by trajectory-uniform terminal reward.
- `recency`: GRPO fed by recency-refilled step rewards.
- `evidence`: GRPO fed by evidence-refilled step rewards.
- `gated`: GRPO fed by gated ECR-refilled step rewards.

On the Linux server:

```bash
cd ~/hjh_playbook/ecr_grpo_server
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
chmod +x scripts/run_stage_b_server.sh

SETS="fair" GPU_IDS="0 1" JOBS=2 bash scripts/run_stage_b_server.sh
```

For a quick learning-path sanity check, rerun one seed and overwrite stale results:

```bash
SEEDS="7" SETS="fair" KERNELS="gated" GPU_IDS="0" JOBS=1 OVERWRITE=1 bash scripts/run_stage_b_server.sh
```

The Stage B HF configs use `learning_rate=1e-5`, `num_updates=60`,
`tasks_per_update=4`, `action_score_normalization="mean"`, and
`action_score_calibration="pmi"`, `use_chat_template=true`, and
`update_score_mode="full_distribution"` by default. They also set
`training.optimizer="grpo"` and route ECR outputs through the GRPO adapter before policy
update. HF updates use `max_grad_norm=1.0`.
During evaluation the trainer writes `eval_action_rankings.jsonl` for first-step top-k
diagnostics and `eval_traces.jsonl` for full greedy rollout traces. If success drops while
first-step rankings remain correct, inspect `eval_traces.jsonl` to find the later step where
the policy diverges.

If memory is still tight, reduce `ECR_GRPO_ACTION_SCORE_BATCH_SIZE` or run one job at a time:

```bash
SETS="fair" GPU_IDS="0" JOBS=1 bash scripts/run_stage_b_server.sh
```

Set `OVERWRITE=1` when you want to rerun configs that already have `eval_metrics.csv`.
Without it, completed runs are skipped.

After the fair comparison is stable, run the supporting sets:

```bash
SETS="ablation" GPU_IDS="0 1" JOBS=2 bash scripts/run_stage_b_server.sh
SETS="lag3" GPU_IDS="0 1" JOBS=2 bash scripts/run_stage_b_server.sh
SETS="lag1" GPU_IDS="0 1" JOBS=2 bash scripts/run_stage_b_server.sh
```

The runner skips a config when its target `eval_metrics.csv` already exists, so interrupted
runs can be resumed with the same command.

Summary files are written under each run root, for example:

```text
runs/hf_lora_stage_b_fair/stage_b_runs.csv
runs/hf_lora_stage_b_fair/stage_b_summary.csv
```

If the server reports an argument error such as `unrecognized arguments: --seeds`, the server
copy is stale. Sync at least these files from local to server:

```text
scripts/run_stage_b_server.sh
scripts/stage_b_plan.py
scripts/make_stage_b_fair_configs.py
scripts/make_stage_b_ablation_configs.py
scripts/make_gated_stage_b_configs.py
scripts/summarize_stage_b.py
src/ecr_grpo/
tests/test_core.py
```

## Interpreting Stage B

The fair comparison should answer two separate questions:

- Learning quality: compare `final_success`, `peak_success`, `logged_mean_success`,
  `final_action_accuracy`, and `final_avg_progress_fraction`.
- Credit quality: compare `target_weight_mean`, `recent_weight_mean`,
  `argmax_target_rate`, `target_top3_rate`, and assignment sharpness metrics such as
  `weight_entropy_mean` and `top_margin_mean`.

A useful Stage B result is not merely "better than recency". The stronger claim is that
`gated` keeps GRPO-like task performance while assigning non-local credit to the right
earlier action under no-oracle metadata. That is the evidence needed before moving to
ALFWorld or other real benchmarks.

## ALFWorld Benchmark Train/Test

Install ALFWorld separately, then install this package with optional dependencies:

```bash
pip install -e ".[alfworld,hf]"
```

If the ALFWorld data is missing on a new server, download it once:

```bash
alfworld-download
```

On the Linux server, first validate the generated benchmark configs without training:

```bash
cd /home/hjh/ecr_grpo_agent/ecr_grpo_agent
export PYTHONPATH=src:${PYTHONPATH:-}
export ALFWORLD_CONFIG=/home/hjh/ecr_grpo_agent/alfworld_src/configs/base_config.yaml
export MODEL_ID=/home/hjh/ecr_grpo_agent/ecr_grpo_agent/models/Qwen/Qwen2.5-1.5B-Instruct

DRY_RUN=1 \
KERNELS="grpo local recency evidence gated" \
BASE_CONFIG=configs/alfworld_gated_lowmem_smoke.json \
OUTPUT_ROOT=runs/alfworld_lowmem_smoke \
SEEDS="7" \
TRAIN_SPLIT=train \
EVAL_SPLIT=eval_out_of_distribution \
NUM_TRAIN_TASKS=2 \
NUM_EVAL_TASKS=2 \
EVAL_SPLITS="eval_in_distribution eval_out_of_distribution" \
MAX_STEPS=15 \
bash scripts/run_alfworld_server.sh
```

Then run the same protocol on one GPU:

```bash
DRY_RUN=0 OVERWRITE=1 GPU_ID=1 \
BASE_CONFIG=configs/alfworld_gated_lowmem_smoke.json \
OUTPUT_ROOT=runs/alfworld_lowmem_smoke \
KERNELS="grpo local recency evidence gated" \
SEEDS="7" \
TRAIN_SPLIT=train \
EVAL_SPLIT=eval_out_of_distribution \
NUM_TRAIN_TASKS=2 \
NUM_EVAL_TASKS=2 \
EVAL_SPLITS="eval_in_distribution eval_out_of_distribution" \
MAX_STEPS=15 \
bash scripts/run_alfworld_server.sh
```

The runner binds every rollout group to one real ALFWorld game and evaluates fixed
`eval_in_distribution` and `eval_out_of_distribution` manifests. `CLEAN_EVAL=1`
disables artificial delay, missing rewards, timeouts, and shaping during benchmark
evaluation. Training still uses the configured asynchronous event stream. Do not add a
space after a shell line-continuation backslash.

You can also call the Python runner directly:

```bash
python -m ecr_grpo.run_alfworld \
  --base-config configs/alfworld_gated_smoke.json \
  --output-root runs/alfworld_fair \
  --alfworld-config /home/hjh/ecr_grpo_agent/alfworld_src/configs/base_config.yaml \
  --model-id /home/hjh/ecr_grpo_agent/ecr_grpo_agent/models/Qwen/Qwen2.5-1.5B-Instruct \
  --kernels grpo recency evidence gated \
  --seeds 7 \
  --train-split train \
  --eval-split eval_out_of_distribution \
  --eval-splits eval_in_distribution eval_out_of_distribution \
  --num-train-tasks 32 \
  --num-eval-tasks 32 \
  --max-steps 50 \
  --clean-eval \
  --dry-run
```

Generated configs are written to `runs/alfworld_fair/_generated_configs/`.
The runner skips a run if `eval_metrics.csv` already exists unless `--overwrite` is set.
The comparison summary is written to `runs/alfworld_fair/alfworld_summary.csv`. During eval,
`eval_traces.jsonl` and `eval_action_rankings.jsonl` include the ALFWorld split and the actual
gamefile-derived task id, which is useful for inspecting failed benchmark episodes.

### Formal ALFWorld comparison

After the low-memory smoke succeeds, use the reportable configuration:

```bash
BASE_CONFIG=configs/alfworld_gated_benchmark.json \
OUTPUT_ROOT=runs/alfworld_fair \
DRY_RUN=0 OVERWRITE=1 GPU_ID=1 \
KERNELS="grpo local recency evidence gated" \
SEEDS="7 13 21" \
TRAIN_SPLIT=train \
EVAL_SPLIT=eval_out_of_distribution \
EVAL_SPLITS="eval_in_distribution eval_out_of_distribution" \
NUM_TRAIN_TASKS=500 \
NUM_EVAL_TASKS=9999 \
MAX_STEPS=50 \
bash scripts/run_alfworld_server.sh
```

The formal config uses group size 8. Every group is sampled from one fixed game.
The ECR variants use trajectory-grouped credit advantages; trajectory GRPO keeps its
ordinary trajectory advantage. All methods share the same model, task sampler, rollout
budget, optimizer, seeds, and clean evaluation protocol.

Primary benchmark columns are success rate, average steps, successful-episode steps,
average tokens, raw environment return, failure/max-step rates, and per-task-type success
rates. Training diagnostics include zero-advantage fraction, normalized policy entropy,
effective action count, attribution entropy, effective attributed steps, and top margin.
ALFWorld does not provide ground-truth causal steps, so these diagnostics must not be called
causal accuracy.

### Other Text Benchmarks

Set environment name to external, provide a JSON/JSONL task manifest, and set environment
factory to a Python callable in module:function form. Each manifest row should contain a
task id, split, and optional task type. The factory receives split, task metadata, task id,
and seed, and returns an environment with reset(), step(action), and a text action space.
This is the intended thin integration point for ScienceWorld, WebShop, or tool benchmarks;
the trainer, async event wrapper, metrics, and fixed-task protocol remain unchanged.

## Why Tabular Policy First?

The algorithmic contribution is the asynchronous credit-assignment mechanism, not model
serving. The tabular policy lets us verify the full training loop without GPU, network
downloads, or benchmark integration. The policy interface is intentionally small so a
HuggingFace/LoRA policy can replace it later.
