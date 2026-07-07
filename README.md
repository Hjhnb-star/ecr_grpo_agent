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
- Lightweight tabular text-action policy for smoke experiments.
- Optional HuggingFace causal-LM policy with LoRA, candidate-action scoring, and a compact
  clipped GRPO update. The HF path supports memory-safe selected-action updates for 24 GB
  GPUs.
- Optional ALFWorld environment adapter.
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
- The seamless GRPO-style baseline is `grpo`, implemented as
  `credit.kernel="trajectory_uniform"` plus `training.advantage_mode="trajectory"`.
- ECR/Gated runs use step-level event credit, usually
  `credit.kernel="gated_evidence"` plus `training.advantage_mode="step"`.

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
    "update_score_mode": "selected"
  }
}
```

`update_score_mode="selected"` keeps candidate scoring for behavior/action sampling, but
uses the selected action sequence for the PPO/GRPO update. This avoids retaining a full
candidate-distribution computation graph and is the recommended default before benchmark
integration.

## Stage B HF/LoRA Experiments on Server

The fair Stage B comparison now includes:

- `grpo`: trajectory-uniform credit with trajectory-return advantages.
- `recency`: local recent-step credit heuristic.
- `evidence`: evidence attribution for event-to-step credit.
- `gated`: recency for local feedback, evidence attribution for non-local events.

On the Linux server:

```bash
cd ~/hjh_playbook/ecr_grpo_server
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
chmod +x scripts/run_stage_b_server.sh

SETS="fair" GPU_IDS="0 1" JOBS=2 bash scripts/run_stage_b_server.sh
```

If memory is still tight, run one job at a time:

```bash
SETS="fair" GPU_IDS="0" JOBS=1 bash scripts/run_stage_b_server.sh
```

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

- Learning quality: compare `final_success`, `peak_success`, and `logged_mean_success`.
- Credit quality: compare `target_weight_mean`, `recent_weight_mean`,
  `argmax_target_rate`, `target_top3_rate`, and assignment sharpness metrics such as
  `weight_entropy_mean` and `top_margin_mean`.

A useful Stage B result is not merely "better than recency". The stronger claim is that
`gated` keeps GRPO-like task performance while assigning non-local credit to the right
earlier action under no-oracle metadata. That is the evidence needed before moving to
ALFWorld or other real benchmarks.

## ALFWorld Placeholder

Install ALFWorld separately, then install this package with optional dependencies:

```powershell
pip install -e ".[alfworld,hf]"
```

Edit `configs\alfworld_hf_lora_placeholder.json` and replace:

```text
REPLACE_WITH_ALFWORLD_CONFIG.yaml
REPLACE_WITH_HF_MODEL_ID_OR_LOCAL_PATH
```

Then run:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m ecr_grpo.trainer --config configs\alfworld_hf_lora_placeholder.json
```

## Why Tabular Policy First?

The algorithmic contribution is the asynchronous credit-assignment mechanism, not model
serving. The tabular policy lets us verify the full training loop without GPU, network
downloads, or benchmark integration. The policy interface is intentionally small so a
HuggingFace/LoRA policy can replace it later.
