# ECR-GRPO Agent

This is a lightweight research codebase for:

> **ECR-GRPO: Event-Conditioned Credit Refill GRPO for long-horizon and asynchronous LLM agents.**

The first runnable version focuses on a controlled synthetic async benchmark. It validates
the core algorithmic mechanism before integrating heavier agent benchmarks such as
ALFWorld, ScienceWorld, WebShop, or tool-orchestration tasks.

## What Is Implemented

- Synthetic long-horizon agent environment.
- Async wrapper with delay, timeout, missing reward, and interruption events.
- Pending step buffer.
- Event-conditioned credit refill kernels:
  - trajectory
  - uniform
  - recency
  - dependency-aware
  - evidence attribution without oracle step links
- GRPO-style group-relative step advantages.
- Lightweight tabular text-action policy for smoke experiments.
- Optional HuggingFace causal-LM policy with LoRA and a compact clipped GRPO update.
- Optional ALFWorld environment adapter.
- Train/eval CLI.
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
