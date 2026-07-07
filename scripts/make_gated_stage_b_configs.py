from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage_b_plan import DEFAULT_NONLOCAL_REWARD, load_seeded_configs, normalize_stage_b_config, parse_seed_list


GATED_CREDIT_DEFAULTS: dict[str, Any] = {
    "kernel": "gated_evidence",
    "lambda": 0.2,
    "local_lambda": 1.0,
    "max_pending_age": 16,
    "timeout_penalty": -0.25,
    "temporal_weight": 0.2,
    "exact_step_weight": 0.0,
    "tool_weight": 0.0,
    "subgoal_weight": 0.0,
    "tag_weight": 8.0,
    "text_weight": 4.0,
    "evidence_top_k": 3,
    "evidence_temperature": 0.7,
    "local_recency_weight": 1.0,
    "local_evidence_weight": 0.0,
    "nonlocal_evidence_weight": 0.9,
    "nonlocal_recency_weight": 0.1,
    "terminal_success_uniform_weight": 0.4,
    "terminal_success_evidence_weight": 0.4,
    "terminal_success_recency_weight": 0.2,
    "terminal_failure_recency_weight": 0.7,
    "terminal_failure_evidence_weight": 0.3,
    "ambiguous_evidence_weight": 0.5,
    "ambiguous_recency_weight": 0.5,
}


def make_gated_config(
    config: dict[str, Any],
    *,
    seed: int,
    out_root: str,
    lag: int,
    nonlocal_reward: float,
) -> dict[str, Any]:
    normalize_stage_b_config(config, seed=seed, lag=lag, reward=nonlocal_reward)
    config["experiment_name"] = f"hf_lora_nonlocal_gated_seed{seed}"
    config["output_dir"] = f"{out_root}/gated/seed={seed}"

    base_credit = dict(config.get("credit", {}))
    gated_credit = dict(GATED_CREDIT_DEFAULTS)
    if "timeout_penalty" in base_credit:
        gated_credit["timeout_penalty"] = base_credit["timeout_penalty"]
    if "max_pending_age" in base_credit:
        gated_credit["max_pending_age"] = base_credit["max_pending_age"]
    config["credit"] = gated_credit
    policy = config.setdefault("policy", {})
    if str(policy.get("kind", "")).lower() in {"hf", "hf_lora", "lora"}:
        policy["action_selection"] = "score"
        policy.setdefault("action_score_batch_size", 8)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default="configs/hf_lora_stage_b_nonlocal_hard")
    parser.add_argument("--pattern", default="evidence_seed*.json")
    parser.add_argument("--out-dir", default="configs/hf_lora_stage_b_nonlocal_gated")
    parser.add_argument("--out-root", default="runs/hf_lora_stage_b_nonlocal_gated")
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--nonlocal-reward", type=float, default=DEFAULT_NONLOCAL_REWARD)
    parser.add_argument("--seeds", nargs="*", default=None)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    src_paths = sorted(src_dir.glob(args.pattern))
    if not src_paths:
        raise SystemExit(f"No configs matched {src_dir / args.pattern}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for seed, base_config, src_path in load_seeded_configs(src_paths, parse_seed_list(args.seeds)):
        config = make_gated_config(
            base_config,
            seed=seed,
            out_root=args.out_root,
            lag=args.lag,
            nonlocal_reward=args.nonlocal_reward,
        )
        out_path = out_dir / f"gated_seed{seed}.json"
        out_path.write_text(json.dumps(config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"{src_path} -> {out_path}")


if __name__ == "__main__":
    main()
