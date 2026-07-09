from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from stage_b_plan import DEFAULT_NONLOCAL_REWARD, load_seeded_configs, normalize_stage_b_config, parse_seed_list

GATED_KEYS = {
    "kernel",
    "lambda",
    "local_lambda",
    "max_pending_age",
    "timeout_penalty",
    "temporal_weight",
    "exact_step_weight",
    "tool_weight",
    "subgoal_weight",
    "tag_weight",
    "text_weight",
    "evidence_top_k",
    "evidence_temperature",
    "local_recency_weight",
    "local_evidence_weight",
    "nonlocal_evidence_weight",
    "nonlocal_recency_weight",
    "terminal_success_uniform_weight",
    "terminal_success_evidence_weight",
    "terminal_success_recency_weight",
    "terminal_failure_recency_weight",
    "terminal_failure_evidence_weight",
    "ambiguous_evidence_weight",
    "ambiguous_recency_weight",
}

EVIDENCE_KEYS = {
    "lambda",
    "max_pending_age",
    "timeout_penalty",
    "temporal_weight",
    "exact_step_weight",
    "tool_weight",
    "subgoal_weight",
    "tag_weight",
    "text_weight",
}

RECENCY_KEYS = {
    "lambda",
    "max_pending_age",
    "timeout_penalty",
}

TRAJECTORY_KEYS = {
    "max_pending_age",
    "timeout_penalty",
}


def filtered_credit(base_credit: dict[str, Any], keys: set[str], kernel: str) -> dict[str, Any]:
    credit = {key: deepcopy(value) for key, value in base_credit.items() if key in keys}
    credit["kernel"] = kernel
    return credit


def make_credit(base_credit: dict[str, Any], kernel: str) -> dict[str, Any]:
    if kernel in {"grpo", "trajectory", "trajectory_uniform"}:
        credit = filtered_credit(base_credit, TRAJECTORY_KEYS, "trajectory_uniform")
        credit.setdefault("max_pending_age", base_credit.get("max_pending_age", 16))
        return credit
    if kernel == "recency":
        credit = filtered_credit(base_credit, RECENCY_KEYS, "recency")
        credit.setdefault("lambda", base_credit.get("lambda", 0.2))
        credit.setdefault("max_pending_age", base_credit.get("max_pending_age", 16))
        return credit
    if kernel == "evidence":
        credit = filtered_credit(base_credit, EVIDENCE_KEYS, "evidence")
        credit.setdefault("lambda", base_credit.get("lambda", 0.2))
        credit.setdefault("max_pending_age", base_credit.get("max_pending_age", 16))
        credit.setdefault("temporal_weight", base_credit.get("temporal_weight", 0.2))
        credit.setdefault("exact_step_weight", base_credit.get("exact_step_weight", 0.0))
        credit.setdefault("tool_weight", base_credit.get("tool_weight", 0.0))
        credit.setdefault("subgoal_weight", base_credit.get("subgoal_weight", 0.0))
        credit.setdefault("tag_weight", base_credit.get("tag_weight", 8.0))
        credit.setdefault("text_weight", base_credit.get("text_weight", 4.0))
        return credit
    if kernel == "gated":
        credit = filtered_credit(base_credit, GATED_KEYS, "gated_evidence")
        credit.setdefault("lambda", base_credit.get("lambda", 0.2))
        credit.setdefault("local_lambda", base_credit.get("local_lambda", 1.0))
        credit.setdefault("max_pending_age", base_credit.get("max_pending_age", 16))
        credit.setdefault("temporal_weight", base_credit.get("temporal_weight", 0.2))
        credit.setdefault("exact_step_weight", base_credit.get("exact_step_weight", 0.0))
        credit.setdefault("tool_weight", base_credit.get("tool_weight", 0.0))
        credit.setdefault("subgoal_weight", base_credit.get("subgoal_weight", 0.0))
        credit.setdefault("tag_weight", base_credit.get("tag_weight", 8.0))
        credit.setdefault("text_weight", base_credit.get("text_weight", 4.0))
        credit.setdefault("evidence_top_k", base_credit.get("evidence_top_k", 3))
        credit.setdefault("evidence_temperature", base_credit.get("evidence_temperature", 0.7))
        credit.setdefault("local_recency_weight", base_credit.get("local_recency_weight", 1.0))
        credit.setdefault("local_evidence_weight", base_credit.get("local_evidence_weight", 0.0))
        credit.setdefault("nonlocal_evidence_weight", base_credit.get("nonlocal_evidence_weight", 0.9))
        credit.setdefault("nonlocal_recency_weight", base_credit.get("nonlocal_recency_weight", 0.1))
        return credit
    raise ValueError(f"Unknown kernel: {kernel}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="configs/hf_lora_stage_b_nonlocal_gated")
    parser.add_argument("--pattern", default="gated_seed*.json")
    parser.add_argument("--out-dir", default="configs/hf_lora_stage_b_fair")
    parser.add_argument("--out-root", default="runs/hf_lora_stage_b_fair")
    parser.add_argument("--kernels", nargs="+", default=["grpo", "recency", "evidence", "gated"])
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--nonlocal-reward", type=float, default=DEFAULT_NONLOCAL_REWARD)
    parser.add_argument("--seeds", nargs="*", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    base_paths = sorted(base_dir.glob(args.pattern))
    if not base_paths:
        raise SystemExit(f"No base configs matched {base_dir / args.pattern}")

    for seed, base_config, base_path in load_seeded_configs(base_paths, parse_seed_list(args.seeds)):
        base_credit = dict(base_config.get("credit", {}))
        for kernel in args.kernels:
            config = deepcopy(base_config)
            normalize_stage_b_config(
                config,
                seed=seed,
                lag=args.lag,
                reward=args.nonlocal_reward,
            )
            config["experiment_name"] = f"hf_lora_stage_b_fair_{kernel}_seed{seed}"
            config["output_dir"] = f"{args.out_root}/{kernel}/seed={seed}"
            config["credit"] = make_credit(base_credit, kernel)
            training = config.setdefault("training", {})
            reward_unit = "trajectory" if kernel in {"grpo", "trajectory", "trajectory_uniform"} else "step"
            training["optimizer"] = "grpo"
            training["grpo_reward_unit"] = reward_unit
            training["advantage_mode"] = reward_unit

            kernel_dir = out_dir / kernel
            kernel_dir.mkdir(parents=True, exist_ok=True)
            out_path = kernel_dir / f"{kernel}_seed{seed}.json"
            out_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            print(f"{base_path} -> {out_path}")


if __name__ == "__main__":
    main()
