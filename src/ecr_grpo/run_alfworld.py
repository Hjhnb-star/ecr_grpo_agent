from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from ecr_grpo.io import load_config, write_csv
from ecr_grpo.trainer import ECRGRPOTrainer


DEFAULT_KERNELS = ["grpo", "recency", "evidence", "gated", "dependency"]
DEFAULT_SEEDS = [7, 13, 21]


def parse_ints(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return list(default)
    out: list[int] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            out.append(int(part))
    return out


def make_credit(base_credit: dict[str, Any], kernel: str) -> dict[str, Any]:
    kernel = kernel.lower()
    common = {
        "max_pending_age": base_credit.get("max_pending_age", 16),
        "timeout_penalty": base_credit.get("timeout_penalty", -0.2),
    }
    if kernel in {"grpo", "trajectory", "trajectory_uniform"}:
        return {**common, "kernel": "trajectory_uniform"}
    if kernel == "recency":
        return {
            **common,
            "kernel": "recency",
            "lambda": base_credit.get("lambda", 0.25),
        }
    if kernel == "dependency":
        return {
            **common,
            "kernel": "dependency",
            "lambda": base_credit.get("lambda", 0.25),
            "tool_match_bonus": base_credit.get("tool_match_bonus", 1.2),
            "subgoal_match_bonus": base_credit.get("subgoal_match_bonus", 1.2),
        }
    if kernel == "evidence":
        return {
            **common,
            "kernel": "evidence",
            "lambda": base_credit.get("lambda", 0.25),
            "temporal_weight": base_credit.get("temporal_weight", 0.5),
            "exact_step_weight": 0.0,
            "tool_weight": base_credit.get("tool_weight", 0.5),
            "subgoal_weight": base_credit.get("subgoal_weight", 0.5),
            "tag_weight": base_credit.get("tag_weight", 2.0),
            "text_weight": base_credit.get("text_weight", 1.0),
        }
    if kernel == "gated":
        return {
            **common,
            "kernel": "gated_evidence",
            "lambda": base_credit.get("lambda", 0.25),
            "local_lambda": base_credit.get("local_lambda", 1.0),
            "temporal_weight": base_credit.get("temporal_weight", 0.5),
            "exact_step_weight": 0.0,
            "tool_weight": base_credit.get("tool_weight", 0.5),
            "subgoal_weight": base_credit.get("subgoal_weight", 0.5),
            "tag_weight": base_credit.get("tag_weight", 2.0),
            "text_weight": base_credit.get("text_weight", 1.0),
            "evidence_top_k": base_credit.get("evidence_top_k", 5),
            "evidence_temperature": base_credit.get("evidence_temperature", 0.8),
            "local_recency_weight": base_credit.get("local_recency_weight", 1.0),
            "local_evidence_weight": base_credit.get("local_evidence_weight", 0.0),
            "nonlocal_evidence_weight": base_credit.get("nonlocal_evidence_weight", 0.85),
            "nonlocal_recency_weight": base_credit.get("nonlocal_recency_weight", 0.15),
            "terminal_success_uniform_weight": base_credit.get("terminal_success_uniform_weight", 0.4),
            "terminal_success_evidence_weight": base_credit.get("terminal_success_evidence_weight", 0.4),
            "terminal_success_recency_weight": base_credit.get("terminal_success_recency_weight", 0.2),
            "terminal_failure_recency_weight": base_credit.get("terminal_failure_recency_weight", 0.7),
            "terminal_failure_evidence_weight": base_credit.get("terminal_failure_evidence_weight", 0.3),
            "ambiguous_evidence_weight": base_credit.get("ambiguous_evidence_weight", 0.5),
            "ambiguous_recency_weight": base_credit.get("ambiguous_recency_weight", 0.5),
        }
    raise ValueError(f"Unknown ALFWorld kernel: {kernel}")


def build_run_config(
    base_config: dict[str, Any],
    *,
    kernel: str,
    seed: int,
    output_root: Path,
    alfworld_config: str | None = None,
    model_id: str | None = None,
    train_split: str | None = None,
    eval_split: str | None = None,
    num_train_tasks: int | None = None,
    num_eval_tasks: int | None = None,
    max_steps: int | None = None,
    clean_eval: bool = False,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["seed"] = seed
    config["experiment_name"] = f"alfworld_{kernel}_seed{seed}"
    config["output_dir"] = str(output_root / kernel / f"seed={seed}")

    env = config.setdefault("environment", {})
    env["name"] = "alfworld"
    env.setdefault("reuse_env", True)
    if alfworld_config:
        env["alfworld_config"] = alfworld_config
    if train_split:
        env["split"] = train_split
        env["train_split"] = train_split
    if eval_split:
        env["eval_split"] = eval_split
    if num_train_tasks is not None:
        env["num_tasks"] = int(num_train_tasks)
        env["num_train_tasks"] = int(num_train_tasks)
    if max_steps is not None:
        env["max_steps"] = int(max_steps)

    evaluation = config.setdefault("evaluation", {})
    if "eval_split" in env:
        evaluation.setdefault("split", env["eval_split"])
    if num_eval_tasks is not None:
        evaluation["num_eval_tasks"] = int(num_eval_tasks)
    if max_steps is not None:
        evaluation.setdefault("max_steps", int(max_steps))
    if clean_eval:
        evaluation["async"] = {
            "enabled": False,
            "delay_prob": 0.0,
            "max_delay_steps": 0,
            "timeout_prob": 0.0,
            "interruption_prob": 0.0,
            "missing_reward_prob": 0.0,
            "terminal_reward_delay": 0,
        }

    async_cfg = config.setdefault("async", {})
    async_cfg["use_oracle_event_links"] = False
    async_cfg["strip_diagnostic_metadata"] = True

    reward_unit = "trajectory" if kernel in {"grpo", "trajectory", "trajectory_uniform"} else "step"
    base_credit = dict(config.get("credit", {}))
    config["credit"] = make_credit(base_credit, kernel)
    config["credit"]["output"] = f"{reward_unit}_reward"

    optimizer = config.setdefault("optimizer", {})
    optimizer["name"] = "grpo"
    optimizer["advantage_mode"] = reward_unit
    optimizer["update_impl"] = "standard_grpo"

    training = config.setdefault("training", {})
    training["optimizer"] = "grpo"
    training["grpo_reward_unit"] = reward_unit
    training["advantage_mode"] = reward_unit

    policy = config.setdefault("policy", {})
    if model_id:
        policy["model_id"] = model_id
    if str(policy.get("kind", "")).lower() in {"hf", "hf_lora", "lora"}:
        policy.setdefault("action_selection", "score")
        policy.setdefault("update_score_mode", "full_distribution")
        policy.setdefault("action_score_batch_size", 2)
    return config


def validate_config(config: dict[str, Any], *, require_files: bool) -> list[str]:
    errors: list[str] = []
    env = config.get("environment", {})
    if str(env.get("name", "")).lower() != "alfworld":
        errors.append("environment.name must be 'alfworld'")

    alfworld_config = str(env.get("alfworld_config", ""))
    if not alfworld_config or "REPLACE_WITH" in alfworld_config:
        errors.append("environment.alfworld_config must point to an ALFWorld YAML config")
    elif require_files and not Path(alfworld_config).expanduser().exists():
        errors.append(f"ALFWorld config does not exist: {alfworld_config}")

    policy = config.get("policy", {})
    if str(policy.get("kind", "")).lower() in {"hf", "hf_lora", "lora"}:
        model_id = str(policy.get("model_id", ""))
        if not model_id or "REPLACE_WITH" in model_id:
            errors.append("policy.model_id must point to a HF model id or local model path")
        elif require_files and looks_like_local_path(model_id) and not Path(model_id).expanduser().exists():
            errors.append(f"local policy.model_id does not exist: {model_id}")

    if int(config.get("training", {}).get("group_size", 1)) < 1:
        errors.append("training.group_size must be >= 1")
    if int(config.get("environment", {}).get("max_steps", 0)) <= 0:
        errors.append("environment.max_steps must be > 0")
    if int(config.get("evaluation", {}).get("num_eval_tasks", 1)) <= 0:
        errors.append("evaluation.num_eval_tasks must be > 0")
    return errors


def looks_like_local_path(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or value.startswith((".", "~")) or (len(value) >= 2 and value[1] == ":")


def final_row(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def summarize_run(run_dir: Path, *, kernel: str, seed: int, status: str) -> dict[str, Any]:
    eval_row = final_row(run_dir / "eval_metrics.csv")
    train_row = final_row(run_dir / "train_metrics.csv")
    return {
        "kernel": kernel,
        "seed": seed,
        "status": status,
        "run_dir": str(run_dir),
        "final_success": eval_row.get("success_rate", ""),
        "final_avg_steps": eval_row.get("avg_steps", ""),
        "final_avg_env_return": eval_row.get("avg_env_return", ""),
        "final_credit_causal": train_row.get("credit_mass_on_causal_steps", ""),
        "num_events": train_row.get("num_events", ""),
        "num_assignments": train_row.get("num_assignments", ""),
        "entropy": train_row.get("entropy", ""),
        "eval_split": eval_row.get("eval_split", ""),
        "num_eval_tasks": eval_row.get("num_eval_tasks", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fair ALFWorld ECR-GRPO smoke/comparison suite.")
    parser.add_argument("--base-config", default="configs/alfworld_gated_smoke.json")
    parser.add_argument("--output-root", default="runs/alfworld_fair")
    parser.add_argument("--kernels", nargs="+", default=DEFAULT_KERNELS)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--alfworld-config", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--train-split", default=None, help="ALFWorld split used for policy updates, e.g. train.")
    parser.add_argument("--eval-split", default=None, help="ALFWorld split used for held-out evaluation.")
    parser.add_argument("--num-train-tasks", type=int, default=None)
    parser.add_argument("--num-eval-tasks", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--clean-eval", action="store_true", help="Disable async perturbations during benchmark evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Only write generated configs and validate them.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun configs even if eval_metrics.csv exists.")
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    output_root = Path(args.output_root)
    generated_dir = output_root / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seeds = parse_ints(args.seeds, DEFAULT_SEEDS)
    for kernel in args.kernels:
        for seed in seeds:
            config = build_run_config(
                base_config,
                kernel=kernel,
                seed=seed,
                output_root=output_root,
                alfworld_config=args.alfworld_config,
                model_id=args.model_id,
                train_split=args.train_split,
                eval_split=args.eval_split,
                num_train_tasks=args.num_train_tasks,
                num_eval_tasks=args.num_eval_tasks,
                max_steps=args.max_steps,
                clean_eval=args.clean_eval,
            )
            errors = validate_config(config, require_files=not args.dry_run)
            config_path = generated_dir / f"{kernel}_seed{seed}.json"
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            run_dir = Path(config["output_dir"])

            if errors:
                rows.append(
                    {
                        "kernel": kernel,
                        "seed": seed,
                        "status": "invalid",
                        "run_dir": str(run_dir),
                        "config_path": str(config_path),
                        "errors": "; ".join(errors),
                    }
                )
                print(f"[invalid] {config_path}: {'; '.join(errors)}")
                continue
            if args.dry_run:
                rows.append(
                    {
                        "kernel": kernel,
                        "seed": seed,
                        "status": "planned",
                        "run_dir": str(run_dir),
                        "config_path": str(config_path),
                    }
                )
                print(f"[plan] {config_path} -> {run_dir}")
                continue
            if not args.overwrite and (run_dir / "eval_metrics.csv").exists():
                row = summarize_run(run_dir, kernel=kernel, seed=seed, status="skipped")
                row["config_path"] = str(config_path)
                rows.append(row)
                print(f"[skip] {config_path} -> {run_dir}")
                continue

            print(f"[run] {config_path} -> {run_dir}")
            trainer = ECRGRPOTrainer(config)
            trainer.train()
            row = summarize_run(run_dir, kernel=kernel, seed=seed, status="done")
            row["config_path"] = str(config_path)
            rows.append(row)

    write_csv(output_root / "alfworld_summary.csv", rows)
    print(f"wrote {output_root / 'alfworld_summary.csv'}")


if __name__ == "__main__":
    main()
