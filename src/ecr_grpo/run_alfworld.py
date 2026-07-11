from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from ecr_grpo.io import load_config, write_csv
from ecr_grpo.trainer import ECRGRPOTrainer


DEFAULT_KERNELS = [
    "grpo",
    "local",
    "recency",
    "evidence",
    "gated",
    "residual",
    "dependency",
]
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
    if kernel in {"local", "latest", "latest_step", "step_local"}:
        return {**common, "kernel": "latest_step"}
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
    if kernel in {"gated", "residual"}:
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
            "adaptive_evidence": base_credit.get("adaptive_evidence", True),
            "evidence_confidence_floor": base_credit.get(
                "evidence_confidence_floor", 0.25
            ),
            "evidence_confidence_power": base_credit.get(
                "evidence_confidence_power", 1.0
            ),
            "delayed_event_threshold": base_credit.get("delayed_event_threshold", 1),
            "local_window": base_credit.get("local_window", 3),
            "delayed_window": base_credit.get("delayed_window", 8),
            "nonlocal_window": base_credit.get("nonlocal_window", 12),
            "terminal_failure_window": base_credit.get(
                "terminal_failure_window", 8
            ),
            "ambiguous_window": base_credit.get("ambiguous_window", 6),
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
    adapter_path: str | None = None,
    train_split: str | None = None,
    eval_split: str | None = None,
    eval_splits: list[str] | None = None,
    num_train_tasks: int | None = None,
    num_eval_tasks: int | None = None,
    max_steps: int | None = None,
    num_updates: int | None = None,
    tasks_per_update: int | None = None,
    group_size: int | None = None,
    eval_every: int | None = None,
    train_delay_prob: float | None = None,
    terminal_reward_delay: int | None = None,
    missing_reward_prob: float | None = None,
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
    if eval_splits:
        evaluation["splits"] = list(dict.fromkeys(eval_splits))
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
    if train_delay_prob is not None:
        async_cfg["delay_prob"] = float(train_delay_prob)
    if terminal_reward_delay is not None:
        async_cfg["terminal_reward_delay"] = int(terminal_reward_delay)
    if missing_reward_prob is not None:
        async_cfg["missing_reward_prob"] = float(missing_reward_prob)

    reward_unit = "trajectory" if kernel in {"grpo", "trajectory", "trajectory_uniform"} else "step"
    base_credit = dict(config.get("credit", {}))
    config["credit"] = make_credit(base_credit, kernel)
    config["credit"]["output"] = f"{reward_unit}_reward"

    optimizer = config.setdefault("optimizer", {})
    optimizer["name"] = "grpo"
    optimizer["advantage_mode"] = reward_unit
    optimizer["update_impl"] = "standard_grpo"
    if reward_unit == "step":
        estimator = (
            "trajectory_grouped_residual"
            if kernel == "residual"
            else "trajectory_grouped_credit"
        )
        optimizer["step_advantage_estimator"] = estimator
        optimizer.setdefault("credit_weight_floor", 0.05)
        optimizer.setdefault("max_credit_multiplier", 4.0)
        if kernel == "residual":
            optimizer.setdefault("residual_beta", 0.5)
            optimizer.setdefault("residual_clip", 2.0)
            optimizer.setdefault("residual_use_confidence", True)
    else:
        optimizer.pop("step_advantage_estimator", None)
        optimizer.pop("credit_weight_floor", None)
        optimizer.pop("max_credit_multiplier", None)
        optimizer.pop("residual_beta", None)
        optimizer.pop("residual_clip", None)
        optimizer.pop("residual_use_confidence", None)

    training = config.setdefault("training", {})
    if num_updates is not None:
        training["num_updates"] = int(num_updates)
    if tasks_per_update is not None:
        training["tasks_per_update"] = int(tasks_per_update)
    if group_size is not None:
        training["group_size"] = int(group_size)
    if eval_every is not None:
        evaluation["every_updates"] = int(eval_every)
    training["optimizer"] = "grpo"
    training["grpo_reward_unit"] = reward_unit
    training["advantage_mode"] = reward_unit

    if reward_unit == "step":
        training["step_advantage_estimator"] = optimizer["step_advantage_estimator"]
    else:
        training.pop("step_advantage_estimator", None)
    policy = config.setdefault("policy", {})
    if model_id:
        policy["model_id"] = model_id
    if adapter_path:
        policy["adapter_path"] = adapter_path
    if str(policy.get("kind", "")).lower() in {"hf", "hf_lora", "lora"}:
        policy.setdefault("action_selection", "score")
        policy.setdefault("update_score_mode", "streaming_distribution")
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
        adapter_path = str(policy.get("adapter_path") or "")
        if (
            adapter_path
            and require_files
            and looks_like_local_path(adapter_path)
            and not Path(adapter_path).expanduser().exists()
        ):
            errors.append(f"local policy.adapter_path does not exist: {adapter_path}")

    training = config.get("training", {})
    optimizer = config.get("optimizer", {})
    group_size = int(training.get("group_size", 1))
    if group_size < 1:
        errors.append("training.group_size must be >= 1")
    reward_unit = str(training.get("grpo_reward_unit", optimizer.get("advantage_mode", "step")))
    estimator = str(
        optimizer.get(
            "step_advantage_estimator",
            training.get("step_advantage_estimator", "prompt_group"),
        )
    )
    if reward_unit.startswith("step") and estimator != "prompt_group" and group_size < 2:
        errors.append("trajectory-grouped step credit requires training.group_size >= 2")
    if (
        str(policy.get("update_score_mode", "")).lower() == "selected"
        and not bool(policy.get("allow_approximate_update", False))
    ):
        errors.append("policy.update_score_mode='selected' is smoke-only; use streaming_distribution")
    if int(config.get("environment", {}).get("max_steps", 0)) <= 0:
        errors.append("environment.max_steps must be > 0")
    if int(config.get("evaluation", {}).get("num_eval_tasks", 1)) <= 0:
        errors.append("evaluation.num_eval_tasks must be > 0")
    return errors


def looks_like_local_path(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or value.startswith((".", "~")) or (len(value) >= 2 and value[1] == ":")


def final_row(csv_path: Path, eval_split: str | None = None) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if eval_split is not None:
        rows = [row for row in rows if row.get("eval_split") == eval_split]
    rows.sort(key=lambda row: int(float(row.get("update", 0) or 0)))
    return rows[-1] if rows else {}


def summarize_run(run_dir: Path, *, kernel: str, seed: int, status: str) -> dict[str, Any]:
    eval_path = run_dir / "eval_metrics.csv"
    eval_ood = final_row(eval_path, "eval_out_of_distribution")
    eval_seen = final_row(eval_path, "eval_in_distribution")
    eval_row = eval_ood or final_row(eval_path)
    train_row = final_row(run_dir / "train_metrics.csv")
    return {
        "kernel": kernel,
        "seed": seed,
        "status": status,
        "run_dir": str(run_dir),
        "final_success": eval_row.get("success_rate", ""),
        "final_avg_steps": eval_row.get("avg_steps", ""),
        "final_success_ood": eval_ood.get("success_rate", ""),
        "final_success_seen": eval_seen.get("success_rate", ""),
        "final_avg_env_return": eval_row.get("avg_env_return", ""),
        "attribution_entropy": train_row.get("attribution_entropy", ""),
        "attribution_top_margin": train_row.get("attribution_top_margin", ""),
        "attribution_routing_confidence": train_row.get(
            "attribution_routing_confidence", ""
        ),
        "weak_routing_frac": train_row.get("weak_routing_frac", ""),
        "avg_abs_step_residual": train_row.get("avg_abs_step_residual", ""),
        "residual_active_frac": train_row.get("residual_active_frac", ""),
        "zero_advantage_frac": train_row.get("zero_advantage_frac", ""),
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
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional per-process summary path for concurrent multi-GPU runs.",
    )
    parser.add_argument("--kernels", nargs="+", default=DEFAULT_KERNELS)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--alfworld-config", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--train-split", default=None, help="ALFWorld split used for policy updates, e.g. train.")
    parser.add_argument("--eval-splits", nargs="+", default=None, help="Additional fixed ALFWorld evaluation splits.")
    parser.add_argument("--eval-split", default=None, help="ALFWorld split used for held-out evaluation.")
    parser.add_argument("--num-train-tasks", type=int, default=None)
    parser.add_argument("--num-eval-tasks", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-updates", type=int, default=None)
    parser.add_argument("--tasks-per-update", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--train-delay-prob", type=float, default=None)
    parser.add_argument("--terminal-reward-delay", type=int, default=None)
    parser.add_argument("--missing-reward-prob", type=float, default=None)
    parser.add_argument("--clean-eval", action="store_true", help="Disable async perturbations during benchmark evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Only write generated configs and validate them.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run from update 1.")
    parser.add_argument("--resume", action="store_true", help="Continue each incomplete run from checkpoints/latest.")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")

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
                adapter_path=args.adapter_path,
                train_split=args.train_split,
                eval_splits=args.eval_splits,
                eval_split=args.eval_split,
                num_train_tasks=args.num_train_tasks,
                num_eval_tasks=args.num_eval_tasks,
                max_steps=args.max_steps,
                num_updates=args.num_updates,
                tasks_per_update=args.tasks_per_update,
                group_size=args.group_size,
                eval_every=args.eval_every,
                train_delay_prob=args.train_delay_prob,
                terminal_reward_delay=args.terminal_reward_delay,
                missing_reward_prob=args.missing_reward_prob,
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
            if not args.overwrite and not args.resume and (run_dir / "COMPLETED.json").exists():
                row = summarize_run(run_dir, kernel=kernel, seed=seed, status="skipped")
                row["config_path"] = str(config_path)
                rows.append(row)
                print(f"[skip] {config_path} -> {run_dir}")
                continue

            if run_dir.exists() and any(run_dir.iterdir()) and not (args.overwrite or args.resume):
                row = summarize_run(run_dir, kernel=kernel, seed=seed, status="incomplete")
                row["config_path"] = str(config_path)
                row["errors"] = "Incomplete run exists; pass --resume or --overwrite"
                rows.append(row)
                print(f"[incomplete] {config_path} -> {run_dir}; use --resume or --overwrite")
                continue

            print(f"[run] {config_path} -> {run_dir}")
            trainer = ECRGRPOTrainer(config, resume=args.resume, overwrite=args.overwrite)
            trainer.train()
            row = summarize_run(run_dir, kernel=kernel, seed=seed, status="done")
            row["config_path"] = str(config_path)
            rows.append(row)

    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else output_root / "alfworld_summary.csv"
    )
    write_csv(summary_path, rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
