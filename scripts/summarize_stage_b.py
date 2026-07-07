from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ecr_grpo.analyze_credit import analyze_run


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def discover_run_dirs(run_root: Path) -> list[Path]:
    return sorted(path.parent for path in run_root.rglob("eval_metrics.csv"))


def run_kernel(run_dir: Path) -> str:
    parts = run_dir.parts
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def run_seed(run_dir: Path) -> str:
    name = run_dir.name
    if name.startswith("seed="):
        return name.split("=", 1)[1]
    return name


def summarize_run(run_dir: Path) -> dict[str, Any]:
    eval_rows = read_csv(run_dir / "eval_metrics.csv")
    train_rows = read_csv(run_dir / "train_metrics.csv")
    if not eval_rows:
        raise ValueError(f"Missing eval rows in {run_dir}")

    success_values = [numeric(row, "success_rate") for row in eval_rows]
    final_eval = eval_rows[-1]
    final_train = train_rows[-1] if train_rows else {}
    credit = analyze_run(run_dir)

    return {
        "kernel": run_kernel(run_dir),
        "seed": run_seed(run_dir),
        "run_dir": str(run_dir),
        "final_success": numeric(final_eval, "success_rate"),
        "peak_success": max(success_values),
        "logged_mean_success": mean(success_values),
        "final_avg_env_return": numeric(final_eval, "avg_env_return"),
        "final_credit_causal": numeric(final_train, "credit_mass_on_causal_steps"),
        "final_entropy": numeric(final_train, "entropy"),
        "non_local_events": credit["non_local_events"],
        "analyzed_events": credit["analyzed_events"],
        "target_weight_mean": credit["target_weight_mean"],
        "recent_weight_mean": credit["recent_weight_mean"],
        "target_credit_fraction": credit["target_credit_fraction"],
        "weight_entropy_mean": credit["weight_entropy_mean"],
        "effective_steps_mean": credit["effective_steps_mean"],
        "top_weight_mean": credit["top_weight_mean"],
        "top_margin_mean": credit["top_margin_mean"],
        "argmax_target_rate": credit["argmax_target_rate"],
        "target_top3_rate": credit["target_top3_rate"],
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["kernel"])].append(row)

    metrics = [
        "final_success",
        "peak_success",
        "logged_mean_success",
        "final_avg_env_return",
        "final_credit_causal",
        "final_entropy",
        "non_local_events",
        "analyzed_events",
        "target_weight_mean",
        "recent_weight_mean",
        "target_credit_fraction",
        "weight_entropy_mean",
        "effective_steps_mean",
        "top_weight_mean",
        "top_margin_mean",
        "argmax_target_rate",
        "target_top3_rate",
    ]
    out: list[dict[str, Any]] = []
    for kernel, kernel_rows in sorted(grouped.items()):
        agg: dict[str, Any] = {"kernel": kernel, "num_seeds": len(kernel_rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in kernel_rows]
            agg[f"{metric}_mean"] = mean(values)
            agg[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(agg)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/hf_lora_stage_b_fair")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root
    run_dirs = discover_run_dirs(run_root)
    if not run_dirs:
        raise SystemExit(f"No eval_metrics.csv files found under {run_root}")

    rows = [summarize_run(path) for path in run_dirs]
    agg = aggregate(rows)
    write_csv(output_dir / "stage_b_runs.csv", rows)
    write_csv(output_dir / "stage_b_summary.csv", agg)

    print(f"wrote {output_dir / 'stage_b_runs.csv'}")
    print(f"wrote {output_dir / 'stage_b_summary.csv'}")
    for row in agg:
        print(
            f"{row['kernel']} final={row['final_success_mean']:.3f} "
            f"auc={row['logged_mean_success_mean']:.3f} "
            f"target_w={row['target_weight_mean_mean']:.3f} "
            f"recent_w={row['recent_weight_mean_mean']:.3f} "
            f"argmax={row['argmax_target_rate_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
