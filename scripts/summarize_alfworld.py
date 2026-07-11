from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ecr_grpo.io import write_csv
from ecr_grpo.run_alfworld import summarize_run


def numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "done":
            grouped[str(row["kernel"])].append(row)

    metrics = (
        "final_success_seen",
        "final_success_ood",
        "final_success",
        "final_avg_steps",
        "zero_advantage_frac",
        "attribution_routing_confidence",
        "weak_routing_frac",
        "avg_abs_step_residual",
        "residual_active_frac",
    )
    output: list[dict[str, Any]] = []
    for kernel, kernel_rows in sorted(grouped.items()):
        item: dict[str, Any] = {"kernel": kernel, "num_seeds": len(kernel_rows)}
        for metric in metrics:
            values = [
                value
                for row in kernel_rows
                if (value := numeric(row, metric)) is not None
            ]
            item[f"{metric}_mean"] = mean(values) if values else ""
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else ""
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize concurrent ALFWorld runs.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(run_root.glob("*/seed=*")):
        if not run_dir.is_dir():
            continue
        kernel = run_dir.parent.name
        seed = int(run_dir.name.split("=", 1)[1])
        status = "done" if (run_dir / "COMPLETED.json").exists() else "incomplete"
        rows.append(summarize_run(run_dir, kernel=kernel, seed=seed, status=status))

    if not rows:
        raise SystemExit(f"No ALFWorld run directories found under {run_root}")
    write_csv(output_dir / "alfworld_runs.csv", rows)
    write_csv(output_dir / "alfworld_aggregate.csv", aggregate(rows))
    print(f"wrote {output_dir / 'alfworld_runs.csv'}")
    print(f"wrote {output_dir / 'alfworld_aggregate.csv'}")


if __name__ == "__main__":
    main()
