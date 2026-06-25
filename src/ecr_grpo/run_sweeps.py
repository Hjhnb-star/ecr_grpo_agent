from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any
import math

from ecr_grpo.io import ensure_dir, load_config, write_csv
from ecr_grpo.trainer import ECRGRPOTrainer


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        as_float = float(raw)
    except ValueError:
        return raw
    if as_float.is_integer() and "." not in raw:
        return int(as_float)
    return as_float


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = config
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _safe_name(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("/", "_").replace("\\", "_")


def run_sweep(
    config: dict,
    *,
    param: str,
    values: list[Any],
    kernels: list[str],
    num_updates: int | None,
    output_root: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = ensure_dir(output_root)

    for value in values:
        for kernel in kernels:
            cfg = copy.deepcopy(config)
            _set_nested(cfg, param, value)
            cfg.setdefault("credit", {})["kernel"] = kernel
            if num_updates is not None:
                cfg.setdefault("training", {})["num_updates"] = num_updates
            run_dir = root / f"{param.replace('.', '_')}={_safe_name(value)}" / kernel
            cfg["output_dir"] = str(run_dir)

            print(f"\n=== Sweep {param}={value} kernel={kernel} ===")
            trainer = ECRGRPOTrainer(cfg)
            trainer.train()

            final_eval = trainer.eval_rows[-1] if trainer.eval_rows else {}
            final_train = trainer.train_rows[-1] if trainer.train_rows else {}
            row = {
                "param": param,
                "value": value,
                "kernel": kernel,
                "num_updates": cfg["training"]["num_updates"],
                "success_rate": final_eval.get("success_rate", 0.0),
                "avg_steps": final_eval.get("avg_steps", 0.0),
                "avg_env_return": final_eval.get("avg_env_return", 0.0),
                "credit_mass_on_causal_steps": final_train.get("credit_mass_on_causal_steps", 0.0),
                "positive_credit_frac": final_train.get("positive_credit_frac", 0.0),
                "entropy": final_train.get("entropy", 0.0),
                "output_dir": str(run_dir),
            }
            rows.append(row)
            write_csv(root / "sweep_summary.csv", rows)

    return rows


def aggregate_by_kernel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["kernel"]), []).append(row)

    metrics = [
        "success_rate",
        "avg_env_return",
        "credit_mass_on_causal_steps",
        "positive_credit_frac",
        "entropy",
    ]
    out: list[dict[str, Any]] = []
    for kernel, kernel_rows in grouped.items():
        item: dict[str, Any] = {"kernel": kernel, "n": len(kernel_rows)}
        for metric in metrics:
            values = [float(row.get(metric, 0.0)) for row in kernel_rows]
            mean = sum(values) / max(1, len(values))
            var = sum((v - mean) ** 2 for v in values) / max(1, len(values))
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = math.sqrt(var)
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--param", required=True, help="Dotted config key, e.g. async.timeout_prob")
    parser.add_argument("--values", nargs="+", required=True)
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["trajectory", "uniform", "recency", "evidence"],
    )
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--output-root", default="runs/sweeps")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = run_sweep(
        config,
        param=args.param,
        values=[_parse_value(v) for v in args.values],
        kernels=args.kernels,
        num_updates=args.updates,
        output_root=args.output_root,
    )
    aggregate_rows = aggregate_by_kernel(rows)
    write_csv(Path(args.output_root) / "aggregate_by_kernel.csv", aggregate_rows)

    print("\nSweep summary:")
    for row in rows:
        print(
            f"{row['param']}={row['value']} {row['kernel']:>10s} "
            f"success={row['success_rate']:.3f} "
            f"return={row['avg_env_return']:.3f} "
            f"credit_causal={row['credit_mass_on_causal_steps']:.3f}"
        )
    print(f"\nAggregate written to: {Path(args.output_root) / 'aggregate_by_kernel.csv'}")


if __name__ == "__main__":
    main()
