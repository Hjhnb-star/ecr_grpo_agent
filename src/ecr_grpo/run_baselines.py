from __future__ import annotations

import argparse
import copy
from pathlib import Path

from ecr_grpo.io import ensure_dir, load_config, write_csv
from ecr_grpo.trainer import ECRGRPOTrainer


def run_baselines(
    config: dict,
    *,
    kernels: list[str],
    num_updates: int | None = None,
    output_root: str | Path = "runs/baselines",
) -> list[dict]:
    rows: list[dict] = []
    root = ensure_dir(output_root)

    for kernel in kernels:
        cfg = copy.deepcopy(config)
        cfg.setdefault("credit", {})["kernel"] = kernel
        if num_updates is not None:
            cfg.setdefault("training", {})["num_updates"] = num_updates
        cfg["output_dir"] = str(root / kernel)
        print(f"\n=== Running baseline: {kernel} ===")
        trainer = ECRGRPOTrainer(cfg)
        trainer.train()
        final_eval = trainer.eval_rows[-1] if trainer.eval_rows else {"success_rate": 0.0}
        rows.append(
            {
                "kernel": kernel,
                "num_updates": cfg["training"]["num_updates"],
                "success_rate": final_eval.get("success_rate", 0.0),
                "avg_steps": final_eval.get("avg_steps", 0.0),
                "avg_env_return": final_eval.get("avg_env_return", 0.0),
                "output_dir": cfg["output_dir"],
            }
        )
        write_csv(root / "comparison.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["trajectory", "uniform", "recency", "dependency"],
    )
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--output-root", default="runs/baselines")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = run_baselines(
        config,
        kernels=args.kernels,
        num_updates=args.updates,
        output_root=args.output_root,
    )
    print("\nFinal comparison:")
    for row in rows:
        print(
            f"{row['kernel']:>10s} success={row['success_rate']:.3f} "
            f"steps={row['avg_steps']:.2f} return={row['avg_env_return']:.3f}"
        )


if __name__ == "__main__":
    main()

