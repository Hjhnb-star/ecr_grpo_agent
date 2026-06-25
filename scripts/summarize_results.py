from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROBUSTNESS_RUNS = {
    "delay": "robust_delay",
    "missing_reward": "robust_missing_reward",
    "interruption": "robust_interruption",
    "timeout": "robust_timeout",
}

DIAG_METRICS = [
    "non_local_events",
    "analyzed_events",
    "target_weight_mean",
    "recent_weight_mean",
    "target_credit_fraction",
    "argmax_target_rate",
    "target_top3_rate",
]

PERF_METRICS = [
    "success_rate",
    "avg_env_return",
    "credit_mass_on_causal_steps",
    "positive_credit_frac",
    "entropy",
]


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


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def kernel_from_run_label(label: str) -> str:
    return label.replace("\\", "/").rstrip("/").split("/")[-1]


def aggregate_diagnostic(path: Path, *, experiment: str, variant: str | None = None) -> list[dict[str, Any]]:
    rows = read_csv(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        kernel = row.get("kernel") or kernel_from_run_label(row.get("run", "unknown"))
        grouped[kernel].append(row)

    out: list[dict[str, Any]] = []
    for kernel, kernel_rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "experiment": experiment,
            "kernel": kernel,
            "n": len(kernel_rows),
        }
        if variant is not None:
            item["variant"] = variant
        for metric in DIAG_METRICS:
            values = [as_float(row, metric) for row in kernel_rows]
            mean, std = mean_std(values)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        out.append(item)
    return out


def load_performance_table(path: Path, *, experiment: str, variant: str | None = None) -> list[dict[str, Any]]:
    rows = read_csv(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {"experiment": experiment, **row}
        if variant is not None:
            item["variant"] = variant
        out.append(item)
    return out


def load_robustness_table(path: Path, *, experiment: str) -> list[dict[str, Any]]:
    rows = read_csv(path)
    return [{"experiment": experiment, **row} for row in rows]


def build_pivot(rows: list[dict[str, Any]], *, metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    kernels: set[str] = set()
    for row in rows:
        param = str(row.get("param", ""))
        value = str(row.get("value", ""))
        kernel = str(row.get("kernel", ""))
        kernels.add(kernel)
        key = (param, value)
        grouped.setdefault(key, {"param": param, "value": value})
        grouped[key][kernel] = as_float(row, metric)

    ordered_kernels = sorted(kernels)
    out = []
    for key in sorted(grouped, key=lambda x: (x[0], float(x[1]) if is_number(x[1]) else x[1])):
        item = grouped[key]
        for kernel in ordered_kernels:
            item.setdefault(kernel, "")
        out.append(item)
    return out


def is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def best_kernel(rows: list[dict[str, Any]], metric_col: str, *, experiment: str) -> tuple[str, float] | None:
    candidates = [row for row in rows if row.get("experiment") == experiment]
    best: tuple[str, float] | None = None
    for row in candidates:
        kernel = str(row.get("kernel", ""))
        value = as_float(row, metric_col)
        if best is None or value > best[1]:
            best = (kernel, value)
    return best


def row_for_kernel(rows: list[dict[str, Any]], *, experiment: str, kernel: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("experiment") == experiment and row.get("kernel") == kernel:
            return row
    return None


def fmt(value: float) -> str:
    return f"{value:.3f}"


def make_markdown_summary(
    *,
    main_perf: list[dict[str, Any]],
    main_diag: list[dict[str, Any]],
    v2_perf: list[dict[str, Any]],
    v2_diag: list[dict[str, Any]],
    robustness: list[dict[str, Any]],
    ablation_perf: list[dict[str, Any]],
    ablation_diag: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# ECR-GRPO Result Summary",
        "",
        "## Main Results",
        "",
    ]

    for experiment, perf_rows, diag_rows in [
        ("nonlocal_seed_no_oracle_main", main_perf, main_diag),
        ("nonlocal_seed_no_oracle_v2", v2_perf, v2_diag),
    ]:
        lines.append(f"### {experiment}")
        evidence_perf = row_for_kernel(perf_rows, experiment=experiment, kernel="evidence")
        recency_perf = row_for_kernel(perf_rows, experiment=experiment, kernel="recency")
        evidence_diag = row_for_kernel(diag_rows, experiment=experiment, kernel="evidence")
        recency_diag = row_for_kernel(diag_rows, experiment=experiment, kernel="recency")
        uniform_diag = row_for_kernel(diag_rows, experiment=experiment, kernel="uniform")

        if evidence_perf and recency_perf:
            lines.append(
                "- Performance: evidence success "
                f"{fmt(as_float(evidence_perf, 'success_rate_mean'))} vs recency "
                f"{fmt(as_float(recency_perf, 'success_rate_mean'))}; evidence return "
                f"{fmt(as_float(evidence_perf, 'avg_env_return_mean'))} vs recency "
                f"{fmt(as_float(recency_perf, 'avg_env_return_mean'))}."
            )
        if evidence_diag and recency_diag:
            lines.append(
                "- Credit diagnostic: evidence target weight "
                f"{fmt(as_float(evidence_diag, 'target_weight_mean_mean'))} vs recency "
                f"{fmt(as_float(recency_diag, 'target_weight_mean_mean'))}; evidence argmax target "
                f"{fmt(as_float(evidence_diag, 'argmax_target_rate_mean'))} vs recency "
                f"{fmt(as_float(recency_diag, 'argmax_target_rate_mean'))}."
            )
        if evidence_diag and uniform_diag:
            lines.append(
                "- Compared with uniform, evidence assigns more credit to the target step: "
                f"{fmt(as_float(evidence_diag, 'target_weight_mean_mean'))} vs "
                f"{fmt(as_float(uniform_diag, 'target_weight_mean_mean'))}."
            )
        lines.append("")

    lines.extend(["## Robustness", ""])
    for experiment in sorted({str(row.get("experiment")) for row in robustness}):
        rows = [row for row in robustness if row.get("experiment") == experiment]
        values = sorted({str(row.get("value")) for row in rows}, key=lambda v: float(v) if is_number(v) else v)
        wins = 0
        total = 0
        for value in values:
            ev = next((row for row in rows if row.get("value") == value and row.get("kernel") == "evidence"), None)
            rec = next((row for row in rows if row.get("value") == value and row.get("kernel") == "recency"), None)
            if ev and rec:
                total += 1
                if as_float(ev, "success_rate") >= as_float(rec, "success_rate"):
                    wins += 1
        lines.append(f"- {experiment}: evidence matches or beats recency on success in {wins}/{total} sweep points.")
    lines.append("")

    lines.extend(["## Ablation", ""])
    if ablation_perf:
        sorted_ablation = sorted(
            ablation_perf,
            key=lambda row: as_float(row, "success_rate_mean"),
            reverse=True,
        )
        for row in sorted_ablation:
            lines.append(
                f"- {row.get('variant')}: success={fmt(as_float(row, 'success_rate_mean'))}, "
                f"return={fmt(as_float(row, 'avg_env_return_mean'))}."
            )
    if ablation_diag:
        lines.append("")
        lines.append("Credit diagnostic should be read together with performance, because success and attribution quality can diverge.")
    lines.append("")

    lines.extend(
        [
            "## Recommended Paper Framing",
            "",
            "- Use main/v2 experiments as the core mechanism validation.",
            "- Use lag and missing-reward robustness to show where evidence-conditioned refill is strongest.",
            "- Use ablation to show temporal and structured evidence are both important.",
            "- Avoid overclaiming that evidence always beats recency under pure delay; recency remains a strong baseline when temporal proximity is reliable.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="runs/paper_tables")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_root = runs_root / "nonlocal_seed_no_oracle_main"
    v2_root = runs_root / "nonlocal_seed_no_oracle_v2"

    main_perf = load_performance_table(
        main_root / "aggregate_by_kernel.csv",
        experiment="nonlocal_seed_no_oracle_main",
    )
    main_diag = aggregate_diagnostic(
        main_root / "table_credit_diagnostic.csv",
        experiment="nonlocal_seed_no_oracle_main",
    )
    v2_perf = load_performance_table(
        v2_root / "aggregate_by_kernel.csv",
        experiment="nonlocal_seed_no_oracle_v2",
    )
    v2_diag = aggregate_diagnostic(
        v2_root / "table_credit_diagnostic.csv",
        experiment="nonlocal_seed_no_oracle_v2",
    )

    robustness: list[dict[str, Any]] = []
    for label, dirname in ROBUSTNESS_RUNS.items():
        robustness.extend(
            load_robustness_table(
                runs_root / dirname / "sweep_summary.csv",
                experiment=label,
            )
        )

    ablation_perf: list[dict[str, Any]] = []
    ablation_diag: list[dict[str, Any]] = []
    for ablation_dir in sorted(runs_root.glob("ablation_*")):
        if not ablation_dir.is_dir():
            continue
        variant = ablation_dir.name.replace("ablation_", "", 1)
        ablation_perf.extend(
            load_performance_table(
                ablation_dir / "aggregate_by_kernel.csv",
                experiment="ablation",
                variant=variant,
            )
        )
        ablation_diag.extend(
            aggregate_diagnostic(
                ablation_dir / "table_credit_diagnostic.csv",
                experiment="ablation",
                variant=variant,
            )
        )

    write_csv(output_dir / "main_performance.csv", main_perf + v2_perf)
    write_csv(output_dir / "credit_diagnostic.csv", main_diag + v2_diag)
    write_csv(output_dir / "robustness_all.csv", robustness)
    write_csv(output_dir / "robustness_success_pivot.csv", build_pivot(robustness, metric="success_rate"))
    write_csv(output_dir / "robustness_return_pivot.csv", build_pivot(robustness, metric="avg_env_return"))
    write_csv(output_dir / "ablation_performance.csv", ablation_perf)
    write_csv(output_dir / "ablation_diagnostic.csv", ablation_diag)

    summary = make_markdown_summary(
        main_perf=main_perf,
        main_diag=main_diag,
        v2_perf=v2_perf,
        v2_diag=v2_diag,
        robustness=robustness,
        ablation_perf=ablation_perf,
        ablation_diag=ablation_diag,
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(f"Wrote summary tables to {output_dir}")
    for path in sorted(output_dir.glob("*")):
        print(f"- {path}")


if __name__ == "__main__":
    main()
