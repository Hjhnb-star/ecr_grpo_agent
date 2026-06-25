from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ecr_grpo.io import write_csv


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _step_key(value: Any) -> tuple[str, str, int]:
    return (str(value[0]), str(value[1]), int(value[2]))


def _step_record_key(step: dict[str, Any]) -> tuple[str, str, int] | None:
    if "key" in step:
        return _step_key(step["key"])
    if {"task_id", "episode_id", "step_id"}.issubset(step):
        return (str(step["task_id"]), str(step["episode_id"]), int(step["step_id"]))
    return None


def _run_label(run_dir: Path) -> str:
    parts = run_dir.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return str(run_dir)


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    events = _load_jsonl(root / "train_events.jsonl")
    assignments = _load_jsonl(root / "credit_assignments.jsonl")
    steps = _load_jsonl(root / "train_steps.jsonl")

    steps_by_key = {}
    for step in steps:
        key = _step_record_key(step)
        if key is not None:
            steps_by_key[key] = step
    assignments_by_event: dict[str, list[dict[str, Any]]] = {}
    for assignment in assignments:
        assignments_by_event.setdefault(str(assignment["event_id"]), []).append(assignment)

    non_local_events = [
        event
        for event in events
        if "non_local_support" in str(event.get("observation_delta", ""))
    ]

    target_weight_sum = 0.0
    recent_weight_sum = 0.0
    target_credit_sum = 0.0
    total_credit_sum = 0.0
    argmax_target = 0
    target_top3 = 0
    analyzed = 0

    for event in non_local_events:
        event_id = str(event["event_id"])
        target_action = event.get("metadata", {}).get("target_action")
        event_assignments = assignments_by_event.get(event_id, [])
        if not target_action or not event_assignments:
            continue

        enriched = []
        for assignment in event_assignments:
            step = steps_by_key.get(_step_key(assignment["step_key"]))
            if step is None:
                continue
            enriched.append((assignment, step))
        if not enriched:
            continue

        analyzed += 1
        ranked = sorted(enriched, key=lambda item: float(item[0]["kernel_weight"]), reverse=True)
        top_assignment, top_step = ranked[0]
        if top_step.get("action") == target_action:
            argmax_target += 1
        if any(step.get("action") == target_action for _, step in ranked[:3]):
            target_top3 += 1

        max_step_id = max(int(step["step_id"]) for _, step in enriched)
        for assignment, step in enriched:
            weight = float(assignment["kernel_weight"])
            credit = float(assignment["assigned_credit"])
            total_credit_sum += credit
            if step.get("action") == target_action:
                target_weight_sum += weight
                target_credit_sum += credit
            if int(step["step_id"]) == max_step_id:
                recent_weight_sum += weight

    denom = max(1, analyzed)
    total_credit_abs = max(1e-12, total_credit_sum)
    return {
        "run": _run_label(root),
        "run_dir": str(root),
        "non_local_events": len(non_local_events),
        "analyzed_events": analyzed,
        "target_weight_mean": target_weight_sum / denom,
        "recent_weight_mean": recent_weight_sum / denom,
        "target_credit_fraction": target_credit_sum / total_credit_abs,
        "argmax_target_rate": argmax_target / denom,
        "target_top3_rate": target_top3 / denom,
    }


def _discover_run_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("credit_assignments.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", default=[])
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs]
    if args.run_root:
        run_dirs.extend(_discover_run_dirs(Path(args.run_root)))
    if not run_dirs:
        raise SystemExit("Provide --run-dirs or --run-root")

    rows = [analyze_run(path) for path in run_dirs]
    if args.output:
        write_csv(args.output, rows)

    for row in rows:
        print(
            f"{row['run']} non_local={row['non_local_events']} "
            f"target_w={row['target_weight_mean']:.3f} "
            f"recent_w={row['recent_weight_mean']:.3f} "
            f"argmax_target={row['argmax_target_rate']:.3f} "
            f"top3_target={row['target_top3_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
