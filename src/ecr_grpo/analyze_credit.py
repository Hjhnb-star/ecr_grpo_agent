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


def _metadata_tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        tokens: set[str] = set()
        for item in value.values():
            tokens.update(_metadata_tokens(item))
        return tokens
    if isinstance(value, (list, tuple, set)):
        tokens: set[str] = set()
        for item in value:
            tokens.update(_metadata_tokens(item))
        return tokens
    return {str(value).lower()}


def is_non_local_event(event: dict[str, Any]) -> bool:
    metadata = event.get("metadata", {})
    diagnostic_metadata = event.get("diagnostic_metadata", {})
    delta = str(event.get("observation_delta", "")).lower()
    tags = _metadata_tokens(metadata.get("tags", metadata.get("evidence_tags", [])))
    return (
        metadata.get("credit_route") == "non_local"
        or "non_local_support" in delta
        or "non_local_support" in tags
        or diagnostic_metadata.get("target_action") is not None
        or diagnostic_metadata.get("target_lag") is not None
    )


def event_target_action(event: dict[str, Any]) -> str | None:
    metadata = event.get("metadata", {})
    diagnostic_metadata = event.get("diagnostic_metadata", {})
    target = diagnostic_metadata.get("target_action")
    if target is not None:
        return str(target)
    target = metadata.get("target_action")
    if target is not None:
        return str(target)
    delta = str(event.get("observation_delta", ""))
    marker = "non_local_support:"
    if marker in delta:
        rest = delta.split(marker, 1)[1]
        return rest.split(":", 1)[0]
    return None


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
        if is_non_local_event(event)
    ]

    target_weight_sum = 0.0
    recent_weight_sum = 0.0
    target_credit_sum = 0.0
    total_credit_sum = 0.0
    weight_entropy_sum = 0.0
    effective_steps_sum = 0.0
    top_weight_sum = 0.0
    top_margin_sum = 0.0
    argmax_target = 0
    target_top3 = 0
    analyzed = 0

    for event in non_local_events:
        event_id = str(event["event_id"])
        target_action = event_target_action(event)
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
        first_assignment = event_assignments[0]
        weight_entropy_sum += float(first_assignment.get("weight_entropy", 0.0))
        effective_steps_sum += float(first_assignment.get("effective_steps", 0.0))
        top_weight_sum += float(first_assignment.get("top_weight", 0.0))
        top_margin_sum += float(first_assignment.get("top_margin", 0.0))
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
        "weight_entropy_mean": weight_entropy_sum / denom,
        "effective_steps_mean": effective_steps_sum / denom,
        "top_weight_mean": top_weight_sum / denom,
        "top_margin_mean": top_margin_sum / denom,
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
