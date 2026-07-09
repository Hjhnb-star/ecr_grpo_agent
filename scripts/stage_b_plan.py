from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_STAGE_B_SEEDS = [7, 13, 21, 42, 100]
DEFAULT_NONLOCAL_LAGS = [1, 2, 3]
DEFAULT_NONLOCAL_REWARD = 0.4

SEED_RE = re.compile(r"seed(?:=)?(\d+)")


def model_id_override() -> str | None:
    value = os.environ.get("ECR_GRPO_MODEL_ID") or os.environ.get("MODEL_ID")
    if value is None or not value.strip():
        return None
    return value.strip()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def parse_seed_list(raw: list[str] | None) -> list[int]:
    if not raw:
        return list(DEFAULT_STAGE_B_SEEDS)
    seeds: list[int] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            seeds.append(int(part))
    return seeds


def infer_seed(path: Path, config: dict[str, Any]) -> int:
    if "seed" in config:
        return int(config["seed"])
    match = SEED_RE.search(path.stem) or SEED_RE.search(str(path.parent))
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer seed for {path}")


def load_seeded_configs(base_paths: list[Path], seeds: list[int]) -> list[tuple[int, dict[str, Any], Path]]:
    if not base_paths:
        raise ValueError("No base config paths were provided")

    by_seed: dict[int, tuple[dict[str, Any], Path]] = {}
    first_config: dict[str, Any] | None = None
    first_path: Path | None = None
    for path in base_paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        seed = infer_seed(path, config)
        by_seed[seed] = (config, path)
        if first_config is None:
            first_config = config
            first_path = path

    if first_config is None or first_path is None:
        raise ValueError("No readable base configs were provided")

    out: list[tuple[int, dict[str, Any], Path]] = []
    for seed in seeds:
        if seed in by_seed:
            config, path = by_seed[seed]
            out.append((seed, deepcopy(config), path))
        else:
            config = deepcopy(first_config)
            config["seed"] = seed
            out.append((seed, config, first_path))
    return out


def normalize_stage_b_config(
    config: dict[str, Any],
    *,
    seed: int,
    lag: int | None,
    reward: float | None,
) -> None:
    config["seed"] = seed
    env = config.setdefault("environment", {})
    non_local = env.setdefault("non_local_credit", {})
    non_local["enabled"] = True
    non_local["prob"] = 1.0
    if lag is not None:
        non_local["lag"] = lag
    if reward is not None:
        non_local["reward"] = reward

    async_cfg = config.setdefault("async", {})
    async_cfg["use_oracle_event_links"] = False
    async_cfg["strip_diagnostic_metadata"] = True

    policy = config.setdefault("policy", {})
    if str(policy.get("kind", "")).lower() in {"hf", "hf_lora", "lora"}:
        override = model_id_override()
        if override is not None:
            policy["model_id"] = override
        policy["action_selection"] = "score"
        policy["action_score_batch_size"] = env_int(
            "ECR_GRPO_ACTION_SCORE_BATCH_SIZE",
            int(policy.get("action_score_batch_size", 2)),
        )
        policy["temperature"] = env_float("ECR_GRPO_TEMPERATURE", max(float(policy.get("temperature", 1.0)), 1.0))
        policy["action_score_normalization"] = str(policy.get("action_score_normalization", "mean"))
        policy["action_score_calibration"] = str(policy.get("action_score_calibration", "pmi"))
        policy["use_chat_template"] = bool(policy.get("use_chat_template", True))
        policy["update_score_mode"] = str(
            os.environ.get(
                "ECR_GRPO_UPDATE_SCORE_MODE",
                policy.get("update_score_mode", "full_distribution"),
            )
        )

        training = config.setdefault("training", {})
        training["learning_rate"] = env_float(
            "ECR_GRPO_STAGE_B_LEARNING_RATE",
            float(training.get("stage_b_learning_rate", training.get("learning_rate", 1e-5))),
        )
        training["clip_eps"] = env_float("ECR_GRPO_CLIP_EPS", float(training.get("clip_eps", 0.2)))
        training["num_updates"] = max(int(training.get("num_updates", 60)), 60)
        training["tasks_per_update"] = max(int(training.get("tasks_per_update", 4)), 4)
        training["group_size"] = max(int(training.get("group_size", 2)), 2)
        training["grad_accum_steps"] = max(int(training.get("grad_accum_steps", 4)), 4)
        training["max_grad_norm"] = float(training.get("max_grad_norm", 1.0))
        training["optimizer"] = "grpo"
        training.setdefault("grpo_reward_unit", str(training.get("advantage_mode", "step")))

        evaluation = config.setdefault("evaluation", {})
        evaluation["rank_num_tasks"] = env_int("ECR_GRPO_RANK_NUM_TASKS", int(evaluation.get("rank_num_tasks", 6)))
        evaluation["rank_top_k"] = env_int("ECR_GRPO_RANK_TOP_K", int(evaluation.get("rank_top_k", 5)))
        evaluation["trace_num_tasks"] = env_int("ECR_GRPO_TRACE_NUM_TASKS", int(evaluation.get("trace_num_tasks", 4)))
        evaluation["trace_top_k"] = env_int("ECR_GRPO_TRACE_TOP_K", int(evaluation.get("trace_top_k", 3)))
