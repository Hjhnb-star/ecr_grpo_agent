from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_STAGE_B_SEEDS = [7, 13, 21, 42, 100]
DEFAULT_NONLOCAL_LAGS = [1, 2, 3]
DEFAULT_NONLOCAL_REWARD = 0.4

SEED_RE = re.compile(r"seed(?:=)?(\d+)")


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
        policy["action_selection"] = "score"
        policy["action_score_batch_size"] = int(policy.get("action_score_batch_size", 2))
        if policy["action_score_batch_size"] > 2:
            policy["action_score_batch_size"] = 2
        policy["update_score_mode"] = str(policy.get("update_score_mode", "selected"))
