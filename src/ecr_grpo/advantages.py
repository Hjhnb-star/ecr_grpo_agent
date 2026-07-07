from __future__ import annotations

import math
from collections import defaultdict

from ecr_grpo.types import StepRecord


def compute_group_advantages(
    steps: list[StepRecord],
    eps: float = 1e-8,
    *,
    mode: str = "step",
) -> dict[tuple[str, str, int], float]:
    mode = mode.lower()
    if mode in {"step", "step_return", "ecr"}:
        return compute_step_group_advantages(steps, eps=eps)
    if mode in {"trajectory", "episode", "grpo", "trajectory_return"}:
        return compute_trajectory_group_advantages(steps, eps=eps)
    raise ValueError(f"Unknown advantage mode: {mode}")


def compute_step_group_advantages(
    steps: list[StepRecord],
    eps: float = 1e-8,
) -> dict[tuple[str, str, int], float]:
    by_group: dict[str, list[StepRecord]] = defaultdict(list)
    for step in steps:
        by_group[step.group_id].append(step)

    out: dict[tuple[str, str, int], float] = {}
    for group_steps in by_group.values():
        returns = [s.return_estimate for s in group_steps]
        mean = sum(returns) / max(1, len(returns))
        var = sum((r - mean) ** 2 for r in returns) / max(1, len(returns))
        std = math.sqrt(var)
        for step, ret in zip(group_steps, returns):
            adv = 0.0 if std < eps else (ret - mean) / (std + eps)
            step.advantage = adv
            out[step.key] = adv
    return out


def compute_trajectory_group_advantages(
    steps: list[StepRecord],
    eps: float = 1e-8,
) -> dict[tuple[str, str, int], float]:
    """GRPO-style episode-level advantages broadcast to each step.

    This keeps the policy update path identical to ECR step advantages while
    matching the standard trajectory-outcome baseline: every action in an
    episode receives the same group-relative trajectory advantage.
    """

    by_group_episode: dict[str, dict[str, list[StepRecord]]] = defaultdict(lambda: defaultdict(list))
    for step in steps:
        by_group_episode[step.group_id][step.episode_id].append(step)

    out: dict[tuple[str, str, int], float] = {}
    for episodes in by_group_episode.values():
        episode_items = sorted(episodes.items())
        episode_returns = [
            sum(step.return_estimate for step in episode_steps)
            for _, episode_steps in episode_items
        ]
        mean = sum(episode_returns) / max(1, len(episode_returns))
        var = sum((ret - mean) ** 2 for ret in episode_returns) / max(1, len(episode_returns))
        std = math.sqrt(var)
        for (_, episode_steps), ret in zip(episode_items, episode_returns):
            adv = 0.0 if std < eps else (ret - mean) / (std + eps)
            for step in episode_steps:
                step.advantage = adv
                out[step.key] = adv
    return out
