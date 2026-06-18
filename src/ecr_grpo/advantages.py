from __future__ import annotations

import math
from collections import defaultdict

from ecr_grpo.types import StepRecord


def compute_group_advantages(steps: list[StepRecord], eps: float = 1e-8) -> dict[tuple[str, str, int], float]:
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

