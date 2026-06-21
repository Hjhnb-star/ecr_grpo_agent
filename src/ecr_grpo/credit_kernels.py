from __future__ import annotations

import math
from typing import Protocol

from ecr_grpo.attribution import EvidenceAttributionScorer, normalize_weights
from ecr_grpo.types import AsyncEvent, StepRecord


EPS = 1e-12


class CreditKernel(Protocol):
    name: str

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        ...


def _normalize(values: list[float]) -> list[float]:
    total = sum(abs(v) for v in values)
    if total <= EPS:
        return [1.0 / len(values) for _ in values] if values else []
    return [v / total for v in values]


class TrajectoryKernel:
    name = "trajectory"

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        if not event.terminal:
            return [0.0 for _ in steps]
        return [1.0 for _ in steps]


class UniformKernel:
    name = "uniform"

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        if not steps:
            return []
        return [1.0 / len(steps) for _ in steps]


class RecencyDecayKernel:
    name = "recency"

    def __init__(self, lambda_: float = 0.3) -> None:
        self.lambda_ = lambda_

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        raw = [
            math.exp(-self.lambda_ * max(0, event.event_time - step.env_time))
            for step in steps
        ]
        return _normalize(raw)


class DependencyAwareKernel:
    name = "dependency"

    def __init__(
        self,
        lambda_: float = 0.3,
        tool_match_bonus: float = 1.5,
        subgoal_match_bonus: float = 2.0,
    ) -> None:
        self.lambda_ = lambda_
        self.tool_match_bonus = tool_match_bonus
        self.subgoal_match_bonus = subgoal_match_bonus

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        raw: list[float] = []
        for step in steps:
            distance = max(0, event.event_time - step.env_time)
            score = math.exp(-self.lambda_ * distance)
            if event.related_step_id is not None and step.step_id == event.related_step_id:
                score *= 2.0
            if event.related_tool and step.tool_name == event.related_tool:
                score *= self.tool_match_bonus
            if event.related_subgoal and step.subgoal_id == event.related_subgoal:
                score *= self.subgoal_match_bonus
            raw.append(score)
        return _normalize(raw)


class EvidenceKernel:
    name = "evidence"

    def __init__(
        self,
        lambda_: float = 0.3,
        temporal_weight: float = 1.0,
        exact_step_weight: float = 2.0,
        tool_weight: float = 1.0,
        subgoal_weight: float = 1.0,
        tag_weight: float = 1.5,
        text_weight: float = 0.75,
    ) -> None:
        self.scorer = EvidenceAttributionScorer(
            lambda_=lambda_,
            temporal_weight=temporal_weight,
            exact_step_weight=exact_step_weight,
            tool_weight=tool_weight,
            subgoal_weight=subgoal_weight,
            tag_weight=tag_weight,
            text_weight=text_weight,
        )
        self.last_reasons: list[str] = []

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        scored = [self.scorer.score(event, step) for step in steps]
        self.last_reasons = [reason for _, reason in scored]
        return normalize_weights([score for score, _ in scored])


def build_credit_kernel(config: dict) -> CreditKernel:
    name = str(config.get("kernel", "dependency")).lower()
    if name == "trajectory":
        return TrajectoryKernel()
    if name == "uniform":
        return UniformKernel()
    if name == "recency":
        return RecencyDecayKernel(lambda_=float(config.get("lambda", 0.3)))
    if name == "dependency":
        return DependencyAwareKernel(
            lambda_=float(config.get("lambda", 0.3)),
            tool_match_bonus=float(config.get("tool_match_bonus", 1.5)),
            subgoal_match_bonus=float(config.get("subgoal_match_bonus", 2.0)),
        )
    if name == "evidence":
        return EvidenceKernel(
            lambda_=float(config.get("lambda", 0.3)),
            temporal_weight=float(config.get("temporal_weight", 1.0)),
            exact_step_weight=float(config.get("exact_step_weight", 2.0)),
            tool_weight=float(config.get("tool_weight", 1.0)),
            subgoal_weight=float(config.get("subgoal_weight", 1.0)),
            tag_weight=float(config.get("tag_weight", 1.5)),
            text_weight=float(config.get("text_weight", 0.75)),
        )
    raise ValueError(f"Unknown credit kernel: {name}")
