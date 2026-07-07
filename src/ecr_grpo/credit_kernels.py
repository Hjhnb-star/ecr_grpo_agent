from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from ecr_grpo.attribution import EvidenceAttributionScorer, metadata_tags, normalize_weights
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
    name = "trajectory_broadcast"

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        if not event.terminal:
            return [0.0 for _ in steps]
        return [1.0 for _ in steps]


class TrajectoryUniformKernel:
    name = "trajectory_uniform"

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        if not event.terminal or not steps:
            return [0.0 for _ in steps]
        return [1.0 / len(steps) for _ in steps]


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


def classify_credit_event(event: AsyncEvent) -> str:
    """Route feedback events by the credit question they ask."""
    delta = str(event.observation_delta or "").lower()
    tags = metadata_tags(event.metadata)
    event_type = str(event.event_type)
    route = str(event.metadata.get("credit_route", "")).lower()

    if (
        route == "non_local"
        or "non_local_support" in delta
        or "non_local_support" in tags
    ):
        return "non_local"
    if event_type in {"timeout", "interruption"}:
        return "local_negative"
    if event_type == "terminal_success" or (event.terminal and event.reward >= 0.0):
        return "terminal_success"
    if event_type == "terminal_failure" or (event.terminal and event.reward < 0.0):
        return "terminal_failure"
    if event_type == "partial_reward":
        if delta.startswith("completed:") or event.reward >= 0.0:
            return "local_positive"
        if delta.startswith("wrong:") or event.reward < 0.0:
            return "local_negative"
    return "ambiguous"


class GatedEvidenceKernel:
    """Event-routed credit refill.

    Local step feedback stays sharp and recency-based, while non-local feedback is
    assigned with the evidence scorer. Terminal or ambiguous events can use a
    configurable mixture.
    """

    name = "gated_evidence"

    def __init__(
        self,
        *,
        lambda_: float = 0.3,
        local_lambda: float = 1.0,
        temporal_weight: float = 1.0,
        exact_step_weight: float = 2.0,
        tool_weight: float = 1.0,
        subgoal_weight: float = 1.0,
        tag_weight: float = 1.5,
        text_weight: float = 0.75,
        evidence_temperature: float = 1.0,
        evidence_top_k: int = 0,
        local_recency_weight: float = 1.0,
        local_evidence_weight: float = 0.0,
        nonlocal_evidence_weight: float = 0.85,
        nonlocal_recency_weight: float = 0.15,
        terminal_success_uniform_weight: float = 0.4,
        terminal_success_evidence_weight: float = 0.4,
        terminal_success_recency_weight: float = 0.2,
        terminal_failure_recency_weight: float = 0.7,
        terminal_failure_evidence_weight: float = 0.3,
        ambiguous_evidence_weight: float = 0.5,
        ambiguous_recency_weight: float = 0.5,
    ) -> None:
        self.evidence = EvidenceKernel(
            lambda_=lambda_,
            temporal_weight=temporal_weight,
            exact_step_weight=exact_step_weight,
            tool_weight=tool_weight,
            subgoal_weight=subgoal_weight,
            tag_weight=tag_weight,
            text_weight=text_weight,
        )
        self.recency = RecencyDecayKernel(lambda_=lambda_)
        self.local_recency = RecencyDecayKernel(lambda_=local_lambda)
        self.uniform = UniformKernel()
        self.evidence_temperature = max(evidence_temperature, 1e-6)
        self.evidence_top_k = max(0, evidence_top_k)
        self.routes = {
            "local_positive": [
                ("recency", local_recency_weight),
                ("evidence", local_evidence_weight),
            ],
            "local_negative": [
                ("recency", local_recency_weight),
                ("evidence", local_evidence_weight),
            ],
            "non_local": [
                ("evidence", nonlocal_evidence_weight),
                ("recency", nonlocal_recency_weight),
            ],
            "terminal_success": [
                ("uniform", terminal_success_uniform_weight),
                ("evidence", terminal_success_evidence_weight),
                ("recency", terminal_success_recency_weight),
            ],
            "terminal_failure": [
                ("recency", terminal_failure_recency_weight),
                ("evidence", terminal_failure_evidence_weight),
            ],
            "ambiguous": [
                ("evidence", ambiguous_evidence_weight),
                ("recency", ambiguous_recency_weight),
            ],
        }
        self.last_category = "ambiguous"
        self.last_reasons: list[str] = []

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        if not steps:
            self.last_reasons = []
            return []
        category = classify_credit_event(event)
        self.last_category = category
        components = self.routes.get(category, self.routes["ambiguous"])
        weights, reason_parts = self._blend(event, steps, components, local=category.startswith("local"))
        self.last_reasons = [
            f"{category}:{';'.join(parts) if parts else 'no_component'}"
            for parts in reason_parts
        ]
        return weights

    def _blend(
        self,
        event: AsyncEvent,
        steps: list[StepRecord],
        components: Sequence[tuple[str, float]],
        *,
        local: bool,
    ) -> tuple[list[float], list[list[str]]]:
        active = [(name, weight) for name, weight in components if weight > 0.0]
        total = sum(weight for _, weight in active)
        if total <= EPS:
            active = [("uniform", 1.0)]
            total = 1.0

        out = [0.0 for _ in steps]
        reason_parts: list[list[str]] = [[] for _ in steps]
        for name, alpha in active:
            component_weight = alpha / total
            sub_weights, sub_reasons = self._component_weights(event, steps, name, local=local)
            for idx, weight in enumerate(sub_weights):
                out[idx] += component_weight * weight
                if weight > 0.0:
                    reason_parts[idx].append(f"{name}:{component_weight:.2f}:{sub_reasons[idx]}")
        return normalize_weights(out), reason_parts

    def _component_weights(
        self,
        event: AsyncEvent,
        steps: list[StepRecord],
        name: str,
        *,
        local: bool,
    ) -> tuple[list[float], list[str]]:
        if name == "evidence":
            return self._evidence_weights(event, steps)
        if name == "recency":
            kernel = self.local_recency if local else self.recency
            return kernel.weights(event, steps), ["recency" for _ in steps]
        if name == "uniform":
            return self.uniform.weights(event, steps), ["uniform" for _ in steps]
        raise ValueError(f"Unknown gated evidence component: {name}")

    def _evidence_weights(
        self,
        event: AsyncEvent,
        steps: list[StepRecord],
    ) -> tuple[list[float], list[str]]:
        scored = [self.evidence.scorer.score(event, step) for step in steps]
        scores = [max(0.0, score) for score, _ in scored]
        reasons = [reason for _, reason in scored]

        if self.evidence_top_k > 0 and self.evidence_top_k < len(scores):
            keep = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[: self.evidence_top_k]
            keep_set = set(keep)
            scores = [score if idx in keep_set else 0.0 for idx, score in enumerate(scores)]
            reasons = [
                f"{reason}+topk" if idx in keep_set else f"{reason}+pruned"
                for idx, reason in enumerate(reasons)
            ]

        if abs(self.evidence_temperature - 1.0) > 1e-9:
            power = 1.0 / self.evidence_temperature
            scores = [score**power for score in scores]
            reasons = [f"{reason}+temp:{self.evidence_temperature:.2f}" for reason in reasons]

        return normalize_weights(scores), reasons


def build_credit_kernel(config: dict) -> CreditKernel:
    name = str(config.get("kernel", "dependency")).lower()
    if name in {"trajectory", "trajectory_broadcast"}:
        return TrajectoryKernel()
    if name in {"trajectory_uniform", "trajectory_conserved"}:
        return TrajectoryUniformKernel()
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
    if name in {"gated_evidence", "gated", "hybrid_evidence"}:
        return GatedEvidenceKernel(
            lambda_=float(config.get("lambda", config.get("evidence_lambda", 0.3))),
            local_lambda=float(config.get("local_lambda", 1.0)),
            temporal_weight=float(config.get("temporal_weight", 1.0)),
            exact_step_weight=float(config.get("exact_step_weight", 2.0)),
            tool_weight=float(config.get("tool_weight", 1.0)),
            subgoal_weight=float(config.get("subgoal_weight", 1.0)),
            tag_weight=float(config.get("tag_weight", 1.5)),
            text_weight=float(config.get("text_weight", 0.75)),
            evidence_temperature=float(config.get("evidence_temperature", 1.0)),
            evidence_top_k=int(config.get("evidence_top_k", 0)),
            local_recency_weight=float(config.get("local_recency_weight", 1.0)),
            local_evidence_weight=float(config.get("local_evidence_weight", 0.0)),
            nonlocal_evidence_weight=float(config.get("nonlocal_evidence_weight", 0.85)),
            nonlocal_recency_weight=float(config.get("nonlocal_recency_weight", 0.15)),
            terminal_success_uniform_weight=float(config.get("terminal_success_uniform_weight", 0.4)),
            terminal_success_evidence_weight=float(config.get("terminal_success_evidence_weight", 0.4)),
            terminal_success_recency_weight=float(config.get("terminal_success_recency_weight", 0.2)),
            terminal_failure_recency_weight=float(config.get("terminal_failure_recency_weight", 0.7)),
            terminal_failure_evidence_weight=float(config.get("terminal_failure_evidence_weight", 0.3)),
            ambiguous_evidence_weight=float(config.get("ambiguous_evidence_weight", 0.5)),
            ambiguous_recency_weight=float(config.get("ambiguous_recency_weight", 0.5)),
        )
    raise ValueError(f"Unknown credit kernel: {name}")
