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


class LatestStepKernel:
    name = "latest_step"

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        if not steps:
            return []
        weights = [0.0 for _ in steps]
        weights[-1] = 1.0
        return weights


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


def classify_credit_event(event: AsyncEvent, delayed_event_threshold: int = 1) -> str:
    """Route feedback events by the credit question they ask."""
    delta = str(event.observation_delta or "").lower()
    tags = metadata_tags(event.metadata)
    event_type = str(event.event_type)
    route = str(event.metadata.get("credit_route", "")).lower()
    try:
        delay = int(event.metadata.get("delay", 0))
    except (TypeError, ValueError):
        delay = 0

    if (
        route == "non_local"
        or "non_local_support" in delta
        or "non_local_support" in tags
    ):
        return "non_local"
    if event_type == "partial_reward" and delay >= max(1, delayed_event_threshold):
        return "delayed_feedback"
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
        adaptive_evidence: bool = True,
        evidence_confidence_floor: float = 0.25,
        evidence_confidence_power: float = 1.0,
        delayed_event_threshold: int = 1,
        local_window: int = 3,
        delayed_window: int = 8,
        nonlocal_window: int = 12,
        terminal_failure_window: int = 8,
        ambiguous_window: int = 6,
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
        self.adaptive_evidence = bool(adaptive_evidence)
        self.evidence_confidence_floor = min(1.0, max(0.0, evidence_confidence_floor))
        self.evidence_confidence_power = max(0.0, evidence_confidence_power)
        self.delayed_event_threshold = max(1, int(delayed_event_threshold))
        self.category_windows = {
            "local_positive": max(0, int(local_window)),
            "local_negative": max(0, int(local_window)),
            "delayed_feedback": max(0, int(delayed_window)),
            "non_local": max(0, int(nonlocal_window)),
            "terminal_success": 0,
            "terminal_failure": max(0, int(terminal_failure_window)),
            "ambiguous": max(0, int(ambiguous_window)),
        }
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
            "delayed_feedback": [
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
        self.last_confidence = 0.0
        self._last_evidence_confidence = 0.0

    def weights(self, event: AsyncEvent, steps: list[StepRecord]) -> list[float]:
        if not steps:
            self.last_reasons = []
            self.last_confidence = 0.0
            return []
        category = classify_credit_event(event, self.delayed_event_threshold)
        self.last_category = category
        components = self.routes.get(category, self.routes["ambiguous"])
        eligible_indices = self._eligible_indices(category, len(steps))
        eligible_steps = [steps[idx] for idx in eligible_indices]
        local = category.startswith("local")
        weights, reason_parts = self._blend(
            event,
            eligible_steps,
            components,
            local=local,
        )
        expanded_weights = [0.0 for _ in steps]
        expanded_reasons = [f"{category}:outside_window" for _ in steps]
        for source_idx, target_idx in enumerate(eligible_indices):
            expanded_weights[target_idx] = weights[source_idx]
            parts = reason_parts[source_idx]
            expanded_reasons[target_idx] = (
                f"{category}:{';'.join(parts) if parts else 'no_component'}"
            )
        self.last_reasons = expanded_reasons
        return expanded_weights

    def _eligible_indices(self, category: str, length: int) -> list[int]:
        window = self.category_windows.get(category, 0)
        if window <= 0 or length <= window:
            return list(range(length))
        return list(range(length - window, length))

    def _blend(
        self,
        event: AsyncEvent,
        steps: list[StepRecord],
        components: Sequence[tuple[str, float]],
        *,
        local: bool,
    ) -> tuple[list[float], list[list[str]]]:
        active = [(name, weight) for name, weight in components if weight > 0.0]
        if not active:
            active = [("uniform", 1.0)]

        prepared = []
        self.last_confidence = 1.0 if local else 0.0
        for name, alpha in active:
            sub_weights, sub_reasons = self._component_weights(
                event,
                steps,
                name,
                local=local,
            )
            adjusted_alpha = alpha
            if name == "evidence":
                self.last_confidence = self._last_evidence_confidence
                if self.adaptive_evidence:
                    confidence_scale = self.evidence_confidence_floor + (
                        1.0 - self.evidence_confidence_floor
                    ) * self._last_evidence_confidence**self.evidence_confidence_power
                    adjusted_alpha *= confidence_scale
            prepared.append((name, adjusted_alpha, sub_weights, sub_reasons))

        total = sum(alpha for _, alpha, _, _ in prepared)
        if total <= EPS:
            prepared = [
                (
                    "uniform",
                    1.0,
                    self.uniform.weights(event, steps),
                    ["uniform_fallback" for _ in steps],
                )
            ]
            total = 1.0
            self.last_confidence = 0.0

        out = [0.0 for _ in steps]
        reason_parts: list[list[str]] = [[] for _ in steps]
        for name, alpha, sub_weights, sub_reasons in prepared:
            component_weight = alpha / total
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

        weights = normalize_weights(scores)
        self._last_evidence_confidence = self._evidence_confidence(weights, reasons)
        return weights, reasons

    def _evidence_confidence(self, weights: list[float], reasons: list[str]) -> float:
        if not weights:
            return 0.0
        semantic_markers = ("exact_step", "tool", "subgoal", "tags:", "text:")
        semantic_mass = sum(
            weight
            for weight, reason in zip(weights, reasons)
            if any(marker in reason for marker in semantic_markers)
        )
        if semantic_mass <= EPS:
            return 0.0
        if len(weights) == 1:
            return 1.0
        entropy = -sum(weight * math.log(max(weight, EPS)) for weight in weights)
        normalized_entropy = entropy / math.log(len(weights))
        ranked = sorted(weights, reverse=True)
        margin = ranked[0] - ranked[1]
        concentration = max(0.0, 1.0 - normalized_entropy)
        return min(1.0, semantic_mass * max(concentration, margin))


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
    if name in {"local", "latest", "latest_step", "step_local"}:
        return LatestStepKernel()
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
            adaptive_evidence=bool(config.get("adaptive_evidence", True)),
            evidence_confidence_floor=float(config.get("evidence_confidence_floor", 0.25)),
            evidence_confidence_power=float(config.get("evidence_confidence_power", 1.0)),
            local_recency_weight=float(config.get("local_recency_weight", 1.0)),
            delayed_event_threshold=int(config.get("delayed_event_threshold", 1)),
            local_window=int(config.get("local_window", 3)),
            delayed_window=int(config.get("delayed_window", 8)),
            nonlocal_window=int(config.get("nonlocal_window", 12)),
            terminal_failure_window=int(config.get("terminal_failure_window", 8)),
            ambiguous_window=int(config.get("ambiguous_window", 6)),
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
