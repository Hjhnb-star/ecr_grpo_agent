from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from ecr_grpo.types import AsyncEvent, StepRecord


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
EPS = 1e-12


def event_source_time(event: AsyncEvent) -> int:
    value = event.metadata.get("source_time", event.event_time)
    try:
        return int(value)
    except (TypeError, ValueError):
        return event.event_time


def normalize_weights(values: list[float]) -> list[float]:
    total = sum(max(0.0, v) for v in values)
    if total <= EPS:
        return [1.0 / len(values) for _ in values] if values else []
    return [max(0.0, v) / total for v in values]


def text_tokens(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            iterable: Iterable[Any] = value.values()
        elif isinstance(value, (list, tuple, set)):
            iterable = value
        else:
            iterable = [value]
        for item in iterable:
            if item is None:
                continue
            tokens.update(t.lower() for t in TOKEN_RE.findall(str(item)))
    return tokens


def metadata_tags(metadata: dict[str, Any]) -> set[str]:
    raw = metadata.get("tags", metadata.get("evidence_tags", []))
    if isinstance(raw, str):
        return text_tokens(raw)
    if isinstance(raw, (list, tuple, set)):
        return text_tokens(*raw)
    return set()


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class EvidenceAttributionScorer:
    """Environment-agnostic event-to-step affinity model.

    The scorer treats explicit environment links as optional evidence, not as a
    requirement. When those links are absent, it still assigns credit from time,
    action/effect text, and generic metadata tags.
    """

    def __init__(
        self,
        *,
        lambda_: float = 0.3,
        temporal_weight: float = 1.0,
        exact_step_weight: float = 2.0,
        tool_weight: float = 1.0,
        subgoal_weight: float = 1.0,
        tag_weight: float = 1.5,
        text_weight: float = 0.75,
    ) -> None:
        self.lambda_ = lambda_
        self.temporal_weight = temporal_weight
        self.exact_step_weight = exact_step_weight
        self.tool_weight = tool_weight
        self.subgoal_weight = subgoal_weight
        self.tag_weight = tag_weight
        self.text_weight = text_weight

    def score(self, event: AsyncEvent, step: StepRecord) -> tuple[float, str]:
        source_time = event_source_time(event)
        distance = max(0, source_time - step.env_time)
        score = self.temporal_weight * math.exp(-self.lambda_ * distance)
        reasons = [f"temporal:{distance}"]

        if event.related_step_id is not None and step.step_id == event.related_step_id:
            score += self.exact_step_weight
            reasons.append("exact_step")

        event_tool = event.related_tool or event.metadata.get("tool") or event.metadata.get("action")
        step_tool = step.tool_name or step.metadata.get("tool") or step.action
        if event_tool and step_tool and str(event_tool).lower() == str(step_tool).lower():
            score += self.tool_weight
            reasons.append("tool")

        event_subgoal = event.related_subgoal or event.metadata.get("subgoal")
        step_subgoal = step.subgoal_id or step.metadata.get("subgoal")
        if event_subgoal and step_subgoal and str(event_subgoal).lower() == str(step_subgoal).lower():
            score += self.subgoal_weight
            reasons.append("subgoal")

        event_tags = metadata_tags(event.metadata)
        step_tags = metadata_tags(step.metadata)
        tag_overlap = jaccard(event_tags, step_tags)
        if tag_overlap > 0.0:
            score += self.tag_weight * tag_overlap
            reasons.append(f"tags:{tag_overlap:.2f}")

        event_text = text_tokens(
            event.event_type,
            event.observation_delta,
            event.related_tool,
            event.related_subgoal,
            event.metadata,
        )
        step_text = text_tokens(step.action, step.tool_name, step.subgoal_id, step.observation, step.metadata)
        text_overlap = jaccard(event_text, step_text)
        if text_overlap > 0.0:
            score += self.text_weight * text_overlap
            reasons.append(f"text:{text_overlap:.2f}")

        return score, "+".join(reasons)
