from __future__ import annotations

import math

from ecr_grpo.credit_kernels import CreditKernel
from ecr_grpo.types import AsyncEvent, CreditAssignment, StepRecord


class PendingStepBuffer:
    def __init__(self, max_age: int = 8) -> None:
        self.max_age = max_age
        self.steps: dict[tuple[str, str, int], StepRecord] = {}

    def add_step(self, step: StepRecord) -> None:
        self.steps[step.key] = step

    def related_steps(self, event: AsyncEvent) -> list[StepRecord]:
        cutoff_step_id = event.related_step_id
        candidates = [
            step
            for step in self.steps.values()
            if step.task_id == event.task_id
            and step.episode_id == event.episode_id
            and step.env_time <= event.event_time
            and (cutoff_step_id is None or step.step_id <= cutoff_step_id)
            and step.status in {"pending", "credited"}
        ]
        candidates.sort(key=lambda s: s.step_id)
        return candidates

    def assign_event(self, event: AsyncEvent, kernel: CreditKernel) -> list[CreditAssignment]:
        steps = self.related_steps(event)
        weights = kernel.weights(event, steps)
        assignments: list[CreditAssignment] = []
        reasons = getattr(kernel, "last_reasons", None)
        route = str(getattr(kernel, "last_category", "unknown"))
        routing_confidence = float(getattr(kernel, "last_confidence", 0.0))
        weight_stats = self._weight_stats(weights)
        for idx, (step, weight) in enumerate(zip(steps, weights)):
            if weight == 0.0:
                continue
            credit = event.reward * weight
            step.filled_credit += credit
            abs_credit = abs(credit)
            step.metadata["credit_abs_mass"] = float(
                step.metadata.get("credit_abs_mass", 0.0)
            ) + abs_credit
            step.metadata["credit_confidence_mass"] = float(
                step.metadata.get("credit_confidence_mass", 0.0)
            ) + abs_credit * routing_confidence
            step.status = "terminal" if event.terminal else "credited"
            reason = kernel.name
            if reasons and idx < len(reasons):
                reason = f"{kernel.name}:{reasons[idx]}"
            assignments.append(
                CreditAssignment(
                    step_key=step.key,
                    event_id=event.event_id,
                    raw_reward=event.reward,
                    kernel_weight=weight,
                    assigned_credit=credit,
                    reason=reason,
                    route=route,
                    weight_entropy=weight_stats["weight_entropy"],
                    effective_steps=weight_stats["effective_steps"],
                    top_weight=weight_stats["top_weight"],
                    top_margin=weight_stats["top_margin"],
                    routing_confidence=routing_confidence,
                )
            )
        return assignments

    def _weight_stats(self, weights: list[float]) -> dict[str, float]:
        probs = [max(0.0, weight) for weight in weights]
        total = sum(probs)
        if total <= 1e-12:
            return {
                "weight_entropy": 0.0,
                "effective_steps": 0.0,
                "top_weight": 0.0,
                "top_margin": 0.0,
            }
        probs = [weight / total for weight in probs]
        entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
        normalized_entropy = entropy / math.log(len(probs)) if len(probs) > 1 else 0.0
        ranked = sorted(probs, reverse=True)
        top_weight = ranked[0]
        second_weight = ranked[1] if len(ranked) > 1 else 0.0
        return {
            "weight_entropy": normalized_entropy,
            "effective_steps": math.exp(entropy),
            "top_weight": top_weight,
            "top_margin": top_weight - second_weight,
        }

    def finalize_ready(self, current_time: int) -> list[StepRecord]:
        ready: list[StepRecord] = []
        for key, step in list(self.steps.items()):
            if step.status == "terminal":
                ready.append(self.steps.pop(key))
            elif current_time - step.env_time >= self.max_age:
                step.status = "expired"
                ready.append(self.steps.pop(key))
        return ready

    def flush_episode(self, episode_id: str) -> list[StepRecord]:
        ready: list[StepRecord] = []
        for key, step in list(self.steps.items()):
            if step.episode_id == episode_id:
                ready.append(self.steps.pop(key))
        return ready
