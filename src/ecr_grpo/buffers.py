from __future__ import annotations

from ecr_grpo.attribution import event_source_time
from ecr_grpo.credit_kernels import CreditKernel
from ecr_grpo.types import AsyncEvent, CreditAssignment, StepRecord


class PendingStepBuffer:
    def __init__(self, max_age: int = 8) -> None:
        self.max_age = max_age
        self.steps: dict[tuple[str, str, int], StepRecord] = {}

    def add_step(self, step: StepRecord) -> None:
        self.steps[step.key] = step

    def related_steps(self, event: AsyncEvent) -> list[StepRecord]:
        source_time = event_source_time(event)
        cutoff_step_id = event.related_step_id
        candidates = [
            step
            for step in self.steps.values()
            if step.task_id == event.task_id
            and step.episode_id == event.episode_id
            and step.env_time <= source_time
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
        for idx, (step, weight) in enumerate(zip(steps, weights)):
            if weight == 0.0:
                continue
            credit = event.reward * weight
            step.filled_credit += credit
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
                )
            )
        return assignments

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
