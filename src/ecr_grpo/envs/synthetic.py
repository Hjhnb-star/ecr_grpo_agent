from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ecr_grpo.types import AsyncEvent


@dataclass
class SyntheticTask:
    task_id: str
    sequence: list[str]


class SyntheticLongHorizonEnv:
    """Small long-horizon text-action environment with known causal steps."""

    def __init__(
        self,
        tasks: list[SyntheticTask],
        action_space: list[str],
        max_steps: int,
        seed: int = 0,
        non_local_credit: dict[str, Any] | None = None,
    ) -> None:
        self.tasks = tasks
        self.action_space = action_space
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.non_local_credit = non_local_credit or {}
        self.task = tasks[0]
        self.progress = 0
        self.step_count = 0
        self.episode_id = ""
        self.correct_history: list[tuple[int, str]] = []

    def reset(self, task_id: str | None = None, episode_id: str | None = None) -> str:
        if task_id is None:
            self.task = self.rng.choice(self.tasks)
        else:
            matches = [t for t in self.tasks if t.task_id == task_id]
            if not matches:
                raise ValueError(f"Unknown task_id: {task_id}")
            self.task = matches[0]
        self.progress = 0
        self.step_count = 0
        self.episode_id = episode_id or f"episode_{self.rng.randrange(10**9)}"
        self.correct_history = []
        return self._observation()

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        self.step_count += 1
        expected = self._expected_action()
        correct = action == expected
        reward = 0.0
        done = False
        events: list[AsyncEvent] = []

        if correct:
            current_step_id = self.step_count - 1
            self.progress += 1
            self.correct_history.append((current_step_id, expected))
            reward = 0.1
            events.append(
                self._event(
                    "partial_reward",
                    reward=0.1,
                    related_step_id=current_step_id,
                    related_subgoal=expected,
                    observation_delta=f"completed:{expected}",
                    terminal=False,
                )
            )
            non_local_event = self._maybe_non_local_event(current_step_id)
            if non_local_event is not None:
                events.append(non_local_event)
        elif action != "wait":
            reward = -0.03
            events.append(
                self._event(
                    "partial_reward",
                    reward=-0.03,
                    related_step_id=self.step_count - 1,
                    related_subgoal=expected,
                    observation_delta=f"wrong:{action}:expected:{expected}",
                    terminal=False,
                )
            )

        if self.progress >= len(self.task.sequence):
            done = True
            events.append(
                self._event(
                    "terminal_success",
                    reward=1.0,
                    related_step_id=self.step_count - 1,
                    related_subgoal=expected,
                    observation_delta="task_success",
                    terminal=True,
                )
            )
        elif self.step_count >= self.max_steps:
            done = True
            events.append(
                self._event(
                    "terminal_failure",
                    reward=-0.5,
                    related_step_id=self.step_count - 1,
                    related_subgoal=expected,
                    observation_delta="task_failure",
                    terminal=True,
                )
            )

        info = {
            "task_id": self.task.task_id,
            "episode_id": self.episode_id,
            "expected_action": expected,
            "progress": self.progress,
            "step_id": self.step_count - 1,
            "events": events,
            "success": self.progress >= len(self.task.sequence),
            "causal_action": correct,
            "tags": [action, expected, f"progress_{self.progress}", "correct" if correct else "incorrect"],
        }
        return self._observation(), reward, done, info

    def _expected_action(self) -> str:
        if self.progress >= len(self.task.sequence):
            return self.task.sequence[-1]
        return self.task.sequence[self.progress]

    def _observation(self) -> str:
        remaining = len(self.task.sequence) - self.progress
        hint = self._expected_action()
        return (
            f"task={self.task.task_id}; progress={self.progress}/{len(self.task.sequence)}; "
            f"remaining={remaining}; hint={hint}"
        )

    def _maybe_non_local_event(self, current_step_id: int) -> AsyncEvent | None:
        if not self.non_local_credit.get("enabled", False):
            return None
        if self.rng.random() >= float(self.non_local_credit.get("prob", 1.0)):
            return None
        lag = int(self.non_local_credit.get("lag", 2))
        if len(self.correct_history) <= lag:
            return None
        target_step_id, target_action = self.correct_history[-(lag + 1)]
        reward = float(self.non_local_credit.get("reward", 0.15))
        event = self._event(
            "partial_reward",
            reward=reward,
            related_step_id=target_step_id,
            related_subgoal=target_action,
            observation_delta=f"non_local_support:{target_action}:confirmed_after_step_{current_step_id}",
            terminal=False,
        )
        event.metadata.update(
            {
                "tags": [target_action, "non_local_support"],
                "target_action": target_action,
                "target_lag": lag,
                "generated_at_step": current_step_id,
            }
        )
        return event

    def _event(
        self,
        event_type: str,
        *,
        reward: float,
        related_step_id: int,
        related_subgoal: str,
        observation_delta: str,
        terminal: bool,
    ) -> AsyncEvent:
        return AsyncEvent(
            task_id=self.task.task_id,
            episode_id=self.episode_id,
            event_id=f"{self.episode_id}_event_{self.step_count}_{len(observation_delta)}",
            event_type=event_type,  # type: ignore[arg-type]
            event_time=self.step_count,
            reward=reward,
            related_step_id=related_step_id,
            related_tool=related_subgoal,
            related_subgoal=related_subgoal,
            observation_delta=observation_delta,
            terminal=terminal,
            metadata={"tags": [related_subgoal, event_type]},
        )


def build_synthetic_tasks(config: dict) -> list[SyntheticTask]:
    sequences = config["environment"]["sequences"]
    num_tasks = int(config["environment"].get("num_tasks", len(sequences)))
    tasks: list[SyntheticTask] = []
    for i in range(num_tasks):
        seq = list(sequences[i % len(sequences)])
        tasks.append(SyntheticTask(task_id=f"task_{i:04d}", sequence=seq))
    return tasks
