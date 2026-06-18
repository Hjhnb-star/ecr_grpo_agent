from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field

from ecr_grpo.types import AsyncEvent


@dataclass(order=True)
class ScheduledEvent:
    due_time: int
    order: int
    event: AsyncEvent = field(compare=False)


class AsyncEnvWrapper:
    """Injects delayed, missing, timeout, and interruption events."""

    def __init__(self, env, config: dict, seed: int = 0) -> None:
        self.env = env
        self.config = config
        self.rng = random.Random(seed)
        self.current_time = 0
        self.counter = 0
        self.queue: list[ScheduledEvent] = []
        self.task_id = ""
        self.episode_id = ""

    @property
    def action_space(self) -> list[str]:
        return self.env.action_space

    def reset(self, task_id: str | None = None, episode_id: str | None = None) -> str:
        self.current_time = 0
        self.counter = 0
        self.queue.clear()
        obs = self.env.reset(task_id=task_id, episode_id=episode_id)
        self.task_id = getattr(getattr(self.env, "task", None), "task_id", None) or getattr(
            self.env, "task_id", None
        ) or task_id or "task_unknown"
        self.episode_id = getattr(self.env, "episode_id", None) or episode_id or "episode_unknown"
        return obs

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        self.current_time += 1
        obs, reward, done, info = self.env.step(action)
        self.task_id = info["task_id"]
        self.episode_id = info["episode_id"]

        if self._coin("interruption_prob"):
            done = True
            self._schedule(
                AsyncEvent(
                    task_id=self.task_id,
                    episode_id=self.episode_id,
                    event_id=f"{self.episode_id}_interruption_{self.current_time}",
                    event_type="interruption",
                    event_time=self.current_time,
                    reward=-0.4,
                    related_step_id=info.get("step_id"),
                    terminal=True,
                    observation_delta="rollout_interrupted",
                ),
                delay=0,
            )

        for event in info.get("events", []):
            if self._coin("missing_reward_prob") and not event.terminal:
                continue
            delay = self._event_delay(event)
            self._schedule(event, delay=delay)

        if self._coin("timeout_prob") and not done:
            self._schedule(
                AsyncEvent(
                    task_id=self.task_id,
                    episode_id=self.episode_id,
                    event_id=f"{self.episode_id}_timeout_{self.current_time}",
                    event_type="timeout",
                    event_time=self.current_time,
                    reward=float(self.config.get("timeout_penalty", -0.2)),
                    related_step_id=info.get("step_id"),
                    related_tool=action,
                    related_subgoal=info.get("expected_action"),
                    terminal=False,
                    observation_delta="timeout",
                ),
                delay=0,
            )

        info = dict(info)
        info["async_time"] = self.current_time
        return obs, reward, done, info

    def pop_events(self) -> list[AsyncEvent]:
        ready: list[AsyncEvent] = []
        while self.queue and self.queue[0].due_time <= self.current_time:
            ready.append(heapq.heappop(self.queue).event)
        return ready

    def drain_events(self) -> list[AsyncEvent]:
        events: list[AsyncEvent] = []
        while self.queue:
            self.current_time = max(self.current_time, self.queue[0].due_time)
            events.extend(self.pop_events())
        return events

    def _schedule(self, event: AsyncEvent, *, delay: int) -> None:
        self.counter += 1
        due = self.current_time + max(0, delay)
        delayed = AsyncEvent(
            task_id=event.task_id,
            episode_id=event.episode_id,
            event_id=event.event_id,
            event_type=event.event_type,
            event_time=due,
            reward=event.reward,
            related_step_id=event.related_step_id,
            related_tool=event.related_tool,
            related_subgoal=event.related_subgoal,
            observation_delta=event.observation_delta,
            terminal=event.terminal,
            metadata={**event.metadata, "source_time": event.event_time, "delay": delay},
        )
        heapq.heappush(self.queue, ScheduledEvent(due, self.counter, delayed))

    def _event_delay(self, event: AsyncEvent) -> int:
        if event.terminal:
            return int(self.config.get("terminal_reward_delay", 0))
        if not self._coin("delay_prob"):
            return 0
        return self.rng.randint(1, int(self.config.get("max_delay_steps", 1)))

    def _coin(self, key: str) -> bool:
        return self.rng.random() < float(self.config.get(key, 0.0))
