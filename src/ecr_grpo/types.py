from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


StepStatus = Literal["pending", "credited", "expired", "terminal"]
EventType = Literal[
    "partial_reward",
    "tool_return",
    "timeout",
    "terminal_success",
    "terminal_failure",
    "interruption",
]


@dataclass
class StepRecord:
    task_id: str
    episode_id: str
    group_id: str
    step_id: int
    env_time: int
    observation: str
    observation_key: str
    action: str
    old_logprob: float
    action_space: list[str]
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    tool_name: str | None = None
    subgoal_id: str | None = None
    status: StepStatus = "pending"
    immediate_reward: float = 0.0
    filled_credit: float = 0.0
    advantage: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.task_id, self.episode_id, self.step_id)

    @property
    def return_estimate(self) -> float:
        return self.immediate_reward + self.filled_credit


@dataclass
class AsyncEvent:
    task_id: str
    episode_id: str
    event_id: str
    event_type: EventType
    event_time: int
    reward: float
    related_step_id: int | None = None
    related_tool: str | None = None
    related_subgoal: str | None = None
    observation_delta: str | None = None
    terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CreditAssignment:
    step_key: tuple[str, str, int]
    event_id: str
    raw_reward: float
    kernel_weight: float
    assigned_credit: float
    reason: str


@dataclass
class RolloutGroup:
    group_id: str
    task_id: str
    episodes: list[str]
    steps: list[StepRecord]
    events: list[AsyncEvent]
    assignments: list[CreditAssignment]


@dataclass
class PolicyAction:
    text: str
    old_logprob: float
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)


@dataclass
class BenchmarkTask:
    task_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
