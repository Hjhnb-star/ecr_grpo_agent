from __future__ import annotations

from typing import Protocol


class AgentEnv(Protocol):
    action_space: list[str]

    def reset(self, task_id: str | None = None) -> str:
        ...

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        ...

