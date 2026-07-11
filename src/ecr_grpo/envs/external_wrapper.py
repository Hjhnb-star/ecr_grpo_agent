from __future__ import annotations

import importlib
import inspect
from typing import Any

from ecr_grpo.types import AsyncEvent


class ExternalTextBenchmarkEnv:
    """Adapter for manifest-backed text benchmarks such as ScienceWorld or WebShop."""

    def __init__(
        self,
        *,
        factory_path: str,
        split: str,
        task_metadata: dict[str, Any],
        fallback_action_space: list[str],
        factory_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.split = split
        self.task_metadata = dict(task_metadata)
        self.fallback_action_space = list(fallback_action_space)
        self.seed = seed
        factory = _load_object(factory_path)
        kwargs = {
            **dict(factory_kwargs or {}),
            "split": split,
            "task": self.task_metadata,
            "task_id": self.task_metadata.get("task_id"),
            "seed": seed,
        }
        self.env = _call_supported(factory, **kwargs)
        self.latest_actions = list(fallback_action_space)
        self.task_id = str(self.task_metadata.get("task_id", "external_task"))
        self.actual_task_id = str(self.task_metadata.get("actual_task_id", self.task_id))
        self.task_type = str(self.task_metadata.get("task_type", "unknown"))
        self.episode_id = "external_episode"
        self.step_count = 0

    @property
    def action_space(self) -> list[str]:
        return self.latest_actions or self.fallback_action_space

    def reset(self, task_id: str | None = None, episode_id: str | None = None) -> str:
        result = _call_supported(
            self.env.reset,
            task_id=self.actual_task_id,
            task=self.task_metadata,
            split=self.split,
            seed=self.seed,
        )
        observation, info = _unpack_reset(result)
        self.task_id = task_id or self.task_id
        self.episode_id = episode_id or f"{self.task_id}_episode"
        self.step_count = 0
        self._update_actions(info)
        return str(observation)

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        self.step_count += 1
        observation, reward, done, info = _unpack_step(self.env.step(action))
        info = dict(info or {})
        self._update_actions(info)
        success = bool(info.get("success", info.get("won", done and reward > 0.0)))
        events = list(info.get("events", []))
        if not events and (reward != 0.0 or done):
            events.append(
                AsyncEvent(
                    task_id=self.task_id,
                    episode_id=self.episode_id,
                    event_id=f"{self.episode_id}_event_{self.step_count}",
                    event_type=(
                        "terminal_success"
                        if done and success
                        else "terminal_failure"
                        if done
                        else "partial_reward"
                    ),
                    event_time=self.step_count,
                    reward=float(reward),
                    related_step_id=self.step_count - 1,
                    related_tool=action,
                    observation_delta=_compact(observation),
                    terminal=done,
                    metadata={"tags": ["external_benchmark", self.task_type]},
                    diagnostic_metadata={
                        "source_action": action,
                        "source_step_id": self.step_count - 1,
                    },
                )
            )
        normalized_info = {
            **info,
            "task_id": self.task_id,
            "actual_task_id": self.actual_task_id,
            "task_type": self.task_type,
            "episode_id": self.episode_id,
            "step_id": self.step_count - 1,
            "events": events,
            "success": success,
            "tool_name": action,
            "public_tags": [self.task_type],
            "env_reward": float(reward),
            "shaping_reward": 0.0,
            "admissible_commands": self.latest_actions,
        }
        return str(observation), float(reward), done, normalized_info

    def _update_actions(self, info: dict[str, Any]) -> None:
        for key in ("admissible_commands", "valid_actions", "actions"):
            value = info.get(key)
            if isinstance(value, (list, tuple)) and value:
                self.latest_actions = [str(item) for item in value]
                return


def _load_object(path: str):
    if ":" not in path:
        raise ValueError("environment.factory must use 'module:function' syntax")
    module_name, object_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _call_supported(callable_obj, **kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(**kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable_obj(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return callable_obj(**supported)


def _unpack_reset(result) -> tuple[Any, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0], dict(result[1])
    return result, {}


def _unpack_step(result) -> tuple[Any, float, bool, dict[str, Any]]:
    if not isinstance(result, tuple):
        raise TypeError("External benchmark step() must return a tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated or truncated), dict(info or {})
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, float(reward), bool(done), dict(info or {})
    raise TypeError("External benchmark step() must return 4 or 5 values")


def _compact(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value).strip().split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

