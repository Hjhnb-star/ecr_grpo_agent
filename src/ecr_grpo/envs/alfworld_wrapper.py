from __future__ import annotations

from typing import Any

from ecr_grpo.types import AsyncEvent


class ALFWorldEnv:
    """Thin ALFWorld adapter following the same text-action interface as synthetic env.

    Expected config:

    ```json
    {
      "environment": {
        "name": "alfworld",
        "alfworld_config": "REPLACE_WITH_ALFWORLD_CONFIG.yaml",
        "split": "eval_out_of_distribution",
        "max_steps": 30,
        "action_space": ["look", "inventory", "... optional fallback ..."]
      }
    }
    ```

    The adapter follows the standard ALFWorld pattern:

    ```python
    import alfworld.agents.environment
    env_cls = getattr(alfworld.agents.environment, config["env"]["type"])
    env = env_cls(config, train_eval=split).init_env(batch_size=1)
    ```
    """

    def __init__(
        self,
        *,
        alfworld_config: str,
        split: str,
        fallback_action_space: list[str],
        shaping_config: dict | None = None,
        seed: int = 0,
        raw_env: Any | None = None,
        loaded_config: dict[str, Any] | None = None,
    ) -> None:
        if raw_env is None:
            try:
                import yaml
                import alfworld.agents.environment as environment
            except ImportError as exc:
                raise RuntimeError(
                    "ALFWorldEnv requires optional dependencies. Install ALFWorld and PyYAML, "
                    "or use the synthetic environment."
                ) from exc

            with open(alfworld_config, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            env_type = self.config["env"]["type"]
            try:
                if hasattr(environment, "get_environment"):
                    env_cls = environment.get_environment(env_type)
                else:
                    env_cls = getattr(environment, env_type)
            except (AttributeError, KeyError) as exc:
                available = [name for name in dir(environment) if name.endswith("Env")]
                raise RuntimeError(
                    f"ALFWorld environment type '{env_type}' is not available. "
                    f"Check env.type in {alfworld_config}. Available direct Env names: {available}. "
                    "If your ALFWorld version provides get_environment(), this wrapper will use it."
                ) from exc
            self.env = env_cls(self.config, train_eval=split).init_env(batch_size=1)
        else:
            self.config = loaded_config or {"env": {"type": "mock"}}
            self.env = raw_env
        self.fallback_action_space = list(fallback_action_space)
        self.latest_admissible = list(fallback_action_space)
        self.task_id = "alfworld_task"
        self.actual_task_id = "alfworld_task"
        self.episode_id = "alfworld_episode"
        self.step_count = 0
        self.seed = seed
        self.shaping_config = shaping_config or {}
        self.previous_observation = ""
        self.previous_action = ""
        self.seen_observations: set[str] = set()

    @property
    def action_space(self) -> list[str]:
        return self.latest_admissible or self.fallback_action_space

    def reset(self, task_id: str | None = None, episode_id: str | None = None) -> str:
        obs, infos = self.env.reset()
        obs_text = self._first(obs, "")
        self.step_count = 0
        self.actual_task_id = self._read_task_id(infos)
        self.task_id = task_id or self.actual_task_id
        self.episode_id = episode_id or f"{self.task_id}_episode"
        self.latest_admissible = self._read_admissible(infos)
        self.previous_observation = self._normalize_obs(obs_text)
        self.previous_action = ""
        self.seen_observations = {self.previous_observation}
        return self._format_observation(obs_text, infos)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        self.step_count += 1
        prev_admissible = list(self.latest_admissible)
        obs, scores, dones, infos = self.env.step([action])
        obs_text = self._first(obs, "")
        env_reward = self._as_float(scores, default=0.0)
        done = self._as_bool(dones, default=False)
        self.latest_admissible = self._read_admissible(infos)
        success = self._as_bool(self._read_info_value(infos, "won", default=False), default=False)
        event_type = "terminal_success" if success else "terminal_failure"
        events: list[AsyncEvent] = []
        next_obs_norm = self._normalize_obs(obs_text)
        shaping_reward, shaping_reason = self._shaping_reward(
            action=action,
            prev_admissible=prev_admissible,
            next_obs_norm=next_obs_norm,
        )
        total_reward = env_reward + shaping_reward
        tags = self._event_tags(action, shaping_reason)
        if env_reward != 0.0:
            events.append(
                AsyncEvent(
                    task_id=self.task_id,
                    episode_id=self.episode_id,
                    event_id=f"{self.episode_id}_reward_{self.step_count}",
                    event_type="partial_reward",
                    event_time=self.step_count,
                    reward=env_reward,
                    related_step_id=self.step_count - 1,
                    related_tool=action,
                    related_subgoal=action,
                    observation_delta="alfworld_score",
                    terminal=False,
                    metadata={
                        "action": action,
                        "tags": ["alfworld_score", "env_reward", *tags],
                    },
                )
            )
        if shaping_reward != 0.0:
            events.append(
                AsyncEvent(
                    task_id=self.task_id,
                    episode_id=self.episode_id,
                    event_id=f"{self.episode_id}_shape_{self.step_count}",
                    event_type="partial_reward",
                    event_time=self.step_count,
                    reward=shaping_reward,
                    related_step_id=self.step_count - 1,
                    related_tool=action,
                    related_subgoal=action,
                    observation_delta=shaping_reason,
                    terminal=False,
                    metadata={
                        "action": action,
                        "tags": ["alfworld_shaping", *tags],
                    },
                )
            )
        if done:
            terminal_reward = 1.0 if success else -0.5
            events.append(
                AsyncEvent(
                    task_id=self.task_id,
                    episode_id=self.episode_id,
                    event_id=f"{self.episode_id}_terminal_{self.step_count}",
                    event_type=event_type,
                    event_time=self.step_count,
                    reward=terminal_reward,
                    related_step_id=self.step_count - 1,
                    related_tool=action,
                    related_subgoal=action,
                    observation_delta="alfworld_done",
                    terminal=True,
                    metadata={
                        "action": action,
                        "tags": [
                            "alfworld_done",
                            "success" if success else "failure",
                            *tags,
                        ],
                    },
                )
            )
        self.previous_observation = next_obs_norm
        self.previous_action = action
        self.seen_observations.add(next_obs_norm)
        actual_task_id = self._read_task_id(infos)
        self.actual_task_id = actual_task_id
        info = {
            "task_id": self.task_id,
            "actual_task_id": actual_task_id,
            "episode_id": self.episode_id,
            "step_id": self.step_count - 1,
            "events": events,
            "success": success,
            "causal_action": total_reward > 0.0,
            "expected_action": action,
            "tool_name": action,
            "subgoal_id": action,
            "public_tags": tags,
            "admissible_commands": self.latest_admissible,
            "env_reward": env_reward,
            "shaping_reward": shaping_reward,
            "shaping_reason": shaping_reason,
        }
        return self._format_observation(obs_text, infos), total_reward, done, info

    def _read_admissible(self, infos) -> list[str]:
        commands = self._read_info_value(infos, "admissible_commands", default=None)
        if commands and isinstance(commands, list):
            first = commands[0]
            if isinstance(first, list):
                return [str(x) for x in first]
            return [str(x) for x in commands]
        return list(self.fallback_action_space)

    def _read_task_id(self, infos) -> str:
        value = self._read_info_value(infos, "extra.gamefile", default=None)
        if value is None:
            value = self._read_info_value(infos, "gamefile", default="alfworld_task")
        value = self._first(value, "alfworld_task")
        return str(value).replace("\\", "/").split("/")[-1]

    def _read_info_value(self, infos, key: str, default=None):
        if isinstance(infos, dict) and key in infos:
            return infos[key]
        cur = infos
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def _format_observation(self, obs: str, infos) -> str:
        admissible = "\n".join(f"- {a}" for a in self.latest_admissible[:40])
        return f"{obs}\n\nAdmissible actions:\n{admissible}"

    def _normalize_obs(self, obs: str) -> str:
        return " ".join(str(obs).strip().lower().split())

    def _first(self, value, default=None):
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value

    def _as_bool(self, value, *, default: bool = False) -> bool:
        value = self._first(value, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "won", "success"}
        return bool(value)

    def _as_float(self, value, *, default: float = 0.0) -> float:
        value = self._first(value, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _event_tags(self, action: str, shaping_reason: str) -> list[str]:
        tags = [action]
        for part in str(shaping_reason or "").replace("+", " ").split():
            if part and part != "disabled":
                tags.append(part)
        return tags

    def _shaping_reward(
        self,
        *,
        action: str,
        prev_admissible: list[str],
        next_obs_norm: str,
    ) -> tuple[float, str]:
        if not self.shaping_config.get("enabled", True):
            return 0.0, "disabled"

        reward = 0.0
        reasons: list[str] = []
        if action in prev_admissible:
            reward += float(self.shaping_config.get("valid_action_reward", 0.02))
            reasons.append("valid_action")
        else:
            reward += float(self.shaping_config.get("invalid_action_penalty", -0.05))
            reasons.append("invalid_action")

        if next_obs_norm != self.previous_observation:
            reward += float(self.shaping_config.get("observation_change_reward", 0.03))
            reasons.append("observation_change")
        else:
            reward += float(self.shaping_config.get("stagnation_penalty", -0.02))
            reasons.append("stagnation")

        if next_obs_norm not in self.seen_observations:
            reward += float(self.shaping_config.get("new_state_reward", 0.02))
            reasons.append("new_state")

        if action == self.previous_action:
            reward += float(self.shaping_config.get("repeat_action_penalty", -0.03))
            reasons.append("repeat_action")

        return reward, "+".join(reasons)
