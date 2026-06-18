from __future__ import annotations

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
        seed: int = 0,
    ) -> None:
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
        env_cls = getattr(environment, env_type)
        self.env = env_cls(self.config, train_eval=split).init_env(batch_size=1)
        self.fallback_action_space = list(fallback_action_space)
        self.latest_admissible = list(fallback_action_space)
        self.task_id = "alfworld_task"
        self.episode_id = "alfworld_episode"
        self.step_count = 0
        self.seed = seed

    @property
    def action_space(self) -> list[str]:
        return self.latest_admissible or self.fallback_action_space

    def reset(self, task_id: str | None = None, episode_id: str | None = None) -> str:
        obs, infos = self.env.reset()
        self.step_count = 0
        self.task_id = task_id or self._read_task_id(infos)
        self.episode_id = episode_id or f"{self.task_id}_episode"
        self.latest_admissible = self._read_admissible(infos)
        return self._format_observation(obs[0], infos)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        self.step_count += 1
        obs, scores, dones, infos = self.env.step([action])
        reward = float(scores[0]) if scores is not None else 0.0
        done = bool(dones[0])
        self.latest_admissible = self._read_admissible(infos)
        success = bool(self._read_info_value(infos, "won", default=False))
        event_type = "terminal_success" if success else "terminal_failure"
        events: list[AsyncEvent] = []
        if reward != 0.0:
            events.append(
                AsyncEvent(
                    task_id=self.task_id,
                    episode_id=self.episode_id,
                    event_id=f"{self.episode_id}_reward_{self.step_count}",
                    event_type="partial_reward",
                    event_time=self.step_count,
                    reward=reward,
                    related_step_id=self.step_count - 1,
                    related_tool=action,
                    observation_delta="alfworld_score",
                    terminal=False,
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
                    observation_delta="alfworld_done",
                    terminal=True,
                )
            )
        info = {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "step_id": self.step_count - 1,
            "events": events,
            "success": success,
            "causal_action": reward > 0.0,
            "expected_action": None,
            "admissible_commands": self.latest_admissible,
        }
        return self._format_observation(obs[0], infos), reward, done, info

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
        if isinstance(value, list):
            value = value[0] if value else "alfworld_task"
        return str(value).replace("\\", "/").split("/")[-1]

    def _read_info_value(self, infos, key: str, default=None):
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
