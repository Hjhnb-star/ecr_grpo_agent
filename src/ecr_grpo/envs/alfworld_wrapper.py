from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from ecr_grpo.types import AsyncEvent


ALFWORLD_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
)

ALFWORLD_SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "the",
    "then",
    "there",
    "to",
    "up",
    "with",
    "you",
    "your",
}

ALFWORLD_VERB_ALIASES = {
    "arrive": "go",
    "close": "close",
    "closed": "close",
    "cool": "cool",
    "cooled": "cool",
    "examine": "examine",
    "go": "go",
    "heat": "heat",
    "heated": "heat",
    "look": "examine",
    "open": "open",
    "opened": "open",
    "pick": "take",
    "picked": "take",
    "place": "put",
    "placed": "put",
    "put": "put",
    "slice": "slice",
    "sliced": "slice",
    "take": "take",
    "taken": "take",
    "toggle": "toggle",
    "turn": "toggle",
    "wash": "clean",
    "washed": "clean",
    "clean": "clean",
    "cleaned": "clean",
}


class ALFWorldGameCatalog:
    """Loads one ALFWorld split once and creates deterministic single-game envs."""

    def __init__(self, *, alfworld_config: str, split: str) -> None:
        try:
            import yaml
            import alfworld.agents.environment as environment
        except ImportError as exc:
            raise RuntimeError(
                "ALFWorld requires optional dependencies. Install the alfworld extra."
            ) from exc

        with open(alfworld_config, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.split = split
        env_type = self.config["env"]["type"]
        try:
            if hasattr(environment, "get_environment"):
                env_cls = environment.get_environment(env_type)
            else:
                env_cls = getattr(environment, env_type)
        except (AttributeError, KeyError) as exc:
            available = [name for name in dir(environment) if name.endswith("Env")]
            raise RuntimeError(
                f"ALFWorld environment type '{env_type}' is unavailable. "
                f"Available direct Env names: {available}."
            ) from exc
        self.manager = env_cls(self.config, train_eval=split)
        self.game_files = sorted(str(path) for path in getattr(self.manager, "game_files", []))
        if not self.game_files:
            raise RuntimeError(f"ALFWorld split '{split}' did not expose any game_files")

    def make_raw_env(self, game_file: str | None = None):
        manager = copy.copy(self.manager)
        if game_file is not None:
            manager.game_files = [game_file]
            if hasattr(manager, "num_games"):
                manager.num_games = 1
        return manager.init_env(batch_size=1)

    def task_metadata(self, game_file: str) -> dict[str, str]:
        normalized = str(game_file).replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        split_index = max((idx for idx, part in enumerate(parts) if part == self.split), default=-1)
        relative = "/".join(parts[split_index + 1 :]) if split_index >= 0 else Path(normalized).name
        if relative.endswith("/game.tw-pddl"):
            relative = relative[: -len("/game.tw-pddl")]
        task_type = next((kind for kind in ALFWORLD_TASK_TYPES if kind in relative), "unknown")
        safe_id = re.sub(r"[^a-zA-Z0-9_.=-]+", "__", relative).strip("_")
        return {
            "actual_task_id": relative,
            "task_id": f"{self.split}__{safe_id}",
            "task_type": task_type,
            "split": self.split,
            "game_file": game_file,
        }


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
        catalog: ALFWorldGameCatalog | None = None,
        game_file: str | None = None,
        task_metadata: dict[str, Any] | None = None,
        history_turns: int = 4,
        raw_env: Any | None = None,
        loaded_config: dict[str, Any] | None = None,
    ) -> None:
        if raw_env is None:
            catalog = catalog or ALFWorldGameCatalog(
                alfworld_config=alfworld_config,
                split=split,
            )
            self.config = catalog.config
            self.env = catalog.make_raw_env(game_file)
        else:
            self.config = loaded_config or {"env": {"type": "mock"}}
            self.env = raw_env
        self.split = split
        self.game_file = game_file
        self.task_metadata = dict(task_metadata or {})
        self.history_turns = max(0, int(history_turns))
        self.fallback_action_space = list(fallback_action_space)
        self.latest_admissible = list(fallback_action_space)
        self.task_id = "alfworld_task"
        self.actual_task_id = "alfworld_task"
        self.task_type = str(self.task_metadata.get("task_type", "unknown"))
        self.episode_id = "alfworld_episode"
        self.step_count = 0
        self.seed = seed
        self.shaping_config = shaping_config or {}
        self.previous_score = 0.0
        self.previous_observation = ""
        self.previous_action = ""
        self.seen_observations: set[str] = set()
        self.task_goal = ""
        self.history: list[tuple[str, str]] = []

    @property
    def action_space(self) -> list[str]:
        return self.latest_admissible or self.fallback_action_space

    def reset(self, task_id: str | None = None, episode_id: str | None = None) -> str:
        obs, infos = self.env.reset()
        obs_text = self._first(obs, "")
        self.step_count = 0
        self.previous_score = 0.0
        self.actual_task_id = str(
            self.task_metadata.get("actual_task_id")
            or self._read_task_id(infos)
        )
        self.task_type = str(self.task_metadata.get("task_type", self.task_type))
        self.task_id = task_id or self.actual_task_id
        self.episode_id = episode_id or f"{self.task_id}_episode"
        self.latest_admissible = self._read_admissible(infos)
        self.previous_observation = self._normalize_obs(obs_text)
        self.previous_action = ""
        self.seen_observations = {self.previous_observation}
        self.task_goal = self._extract_task_goal(obs_text)
        self.history = []
        return self._format_observation(obs_text, infos)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        self.step_count += 1
        prev_admissible = list(self.latest_admissible)
        obs, scores, dones, infos = self.env.step([action])
        obs_text = self._first(obs, "")
        raw_score = self._as_float(scores, default=self.previous_score)
        env_reward = raw_score - self.previous_score
        self.previous_score = raw_score
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
        terminal_reward = env_reward
        if done and abs(terminal_reward) <= 1e-12:
            reward_key = "terminal_success_reward" if success else "terminal_failure_reward"
            default_reward = 1.0 if success else -0.5
            terminal_reward = float(self.shaping_config.get(reward_key, default_reward))
        task_reward = terminal_reward if done else env_reward
        total_reward = task_reward + shaping_reward
        tags = self._event_tags(shaping_reason)
        event_observation = self._observation_delta(obs_text)
        effect_tags = self._semantic_tags(event_observation)
        action_tags = self._semantic_tags(action)
        if not done and env_reward != 0.0:
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
                    observation_delta=event_observation,
                    terminal=False,
                    metadata={
                        "tags": ["alfworld_score", "env_reward", *tags, *effect_tags]
                    },
                    diagnostic_metadata={
                        "source_action": action,
                        "source_step_id": self.step_count - 1,
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
                    observation_delta=f"{shaping_reason}: {event_observation}",
                    terminal=False,
                    metadata={
                        "tags": ["alfworld_shaping", *tags, *effect_tags]
                    },
                    diagnostic_metadata={
                        "source_action": action,
                        "source_step_id": self.step_count - 1,
                    },
                )
            )
        if done:
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
                    observation_delta=event_observation,
                    terminal=True,
                    metadata={
                        "tags": [
                            "alfworld_done",
                            "success" if success else "failure",
                            *tags,
                            *effect_tags,
                        ],
                    },
                    diagnostic_metadata={
                        "source_action": action,
                        "source_step_id": self.step_count - 1,
                    },
                )
            )
        self.previous_observation = next_obs_norm
        self.previous_action = action
        self.seen_observations.add(next_obs_norm)
        self.history.append((action, self._compact_text(obs_text)))
        self.history = self.history[-self.history_turns :] if self.history_turns else []
        actual_task_id = str(self.task_metadata.get("actual_task_id") or self._read_task_id(infos))
        self.actual_task_id = actual_task_id
        info = {
            "task_id": self.task_id,
            "actual_task_id": actual_task_id,
            "task_type": self.task_type,
            "episode_id": self.episode_id,
            "step_id": self.step_count - 1,
            "events": events,
            "success": success,
            "positive_transition": task_reward > 0.0 or shaping_reward > 0.0,
            "tool_name": action,
            "subgoal_id": action,
            "public_tags": [*tags, *action_tags],
            "admissible_commands": self.latest_admissible,
            "env_reward": env_reward,
            "raw_score": raw_score,
            "task_reward": task_reward,
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
        sections = []
        if self.task_goal:
            sections.append(f"Task:\n{self.task_goal}")
        sections.append(f"Current observation:\n{obs}")
        if self.history:
            history = "\n".join(
                f"{idx + 1}. {action} -> {result}"
                for idx, (action, result) in enumerate(self.history)
            )
            sections.append(f"Recent interaction history:\n{history}")
        return "\n\n".join(sections)

    def _normalize_obs(self, obs: str) -> str:
        return " ".join(str(obs).strip().lower().split())

    def _extract_task_goal(self, obs: str) -> str:
        match = re.search(r"your task is to:\s*(.+?)(?:\n|$)", str(obs), flags=re.IGNORECASE)
        if match:
            return self._compact_text(match.group(1), limit=320)
        return self._compact_text(obs, limit=320)

    def _compact_text(self, value: str, *, limit: int = 180) -> str:
        compact = " ".join(str(value).strip().split())
        if len(compact) <= limit:
            return compact
        return compact[: max(1, limit - 3)].rstrip() + "..."

    def _observation_delta(self, obs: str) -> str:
        current = self._normalize_obs(obs)
        previous_tokens = set(self.previous_observation.split())
        additions = [token for token in current.split() if token not in previous_tokens]
        if additions:
            return self._compact_text(" ".join(additions), limit=240)
        return self._compact_text(obs, limit=240)

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

    def _event_tags(self, shaping_reason: str) -> list[str]:
        tags: list[str] = []
        for part in str(shaping_reason or "").replace("+", " ").split():
            if part and part != "disabled":
                tags.append(part)
        return tags

    def _semantic_tags(self, text: str) -> list[str]:
        """Extract public ALFWorld action/effect concepts without source-step links."""

        tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", str(text))]
        tags: list[str] = []
        canonical_verbs = {
            ALFWORLD_VERB_ALIASES[token]
            for token in tokens
            if token in ALFWORLD_VERB_ALIASES
        }
        tags.extend(f"verb_{verb}" for verb in sorted(canonical_verbs))

        ignored = {
            *ALFWORLD_SEMANTIC_STOPWORDS,
            *ALFWORLD_VERB_ALIASES,
            *ALFWORLD_VERB_ALIASES.values(),
        }
        for token in tokens:
            if token in ignored or token.isdigit() or len(token) < 3:
                continue
            tag = f"entity_{token}"
            if tag not in tags:
                tags.append(tag)
            if len(tags) >= 10:
                break
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
