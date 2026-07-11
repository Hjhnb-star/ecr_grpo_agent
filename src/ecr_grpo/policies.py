from __future__ import annotations

import math
import random
import hashlib
import pickle
from collections import defaultdict
from pathlib import Path

from ecr_grpo.types import PolicyAction, StepRecord


def format_agent_prompt(observation: str, action_space: list[str] | None = None) -> str:
    actions = "\n".join(f"- {a}" for a in (action_space or []))
    if not actions:
        actions = "- choose the best next action"
    return (
        "You are an agent solving a long-horizon interactive task.\n\n"
        f"Observation:\n{observation}\n\n"
        f"Available actions:\n{actions}\n\n"
        "Return exactly one action from the available actions.\n"
        "Action:"
    )


def observation_key(observation: str) -> str:
    # The synthetic environment exposes the current decision state in a compact text.
    # For real environments this should be replaced with a state encoder or prompt hash.
    parts = [p.strip() for p in observation.split(";")]
    task = next((p for p in parts if p.startswith("task=")), "task=unknown")
    progress = next((p for p in parts if p.startswith("progress=")), "progress=unknown")
    hint = next((p for p in parts if p.startswith("hint=")), "hint=unknown")
    if task != "task=unknown" or progress != "progress=unknown" or hint != "hint=unknown":
        return f"{task}|{progress}|{hint}"
    normalized = " ".join(observation.lower().split())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"text_obs:{digest}"


class TabularSoftmaxPolicy:
    def __init__(
        self,
        action_space: list[str],
        *,
        seed: int = 0,
        temperature: float = 1.0,
        init_scale: float = 0.01,
        entropy_bonus: float = 0.0,
    ) -> None:
        self.action_space = list(action_space)
        self.rng = random.Random(seed)
        self.temperature = temperature
        self.init_scale = init_scale
        self.entropy_bonus = entropy_bonus
        self.logits: dict[str, dict[str, float]] = defaultdict(self._new_logits)

    def act(
        self,
        observation: str,
        action_space: list[str] | None = None,
        *,
        greedy: bool = False,
    ) -> PolicyAction:
        key = observation_key(observation)
        probs = self.probs(key)
        if greedy:
            action = max(probs, key=probs.get)
        else:
            action = self._sample(probs)
        return PolicyAction(text=action, old_logprob=math.log(max(probs[action], 1e-12)))

    def probs(self, key: str) -> dict[str, float]:
        values = self.logits[key]
        max_logit = max(values.values())
        scaled = {
            a: math.exp((v - max_logit) / max(self.temperature, 1e-6))
            for a, v in values.items()
        }
        total = sum(scaled.values())
        return {a: v / total for a, v in scaled.items()}

    def update(self, steps: list[StepRecord], lr: float) -> dict[str, float]:
        total_loss = 0.0
        total_entropy = 0.0
        for step in steps:
            probs = self.probs(step.observation_key)
            action_prob = max(probs.get(step.action, 1e-12), 1e-12)
            total_loss += -math.log(action_prob) * step.advantage
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs.values())
            total_entropy += entropy
            for action in self.action_space:
                grad = (1.0 if action == step.action else 0.0) - probs[action]
                entropy_grad = -probs[action] * (math.log(max(probs[action], 1e-12)) + entropy)
                self.logits[step.observation_key][action] += lr * (
                    step.advantage * grad + self.entropy_bonus * entropy_grad
                )
        denom = max(1, len(steps))
        return {"policy_loss": total_loss / denom, "entropy": total_entropy / denom}

    def update_grpo(self, steps: list[StepRecord], lr: float) -> dict[str, float]:
        return self.update(steps, lr=lr)

    def save(self, path: str) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        state = {
            "action_space": list(self.action_space),
            "temperature": self.temperature,
            "init_scale": self.init_scale,
            "entropy_bonus": self.entropy_bonus,
            "logits": {key: dict(value) for key, value in self.logits.items()},
            "rng_state": self.rng.getstate(),
        }
        with (output / "tabular_policy.pkl").open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        with (Path(path) / "tabular_policy.pkl").open("rb") as handle:
            state = pickle.load(handle)
        if list(state["action_space"]) != self.action_space:
            raise ValueError("Checkpoint action space does not match the current policy")
        self.temperature = float(state["temperature"])
        self.init_scale = float(state["init_scale"])
        self.entropy_bonus = float(state["entropy_bonus"])
        self.logits = defaultdict(
            self._new_logits,
            {key: dict(value) for key, value in state["logits"].items()},
        )
        self.rng.setstate(state["rng_state"])

    def save_training_state(self, path: str) -> None:
        # The tabular policy has no optimizer; weights and RNG are stored by save().
        Path(path).write_bytes(b"")

    def load_training_state(self, path: str, lr: float) -> None:
        del path, lr

    def _new_logits(self) -> dict[str, float]:
        return {
            action: self.rng.uniform(-self.init_scale, self.init_scale)
            for action in self.action_space
        }

    def _sample(self, probs: dict[str, float]) -> str:
        r = self.rng.random()
        acc = 0.0
        for action, prob in probs.items():
            acc += prob
            if r <= acc:
                return action
        return self.action_space[-1]


def build_policy(config: dict, action_space: list[str], seed: int = 0):
    policy_cfg = config.get("policy", {})
    kind = str(policy_cfg.get("kind", "tabular")).lower()
    if kind == "tabular":
        return TabularSoftmaxPolicy(
            action_space,
            seed=seed,
            temperature=float(policy_cfg.get("temperature", 1.0)),
            init_scale=float(policy_cfg.get("init_scale", 0.01)),
            entropy_bonus=float(config.get("training", {}).get("entropy_bonus", 0.0)),
        )
    if kind in {"hf", "hf_lora", "lora"}:
        from ecr_grpo.hf_policy import HFLoraPolicy

        return HFLoraPolicy(
            action_space=action_space,
            model_id=str(policy_cfg.get("model_id", "REPLACE_WITH_MODEL_ID")),
            adapter_path=policy_cfg.get("adapter_path"),
            use_lora=bool(policy_cfg.get("use_lora", kind != "hf")),
            lora_r=int(policy_cfg.get("lora_r", 8)),
            lora_alpha=int(policy_cfg.get("lora_alpha", 16)),
            lora_dropout=float(policy_cfg.get("lora_dropout", 0.05)),
            device=policy_cfg.get("device"),
            max_new_tokens=int(policy_cfg.get("max_new_tokens", 8)),
            temperature=float(policy_cfg.get("temperature", 0.7)),
            top_p=float(policy_cfg.get("top_p", 1.0)),
            action_selection=str(policy_cfg.get("action_selection", "score")),
            action_score_batch_size=int(policy_cfg.get("action_score_batch_size", 8)),
            action_score_normalization=str(policy_cfg.get("action_score_normalization", "mean")),
            action_score_calibration=str(policy_cfg.get("action_score_calibration", "pmi")),
            use_chat_template=bool(policy_cfg.get("use_chat_template", True)),
            update_score_mode=str(policy_cfg.get("update_score_mode", "full_distribution")),
            max_prompt_tokens=int(policy_cfg.get("max_prompt_tokens", 1024)),
            gradient_checkpointing=bool(policy_cfg.get("gradient_checkpointing", False)),
            entropy_bonus=float(config.get("training", {}).get("entropy_bonus", 0.0)),
            clip_eps=float(config.get("training", {}).get("clip_eps", 0.2)),
            grad_accum_steps=int(config.get("training", {}).get("grad_accum_steps", 1)),
            max_grad_norm=float(config.get("training", {}).get("max_grad_norm", 1.0)),
            seed=seed,
        )
    raise ValueError(f"Unknown policy kind: {kind}")
