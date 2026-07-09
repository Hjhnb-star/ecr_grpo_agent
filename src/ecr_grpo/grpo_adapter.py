from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ecr_grpo.types import RolloutGroup, StepRecord


StepSource = RolloutGroup | Sequence[StepRecord]


@dataclass
class GRPOSample:
    """A GRPO-consumable sample built after ECR credit construction.

    ECR ends at `reward`: it refills asynchronous events into a scalar reward
    for a step or trajectory. GRPO then owns group-relative normalization and
    the clipped policy update.
    """

    group_id: str
    task_id: str
    episode_id: str
    unit: str
    prompt: str
    completion: str
    reward: float
    advantage: float = 0.0
    old_logprob: float = 0.0
    step_keys: list[tuple[str, str, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_grpo_prompt(observation: str, action_space: list[str] | None = None) -> str:
    actions = "\n".join(f"- {action}" for action in (action_space or []))
    if not actions:
        actions = "- choose the best next action"
    return (
        f"Observation:\n{observation}\n\n"
        f"Available actions:\n{actions}\n\n"
        "Action:"
    )


def _steps_from(source: StepSource) -> list[StepRecord]:
    if isinstance(source, RolloutGroup):
        return list(source.steps)
    return list(source)


def build_step_grpo_samples(source: StepSource) -> list[GRPOSample]:
    """Build step-level GRPO samples from ECR-refilled step returns.

    The GRPO group is the same prompt/observation inside a rollout task group,
    so group-relative normalization compares alternative completions for the
    same decision point rather than mixing unrelated time steps.
    """

    samples: list[GRPOSample] = []
    steps = _steps_from(source)
    for step in sorted(steps, key=lambda s: (s.group_id, s.observation_key, s.episode_id, s.step_id)):
        samples.append(
            GRPOSample(
                group_id=f"{step.group_id}|prompt={step.observation_key}",
                task_id=step.task_id,
                episode_id=step.episode_id,
                unit="step",
                prompt=render_grpo_prompt(step.observation, step.action_space),
                completion=step.action,
                reward=step.return_estimate,
                old_logprob=step.old_logprob,
                step_keys=[step.key],
                metadata={
                    "step_id": step.step_id,
                    "observation_key": step.observation_key,
                    "action": step.action,
                    "action_space_size": len(step.action_space),
                    "immediate_reward": step.immediate_reward,
                    "filled_credit": step.filled_credit,
                    "reward_source": "ecr_refilled_step_return",
                    "optimizer_boundary": "reward_construction_only",
                },
            )
        )
    return samples


def build_trajectory_grpo_samples(source: StepSource) -> list[GRPOSample]:
    """Build trajectory-level GRPO samples from ECR-refilled trajectory returns."""

    steps = _steps_from(source)
    by_group_episode: dict[str, dict[str, list[StepRecord]]] = defaultdict(lambda: defaultdict(list))
    for step in steps:
        by_group_episode[step.group_id][step.episode_id].append(step)

    samples: list[GRPOSample] = []
    for group_id, episodes in sorted(by_group_episode.items()):
        for episode_id, episode_steps in sorted(episodes.items()):
            ordered = sorted(episode_steps, key=lambda s: s.step_id)
            if not ordered:
                continue
            reward = sum(step.return_estimate for step in ordered)
            completion = "\n".join(step.action for step in ordered)
            samples.append(
                GRPOSample(
                    group_id=group_id,
                    task_id=ordered[0].task_id,
                    episode_id=episode_id,
                    unit="trajectory",
                    prompt=render_grpo_prompt(ordered[0].observation, ordered[0].action_space),
                    completion=completion,
                    reward=reward,
                    old_logprob=sum(step.old_logprob for step in ordered),
                    step_keys=[step.key for step in ordered],
                    metadata={
                        "num_steps": len(ordered),
                        "actions": [step.action for step in ordered],
                        "return_estimates": [step.return_estimate for step in ordered],
                        "reward_source": "ecr_refilled_trajectory_return",
                        "optimizer_boundary": "reward_construction_only",
                    },
                )
            )
    return samples


def build_grpo_samples(source: StepSource, *, reward_unit: str) -> list[GRPOSample]:
    unit = normalize_reward_unit(reward_unit)
    if unit == "step":
        return build_step_grpo_samples(source)
    if unit == "trajectory":
        return build_trajectory_grpo_samples(source)
    raise ValueError(f"Unknown GRPO reward unit: {reward_unit}")


def assign_grpo_advantages(
    steps: list[StepRecord],
    *,
    reward_unit: str,
    eps: float = 1e-8,
) -> tuple[list[GRPOSample], dict[str, float]]:
    samples = build_grpo_samples(steps, reward_unit=reward_unit)
    _normalize_group_advantages(samples, eps=eps)
    _write_advantages_to_steps(steps, samples)
    return samples, grpo_batch_stats(samples)


def normalize_reward_unit(reward_unit: str) -> str:
    unit = reward_unit.lower()
    if unit in {"step", "steps", "ecr", "step_return", "step_reward", "step_grpo", "ecr_step_reward"}:
        return "step"
    if unit in {
        "trajectory",
        "episode",
        "grpo",
        "trajectory_return",
        "trajectory_reward",
        "trajectory_grpo",
        "grpo_trajectory_reward",
    }:
        return "trajectory"
    raise ValueError(f"Unknown GRPO reward unit: {reward_unit}")


def grpo_batch_stats(samples: list[GRPOSample]) -> dict[str, float]:
    if not samples:
        return {
            "num_grpo_samples": 0.0,
            "num_grpo_groups": 0.0,
            "avg_grpo_group_size": 0.0,
            "zero_advantage_frac": 0.0,
        }
    group_sizes: dict[str, int] = defaultdict(int)
    for sample in samples:
        group_sizes[sample.group_id] += 1
    zero_advantages = sum(1 for sample in samples if abs(sample.advantage) <= 1e-12)
    return {
        "num_grpo_samples": float(len(samples)),
        "num_grpo_groups": float(len(group_sizes)),
        "avg_grpo_group_size": sum(group_sizes.values()) / max(1, len(group_sizes)),
        "zero_advantage_frac": zero_advantages / max(1, len(samples)),
    }


def _normalize_group_advantages(samples: list[GRPOSample], *, eps: float) -> None:
    by_group: dict[str, list[GRPOSample]] = defaultdict(list)
    for sample in samples:
        by_group[sample.group_id].append(sample)

    for group_samples in by_group.values():
        rewards = [sample.reward for sample in group_samples]
        mean = sum(rewards) / max(1, len(rewards))
        var = sum((reward - mean) ** 2 for reward in rewards) / max(1, len(rewards))
        std = math.sqrt(var)
        for sample, reward in zip(group_samples, rewards):
            sample.advantage = 0.0 if std < eps else (reward - mean) / (std + eps)


def _write_advantages_to_steps(steps: list[StepRecord], samples: list[GRPOSample]) -> None:
    by_key = {step.key: step for step in steps}
    for sample in samples:
        for key in sample.step_keys:
            if key in by_key:
                by_key[key].advantage = sample.advantage
