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
    step_advantage_estimator: str = "prompt_group",
    credit_weight_floor: float = 0.05,
    max_credit_multiplier: float = 4.0,
    residual_beta: float = 0.5,
    residual_clip: float = 2.0,
    residual_use_confidence: bool = True,
    eps: float = 1e-8,
) -> tuple[list[GRPOSample], dict[str, float]]:
    samples = build_grpo_samples(steps, reward_unit=reward_unit)
    unit = normalize_reward_unit(reward_unit)
    estimator = normalize_step_advantage_estimator(step_advantage_estimator)
    extra_stats: dict[str, float] = {}
    if unit == "step" and estimator == "trajectory_grouped_credit":
        extra_stats = _assign_trajectory_grouped_credit_advantages(
            steps,
            samples,
            credit_weight_floor=max(0.0, credit_weight_floor),
            max_credit_multiplier=max(1.0, max_credit_multiplier),
            eps=eps,
        )
    elif unit == "step" and estimator == "trajectory_grouped_residual":
        extra_stats = _assign_trajectory_grouped_residual_advantages(
            steps,
            samples,
            residual_beta=max(0.0, residual_beta),
            residual_clip=max(0.0, residual_clip),
            residual_use_confidence=bool(residual_use_confidence),
            eps=eps,
        )
    else:
        _normalize_group_advantages(samples, eps=eps)
    _write_advantages_to_steps(steps, samples)
    stats = grpo_batch_stats(samples)
    stats.update(extra_stats)
    stats["step_advantage_estimator"] = estimator
    return samples, stats


def normalize_step_advantage_estimator(value: str) -> str:
    estimator = str(value).lower()
    if estimator in {"prompt", "prompt_group", "exact_prompt", "observation_group"}:
        return "prompt_group"
    if estimator in {
        "trajectory_grouped_credit",
        "trajectory_credit",
        "grouped_credit",
        "ecr_grouped",
    }:
        return "trajectory_grouped_credit"
    if estimator in {
        "trajectory_grouped_residual",
        "trajectory_residual",
        "ecr_residual",
        "step_residual",
    }:
        return "trajectory_grouped_residual"
    raise ValueError(f"Unknown step advantage estimator: {value}")


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


def _assign_trajectory_grouped_credit_advantages(
    steps: list[StepRecord],
    samples: list[GRPOSample],
    *,
    credit_weight_floor: float,
    max_credit_multiplier: float,
    eps: float,
) -> dict[str, float]:
    """Normalize trajectory returns, then localize the advantage with ECR credit."""

    by_group_episode: dict[str, dict[str, list[StepRecord]]] = defaultdict(lambda: defaultdict(list))
    for step in steps:
        by_group_episode[step.group_id][step.episode_id].append(step)

    advantage_by_key: dict[tuple[str, str, int], float] = {}
    multiplier_by_key: dict[tuple[str, str, int], float] = {}
    group_sizes: list[int] = []
    singleton_groups = 0
    zero_variance_groups = 0

    for episodes in by_group_episode.values():
        ordered_episodes = {
            episode_id: sorted(episode_steps, key=lambda item: item.step_id)
            for episode_id, episode_steps in episodes.items()
        }
        episode_ids = sorted(ordered_episodes)
        returns = [
            sum(item.return_estimate for item in ordered_episodes[episode_id])
            for episode_id in episode_ids
        ]
        group_sizes.append(len(episode_ids))
        if len(episode_ids) <= 1:
            singleton_groups += 1
        mean = sum(returns) / max(1, len(returns))
        var = sum((value - mean) ** 2 for value in returns) / max(1, len(returns))
        std = math.sqrt(var)
        if std < eps:
            zero_variance_groups += 1

        for episode_id, episode_return in zip(episode_ids, returns):
            episode_steps = ordered_episodes[episode_id]
            trajectory_advantage = 0.0 if std < eps else (episode_return - mean) / (std + eps)
            masses = [abs(item.return_estimate) for item in episode_steps]
            positive_masses = [mass for mass in masses if mass > eps]
            reference_mass = (
                sum(positive_masses) / len(positive_masses)
                if positive_masses
                else 1.0
            )
            weights = [mass + credit_weight_floor * reference_mass for mass in masses]
            total = sum(weights)
            if total <= eps:
                multipliers = [1.0 for _ in episode_steps]
            else:
                multipliers = [len(episode_steps) * weight / total for weight in weights]
                multipliers = [min(max_credit_multiplier, value) for value in multipliers]
                scale = len(episode_steps) / max(eps, sum(multipliers))
                multipliers = [value * scale for value in multipliers]

            for item, multiplier in zip(episode_steps, multipliers):
                advantage_by_key[item.key] = trajectory_advantage * multiplier
                multiplier_by_key[item.key] = multiplier

    for sample in samples:
        if not sample.step_keys:
            continue
        key = sample.step_keys[0]
        sample.advantage = advantage_by_key.get(key, 0.0)
        sample.metadata["step_advantage_estimator"] = "trajectory_grouped_credit"
        sample.metadata["credit_multiplier"] = multiplier_by_key.get(key, 0.0)
        sample.group_id = sample.group_id.split("|prompt=", 1)[0]

    return {
        "num_trajectory_groups": float(len(group_sizes)),
        "avg_trajectory_group_size": sum(group_sizes) / max(1, len(group_sizes)),
        "singleton_trajectory_group_frac": singleton_groups / max(1, len(group_sizes)),
        "zero_variance_trajectory_group_frac": zero_variance_groups / max(1, len(group_sizes)),
    }


def _assign_trajectory_grouped_residual_advantages(
    steps: list[StepRecord],
    samples: list[GRPOSample],
    *,
    residual_beta: float,
    residual_clip: float,
    residual_use_confidence: bool,
    eps: float,
) -> dict[str, float]:
    """Add a centered ECR step residual to the trajectory-group advantage."""

    by_group_episode: dict[str, dict[str, list[StepRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for step in steps:
        by_group_episode[step.group_id][step.episode_id].append(step)

    advantage_by_key: dict[tuple[str, str, int], float] = {}
    residual_by_key: dict[tuple[str, str, int], float] = {}
    confidence_by_key: dict[tuple[str, str, int], float] = {}
    group_sizes: list[int] = []
    zero_variance_groups = 0
    active_residuals = 0
    sign_flips = 0
    total_steps = 0
    abs_residual_sum = 0.0
    confidence_sum = 0.0

    for episodes in by_group_episode.values():
        ordered_episodes = {
            episode_id: sorted(episode_steps, key=lambda item: item.step_id)
            for episode_id, episode_steps in episodes.items()
        }
        episode_ids = sorted(ordered_episodes)
        returns = [
            sum(item.return_estimate for item in ordered_episodes[episode_id])
            for episode_id in episode_ids
        ]
        group_sizes.append(len(episode_ids))
        mean_return = sum(returns) / max(1, len(returns))
        return_var = sum((value - mean_return) ** 2 for value in returns) / max(
            1, len(returns)
        )
        return_std = math.sqrt(return_var)
        if return_std < eps:
            zero_variance_groups += 1

        centered_by_episode: dict[str, list[float]] = {}
        all_centered: list[float] = []
        for episode_id in episode_ids:
            episode_steps = ordered_episodes[episode_id]
            credits = [item.return_estimate for item in episode_steps]
            mean_credit = sum(credits) / max(1, len(credits))
            centered = [credit - mean_credit for credit in credits]
            centered_by_episode[episode_id] = centered
            all_centered.extend(centered)
        residual_scale = math.sqrt(
            sum(value * value for value in all_centered) / max(1, len(all_centered))
        )

        for episode_id, episode_return in zip(episode_ids, returns):
            episode_steps = ordered_episodes[episode_id]
            trajectory_advantage = (
                0.0
                if return_std < eps
                else (episode_return - mean_return) / (return_std + eps)
            )
            for item, centered_credit in zip(
                episode_steps,
                centered_by_episode[episode_id],
            ):
                if residual_scale < eps:
                    normalized_residual = 0.0
                else:
                    normalized_residual = centered_credit / (residual_scale + eps)
                    if residual_clip > 0.0:
                        normalized_residual = max(
                            -residual_clip,
                            min(residual_clip, normalized_residual),
                        )
                abs_mass = float(item.metadata.get("credit_abs_mass", 0.0))
                confidence_mass = float(
                    item.metadata.get("credit_confidence_mass", 0.0)
                )
                if residual_use_confidence:
                    confidence = (
                        min(1.0, max(0.0, confidence_mass / abs_mass))
                        if abs_mass > eps
                        else 0.0
                    )
                else:
                    confidence = 1.0
                residual = residual_beta * confidence * normalized_residual
                final_advantage = trajectory_advantage + residual
                advantage_by_key[item.key] = final_advantage
                residual_by_key[item.key] = residual
                confidence_by_key[item.key] = confidence
                total_steps += 1
                confidence_sum += confidence
                abs_residual_sum += abs(residual)
                if abs(residual) > eps:
                    active_residuals += 1
                if (
                    abs(trajectory_advantage) > eps
                    and abs(final_advantage) > eps
                    and trajectory_advantage * final_advantage < 0.0
                ):
                    sign_flips += 1

    for sample in samples:
        if not sample.step_keys:
            continue
        key = sample.step_keys[0]
        sample.advantage = advantage_by_key.get(key, 0.0)
        sample.metadata["step_advantage_estimator"] = "trajectory_grouped_residual"
        sample.metadata["step_residual"] = residual_by_key.get(key, 0.0)
        sample.metadata["credit_confidence"] = confidence_by_key.get(key, 0.0)
        sample.group_id = sample.group_id.split("|prompt=", 1)[0]

    return {
        "num_trajectory_groups": float(len(group_sizes)),
        "avg_trajectory_group_size": sum(group_sizes) / max(1, len(group_sizes)),
        "zero_variance_trajectory_group_frac": zero_variance_groups
        / max(1, len(group_sizes)),
        "residual_beta": residual_beta,
        "residual_active_frac": active_residuals / max(1, total_steps),
        "avg_abs_step_residual": abs_residual_sum / max(1, total_steps),
        "avg_credit_confidence": confidence_sum / max(1, total_steps),
        "step_sign_flip_frac": sign_flips / max(1, total_steps),
    }


def _write_advantages_to_steps(steps: list[StepRecord], samples: list[GRPOSample]) -> None:
    by_key = {step.key: step for step in steps}
    for sample in samples:
        for key in sample.step_keys:
            if key in by_key:
                by_key[key].advantage = sample.advantage
