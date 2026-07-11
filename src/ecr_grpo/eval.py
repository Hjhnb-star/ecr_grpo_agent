from __future__ import annotations

import re
from collections import defaultdict

from ecr_grpo.envs.async_wrapper import AsyncEnvWrapper


def evaluate_policy(
    *,
    tasks: list,
    env_factory,
    policy,
    max_steps: int,
    greedy: bool = True,
) -> dict[str, float]:
    successes = 0
    total_steps = 0
    successful_steps = 0
    successful_episodes = 0
    total_env_return = 0.0
    total_shaped_return = 0.0
    total_tokens = 0
    total_correct = 0
    labeled_actions = 0
    positive_transitions = 0
    labeled_transitions = 0
    total_progress = 0.0
    total_progress_fraction = 0.0
    progress_episodes = 0
    max_step_episodes = 0
    category_successes: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = defaultdict(int)

    for task in tasks:
        env: AsyncEnvWrapper = _make_env(env_factory, task.task_id)
        obs = env.reset(task_id=task.task_id, episode_id=f"eval_{task.task_id}")
        episode_env_return = 0.0
        episode_shaped_return = 0.0
        success = False
        steps = 0
        final_progress = 0.0
        has_progress = False
        done = False
        task_horizon = len(getattr(task, "sequence", []) or []) or max_steps
        for _ in range(max_steps):
            action = policy.act(obs, action_space=list(env.action_space), greedy=greedy)
            obs, reward, done, info = env.step(action.text)
            episode_shaped_return += reward
            episode_env_return += float(info.get("env_reward", reward))
            total_tokens += len(action.prompt_ids) + len(action.response_ids)
            steps += 1
            if "causal_action" in info:
                total_correct += int(bool(info["causal_action"]))
                labeled_actions += 1
            if info.get("positive_transition") is not None:
                positive_transitions += int(bool(info["positive_transition"]))
                labeled_transitions += 1
            if info.get("progress") is not None:
                final_progress = float(info["progress"])
                has_progress = True
            if done:
                success = bool(info.get("success", False))
                break
        successes += int(success)
        total_steps += steps
        total_env_return += episode_env_return
        total_shaped_return += episode_shaped_return
        if success:
            successful_steps += steps
            successful_episodes += 1
        if not done:
            max_step_episodes += 1
        if has_progress:
            progress_episodes += 1
            total_progress += final_progress
            total_progress_fraction += final_progress / max(1, task_horizon)
        metadata = getattr(task, "metadata", {}) or {}
        category = str(
            metadata.get("task_type")
            or getattr(getattr(env, "env", None), "task_type", "unknown")
        )
        category_counts[category] += 1
        category_successes[category] += int(success)

    n = max(1, len(tasks))
    metrics = {
        "success_rate": successes / n,
        "avg_steps": total_steps / n,
        "avg_steps_success": successful_steps / max(1, successful_episodes),
        "avg_env_return": total_env_return / n,
        "avg_shaped_return": total_shaped_return / n,
        "avg_tokens": total_tokens / n,
        "failure_rate": (n - successes) / n,
        "max_step_rate": max_step_episodes / n,
    }
    if labeled_actions:
        metrics["action_accuracy"] = total_correct / labeled_actions
    if labeled_transitions:
        metrics["positive_transition_rate"] = positive_transitions / labeled_transitions
    if progress_episodes:
        metrics["avg_progress"] = total_progress / progress_episodes
        metrics["avg_progress_fraction"] = total_progress_fraction / progress_episodes
    for category, count in sorted(category_counts.items()):
        key = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_") or "unknown"
        metrics[f"success_rate_{key}"] = category_successes[category] / max(1, count)
        metrics[f"num_tasks_{key}"] = float(count)
    return metrics


def _make_env(env_factory, task_id: str):
    try:
        return env_factory(task_id=task_id)
    except TypeError as exc:
        message = str(exc)
        if (
            "unexpected keyword argument" not in message
            and "takes no arguments" not in message
        ):
            raise
        return env_factory()
