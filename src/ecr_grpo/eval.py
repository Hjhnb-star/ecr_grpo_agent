from __future__ import annotations

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
    total_return = 0.0
    total_correct = 0
    total_progress = 0.0
    total_progress_fraction = 0.0

    for task in tasks:
        env: AsyncEnvWrapper = env_factory()
        obs = env.reset(task_id=task.task_id, episode_id=f"eval_{task.task_id}")
        episode_return = 0.0
        success = False
        steps = 0
        final_progress = 0.0
        task_horizon = len(getattr(task, "sequence", []) or []) or max_steps
        for _ in range(max_steps):
            action = policy.act(obs, action_space=list(env.action_space), greedy=greedy)
            obs, reward, done, info = env.step(action.text)
            episode_return += reward
            steps += 1
            total_correct += int(bool(info.get("causal_action", False)))
            final_progress = float(info.get("progress", final_progress))
            if done:
                success = bool(info.get("success", False))
                break
        successes += int(success)
        total_steps += steps
        total_return += episode_return
        total_progress += final_progress
        total_progress_fraction += final_progress / max(1, task_horizon)

    n = max(1, len(tasks))
    step_denom = max(1, total_steps)
    return {
        "success_rate": successes / n,
        "avg_steps": total_steps / n,
        "avg_env_return": total_return / n,
        "action_accuracy": total_correct / step_denom,
        "avg_progress": total_progress / n,
        "avg_progress_fraction": total_progress_fraction / n,
    }
