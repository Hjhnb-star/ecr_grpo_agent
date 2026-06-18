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

    for task in tasks:
        env: AsyncEnvWrapper = env_factory()
        obs = env.reset(task_id=task.task_id, episode_id=f"eval_{task.task_id}")
        episode_return = 0.0
        success = False
        steps = 0
        for _ in range(max_steps):
            action = policy.act(obs, action_space=list(env.action_space), greedy=greedy)
            obs, reward, done, info = env.step(action.text)
            episode_return += reward
            steps += 1
            if done:
                success = bool(info.get("success", False))
                break
        successes += int(success)
        total_steps += steps
        total_return += episode_return

    n = max(1, len(tasks))
    return {
        "success_rate": successes / n,
        "avg_steps": total_steps / n,
        "avg_env_return": total_return / n,
    }
