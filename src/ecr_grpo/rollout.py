from __future__ import annotations

from ecr_grpo.buffers import PendingStepBuffer
from ecr_grpo.credit_kernels import CreditKernel
from ecr_grpo.envs.async_wrapper import AsyncEnvWrapper
from ecr_grpo.policies import observation_key
from ecr_grpo.types import AsyncEvent, CreditAssignment, RolloutGroup, StepRecord


def collect_rollout_group(
    *,
    group_id: str,
    task_id: str,
    group_size: int,
    env_factory,
    policy,
    kernel: CreditKernel,
    max_pending_age: int,
    max_steps: int,
    greedy: bool = False,
) -> RolloutGroup:
    all_steps: list[StepRecord] = []
    all_events: list[AsyncEvent] = []
    all_assignments: list[CreditAssignment] = []
    episodes: list[str] = []

    for sample_idx in range(group_size):
        episode_id = f"{group_id}_ep_{sample_idx:02d}"
        env: AsyncEnvWrapper = env_factory()
        obs = env.reset(task_id=task_id, episode_id=episode_id)
        buffer = PendingStepBuffer(max_age=max_pending_age)
        episodes.append(episode_id)

        for step_id in range(max_steps):
            action = policy.act(obs, action_space=list(env.action_space), greedy=greedy)
            obs_key = observation_key(obs)
            next_obs, reward, done, info = env.step(action.text)
            step = StepRecord(
                task_id=task_id,
                episode_id=episode_id,
                group_id=group_id,
                step_id=step_id,
                env_time=info["async_time"],
                observation=obs,
                observation_key=obs_key,
                action=action.text,
                old_logprob=action.old_logprob,
                action_space=list(env.action_space),
                prompt_ids=action.prompt_ids,
                response_ids=action.response_ids,
                tool_name=info.get("tool_name") or info.get("called_tool"),
                subgoal_id=info.get("subgoal_id"),
                immediate_reward=0.0,
                metadata={
                    "action": action.text,
                    "causal_action": info.get("causal_action", False),
                    "expected_action": info.get("expected_action"),
                    "subgoal": info.get("subgoal_id") or info.get("expected_action"),
                    "tags": info.get("tags", []),
                },
            )
            buffer.add_step(step)

            ready_events = env.pop_events()
            all_events.extend(ready_events)
            for event in ready_events:
                all_assignments.extend(buffer.assign_event(event, kernel))

            all_steps.extend(buffer.finalize_ready(env.current_time))
            obs = next_obs
            if done:
                break

        drained_events = env.drain_events()
        all_events.extend(drained_events)
        for event in drained_events:
            all_assignments.extend(buffer.assign_event(event, kernel))
        all_steps.extend(buffer.flush_episode(episode_id))

    return RolloutGroup(
        group_id=group_id,
        task_id=task_id,
        episodes=episodes,
        steps=all_steps,
        events=all_events,
        assignments=all_assignments,
    )
