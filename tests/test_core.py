from __future__ import annotations

import unittest

from ecr_grpo.advantages import compute_group_advantages
from ecr_grpo.buffers import PendingStepBuffer
from ecr_grpo.credit_kernels import DependencyAwareKernel, EvidenceKernel, RecencyDecayKernel, UniformKernel
from ecr_grpo.types import AsyncEvent, StepRecord


def step(step_id: int, action: str = "a", subgoal: str = "a") -> StepRecord:
    return StepRecord(
        task_id="task",
        episode_id="ep",
        group_id="grp",
        step_id=step_id,
        env_time=step_id,
        observation="obs",
        observation_key=f"obs_{step_id}",
        action=action,
        old_logprob=-1.0,
        action_space=["a", "b"],
        tool_name=action,
        subgoal_id=subgoal,
    )


def event(reward: float = 1.0, related_step_id: int | None = 2) -> AsyncEvent:
    return AsyncEvent(
        task_id="task",
        episode_id="ep",
        event_id="evt",
        event_type="terminal_success",
        event_time=3,
        reward=reward,
        related_step_id=related_step_id,
        related_tool="a",
        related_subgoal="a",
        terminal=True,
    )


class CoreTests(unittest.TestCase):
    def test_uniform_weights_sum_to_one(self) -> None:
        weights = UniformKernel().weights(event(), [step(0), step(1), step(2)])
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_recency_prefers_later_steps(self) -> None:
        weights = RecencyDecayKernel(lambda_=0.5).weights(event(), [step(0), step(1), step(2)])
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[1], weights[0])

    def test_dependency_boosts_matching_step(self) -> None:
        steps = [step(0, "b", "b"), step(1, "b", "b"), step(2, "a", "a")]
        weights = DependencyAwareKernel().weights(event(), steps)
        self.assertEqual(max(range(len(weights)), key=lambda i: weights[i]), 2)

    def test_evidence_kernel_works_without_oracle_links(self) -> None:
        steps = [
            step(0, "search_web", "search"),
            step(1, "extract_fact", "extract"),
            step(2, "answer", "answer"),
        ]
        steps[1].metadata["tags"] = ["extract_fact", "verified"]
        evt = AsyncEvent(
            task_id="task",
            episode_id="ep",
            event_id="evt",
            event_type="partial_reward",
            event_time=3,
            reward=1.0,
            observation_delta="extracted fact verified",
            terminal=False,
            metadata={"source_time": 2, "tags": ["extract_fact", "verified"]},
        )
        weights = EvidenceKernel(lambda_=0.2).weights(evt, steps)
        self.assertEqual(max(range(len(weights)), key=lambda i: weights[i]), 1)

    def test_buffer_assigns_credit(self) -> None:
        buffer = PendingStepBuffer(max_age=5)
        for i in range(3):
            buffer.add_step(step(i))
        assignments = buffer.assign_event(event(), UniformKernel())
        self.assertEqual(len(assignments), 3)
        flushed = buffer.flush_episode("ep")
        self.assertAlmostEqual(sum(s.filled_credit for s in flushed), 1.0)

    def test_group_advantages_zero_mean(self) -> None:
        steps = [step(0), step(1), step(2)]
        steps[0].filled_credit = 0.0
        steps[1].filled_credit = 1.0
        steps[2].filled_credit = 2.0
        compute_group_advantages(steps)
        self.assertAlmostEqual(sum(s.advantage for s in steps), 0.0, places=6)
        self.assertGreater(steps[2].advantage, steps[0].advantage)


if __name__ == "__main__":
    unittest.main()
