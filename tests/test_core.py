from __future__ import annotations

import unittest
from pathlib import Path

from ecr_grpo.advantages import compute_group_advantages
from ecr_grpo.analyze_credit import event_target_action, is_non_local_event
from ecr_grpo.buffers import PendingStepBuffer
from ecr_grpo.credit_kernels import (
    DependencyAwareKernel,
    EvidenceKernel,
    GatedEvidenceKernel,
    RecencyDecayKernel,
    UniformKernel,
    classify_credit_event,
)
from ecr_grpo.envs.async_wrapper import AsyncEnvWrapper
from ecr_grpo.envs.alfworld_wrapper import ALFWorldEnv
from ecr_grpo.rollout import collect_rollout_group
from ecr_grpo.run_alfworld import build_run_config, validate_config
from ecr_grpo.types import AsyncEvent, PolicyAction, StepRecord


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

    def test_gated_evidence_routes_non_local_events_to_evidence(self) -> None:
        steps = [
            step(0, "find_key", "find_key"),
            step(1, "open_box", "open_box"),
            step(2, "submit_code", "submit_code"),
        ]
        steps[0].metadata["tags"] = ["find_key", "correct"]
        evt = AsyncEvent(
            task_id="task",
            episode_id="ep",
            event_id="evt",
            event_type="partial_reward",
            event_time=3,
            reward=0.4,
            observation_delta="non_local_support:find_key:confirmed_after_step_2",
            terminal=False,
            metadata={
                "credit_route": "non_local",
                "source_time": 3,
                "tags": ["find_key", "non_local_support"],
            },
            diagnostic_metadata={"target_action": "find_key"},
        )
        kernel = GatedEvidenceKernel(
            lambda_=0.2,
            nonlocal_evidence_weight=1.0,
            nonlocal_recency_weight=0.0,
            evidence_top_k=1,
        )
        weights = kernel.weights(evt, steps)
        self.assertEqual(classify_credit_event(evt), "non_local")
        self.assertEqual(kernel.last_category, "non_local")
        self.assertEqual(max(range(len(weights)), key=lambda i: weights[i]), 0)

    def test_gated_evidence_keeps_local_partial_reward_recent(self) -> None:
        steps = [
            step(0, "find_key", "find_key"),
            step(1, "open_box", "open_box"),
            step(2, "submit_code", "submit_code"),
        ]
        evt = AsyncEvent(
            task_id="task",
            episode_id="ep",
            event_id="evt",
            event_type="partial_reward",
            event_time=3,
            reward=0.1,
            observation_delta="completed:submit_code",
            terminal=False,
            metadata={"source_time": 3, "tags": ["submit_code", "partial_reward"]},
        )
        weights = GatedEvidenceKernel(local_lambda=1.0).weights(evt, steps)
        self.assertEqual(classify_credit_event(evt), "local_positive")
        self.assertEqual(max(range(len(weights)), key=lambda i: weights[i]), 2)

    def test_credit_analysis_detects_non_local_metadata(self) -> None:
        evt = {
            "event_id": "evt",
            "event_type": "partial_reward",
            "observation_delta": "support arrived",
            "metadata": {"credit_route": "non_local"},
            "diagnostic_metadata": {"target_action": "find_key", "target_lag": 4},
        }
        self.assertTrue(is_non_local_event(evt))
        self.assertEqual(event_target_action(evt), "find_key")

    def test_async_wrapper_strips_diagnostic_metadata_from_public_event(self) -> None:
        evt = AsyncEvent(
            task_id="task",
            episode_id="ep",
            event_id="evt",
            event_type="partial_reward",
            event_time=1,
            reward=0.1,
            related_step_id=0,
            related_tool="find_key",
            related_subgoal="find_key",
            metadata={
                "credit_route": "non_local",
                "target_action": "find_key",
                "tags": ["non_local_support"],
            },
        )

        class OneEventEnv:
            action_space = ["find_key"]

            def reset(self, task_id=None, episode_id=None):
                return "obs"

            def step(self, action):
                return "obs", 0.0, True, {
                    "task_id": "task",
                    "episode_id": "ep",
                    "step_id": 0,
                    "events": [evt],
                }

        env = AsyncEnvWrapper(OneEventEnv(), {"use_oracle_event_links": False}, seed=0)
        env.reset("task", "ep")
        env.step("find_key")
        ready = env.pop_events()
        self.assertNotIn("target_action", ready[0].metadata)
        self.assertEqual(ready[0].diagnostic_metadata["target_action"], "find_key")

    def test_buffer_assigns_credit(self) -> None:
        buffer = PendingStepBuffer(max_age=5)
        for i in range(3):
            buffer.add_step(step(i))
        assignments = buffer.assign_event(event(), UniformKernel())
        self.assertEqual(len(assignments), 3)
        self.assertGreater(assignments[0].effective_steps, 1.0)
        self.assertGreaterEqual(assignments[0].weight_entropy, 0.0)
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

    def test_trajectory_advantage_broadcasts_episode_return(self) -> None:
        steps = [
            step(0),
            step(1),
            step(2),
            step(3),
        ]
        steps[0].episode_id = "ep_a"
        steps[1].episode_id = "ep_a"
        steps[2].episode_id = "ep_b"
        steps[3].episode_id = "ep_b"
        steps[0].filled_credit = 1.0
        steps[1].filled_credit = 1.0
        steps[2].filled_credit = -1.0
        steps[3].filled_credit = -1.0
        compute_group_advantages(steps, mode="trajectory")
        self.assertAlmostEqual(steps[0].advantage, steps[1].advantage)
        self.assertAlmostEqual(steps[2].advantage, steps[3].advantage)
        self.assertGreater(steps[0].advantage, steps[2].advantage)

    def test_alfworld_adapter_handles_batched_infos(self) -> None:
        class MockALFWorldRawEnv:
            def reset(self):
                return ["You are in a room."], {
                    "admissible_commands": [["look", "inventory"]],
                    "extra.gamefile": ["/tmp/pick_and_place/game.ulx"],
                    "won": [False],
                }

            def step(self, actions):
                self.last_actions = actions
                return ["You are still in a room."], [0.0], [False], {
                    "admissible_commands": [["look", "inventory"]],
                    "extra.gamefile": ["/tmp/pick_and_place/game.ulx"],
                    "won": [False],
                }

        env = ALFWorldEnv(
            alfworld_config="mock.yaml",
            split="eval_out_of_distribution",
            fallback_action_space=["look"],
            raw_env=MockALFWorldRawEnv(),
        )
        obs = env.reset(task_id="alfworld_0000", episode_id="ep")
        self.assertIn("Admissible actions", obs)
        self.assertEqual(env.action_space, ["look", "inventory"])

        _, reward, done, info = env.step("look")
        self.assertFalse(done)
        self.assertFalse(info["success"])
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["tool_name"], "look")
        self.assertIn("look", info["public_tags"])
        self.assertTrue(info["events"])
        self.assertIn("look", info["events"][0].metadata["tags"])

    def test_alfworld_adapter_reads_terminal_success(self) -> None:
        class MockDoneRawEnv:
            def reset(self):
                return ["start"], {"admissible_commands": [["look"]], "gamefile": ["game.ulx"]}

            def step(self, actions):
                return ["done"], [1.0], [True], {
                    "admissible_commands": [["look"]],
                    "gamefile": ["game.ulx"],
                    "won": [True],
                }

        env = ALFWorldEnv(
            alfworld_config="mock.yaml",
            split="eval_out_of_distribution",
            fallback_action_space=["look"],
            raw_env=MockDoneRawEnv(),
        )
        env.reset(task_id="alfworld_0000", episode_id="ep")
        _, _, done, info = env.step("look")
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertEqual(info["events"][-1].event_type, "terminal_success")

    def test_run_alfworld_builds_no_oracle_fair_configs(self) -> None:
        base = {
            "environment": {
                "name": "alfworld",
                "alfworld_config": "C:/alfworld/base_config.yaml",
                "max_steps": 5,
                "action_space": ["look"],
            },
            "async": {"delay_prob": 0.1},
            "credit": {"kernel": "gated_evidence", "max_pending_age": 4},
            "policy": {"kind": "hf_lora", "model_id": "Qwen/Qwen2.5-1.5B-Instruct"},
            "training": {"group_size": 1},
        }
        gated = build_run_config(base, kernel="gated", seed=7, output_root=Path("runs/alf"))
        grpo = build_run_config(base, kernel="grpo", seed=7, output_root=Path("runs/alf"))

        self.assertEqual(gated["credit"]["kernel"], "gated_evidence")
        self.assertEqual(gated["training"]["advantage_mode"], "step")
        self.assertFalse(gated["async"]["use_oracle_event_links"])
        self.assertEqual(grpo["credit"]["kernel"], "trajectory_uniform")
        self.assertEqual(grpo["training"]["advantage_mode"], "trajectory")
        self.assertEqual(validate_config(gated, require_files=False), [])

    def test_rollout_records_pre_step_action_space(self) -> None:
        class DynamicActionEnv:
            def __init__(self) -> None:
                self.current_time = 0
                self.actions = ["open door"]

            @property
            def action_space(self):
                return self.actions

            def reset(self, task_id=None, episode_id=None):
                self.task_id = task_id
                self.episode_id = episode_id
                return "obs"

            def step(self, action):
                self.current_time += 1
                self.actions = ["go north"]
                return "next", 0.0, True, {
                    "task_id": self.task_id,
                    "episode_id": self.episode_id,
                    "async_time": self.current_time,
                    "step_id": 0,
                    "events": [],
                }

            def pop_events(self):
                return []

            def drain_events(self):
                return []

        class RecordingPolicy:
            def __init__(self) -> None:
                self.seen_action_space = []

            def act(self, obs, action_space=None, greedy=False):
                self.seen_action_space = list(action_space)
                return PolicyAction(text=action_space[0], old_logprob=-1.0)

        policy = RecordingPolicy()
        group = collect_rollout_group(
            group_id="grp",
            task_id="task",
            group_size=1,
            env_factory=DynamicActionEnv,
            policy=policy,
            kernel=UniformKernel(),
            max_pending_age=4,
            max_steps=2,
        )
        self.assertEqual(policy.seen_action_space, ["open door"])
        self.assertEqual(group.steps[0].action_space, ["open door"])


if __name__ == "__main__":
    unittest.main()
