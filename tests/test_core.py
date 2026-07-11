from __future__ import annotations

import csv
import tempfile
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from ecr_grpo.advantages import compute_group_advantages
from ecr_grpo.analyze_credit import event_target_action, is_non_local_event
from ecr_grpo.buffers import PendingStepBuffer
from ecr_grpo.credit_kernels import (
    DependencyAwareKernel,
    EvidenceKernel,
    GatedEvidenceKernel,
    LatestStepKernel,
    RecencyDecayKernel,
    UniformKernel,
    classify_credit_event,
)
from ecr_grpo.envs.async_wrapper import AsyncEnvWrapper
from ecr_grpo.envs.alfworld_wrapper import ALFWorldEnv, ALFWorldGameCatalog
from ecr_grpo.hf_policy import clipped_policy_logprob_gradient
from ecr_grpo.envs.external_wrapper import ExternalTextBenchmarkEnv
from ecr_grpo.grpo_adapter import (
    assign_grpo_advantages,
    build_step_grpo_samples,
    build_trajectory_grpo_samples,
    normalize_reward_unit,
)
from ecr_grpo.io import write_csv
from ecr_grpo.rollout import collect_rollout_group
from ecr_grpo.run_alfworld import build_run_config, validate_config
from ecr_grpo.types import AsyncEvent, PolicyAction, RolloutGroup, StepRecord
from ecr_grpo.trainer import ECRGRPOTrainer
from stage_b_plan import normalize_stage_b_config


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

    def test_latest_step_kernel_assigns_only_last_candidate(self) -> None:
        weights = LatestStepKernel().weights(event(), [step(0), step(1), step(2)])
        self.assertEqual(weights, [0.0, 0.0, 1.0])

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

    def test_gated_evidence_limits_local_credit_window(self) -> None:
        steps = [step(index, f"action_{index}", f"action_{index}") for index in range(6)]
        evt = AsyncEvent(
            task_id="task",
            episode_id="ep",
            event_id="evt",
            event_type="partial_reward",
            event_time=6,
            reward=-0.1,
            observation_delta="stagnation",
            terminal=False,
            metadata={"delay": 0, "tags": ["stagnation"]},
        )
        kernel = GatedEvidenceKernel(local_window=2, local_lambda=1.0)
        weights = kernel.weights(evt, steps)

        self.assertEqual(weights[:4], [0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(weights[-2:]), 1.0)
        self.assertEqual(kernel.last_confidence, 1.0)

    def test_gated_evidence_downweights_weak_semantic_evidence(self) -> None:
        steps = [step(0, "open_box"), step(1, "wait"), step(2, "submit_code")]
        evt = AsyncEvent(
            task_id="task",
            episode_id="ep",
            event_id="evt",
            event_type="partial_reward",
            event_time=3,
            reward=0.2,
            observation_delta="unrelated_feedback_token",
            terminal=False,
            metadata={"credit_route": "non_local", "delay": 2},
        )
        kernel = GatedEvidenceKernel(
            nonlocal_evidence_weight=0.9,
            nonlocal_recency_weight=0.1,
            adaptive_evidence=True,
        )
        weights = kernel.weights(evt, steps)

        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertEqual(kernel.last_confidence, 0.0)

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
        self.assertNotIn("source_time", ready[0].metadata)
        self.assertEqual(ready[0].diagnostic_metadata["source_time"], 1)

    def test_delayed_event_keeps_intervening_steps_as_candidates(self) -> None:
        buffer = PendingStepBuffer(max_age=8)
        for step_id in range(3):
            buffer.add_step(step(step_id))
        evt = AsyncEvent(
            task_id="task",
            episode_id="ep",
            event_id="delayed",
            event_type="partial_reward",
            event_time=3,
            reward=1.0,
            observation_delta="result arrived",
            terminal=False,
            metadata={"delay": 3},
            diagnostic_metadata={"source_time": 0},
        )

        candidates = buffer.related_steps(evt)

        self.assertEqual([item.step_id for item in candidates], [0, 1, 2])
        self.assertEqual(classify_credit_event(evt), "delayed_feedback")

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

    def test_step_grpo_groups_by_prompt_not_whole_task(self) -> None:
        steps = [step(0), step(1), step(2)]
        for item in steps:
            item.group_id = "upd_0001_task"
        steps[0].episode_id = "ep_a"
        steps[1].episode_id = "ep_b"
        steps[2].episode_id = "ep_c"
        steps[0].observation = "same prompt"
        steps[1].observation = "same prompt"
        steps[2].observation = "different prompt"
        steps[0].observation_key = "prompt_same"
        steps[1].observation_key = "prompt_same"
        steps[2].observation_key = "prompt_other"
        steps[0].filled_credit = 0.0
        steps[1].filled_credit = 2.0
        steps[2].filled_credit = 100.0

        samples, stats = assign_grpo_advantages(steps, reward_unit="step")

        self.assertEqual(len(samples), 3)
        self.assertEqual(stats["num_grpo_groups"], 2.0)
        self.assertLess(steps[0].advantage, 0.0)
        self.assertGreater(steps[1].advantage, 0.0)
        self.assertEqual(steps[2].advantage, 0.0)

    def test_trajectory_grouped_credit_produces_nonzero_step_advantages(self) -> None:
        steps = []
        for episode_id, credits in (("ep_good", [1.0, 0.0]), ("ep_bad", [-1.0, 0.0])):
            for step_id, credit in enumerate(credits):
                item = step(step_id)
                item.group_id = "same_real_task"
                item.episode_id = episode_id
                item.observation_key = f"{episode_id}_{step_id}"
                item.filled_credit = credit
                steps.append(item)

        samples, stats = assign_grpo_advantages(
            steps,
            reward_unit="step",
            step_advantage_estimator="trajectory_grouped_credit",
            credit_weight_floor=0.05,
        )

        good = [item for item in steps if item.episode_id == "ep_good"]
        bad = [item for item in steps if item.episode_id == "ep_bad"]
        self.assertTrue(all(item.advantage > 0.0 for item in good))
        self.assertTrue(all(item.advantage < 0.0 for item in bad))
        self.assertGreater(abs(good[0].advantage), abs(good[1].advantage))
        self.assertEqual(stats["avg_trajectory_group_size"], 2.0)
        self.assertLess(stats["zero_advantage_frac"], 1.0)
        self.assertEqual({sample.group_id for sample in samples}, {"same_real_task"})

    def test_trajectory_grouped_residual_creates_step_direction(self) -> None:
        steps = []
        for episode_id in ("ep_a", "ep_b"):
            for step_id, credit in enumerate((1.0, -1.0)):
                item = step(step_id)
                item.group_id = "same_real_task"
                item.episode_id = episode_id
                item.observation_key = f"{episode_id}_{step_id}"
                item.filled_credit = credit
                item.metadata["credit_abs_mass"] = 1.0
                item.metadata["credit_confidence_mass"] = 1.0
                steps.append(item)

        _, stats = assign_grpo_advantages(
            steps,
            reward_unit="step",
            step_advantage_estimator="trajectory_grouped_residual",
            residual_beta=0.5,
            residual_clip=2.0,
        )

        first_steps = [item for item in steps if item.step_id == 0]
        second_steps = [item for item in steps if item.step_id == 1]
        self.assertTrue(all(item.advantage > 0.0 for item in first_steps))
        self.assertTrue(all(item.advantage < 0.0 for item in second_steps))
        self.assertEqual(stats["zero_variance_trajectory_group_frac"], 1.0)
        self.assertEqual(stats["residual_active_frac"], 1.0)

    def test_clipped_policy_gradient_matches_grpo_clipping_regions(self) -> None:
        loss, ratio, gradient = clipped_policy_logprob_gradient(
            new_logprob=0.0,
            old_logprob=0.0,
            advantage=1.0,
            clip_eps=0.2,
        )
        self.assertAlmostEqual(loss, -1.0)
        self.assertAlmostEqual(ratio, 1.0)
        self.assertAlmostEqual(gradient, -1.0)

        _, _, clipped_positive = clipped_policy_logprob_gradient(
            new_logprob=1.0,
            old_logprob=0.0,
            advantage=1.0,
            clip_eps=0.2,
        )
        _, _, clipped_negative = clipped_policy_logprob_gradient(
            new_logprob=-1.0,
            old_logprob=0.0,
            advantage=-1.0,
            clip_eps=0.2,
        )
        self.assertEqual(clipped_positive, 0.0)
        self.assertEqual(clipped_negative, 0.0)
    def test_step_grpo_sample_exposes_reward_adapter_boundary(self) -> None:
        item = step(0, action="open_box")
        item.filled_credit = 0.5
        samples = build_step_grpo_samples([item])

        self.assertEqual(samples[0].completion, "open_box")
        self.assertAlmostEqual(samples[0].reward, 0.5)
        self.assertEqual(samples[0].unit, "step")
        self.assertIn("Available actions", samples[0].prompt)
        self.assertEqual(samples[0].metadata["optimizer_boundary"], "reward_construction_only")

    def test_trajectory_grpo_sample_uses_refilled_episode_return(self) -> None:
        steps = [step(0, action="find_key"), step(1, action="open_box")]
        steps[0].filled_credit = 0.25
        steps[1].filled_credit = 0.75
        group = RolloutGroup(
            group_id="grp",
            task_id="task",
            episodes=["ep"],
            steps=steps,
            events=[],
            assignments=[],
        )

        samples = build_trajectory_grpo_samples(group)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].unit, "trajectory")
        self.assertEqual(samples[0].group_id, "grp")
        self.assertAlmostEqual(samples[0].reward, 1.0)
        self.assertEqual(samples[0].completion, "find_key\nopen_box")
        self.assertIn("Available actions", samples[0].prompt)
        self.assertEqual(samples[0].metadata["reward_source"], "ecr_refilled_trajectory_return")

    def test_reward_unit_aliases_support_plugin_config_terms(self) -> None:
        self.assertEqual(normalize_reward_unit("step_reward"), "step")
        self.assertEqual(normalize_reward_unit("trajectory_reward"), "trajectory")

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
        self.assertIn("Current observation", obs)
        self.assertEqual(env.action_space, ["look", "inventory"])

        _, reward, done, info = env.step("look")
        self.assertFalse(done)
        self.assertFalse(info["success"])
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["tool_name"], "look")
        self.assertNotIn("look", info["public_tags"])
        self.assertIn("verb_examine", info["public_tags"])
        self.assertTrue(info["events"])
        self.assertNotIn("action", info["events"][0].metadata)
        self.assertEqual(info["events"][0].diagnostic_metadata["source_action"], "look")

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

    def test_alfworld_catalog_builds_unique_game_metadata(self) -> None:
        catalog = ALFWorldGameCatalog.__new__(ALFWorldGameCatalog)
        catalog.split = "eval_out_of_distribution"
        metadata = catalog.task_metadata(
            "/data/json_2.1.1/eval_out_of_distribution/"
            "pick_heat_then_place_in_recep-Apple-Plate-Microwave-1/game.tw-pddl"
        )

        self.assertEqual(metadata["task_type"], "pick_heat_then_place_in_recep")
        self.assertIn("pick_heat_then_place_in_recep", metadata["actual_task_id"])
        self.assertNotEqual(metadata["actual_task_id"], "game.tw-pddl")
        self.assertTrue(metadata["task_id"].startswith("eval_out_of_distribution__"))

    def test_alfworld_uses_score_delta_and_single_terminal_event(self) -> None:
        class CumulativeScoreEnv:
            def __init__(self):
                self.index = 0

            def reset(self):
                return ["start"], {
                    "admissible_commands": [["look"]],
                    "gamefile": ["/tmp/task/game.ulx"],
                    "won": [False],
                }

            def step(self, actions):
                self.index += 1
                if self.index == 1:
                    return ["progress"], [0.4], [False], {
                        "admissible_commands": [["look"]],
                        "gamefile": ["/tmp/task/game.ulx"],
                        "won": [False],
                    }
                return ["done"], [1.0], [True], {
                    "admissible_commands": [["look"]],
                    "gamefile": ["/tmp/task/game.ulx"],
                    "won": [True],
                }

        env = ALFWorldEnv(
            alfworld_config="mock.yaml",
            split="eval_out_of_distribution",
            fallback_action_space=["look"],
            shaping_config={"enabled": False},
            raw_env=CumulativeScoreEnv(),
        )
        env.reset(task_id="task", episode_id="ep")
        _, first_reward, first_done, first_info = env.step("look")
        _, second_reward, second_done, second_info = env.step("look")

        self.assertFalse(first_done)
        self.assertTrue(second_done)
        self.assertAlmostEqual(first_reward, 0.4)
        self.assertAlmostEqual(second_reward, 0.6)
        self.assertAlmostEqual(first_reward + second_reward, 1.0)
        self.assertEqual(len(second_info["events"]), 1)
        self.assertEqual(second_info["events"][0].event_type, "terminal_success")
        self.assertAlmostEqual(second_info["events"][0].reward, 0.6)
    def test_external_benchmark_adapter_eventizes_standard_step_api(self) -> None:
        class RawEnv:
            action_space = ["inspect"]

            def reset(self, task_id=None, **kwargs):
                return "start", {"valid_actions": ["inspect"]}

            def step(self, action):
                return "done", 1.0, True, {
                    "success": True,
                    "valid_actions": ["finish"],
                }

        def factory(**kwargs):
            return RawEnv()

        with patch(
            "ecr_grpo.envs.external_wrapper._load_object",
            return_value=factory,
        ):
            env = ExternalTextBenchmarkEnv(
                factory_path="example:create_env",
                split="test",
                task_metadata={
                    "task_id": "test__task_1",
                    "actual_task_id": "task_1",
                    "task_type": "science",
                },
                fallback_action_space=["inspect"],
                seed=7,
            )

        observation = env.reset(task_id="task", episode_id="ep")
        next_observation, reward, done, info = env.step("inspect")

        self.assertEqual(observation, "start")
        self.assertEqual(next_observation, "done")
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertEqual(info["task_type"], "science")
        self.assertEqual(env.action_space, ["finish"])
        self.assertEqual(info["events"][0].event_type, "terminal_success")
        self.assertEqual(info["events"][0].diagnostic_metadata["source_action"], "inspect")

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
            "training": {"group_size": 2},
        }
        gated = build_run_config(
            base,
            kernel="gated",
            seed=7,
            output_root=Path("runs/alf"),
            train_split="train",
            eval_split="eval_out_of_distribution",
            num_train_tasks=32,
            num_eval_tasks=12,
            max_steps=40,
            adapter_path="runs/warmup/checkpoints/latest",
            num_updates=20,
            tasks_per_update=2,
            group_size=4,
            eval_every=5,
            train_delay_prob=0.2,
            terminal_reward_delay=1,
            missing_reward_prob=0.05,
            clean_eval=True,
        )
        grpo = build_run_config(base, kernel="grpo", seed=7, output_root=Path("runs/alf"))
        residual = build_run_config(
            base,
            kernel="residual",
            seed=7,
            output_root=Path("runs/alf"),
        )

        self.assertEqual(gated["environment"]["split"], "train")
        self.assertEqual(gated["environment"]["train_split"], "train")
        self.assertEqual(gated["environment"]["eval_split"], "eval_out_of_distribution")
        self.assertEqual(gated["environment"]["num_train_tasks"], 32)
        self.assertEqual(gated["environment"]["max_steps"], 40)
        self.assertEqual(gated["evaluation"]["split"], "eval_out_of_distribution")
        self.assertEqual(gated["evaluation"]["num_eval_tasks"], 12)
        self.assertFalse(gated["evaluation"]["async"]["enabled"])
        self.assertEqual(gated["evaluation"]["async"]["delay_prob"], 0.0)
        self.assertEqual(gated["credit"]["kernel"], "gated_evidence")
        self.assertEqual(gated["credit"]["output"], "step_reward")
        self.assertEqual(gated["optimizer"]["name"], "grpo")
        self.assertEqual(gated["optimizer"]["advantage_mode"], "step")
        self.assertEqual(gated["optimizer"]["update_impl"], "standard_grpo")
        self.assertEqual(gated["training"]["advantage_mode"], "step")
        self.assertEqual(gated["training"]["optimizer"], "grpo")
        self.assertEqual(gated["training"]["grpo_reward_unit"], "step")
        self.assertEqual(gated["training"]["num_updates"], 20)
        self.assertEqual(gated["training"]["tasks_per_update"], 2)
        self.assertEqual(gated["training"]["group_size"], 4)
        self.assertEqual(gated["evaluation"]["every_updates"], 5)
        self.assertEqual(gated["async"]["delay_prob"], 0.2)
        self.assertEqual(gated["async"]["terminal_reward_delay"], 1)
        self.assertEqual(gated["async"]["missing_reward_prob"], 0.05)
        self.assertEqual(gated["policy"]["adapter_path"], "runs/warmup/checkpoints/latest")
        self.assertFalse(gated["async"]["use_oracle_event_links"])
        self.assertEqual(grpo["credit"]["kernel"], "trajectory_uniform")
        self.assertEqual(grpo["credit"]["output"], "trajectory_reward")
        self.assertEqual(grpo["optimizer"]["name"], "grpo")
        self.assertEqual(grpo["optimizer"]["advantage_mode"], "trajectory")
        self.assertEqual(grpo["training"]["advantage_mode"], "trajectory")
        self.assertEqual(grpo["training"]["optimizer"], "grpo")
        self.assertEqual(grpo["training"]["grpo_reward_unit"], "trajectory")
        self.assertEqual(gated["policy"]["update_score_mode"], "streaming_distribution")
        self.assertEqual(gated["optimizer"]["step_advantage_estimator"], "trajectory_grouped_credit")
        self.assertEqual(
            residual["optimizer"]["step_advantage_estimator"],
            "trajectory_grouped_residual",
        )
        self.assertTrue(residual["credit"]["adaptive_evidence"])
        self.assertEqual(validate_config(gated, require_files=False), [])

        gated["training"]["group_size"] = 1
        errors = validate_config(gated, require_files=False)
        self.assertTrue(any("group_size >= 2" in error for error in errors))

        gated["training"]["group_size"] = 2
        gated["policy"]["update_score_mode"] = "selected"
        errors = validate_config(gated, require_files=False)
        self.assertTrue(any("smoke-only" in error for error in errors))

    def test_stage_b_hf_configs_default_to_distribution_ratio(self) -> None:
        config = {
            "environment": {"non_local_credit": {}},
            "async": {},
            "policy": {"kind": "hf_lora", "model_id": "Qwen/Qwen2.5-1.5B-Instruct"},
            "training": {},
            "evaluation": {},
        }

        with patch.dict(os.environ, {}, clear=True):
            normalize_stage_b_config(config, seed=7, lag=2, reward=0.4)

        self.assertEqual(config["policy"]["action_selection"], "score")
        self.assertEqual(config["policy"]["update_score_mode"], "full_distribution")
        self.assertEqual(config["training"]["optimizer"], "grpo")

    def test_stage_b_hf_update_mode_env_override(self) -> None:
        config = {
            "environment": {"non_local_credit": {}},
            "async": {},
            "policy": {"kind": "hf_lora", "model_id": "Qwen/Qwen2.5-1.5B-Instruct"},
            "training": {},
            "evaluation": {},
        }

        with patch.dict(os.environ, {"ECR_GRPO_UPDATE_SCORE_MODE": "selected"}, clear=True):
            normalize_stage_b_config(config, seed=7, lag=2, reward=0.4)

        self.assertEqual(config["policy"]["update_score_mode"], "selected")

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

    def test_csv_writer_unions_dynamic_split_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            write_csv(
                path,
                [
                    {"eval_split": "seen", "success_rate_seen_type": 0.5},
                    {"eval_split": "ood", "success_rate_ood_type": 0.25},
                ],
            )
            with path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
        self.assertIn("success_rate_seen_type", reader.fieldnames)
        self.assertIn("success_rate_ood_type", reader.fieldnames)
        self.assertEqual(len(rows), 2)


    def test_long_run_resume_matches_uninterrupted_training(self) -> None:
        def make_config(output_dir: Path, num_updates: int) -> dict:
            return {
                "experiment_name": "resume_test",
                "output_dir": str(output_dir),
                "seed": 19,
                "environment": {
                    "name": "synthetic",
                    "num_tasks": 4,
                    "max_steps": 4,
                    "sequences": [
                        ["find_key", "open_box", "submit"],
                        ["inspect", "edit", "submit"],
                    ],
                    "action_space": [
                        "find_key",
                        "open_box",
                        "inspect",
                        "edit",
                        "submit",
                        "wait",
                    ],
                },
                "async": {
                    "enabled": True,
                    "delay_prob": 0.5,
                    "max_delay_steps": 2,
                    "timeout_prob": 0.0,
                    "interruption_prob": 0.0,
                    "missing_reward_prob": 0.0,
                    "terminal_reward_delay": 2,
                    "use_oracle_event_links": False,
                },
                "credit": {
                    "kernel": "recency",
                    "lambda": 0.3,
                    "max_pending_age": 6,
                    "timeout_penalty": -0.2,
                },
                "policy": {
                    "kind": "tabular",
                    "temperature": 1.0,
                    "init_scale": 0.01,
                },
                "training": {
                    "num_updates": num_updates,
                    "tasks_per_update": 2,
                    "group_size": 2,
                    "learning_rate": 0.1,
                    "checkpoint_every": 1,
                    "task_sampling": {
                        "mode": "success_aware",
                        "ema_alpha": 0.2,
                    },
                },
                "evaluation": {
                    "every_updates": 100,
                    "num_eval_tasks": 4,
                    "delay_sweep": [],
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resumed_dir = root / "resumed"
            first = ECRGRPOTrainer(
                make_config(resumed_dir, 1),
                overwrite=True,
            )
            first.train()
            samples_path = resumed_dir / "train_grpo_samples.jsonl"
            with samples_path.open("a", encoding="utf-8") as handle:
                handle.write('{"uncommitted": true}\n')

            resumed = ECRGRPOTrainer(
                make_config(resumed_dir, 3),
                resume=True,
            )
            resumed.train()

            uninterrupted = ECRGRPOTrainer(
                make_config(root / "uninterrupted", 3),
                overwrite=True,
            )
            uninterrupted.train()

            self.assertEqual([row["update"] for row in resumed.train_rows], [1, 2, 3])
            self.assertEqual(resumed.train_rows, uninterrupted.train_rows)
            self.assertEqual(dict(resumed.policy.logits), dict(uninterrupted.policy.logits))
            self.assertEqual(resumed._task_success_ema, uninterrupted._task_success_ema)
            self.assertEqual(resumed._task_visits, uninterrupted._task_visits)
            self.assertTrue((resumed_dir / "COMPLETED.json").exists())
            self.assertNotIn("uncommitted", samples_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    unittest.main()
