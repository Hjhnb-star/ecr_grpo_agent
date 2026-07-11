from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import shutil
from pathlib import Path

from ecr_grpo.credit_kernels import build_credit_kernel
from ecr_grpo.envs.alfworld_wrapper import ALFWorldEnv, ALFWorldGameCatalog
from ecr_grpo.envs.async_wrapper import AsyncEnvWrapper
from ecr_grpo.envs.external_wrapper import ExternalTextBenchmarkEnv
from ecr_grpo.envs.synthetic import SyntheticLongHorizonEnv, build_synthetic_tasks
from ecr_grpo.eval import evaluate_policy
from ecr_grpo.grpo_adapter import assign_grpo_advantages, normalize_reward_unit
from ecr_grpo.io import append_jsonl, ensure_dir, load_config, write_csv
from ecr_grpo.policies import build_policy
from ecr_grpo.rollout import collect_rollout_group
from ecr_grpo.types import BenchmarkTask


class ECRGRPOTrainer:
    def __init__(self, config: dict, *, resume: bool = False, overwrite: bool = False) -> None:
        self.config = config
        self.resume_requested = bool(resume or config.get("training", {}).get("resume", False))
        self.overwrite_requested = bool(overwrite)
        if self.resume_requested and self.overwrite_requested:
            raise ValueError("resume and overwrite are mutually exclusive")

        self.seed = int(config.get("seed", 0))
        self.rng = random.Random(self.seed)
        self.output_dir = ensure_dir(config.get("output_dir", "runs/smoke"))
        self._cached_alfworld_envs: dict[tuple[str, str, str], AsyncEnvWrapper] = {}
        self._restored_alfworld_rng_states: dict[tuple[str, str, str], object] = {}
        self._alfworld_cache_order: list[tuple[str, str, str]] = []
        self._alfworld_catalogs: dict[str, ALFWorldGameCatalog] = {}
        self.tasks = self._build_tasks()
        self.action_space = list(config["environment"]["action_space"])
        self.max_steps = int(config["environment"].get("max_steps", 10))
        self.kernel = build_credit_kernel(config.get("credit", {}))
        self.policy = build_policy(config, self.action_space, seed=self.seed)
        self.eval_tasks_by_split = {
            split: self._build_tasks(eval_mode=True, split_override=split)
            for split in self._eval_splits()
        }
        primary_eval_split = self._env_split(eval_mode=True)
        self.eval_tasks = self.eval_tasks_by_split[primary_eval_split]
        self._task_lookup = {
            task.task_id: task
            for task in [
                *self.tasks,
                *(task for tasks in self.eval_tasks_by_split.values() for task in tasks),
            ]
        }
        self.train_rows: list[dict] = []
        self.eval_rows: list[dict] = []
        self._task_success_ema: dict[str, float] = {}
        self._task_visits: dict[str, int] = {}

        self._last_completed_update = 0

    def _task_sampling_config(self) -> dict:
        value = self.config.get("training", {}).get("task_sampling", {"mode": "uniform"})
        if isinstance(value, str):
            return {"mode": value}
        return dict(value)

    def _sample_training_tasks(self, count: int) -> list[BenchmarkTask]:
        count = min(max(0, count), len(self.tasks))
        sampling = self._task_sampling_config()
        if count == 0:
            return []
        if str(sampling.get("mode", "uniform")).lower() != "success_aware":
            return self.rng.sample(self.tasks, k=count)
        unseen_boost = float(sampling.get("unseen_boost", 2.0))
        min_weight = float(sampling.get("min_weight", 0.05))
        uncertainty_weight = float(sampling.get("uncertainty_weight", 0.7))
        pool = list(self.tasks)
        selected: list[BenchmarkTask] = []
        while pool and len(selected) < count:
            weights = []
            for task in pool:
                if self._task_visits.get(task.task_id, 0) == 0:
                    weights.append(max(min_weight, unseen_boost))
                    continue
                success = self._task_success_ema.get(task.task_id, 0.0)
                uncertainty = 4.0 * success * (1.0 - success)
                hardness = 1.0 - success
                weights.append(
                    max(
                        min_weight,
                        uncertainty_weight * uncertainty
                        + (1.0 - uncertainty_weight) * hardness,
                    )
                )
            chosen = self.rng.choices(pool, weights=weights, k=1)[0]
            selected.append(chosen)
            pool.remove(chosen)
        return selected

    def _update_task_sampler(self, task_id: str, group) -> None:
        episodes = set(group.episodes)
        successes = {
            event.episode_id
            for event in group.events
            if event.event_type == "terminal_success"
        }
        success_rate = len(successes & episodes) / max(1, len(episodes))
        alpha = float(self._task_sampling_config().get("ema_alpha", 0.2))
        previous = self._task_success_ema.get(task_id, success_rate)
        self._task_success_ema[task_id] = (1.0 - alpha) * previous + alpha * success_rate
        self._task_visits[task_id] = self._task_visits.get(task_id, 0) + 1

    def train(self) -> None:
        train_cfg = self.config["training"]
        optimizer_cfg = self.config.get("optimizer", {})
        eval_cfg = self.config.get("evaluation", {})
        num_updates = int(train_cfg.get("num_updates", 100))
        tasks_per_update = int(train_cfg.get("tasks_per_update", 8))
        group_size = int(train_cfg.get("group_size", 4))
        lr = float(train_cfg.get("learning_rate", 0.1))
        max_pending_age = int(self.config.get("credit", {}).get("max_pending_age", 8))
        credit_cfg = self.config.get("credit", {})
        advantage_mode = str(
            optimizer_cfg.get(
                "advantage_mode",
                train_cfg.get("advantage_mode", credit_cfg.get("advantage_mode", "step")),
            )
        )
        reward_unit_raw = str(
            train_cfg.get(
                "grpo_reward_unit",
                optimizer_cfg.get("reward_unit", credit_cfg.get("output", advantage_mode)),
            )
        )
        reward_unit = normalize_reward_unit(reward_unit_raw)
        step_advantage_estimator = str(
            optimizer_cfg.get(
                "step_advantage_estimator",
                train_cfg.get("step_advantage_estimator", "prompt_group"),
            )
        )
        credit_weight_floor = float(optimizer_cfg.get("credit_weight_floor", 0.05))
        max_credit_multiplier = float(optimizer_cfg.get("max_credit_multiplier", 4.0))
        residual_beta = float(optimizer_cfg.get("residual_beta", 0.5))
        residual_clip = float(optimizer_cfg.get("residual_clip", 2.0))
        residual_use_confidence = bool(
            optimizer_cfg.get("residual_use_confidence", True)
        )
        if reward_unit == "step" and step_advantage_estimator != "prompt_group" and group_size < 2:
            raise ValueError(
                "Step-level trajectory-grouped credit requires training.group_size >= 2."
            )
        optimizer_name = str(optimizer_cfg.get("name", train_cfg.get("optimizer", "grpo"))).lower()
        update_impl = str(optimizer_cfg.get("update_impl", train_cfg.get("update_impl", "standard_grpo"))).lower()
        update_backend = str(train_cfg.get("update_backend", optimizer_cfg.get("update_backend", "internal"))).lower()
        if optimizer_name != "grpo":
            raise ValueError("ECR-GRPO only supports training.optimizer='grpo'; change reward construction, not the optimizer.")
        if update_impl not in {"standard_grpo", "internal_standard_grpo"}:
            raise ValueError(
                "ECR-StepGRPO keeps the clipped GRPO update; select a supported "
                "advantage estimator instead of changing optimizer.update_impl."
            )
        if update_backend not in {"internal", "grpo_adapter", "samples_only"}:
            raise ValueError("training.update_backend must be 'internal', 'grpo_adapter', or 'samples_only'.")
        eval_every = int(eval_cfg.get("every_updates", 10))
        checkpoint_every = int(train_cfg.get("checkpoint_every", 0))
        start_update = self._prepare_output(lr=lr)

        for update_idx in range(start_update, num_updates + 1):
            chosen_tasks = self._sample_training_tasks(tasks_per_update)
            finalized_steps = []
            group_returns = []
            event_count = 0
            assignment_count = 0
            batch_assignments = []

            for task in chosen_tasks:
                group = collect_rollout_group(
                    group_id=f"upd_{update_idx:04d}_{task.task_id}",
                    task_id=task.task_id,
                    group_size=group_size,
                    env_factory=self._env_factory,
                    policy=self.policy,
                    kernel=self.kernel,
                    max_pending_age=max_pending_age,
                    max_steps=self.max_steps,
                    greedy=False,
                )
                finalized_steps.extend(group.steps)
                group_returns.append(sum(step.return_estimate for step in group.steps))
                self._update_task_sampler(task.task_id, group)
                event_count += len(group.events)
                assignment_count += len(group.assignments)
                batch_assignments.extend(group.assignments)
                for event in group.events:
                    append_jsonl(self.output_dir / "train_events.jsonl", event)
                for assignment in group.assignments:
                    append_jsonl(self.output_dir / "credit_assignments.jsonl", assignment)

            grpo_samples, grpo_stats = assign_grpo_advantages(
                finalized_steps,
                reward_unit=reward_unit,
                step_advantage_estimator=step_advantage_estimator,
                credit_weight_floor=credit_weight_floor,
                max_credit_multiplier=max_credit_multiplier,
                residual_beta=residual_beta,
                residual_clip=residual_clip,
                residual_use_confidence=residual_use_confidence,
            )
            if update_backend in {"grpo_adapter", "samples_only"}:
                stats = {"policy_loss": 0.0, "entropy": 0.0, "policy_updated": 0.0}
            else:
                stats = {**self._update_policy_with_grpo(finalized_steps, lr=lr), "policy_updated": 1.0}
            positive_credit = sum(1 for s in finalized_steps if s.return_estimate > 0)
            assignment_by_event = {}
            for assignment in batch_assignments:
                assignment_by_event.setdefault(assignment.event_id, assignment)
            event_assignments = list(assignment_by_event.values())
            attribution_entropy = (
                sum(item.weight_entropy for item in event_assignments) / max(1, len(event_assignments))
            )
            attribution_effective_steps = (
                sum(item.effective_steps for item in event_assignments) / max(1, len(event_assignments))
            )
            attribution_top_margin = (
                sum(item.top_margin for item in event_assignments) / max(1, len(event_assignments))
            )
            attribution_routing_confidence = (
                sum(item.routing_confidence for item in event_assignments)
                / max(1, len(event_assignments))
            )
            weak_routing_frac = sum(
                1 for item in event_assignments if item.routing_confidence < 0.2
            ) / max(1, len(event_assignments))
            route_counts: dict[str, int] = {}
            for assignment in event_assignments:
                route_counts[assignment.route] = route_counts.get(assignment.route, 0) + 1
            route_metrics = {
                f"route_frac_{route}": count / max(1, len(event_assignments))
                for route, count in sorted(route_counts.items())
            }
            row = {
                "update": update_idx,
                "kernel": self.kernel.name,
                "optimizer": optimizer_name,
                "update_impl": update_impl,
                "update_backend": update_backend,
                "advantage_mode": advantage_mode,
                "grpo_reward_unit": reward_unit,
                "step_advantage_estimator": step_advantage_estimator,
                "policy_update_score_mode": self.config.get("policy", {}).get("update_score_mode", ""),
                "task_sampling_mode": self._task_sampling_config().get("mode", "uniform"),
                "num_steps": len(finalized_steps),
                "num_events": event_count,
                "num_assignments": assignment_count,
                "avg_group_return": sum(group_returns) / max(1, len(group_returns)),
                "positive_credit_frac": positive_credit / max(1, len(finalized_steps)),
                "attribution_entropy": attribution_entropy,
                "attribution_effective_steps": attribution_effective_steps,
                "attribution_top_margin": attribution_top_margin,
                "attribution_routing_confidence": attribution_routing_confidence,
                "weak_routing_frac": weak_routing_frac,
                **route_metrics,
                **grpo_stats,
                **stats,
            }
            self.train_rows.append(row)
            write_csv(self.output_dir / "train_metrics.csv", self.train_rows)
            for sample in grpo_samples:
                append_jsonl(self.output_dir / "train_grpo_samples.jsonl", {"update": update_idx, **sample.to_json_dict()})
            for step in finalized_steps:
                append_jsonl(self.output_dir / "train_steps.jsonl", step)

            if update_idx == 1 or update_idx % eval_every == 0 or update_idx == num_updates:
                for eval_split in self._eval_splits():
                    eval_row = {"update": update_idx, **self.evaluate(eval_split)}
                    self.eval_rows.append(eval_row)
                    print(
                        f"update={update_idx:04d} kernel={self.kernel.name} "
                        f"split={eval_split} "
                        f"success={eval_row['success_rate']:.3f} "
                        f"steps={eval_row['avg_steps']:.1f} "
                        f"attr_entropy={row['attribution_entropy']:.3f} "
                        f"policy_entropy={row['entropy']:.3f}"
                    )
                write_csv(self.output_dir / "eval_metrics.csv", self.eval_rows)
                self._write_eval_action_rankings(update_idx)
                self._write_eval_traces(update_idx)
            archive = checkpoint_every > 0 and (
                update_idx % checkpoint_every == 0 or update_idx == num_updates
            )
            self._last_completed_update = update_idx
            self._save_checkpoint(update_idx, archive=archive)


        self.robustness_sweep()
        final_update = max(self._last_completed_update, num_updates)
        self._save_checkpoint(final_update, archive=False)
        self._write_completion_marker(final_update)

    def _update_policy_with_grpo(self, finalized_steps: list, lr: float) -> dict[str, float]:
        if hasattr(self.policy, "update_grpo"):
            return self.policy.update_grpo(finalized_steps, lr=lr)
        return self.policy.update(finalized_steps, lr=lr)

    def evaluate(self, eval_split: str | None = None) -> dict[str, float]:
        eval_split = eval_split or self._env_split(eval_mode=True)
        num_eval = int(self.config.get("evaluation", {}).get("num_eval_tasks", len(self.tasks)))
        tasks = self.eval_tasks_by_split[eval_split][:num_eval]
        metrics = evaluate_policy(
            tasks=tasks,
            env_factory=lambda task_id=None: self._env_factory(task_id=task_id, eval_mode=True),
            policy=self.policy,
            max_steps=self._max_steps(eval_mode=True),
            greedy=True,
        )
        metrics["num_eval_tasks"] = len(tasks)
        metrics["eval_async_delay_prob"] = float(self._async_config(eval_mode=True).get("delay_prob", 0.0))
        if self._env_name() == "alfworld":
            metrics["eval_split"] = eval_split
        return metrics

    def _write_eval_action_rankings(self, update_idx: int) -> None:
        if not hasattr(self.policy, "rank_actions"):
            return
        eval_cfg = self.config.get("evaluation", {})
        top_k = int(eval_cfg.get("rank_top_k", 5))
        num_tasks = int(eval_cfg.get("rank_num_tasks", min(6, len(self.eval_tasks))))
        if top_k <= 0 or num_tasks <= 0:
            return
        for task in self.eval_tasks[:num_tasks]:
            env: AsyncEnvWrapper = self._env_factory(task_id=task.task_id, eval_mode=True)
            obs = env.reset(task_id=task.task_id, episode_id=f"rank_{update_idx}_{task.task_id}")
            expected = None
            if hasattr(task, "sequence") and getattr(task, "sequence"):
                expected = getattr(task, "sequence")[0]
            benchmark_env = getattr(env, "env", None)
            append_jsonl(
                self.output_dir / "eval_action_rankings.jsonl",
                {
                    "update": update_idx,
                    "task_id": task.task_id,
                    "actual_task_id": getattr(benchmark_env, "actual_task_id", None),
                    "eval_split": self._env_split(eval_mode=True) if self._env_name() == "alfworld" else "",
                    "expected_action": expected,
                    "observation": obs,
                    "top_actions": self.policy.rank_actions(
                        obs,
                        action_space=list(env.action_space),
                        top_k=top_k,
                    ),
                },
            )

    def _write_eval_traces(self, update_idx: int) -> None:
        eval_cfg = self.config.get("evaluation", {})
        num_tasks = int(eval_cfg.get("trace_num_tasks", min(4, len(self.eval_tasks))))
        top_k = int(eval_cfg.get("trace_top_k", 3))
        if num_tasks <= 0:
            return
        for task in self.eval_tasks[:num_tasks]:
            env: AsyncEnvWrapper = self._env_factory(task_id=task.task_id, eval_mode=True)
            obs = env.reset(task_id=task.task_id, episode_id=f"trace_{update_idx}_{task.task_id}")
            benchmark_env = getattr(env, "env", None)
            trace = []
            success = False
            for step_id in range(self._max_steps(eval_mode=True)):
                action_space = list(env.action_space)
                top_actions = []
                if top_k > 0 and hasattr(self.policy, "rank_actions"):
                    top_actions = self.policy.rank_actions(obs, action_space=action_space, top_k=top_k)
                if top_actions:
                    action_text = str(top_actions[0]["action"])
                else:
                    action_text = self.policy.act(obs, action_space=action_space, greedy=True).text
                next_obs, reward, done, info = env.step(action_text)
                trace.append(
                    {
                        "step_id": step_id,
                        "observation": obs,
                        "action": action_text,
                        "reward": reward,
                        "env_reward": info.get("env_reward", reward),
                        "shaping_reward": info.get("shaping_reward", 0.0),
                        "positive_transition": info.get("positive_transition"),
                        "done": done,
                        "success": bool(info.get("success", False)),
                        "actual_task_id": info.get("actual_task_id"),
                        "task_type": info.get("task_type"),
                        "top_actions": top_actions,
                    }
                )
                obs = next_obs
                if done:
                    success = bool(info.get("success", False))
                    break
            append_jsonl(
                self.output_dir / "eval_traces.jsonl",
                {
                    "update": update_idx,
                    "task_id": task.task_id,
                    "actual_task_id": getattr(benchmark_env, "actual_task_id", None),
                    "eval_split": self._env_split(eval_mode=True) if self._env_name() == "alfworld" else "",
                    "success": success,
                    "trace": trace,
                },
            )

    def robustness_sweep(self) -> None:
        rows = []
        eval_cfg = self.config.setdefault("evaluation", {})
        base_eval_async = dict(eval_cfg.get("async", {}))
        for delay in self.config.get("evaluation", {}).get("delay_sweep", [0.0, 0.2, 0.4, 0.6]):
            eval_cfg["async"] = {**base_eval_async, "enabled": True, "delay_prob": delay}
            metrics = self.evaluate()
            rows.append({"delay_prob": delay, **metrics})
        if base_eval_async:
            eval_cfg["async"] = base_eval_async
        else:
            eval_cfg.pop("async", None)
        write_csv(self.output_dir / "robustness_sweep.csv", rows)

    def _env_factory(self, *, task_id: str | None = None, eval_mode: bool = False):
        base_seed = self.rng.randrange(10**9)
        env_name = self._env_name()
        wrapper_config = {
            **self._async_config(eval_mode=eval_mode),
            **self.config.get("credit", {}),
        }
        if env_name == "synthetic":
            base = SyntheticLongHorizonEnv(
                tasks=self.eval_tasks if eval_mode else self.tasks,
                action_space=self.action_space,
                max_steps=self._max_steps(eval_mode=eval_mode),
                seed=base_seed,
                non_local_credit=self.config["environment"].get("non_local_credit", {}),
            )
            return AsyncEnvWrapper(
                base,
                config=wrapper_config,
                seed=base_seed + 1,
            )
        if env_name == "external":
            if not task_id:
                raise ValueError("External benchmark env_factory requires a task_id")
            task = self._task_lookup[task_id]
            metadata = dict(task.metadata)
            env_cfg = self.config["environment"]
            base = ExternalTextBenchmarkEnv(
                factory_path=str(env_cfg["factory"]),
                split=str(metadata.get("split") or self._env_split(eval_mode=eval_mode)),
                task_metadata=metadata,
                fallback_action_space=self.action_space,
                factory_kwargs=dict(env_cfg.get("factory_kwargs", {})),
                seed=base_seed,
            )
            return AsyncEnvWrapper(base, config=wrapper_config, seed=base_seed + 1)
        if env_name == "alfworld":
            env_cfg = self.config["environment"]
            reuse = bool(env_cfg.get("reuse_env", True))
            task = self._task_lookup.get(task_id) if task_id else None
            metadata = dict(getattr(task, "metadata", {}) or {})
            split = str(metadata.get("split") or self._env_split(eval_mode=eval_mode))
            stable_task_id = str(task_id or metadata.get("task_id") or "__sequential__")
            cache_key = ("eval" if eval_mode else "train", split, stable_task_id)
            if reuse and cache_key in self._cached_alfworld_envs:
                self._cached_alfworld_envs[cache_key].config = wrapper_config
                return self._cached_alfworld_envs[cache_key]
            shaping_config = env_cfg.get("shaping", {})
            if eval_mode:
                shaping_config = self.config.get("evaluation", {}).get(
                    "shaping",
                    {"enabled": False},
                )
            base = ALFWorldEnv(
                alfworld_config=str(env_cfg.get("alfworld_config", "REPLACE_WITH_ALFWORLD_CONFIG.yaml")),
                split=split,
                fallback_action_space=self.action_space,
                shaping_config=shaping_config,
                catalog=self._get_alfworld_catalog(split),
                game_file=metadata.get("game_file"),
                task_metadata=metadata,
                history_turns=int(env_cfg.get("history_turns", 4)),
                seed=base_seed,
            )
            wrapped = AsyncEnvWrapper(
                base,
                config=wrapper_config,
                seed=base_seed + 1,
            )
            restored_rng = self._restored_alfworld_rng_states.pop(cache_key, None)
            if restored_rng is not None:
                wrapped.rng.setstate(restored_rng)
            if reuse:
                self._cached_alfworld_envs[cache_key] = wrapped
                if cache_key not in self._alfworld_cache_order:
                    self._alfworld_cache_order.append(cache_key)
                max_cached = max(1, int(env_cfg.get("max_cached_envs", 16)))
                while len(self._alfworld_cache_order) > max_cached:
                    oldest_key = self._alfworld_cache_order.pop(0)
                    self._cached_alfworld_envs.pop(oldest_key, None)
                    self._restored_alfworld_rng_states.pop(oldest_key, None)
            return wrapped
        raise ValueError(f"Unknown environment: {env_name}")

    def _get_alfworld_catalog(self, split: str) -> ALFWorldGameCatalog:
        if split not in self._alfworld_catalogs:
            env_cfg = self.config["environment"]
            self._alfworld_catalogs[split] = ALFWorldGameCatalog(
                alfworld_config=str(
                    env_cfg.get("alfworld_config", "REPLACE_WITH_ALFWORLD_CONFIG.yaml")
                ),
                split=split,
            )
        return self._alfworld_catalogs[split]

    def _async_config(self, *, eval_mode: bool = False) -> dict:
        cfg = dict(self.config.get("async", {}))
        if eval_mode:
            eval_cfg = self.config.get("evaluation", {})
            override = eval_cfg.get("async", eval_cfg.get("async_override", None))
            if override is not None:
                cfg.update(dict(override))
        if not bool(cfg.get("enabled", True)):
            cfg.update(
                {
                    "delay_prob": 0.0,
                    "timeout_prob": 0.0,
                    "interruption_prob": 0.0,
                    "missing_reward_prob": 0.0,
                    "terminal_reward_delay": 0,
                }
            )
        return cfg

    def _env_name(self) -> str:
        return str(self.config["environment"].get("name", "synthetic")).lower()

    def _env_split(self, *, eval_mode: bool = False) -> str:
        env_cfg = self.config["environment"]
        if eval_mode:
            eval_cfg = self.config.get("evaluation", {})
            return str(eval_cfg.get("split", env_cfg.get("eval_split", env_cfg.get("split", "eval_out_of_distribution"))))
        return str(env_cfg.get("train_split", env_cfg.get("split", "train")))

    def _eval_splits(self) -> list[str]:
        primary = self._env_split(eval_mode=True)
        configured = self.config.get("evaluation", {}).get("splits", [])
        if isinstance(configured, str):
            configured = configured.replace(",", " ").split()
        splits = [primary]
        for split in configured:
            value = str(split)
            if value and value not in splits:
                splits.append(value)
        return splits

    def _max_steps(self, *, eval_mode: bool = False) -> int:
        if eval_mode:
            return int(self.config.get("evaluation", {}).get("max_steps", self.max_steps))
        return self.max_steps

    def _build_tasks(self, *, eval_mode: bool = False, split_override: str | None = None):
        env_name = self._env_name()
        if env_name == "synthetic":
            return build_synthetic_tasks(self.config)
        if env_name == "external":
            return self._build_external_tasks(eval_mode=eval_mode, split_override=split_override)
        if env_name != "alfworld":
            raise ValueError(f"Unknown environment: {env_name}")
        env_cfg = self.config["environment"]
        eval_cfg = self.config.get("evaluation", {})
        if eval_mode:
            num_tasks = int(eval_cfg.get("num_eval_tasks", env_cfg.get("eval_num_tasks", env_cfg.get("num_tasks", 16))))
        else:
            num_tasks = int(env_cfg.get("num_train_tasks", env_cfg.get("num_tasks", 16)))
        split = split_override or self._env_split(eval_mode=eval_mode)
        catalog = self._get_alfworld_catalog(split)
        game_files = list(catalog.game_files)
        if num_tasks < len(game_files):
            split_rng = random.Random(f"alfworld:{self.seed}:{split}")
            game_files = sorted(split_rng.sample(game_files, k=num_tasks))
        tasks = []
        for game_file in game_files[:num_tasks]:
            metadata = catalog.task_metadata(game_file)
            tasks.append(
                BenchmarkTask(
                    task_id=metadata["task_id"],
                    metadata=metadata,
                )
            )
        if not tasks:
            raise RuntimeError(f"No ALFWorld tasks were selected for split '{split}'")
        return tasks


    def _build_external_tasks(
        self,
        *,
        eval_mode: bool,
        split_override: str | None,
    ) -> list[BenchmarkTask]:
        env_cfg = self.config["environment"]
        eval_cfg = self.config.get("evaluation", {})
        split = split_override or self._env_split(eval_mode=eval_mode)
        manifest_path = Path(str(env_cfg.get("task_manifest", ""))).expanduser()
        if not manifest_path.exists():
            raise FileNotFoundError(f"External task manifest does not exist: {manifest_path}")
        text = manifest_path.read_text(encoding="utf-8")
        if manifest_path.suffix.lower() == ".json":
            data = json.loads(text)
            records = data.get("tasks", []) if isinstance(data, dict) else data
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        selected = []
        for index, record in enumerate(records):
            metadata = dict(record)
            record_split = str(metadata.get("split", split))
            if record_split != split:
                continue
            actual_task_id = str(
                metadata.get("actual_task_id")
                or metadata.get("task_id")
                or metadata.get("id")
                or f"task_{index:05d}"
            )
            metadata["actual_task_id"] = actual_task_id
            metadata["split"] = record_split
            metadata.setdefault("task_type", "unknown")
            metadata["task_id"] = f"{record_split}__{actual_task_id}"
            selected.append(BenchmarkTask(task_id=metadata["task_id"], metadata=metadata))
        num_tasks = int(
            eval_cfg.get("num_eval_tasks", len(selected))
            if eval_mode
            else env_cfg.get("num_train_tasks", env_cfg.get("num_tasks", len(selected)))
        )
        if num_tasks < len(selected):
            manifest_rng = random.Random(f"external:{self.seed}:{split}")
            selected = manifest_rng.sample(selected, k=num_tasks)
        if not selected:
            raise RuntimeError(f"No external benchmark tasks found for split '{split}'")
        return selected[:num_tasks]

    def _prepare_output(self, *, lr: float) -> int:
        if self.resume_requested:
            return self._restore_training_state(lr=lr)

        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not self.overwrite_requested:
                raise FileExistsError(
                    f"Output directory is not empty: {self.output_dir}. "
                    "Use --resume to continue it or --overwrite to replace it."
                )
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()
        return 1

    def _write_config(self) -> None:
        (self.output_dir / "config.json").write_text(
            json.dumps(self.config, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def _config_signature(self) -> str:
        normalized = json.loads(json.dumps(self.config))
        normalized.pop("output_dir", None)
        training = normalized.setdefault("training", {})
        for key in ("resume", "num_updates", "checkpoint_every"):
            training.pop(key, None)
        payload = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _log_names() -> tuple[str, ...]:
        return (
            "train_events.jsonl",
            "credit_assignments.jsonl",
            "train_grpo_samples.jsonl",
            "train_steps.jsonl",
            "eval_action_rankings.jsonl",
            "eval_traces.jsonl",
        )

    def _log_offsets(self) -> dict[str, int]:
        return {
            name: (self.output_dir / name).stat().st_size
            if (self.output_dir / name).exists()
            else 0
            for name in self._log_names()
        }

    def _truncate_logs(self, offsets: dict[str, int]) -> None:
        for name in self._log_names():
            expected_size = int(offsets.get(name, 0))
            path = self.output_dir / name
            if not path.exists():
                if expected_size:
                    raise RuntimeError(f"Resume log is missing: {path}")
                continue
            if path.stat().st_size < expected_size:
                raise RuntimeError(
                    f"Resume log is shorter than its checkpoint offset: {path}"
                )
            with path.open("r+b") as handle:
                handle.truncate(expected_size)

    def _resume_candidates(self) -> list[Path]:
        ckpt_root = self.output_dir / "checkpoints"
        return [ckpt_root / "latest", ckpt_root / ".latest.prev"]

    def _restore_training_state(self, *, lr: float) -> int:
        failures = []
        expected_signature = self._config_signature()
        for checkpoint in self._resume_candidates():
            state_path = checkpoint / "trainer_state.pkl"
            if not state_path.exists():
                continue
            try:
                with state_path.open("rb") as handle:
                    state = pickle.load(handle)
                if state.get("config_signature") != expected_signature:
                    raise ValueError(
                        "Resume configuration differs from the checkpoint. Only "
                        "output_dir, training.num_updates, training.checkpoint_every, "
                        "and training.resume may change."
                    )
                if not hasattr(self.policy, "load"):
                    raise RuntimeError("The selected policy does not support resume loading")
                self.policy.load(str(checkpoint))
                training_state_path = checkpoint / "policy_training_state.bin"
                if hasattr(self.policy, "load_training_state"):
                    self.policy.load_training_state(str(training_state_path), lr=lr)
                self.rng.setstate(state["trainer_rng_state"])
                self._task_success_ema = dict(state.get("task_success_ema", {}))
                self._task_visits = dict(state.get("task_visits", {}))
                self._restored_alfworld_rng_states = dict(
                    state.get("alfworld_async_rng_states", {})
                )
                self._alfworld_cache_order = list(state.get("alfworld_cache_order", []))

                self.train_rows = list(state.get("train_rows", []))
                self.eval_rows = list(state.get("eval_rows", []))
                self._last_completed_update = int(state["last_completed_update"])
                target_updates = int(self.config.get("training", {}).get("num_updates", 0))
                if target_updates < self._last_completed_update:
                    raise ValueError(
                        f"training.num_updates={target_updates} is below checkpoint update "
                        f"{self._last_completed_update}"
                    )

                self._truncate_logs(dict(state.get("log_offsets", {})))
                write_csv(self.output_dir / "train_metrics.csv", self.train_rows)
                write_csv(self.output_dir / "eval_metrics.csv", self.eval_rows)
                completion = self.output_dir / "COMPLETED.json"
                if completion.exists():
                    completion.unlink()
                self._write_config()
                print(
                    f"[resume] checkpoint={checkpoint} "
                    f"last_update={self._last_completed_update}"
                )
                return self._last_completed_update + 1
            except Exception as exc:
                failures.append(f"{checkpoint}: {exc}")
        detail = "; ".join(failures) if failures else "no checkpoint found"
        raise RuntimeError(f"Unable to resume {self.output_dir}: {detail}")

    def _save_checkpoint(self, update_idx: int, *, archive: bool) -> None:
        if not hasattr(self.policy, "save"):
            raise RuntimeError("The selected policy does not support checkpoint saving")
        ckpt_root = self.output_dir / "checkpoints"
        ckpt_root.mkdir(parents=True, exist_ok=True)
        latest = ckpt_root / "latest"
        previous = ckpt_root / ".latest.prev"
        temporary = ckpt_root / ".latest.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        self.policy.save(str(temporary))
        if hasattr(self.policy, "save_training_state"):
            self.policy.save_training_state(str(temporary / "policy_training_state.bin"))
        state = {
            "version": 1,
            "last_completed_update": int(update_idx),
            "config_signature": self._config_signature(),
            "trainer_rng_state": self.rng.getstate(),
            "task_success_ema": self._task_success_ema,
            "task_visits": self._task_visits,
            "train_rows": self.train_rows,
            "alfworld_async_rng_states": {
                **self._restored_alfworld_rng_states,
                **{
                    key: env.rng.getstate()
                    for key, env in self._cached_alfworld_envs.items()
                },
            },
            "alfworld_cache_order": self._alfworld_cache_order,
            "eval_rows": self.eval_rows,
            "log_offsets": self._log_offsets(),
        }
        with (temporary / "trainer_state.pkl").open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if previous.exists():
            shutil.rmtree(previous)
        if latest.exists():
            latest.replace(previous)
        try:
            temporary.replace(latest)
        except Exception:
            if previous.exists() and not latest.exists():
                previous.replace(latest)
            raise
        if previous.exists():
            shutil.rmtree(previous)

        if archive:
            numbered = ckpt_root / f"update_{update_idx:04d}"
            if numbered.exists():
                shutil.rmtree(numbered)
            shutil.copytree(latest, numbered)

    def _write_completion_marker(self, update_idx: int) -> None:
        marker = self.output_dir / "COMPLETED.json"
        temporary = self.output_dir / ".COMPLETED.json.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "last_completed_update": int(update_idx),
                    "config_signature": self._config_signature(),
                    "checkpoint": "checkpoints/latest",
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(marker)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    trainer = ECRGRPOTrainer(
        load_config(args.config),
        resume=args.resume,
        overwrite=args.overwrite,
    )
    trainer.train()


if __name__ == "__main__":
    main()
