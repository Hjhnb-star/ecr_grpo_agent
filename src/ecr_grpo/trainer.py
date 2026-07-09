from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from ecr_grpo.credit_kernels import build_credit_kernel
from ecr_grpo.envs.alfworld_wrapper import ALFWorldEnv
from ecr_grpo.envs.async_wrapper import AsyncEnvWrapper
from ecr_grpo.envs.synthetic import SyntheticLongHorizonEnv, build_synthetic_tasks
from ecr_grpo.eval import evaluate_policy
from ecr_grpo.grpo_adapter import assign_grpo_advantages, normalize_reward_unit
from ecr_grpo.io import append_jsonl, ensure_dir, load_config, write_csv
from ecr_grpo.policies import build_policy
from ecr_grpo.rollout import collect_rollout_group
from ecr_grpo.types import BenchmarkTask


class ECRGRPOTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.seed = int(config.get("seed", 0))
        self.rng = random.Random(self.seed)
        self.output_dir = ensure_dir(config.get("output_dir", "runs/smoke"))
        self.tasks = self._build_tasks()
        self.action_space = list(config["environment"]["action_space"])
        self.max_steps = int(config["environment"].get("max_steps", 10))
        self.kernel = build_credit_kernel(config.get("credit", {}))
        self.policy = build_policy(config, self.action_space, seed=self.seed)
        self.eval_tasks = self._build_tasks(eval_mode=True)
        self._cached_alfworld_envs = {}
        self.train_rows: list[dict] = []
        self.eval_rows: list[dict] = []

    def train(self) -> None:
        self._prepare_output()
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
        optimizer_name = str(optimizer_cfg.get("name", train_cfg.get("optimizer", "grpo"))).lower()
        update_impl = str(optimizer_cfg.get("update_impl", train_cfg.get("update_impl", "standard_grpo"))).lower()
        update_backend = str(train_cfg.get("update_backend", optimizer_cfg.get("update_backend", "internal"))).lower()
        if optimizer_name != "grpo":
            raise ValueError("ECR-GRPO only supports training.optimizer='grpo'; change reward construction, not the optimizer.")
        if update_impl not in {"standard_grpo", "internal_standard_grpo"}:
            raise ValueError("ECR-GRPO keeps optimizer.update_impl='standard_grpo'; ECR only constructs rewards.")
        if update_backend not in {"internal", "grpo_adapter", "samples_only"}:
            raise ValueError("training.update_backend must be 'internal', 'grpo_adapter', or 'samples_only'.")
        eval_every = int(eval_cfg.get("every_updates", 10))
        checkpoint_every = int(train_cfg.get("checkpoint_every", 0))

        for update_idx in range(1, num_updates + 1):
            chosen_tasks = self.rng.sample(self.tasks, k=min(tasks_per_update, len(self.tasks)))
            finalized_steps = []
            group_returns = []
            event_count = 0
            assignment_count = 0

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
                event_count += len(group.events)
                assignment_count += len(group.assignments)
                for event in group.events:
                    append_jsonl(self.output_dir / "train_events.jsonl", event)
                for assignment in group.assignments:
                    append_jsonl(self.output_dir / "credit_assignments.jsonl", assignment)

            grpo_samples, grpo_stats = assign_grpo_advantages(
                finalized_steps,
                reward_unit=reward_unit,
            )
            if update_backend in {"grpo_adapter", "samples_only"}:
                stats = {"policy_loss": 0.0, "entropy": 0.0, "policy_updated": 0.0}
            else:
                stats = {**self._update_policy_with_grpo(finalized_steps, lr=lr), "policy_updated": 1.0}
            positive_credit = sum(1 for s in finalized_steps if s.return_estimate > 0)
            causal_credit_mass = sum(
                max(0.0, s.return_estimate)
                for s in finalized_steps
                if s.diagnostic_metadata.get("causal_action")
            )
            total_positive_mass = sum(max(0.0, s.return_estimate) for s in finalized_steps)
            row = {
                "update": update_idx,
                "kernel": self.kernel.name,
                "optimizer": optimizer_name,
                "update_impl": update_impl,
                "update_backend": update_backend,
                "advantage_mode": advantage_mode,
                "grpo_reward_unit": reward_unit,
                "num_steps": len(finalized_steps),
                "num_events": event_count,
                "num_assignments": assignment_count,
                "avg_group_return": sum(group_returns) / max(1, len(group_returns)),
                "positive_credit_frac": positive_credit / max(1, len(finalized_steps)),
                "credit_mass_on_causal_steps": causal_credit_mass / max(total_positive_mass, 1e-8),
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
                eval_row = {"update": update_idx, **self.evaluate()}
                self.eval_rows.append(eval_row)
                write_csv(self.output_dir / "eval_metrics.csv", self.eval_rows)
                self._write_eval_action_rankings(update_idx)
                self._write_eval_traces(update_idx)
                print(
                    f"update={update_idx:04d} kernel={self.kernel.name} "
                    f"success={eval_row['success_rate']:.3f} "
                    f"acc={eval_row.get('action_accuracy', 0.0):.3f} "
                    f"progress={eval_row.get('avg_progress_fraction', 0.0):.3f} "
                    f"credit_causal={row['credit_mass_on_causal_steps']:.3f} "
                    f"entropy={row['entropy']:.3f}"
                )
            if checkpoint_every > 0 and (update_idx % checkpoint_every == 0 or update_idx == num_updates):
                self._save_checkpoint(update_idx)

        self.robustness_sweep()
        self._save_checkpoint(num_updates, latest_only=True)

    def _update_policy_with_grpo(self, finalized_steps: list, lr: float) -> dict[str, float]:
        if hasattr(self.policy, "update_grpo"):
            return self.policy.update_grpo(finalized_steps, lr=lr)
        return self.policy.update(finalized_steps, lr=lr)

    def evaluate(self) -> dict[str, float]:
        num_eval = int(self.config.get("evaluation", {}).get("num_eval_tasks", len(self.tasks)))
        tasks = self.eval_tasks[:num_eval]
        metrics = evaluate_policy(
            tasks=tasks,
            env_factory=lambda: self._env_factory(eval_mode=True),
            policy=self.policy,
            max_steps=self._max_steps(eval_mode=True),
            greedy=True,
        )
        metrics["num_eval_tasks"] = len(tasks)
        metrics["eval_async_delay_prob"] = float(self._async_config(eval_mode=True).get("delay_prob", 0.0))
        if self._env_name() == "alfworld":
            metrics["eval_split"] = self._env_split(eval_mode=True)
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
            env: AsyncEnvWrapper = self._env_factory(eval_mode=True)
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
            env: AsyncEnvWrapper = self._env_factory(eval_mode=True)
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
                        "expected_action": info.get("expected_action"),
                        "action": action_text,
                        "correct": bool(info.get("causal_action", False)),
                        "reward": reward,
                        "progress": info.get("progress"),
                        "done": done,
                        "success": bool(info.get("success", False)),
                        "actual_task_id": info.get("actual_task_id"),
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

    def _env_factory(self, *, eval_mode: bool = False):
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
        if env_name == "alfworld":
            env_cfg = self.config["environment"]
            reuse = bool(env_cfg.get("reuse_env", True))
            split = self._env_split(eval_mode=eval_mode)
            cache_key = ("eval" if eval_mode else "train", split)
            if reuse and cache_key in self._cached_alfworld_envs:
                self._cached_alfworld_envs[cache_key].config = wrapper_config
                return self._cached_alfworld_envs[cache_key]
            base = ALFWorldEnv(
                alfworld_config=str(env_cfg.get("alfworld_config", "REPLACE_WITH_ALFWORLD_CONFIG.yaml")),
                split=split,
                fallback_action_space=self.action_space,
                shaping_config=env_cfg.get("shaping", {}),
                seed=base_seed,
            )
            wrapped = AsyncEnvWrapper(
                base,
                config=wrapper_config,
                seed=base_seed + 1,
            )
            if reuse:
                self._cached_alfworld_envs[cache_key] = wrapped
            return wrapped
        raise ValueError(f"Unknown environment: {env_name}")

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

    def _max_steps(self, *, eval_mode: bool = False) -> int:
        if eval_mode:
            return int(self.config.get("evaluation", {}).get("max_steps", self.max_steps))
        return self.max_steps

    def _build_tasks(self, *, eval_mode: bool = False):
        env_name = self._env_name()
        if env_name == "synthetic":
            return build_synthetic_tasks(self.config)
        env_cfg = self.config["environment"]
        eval_cfg = self.config.get("evaluation", {})
        if eval_mode:
            num_tasks = int(eval_cfg.get("num_eval_tasks", env_cfg.get("eval_num_tasks", env_cfg.get("num_tasks", 16))))
            prefix = f"{env_name}_eval"
        else:
            num_tasks = int(env_cfg.get("num_train_tasks", env_cfg.get("num_tasks", 16)))
            prefix = f"{env_name}_train"
        return [BenchmarkTask(task_id=f"{prefix}_{i:04d}") for i in range(num_tasks)]

    def _prepare_output(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.json").write_text(
            json.dumps(self.config, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def _save_checkpoint(self, update_idx: int, *, latest_only: bool = False) -> None:
        if not hasattr(self.policy, "save"):
            return
        ckpt_root = self.output_dir / "checkpoints"
        ckpt_root.mkdir(parents=True, exist_ok=True)
        latest = ckpt_root / "latest"
        if latest.exists():
            shutil.rmtree(latest)
        self.policy.save(str(latest))
        if latest_only:
            return
        numbered = ckpt_root / f"update_{update_idx:04d}"
        if numbered.exists():
            shutil.rmtree(numbered)
        self.policy.save(str(numbered))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    trainer = ECRGRPOTrainer(load_config(args.config))
    trainer.train()


if __name__ == "__main__":
    main()
