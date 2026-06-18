from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from ecr_grpo.advantages import compute_group_advantages
from ecr_grpo.credit_kernels import build_credit_kernel
from ecr_grpo.envs.alfworld_wrapper import ALFWorldEnv
from ecr_grpo.envs.async_wrapper import AsyncEnvWrapper
from ecr_grpo.envs.synthetic import SyntheticLongHorizonEnv, build_synthetic_tasks
from ecr_grpo.eval import evaluate_policy
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
        self.train_rows: list[dict] = []
        self.eval_rows: list[dict] = []

    def train(self) -> None:
        self._prepare_output()
        train_cfg = self.config["training"]
        eval_cfg = self.config.get("evaluation", {})
        num_updates = int(train_cfg.get("num_updates", 100))
        tasks_per_update = int(train_cfg.get("tasks_per_update", 8))
        group_size = int(train_cfg.get("group_size", 4))
        lr = float(train_cfg.get("learning_rate", 0.1))
        max_pending_age = int(self.config.get("credit", {}).get("max_pending_age", 8))
        eval_every = int(eval_cfg.get("every_updates", 10))

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

            compute_group_advantages(finalized_steps)
            stats = self.policy.update(finalized_steps, lr=lr)
            positive_credit = sum(1 for s in finalized_steps if s.return_estimate > 0)
            causal_credit_mass = sum(
                max(0.0, s.return_estimate)
                for s in finalized_steps
                if s.metadata.get("causal_action")
            )
            total_positive_mass = sum(max(0.0, s.return_estimate) for s in finalized_steps)
            row = {
                "update": update_idx,
                "kernel": self.kernel.name,
                "num_steps": len(finalized_steps),
                "num_events": event_count,
                "num_assignments": assignment_count,
                "avg_group_return": sum(group_returns) / max(1, len(group_returns)),
                "positive_credit_frac": positive_credit / max(1, len(finalized_steps)),
                "credit_mass_on_causal_steps": causal_credit_mass / max(total_positive_mass, 1e-8),
                **stats,
            }
            self.train_rows.append(row)
            write_csv(self.output_dir / "train_metrics.csv", self.train_rows)
            for step in finalized_steps:
                append_jsonl(self.output_dir / "train_steps.jsonl", step)

            if update_idx == 1 or update_idx % eval_every == 0 or update_idx == num_updates:
                eval_row = {"update": update_idx, **self.evaluate()}
                self.eval_rows.append(eval_row)
                write_csv(self.output_dir / "eval_metrics.csv", self.eval_rows)
                print(
                    f"update={update_idx:04d} kernel={self.kernel.name} "
                    f"success={eval_row['success_rate']:.3f} "
                    f"credit_causal={row['credit_mass_on_causal_steps']:.3f} "
                    f"entropy={row['entropy']:.3f}"
                )

        self.robustness_sweep()

    def evaluate(self) -> dict[str, float]:
        num_eval = int(self.config.get("evaluation", {}).get("num_eval_tasks", len(self.tasks)))
        return evaluate_policy(
            tasks=self.tasks[:num_eval],
            env_factory=self._env_factory,
            policy=self.policy,
            max_steps=self.max_steps,
            greedy=True,
        )

    def robustness_sweep(self) -> None:
        rows = []
        base_async = dict(self.config.get("async", {}))
        for delay in self.config.get("evaluation", {}).get("delay_sweep", [0.0, 0.2, 0.4, 0.6]):
            self.config["async"] = {**base_async, "delay_prob": delay}
            metrics = self.evaluate()
            rows.append({"delay_prob": delay, **metrics})
        self.config["async"] = base_async
        write_csv(self.output_dir / "robustness_sweep.csv", rows)

    def _env_factory(self):
        base_seed = self.rng.randrange(10**9)
        env_name = str(self.config["environment"].get("name", "synthetic")).lower()
        if env_name == "synthetic":
            base = SyntheticLongHorizonEnv(
                tasks=self.tasks,
                action_space=self.action_space,
                max_steps=self.max_steps,
                seed=base_seed,
            )
        elif env_name == "alfworld":
            env_cfg = self.config["environment"]
            base = ALFWorldEnv(
                alfworld_config=str(env_cfg.get("alfworld_config", "REPLACE_WITH_ALFWORLD_CONFIG.yaml")),
                split=str(env_cfg.get("split", "eval_out_of_distribution")),
                fallback_action_space=self.action_space,
                seed=base_seed,
            )
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        return AsyncEnvWrapper(
            base,
            config={**self.config.get("async", {}), **self.config.get("credit", {})},
            seed=base_seed + 1,
        )

    def _build_tasks(self):
        env_name = str(self.config["environment"].get("name", "synthetic")).lower()
        if env_name == "synthetic":
            return build_synthetic_tasks(self.config)
        num_tasks = int(self.config["environment"].get("num_tasks", 16))
        prefix = env_name
        return [BenchmarkTask(task_id=f"{prefix}_{i:04d}") for i in range(num_tasks)]

    def _prepare_output(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.json").write_text(
            json.dumps(self.config, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    trainer = ECRGRPOTrainer(load_config(args.config))
    trainer.train()


if __name__ == "__main__":
    main()
