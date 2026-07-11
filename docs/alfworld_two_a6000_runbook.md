# ALFWorld 主线实验执行手册（2×A6000）

## 目标

主实验只回答：异步事件回填能否提升训练后 ALFWorld seen/OOD 任务成功率。方法保留三项：`grpo`、`gated`、`residual`。两张 GPU 按 kernel/seed 独立并行，不使用 DDP。

## 0. 服务器准备

```bash
cd /home/hjh/ecr_grpo_agent/ecr_grpo_agent
export PYTHONPATH=src:${PYTHONPATH:-}
export ALFWORLD_CONFIG=/home/hjh/ecr_grpo_agent/alfworld_src/configs/base_config.yaml
export MODEL_ID=/home/hjh/ecr_grpo_agent/ecr_grpo_agent/models/Qwen/Qwen2.5-1.5B-Instruct
export GPU_IDS="0 1"

nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
python -c "import alfworld, transformers, peft; print('dependencies ok')"
test -f "${ALFWORLD_CONFIG}"
test -d "${MODEL_ID}"
```

若 ALFWorld 数据不存在，只执行一次：

```bash
alfworld-download
```

## 1. 配置 dry-run

```bash
BASE_CONFIG=configs/alfworld_gated_lowmem_smoke.json \
OUTPUT_ROOT=runs/alfworld_plan \
KERNELS="gated residual" SEEDS="7" \
NUM_TRAIN_TASKS=2 NUM_EVAL_TASKS=2 MAX_STEPS=15 \
DRY_RUN=1 OVERWRITE=0 RESUME=0 \
bash scripts/run_alfworld_dual_gpu.sh
```

检查：

```bash
find runs/alfworld_plan/_generated_configs -type f -maxdepth 1 -print
grep -R 'trajectory_grouped_residual' runs/alfworld_plan/_generated_configs
grep -R 'REPLACE_WITH' runs/alfworld_plan/_generated_configs && echo "unexpected placeholder"
```

## 2. 双卡真实 smoke

```bash
BASE_CONFIG=configs/alfworld_gated_lowmem_smoke.json \
OUTPUT_ROOT=runs/alfworld_smoke_v2 \
KERNELS="gated residual" SEEDS="7" \
NUM_TRAIN_TASKS=2 NUM_EVAL_TASKS=2 MAX_STEPS=15 \
NUM_UPDATES=2 TASKS_PER_UPDATE=1 GROUP_SIZE=2 EVAL_EVERY=1 \
DRY_RUN=0 OVERWRITE=1 RESUME=0 \
bash scripts/run_alfworld_dual_gpu.sh
```

另开终端监控显存：

```bash
nvidia-smi dmon -s pucm -d 2
```

训练结束检查：

```bash
find runs/alfworld_smoke_v2 -name COMPLETED.json -print
tail -n 5 runs/alfworld_smoke_v2/gated/seed=7/train_metrics.csv
tail -n 5 runs/alfworld_smoke_v2/residual/seed=7/train_metrics.csv
tail -n 5 runs/alfworld_smoke_v2/gated/seed=7/eval_metrics.csv
tail -n 2 runs/alfworld_smoke_v2/residual/seed=7/eval_traces.jsonl
cat runs/alfworld_smoke_v2/alfworld_aggregate.csv
```

通过标准：

- 两个 `COMPLETED.json` 都存在，无 OOM、NaN 或 traceback。
- `avg_trajectory_group_size=2`。
- 至少一个 update 的 `zero_advantage_frac<1`。
- `num_events>0` 且 `num_assignments>0`。
- eval 同时包含 `eval_in_distribution` 和 `eval_out_of_distribution`。
- trace 中 `actual_task_id` 不是单纯的 `game.tw-pddl`。
- residual 的 `residual_active_frac>0`、`avg_abs_step_residual>0`。
- `attribution_routing_confidence` 应位于 `[0,1]`；`weak_routing_frac` 用于诊断，不设硬性优劣阈值。

## 3. 20-update 学习性检查

```bash
BASE_CONFIG=configs/alfworld_gated_benchmark.json \
OUTPUT_ROOT=runs/alfworld_pilot20 \
KERNELS="grpo gated residual" SEEDS="7" \
NUM_TRAIN_TASKS=100 NUM_EVAL_TASKS=50 MAX_STEPS=50 \
NUM_UPDATES=20 TASKS_PER_UPDATE=2 GROUP_SIZE=4 EVAL_EVERY=5 \
DRY_RUN=0 OVERWRITE=1 RESUME=0 \
bash scripts/run_alfworld_dual_gpu.sh
```

检查：

```bash
python scripts/summarize_alfworld.py --run-root runs/alfworld_pilot20
cat runs/alfworld_pilot20/alfworld_aggregate.csv
for method in grpo gated residual; do
  echo "===== ${method} ====="
  tail -n 6 "runs/alfworld_pilot20/${method}/seed=7/train_metrics.csv"
  tail -n 6 "runs/alfworld_pilot20/${method}/seed=7/eval_metrics.csv"
done
```

主要观察：seen/OOD success rate 是否上升、max-step rate 是否下降、`zero_advantage_frac` 是否长期为 1、`normalized_policy_entropy` 是否快速接近 0，以及 residual 是否实际激活。

若三种方法在 20 updates 后 seen/OOD 都为 0，先做第 4 节冷启动；否则跳到第 5 节。

## 4. 必要时做同步 warm-up

只训练一个同步 gated adapter：

```bash
BASE_CONFIG=configs/alfworld_gated_benchmark.json \
OUTPUT_ROOT=runs/alfworld_warmup \
KERNELS="gated" SEEDS="7" GPU_IDS="0" \
NUM_TRAIN_TASKS=100 NUM_EVAL_TASKS=50 MAX_STEPS=50 \
NUM_UPDATES=20 TASKS_PER_UPDATE=2 GROUP_SIZE=4 EVAL_EVERY=5 \
TRAIN_DELAY_PROB=0 TERMINAL_REWARD_DELAY=0 MISSING_REWARD_PROB=0 \
DRY_RUN=0 OVERWRITE=1 RESUME=0 \
bash scripts/run_alfworld_dual_gpu.sh
```

确认 `runs/alfworld_warmup/gated/seed=7/checkpoints/latest/adapter_config.json` 存在。随后让三个主方法从同一 adapter 开始：

```bash
export ADAPTER_PATH=/home/hjh/ecr_grpo_agent/ecr_grpo_agent/runs/alfworld_warmup/gated/seed=7/checkpoints/latest
```

## 5. 三 seed 主实验

先跑 60 updates；如果曲线仍上升，再用断点扩展到 125。

```bash
BASE_CONFIG=configs/alfworld_gated_benchmark.json \
OUTPUT_ROOT=runs/alfworld_mainline \
KERNELS="grpo gated residual" SEEDS="7 13 21" \
NUM_TRAIN_TASKS=500 NUM_EVAL_TASKS=50 MAX_STEPS=50 \
NUM_UPDATES=60 TASKS_PER_UPDATE=2 GROUP_SIZE=4 EVAL_EVERY=20 \
DRY_RUN=0 OVERWRITE=1 RESUME=0 \
bash scripts/run_alfworld_dual_gpu.sh
```

中断后原命令只改：

```bash
OVERWRITE=0 RESUME=1
```

若 60 updates 曲线仍明显上升，保持其他参数完全一致并扩展：

```bash
BASE_CONFIG=configs/alfworld_gated_benchmark.json \
OUTPUT_ROOT=runs/alfworld_mainline \
KERNELS="grpo gated residual" SEEDS="7 13 21" \
NUM_TRAIN_TASKS=500 NUM_EVAL_TASKS=50 MAX_STEPS=50 \
NUM_UPDATES=125 TASKS_PER_UPDATE=2 GROUP_SIZE=4 EVAL_EVERY=20 \
DRY_RUN=0 OVERWRITE=0 RESUME=1 \
bash scripts/run_alfworld_dual_gpu.sh
```

## 6. 最终结果读取

```bash
python scripts/summarize_alfworld.py --run-root runs/alfworld_mainline
cat runs/alfworld_mainline/alfworld_runs.csv
cat runs/alfworld_mainline/alfworld_aggregate.csv
find runs/alfworld_mainline -name COMPLETED.json | wc -l
```

完整三方法三 seed 应有 9 个 `COMPLETED.json`。最终首先比较：

1. `final_success_ood_mean`；
2. `final_success_seen_mean`；
3. success curve 是否更早上升；
4. `final_avg_steps_mean` 与 max-step rate；
5. gated 到 residual 的增益是否伴随非零 `residual_active_frac`。

不要用 attribution confidence 代替任务成功率；它只解释机制是否工作。
