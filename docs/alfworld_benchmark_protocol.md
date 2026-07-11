# ALFWorld 与外部 Benchmark 实验协议

## 目标与边界

本仓库现在把 ALFWorld 作为固定真实 game 的训练/测试协议，而不是顺序轮转环境的 wrapper smoke。目标是提高 ECR-GRPO 在异步长程反馈下的训练效果，并与相同模型、数据预算、rollout 预算和优化器的基线公平比较。代码改进不能预先保证 SOTA；最终结论必须来自多 seed 真实结果。

## 本轮关键修复

1. 每个 rollout group 绑定同一个真实 ALFWorld game_file，train、seen eval、OOD eval 使用各自固定 manifest。
2. ECR step reward 默认使用 trajectory_grouped_credit：先在同任务轨迹组上计算标准化 trajectory advantage，再按轨迹内 ECR credit mass 分配 step gradient。
3. reportable step 配置要求 group_size >= 2；group_size=1 会在配置验证阶段失败。
4. streaming_distribution 先无梯度计算完整候选 softmax，再逐候选重算精确一阶梯度，避免 full_distribution 的峰值显存。
5. ALFWorld 累计 score 转为增量 reward；成功终局只产生一个 terminal event，避免重复奖励。
6. source_time、source action 和精确 related step 默认只保存在 diagnostic_metadata，credit kernel 不可见。
7. 延迟期间的新步骤仍进入候选集合；gated kernel 会把 delayed partial feedback 路由到 evidence/recency 混合。
8. prompt 始终保留任务目标和有限交互历史，不再重复拼接 admissible actions。
9. success-aware sampler 优先未见、学习边界和较难任务；所有对比方法共享相同 sampler。

## 指标

主要任务指标：seen success rate、OOD success rate、六类任务 success rate、平均步数、成功轨迹平均步数、平均 token、原始环境 return、failure rate、max-step rate。

训练效率指标：learning-curve AUC、达到固定 success threshold 的更新数/环境步数/token、wall-clock time、峰值 GPU 显存。

信用诊断：zero_advantage_frac、normalized_policy_entropy、effective_action_count、attribution_entropy、attribution_effective_steps、attribution_top_margin 和 event route 分布。

ALFWorld 没有真实 causal-step 标签。因此 positive_transition 和 attribution 统计只能叫诊断或 proxy，不能叫 causal accuracy。真实 causal localization 只在带隐藏标签的合成或人工注入诊断中报告。

## Linux Smoke

```bash
cd /home/hjh/ecr_grpo_agent/ecr_grpo_agent
export PYTHONPATH=src:${PYTHONPATH:-}
export ALFWORLD_CONFIG=/home/hjh/ecr_grpo_agent/alfworld_src/configs/base_config.yaml
export MODEL_ID=/home/hjh/ecr_grpo_agent/ecr_grpo_agent/models/Qwen/Qwen2.5-1.5B-Instruct

BASE_CONFIG=configs/alfworld_gated_lowmem_smoke.json \
OUTPUT_ROOT=runs/alfworld_lowmem_smoke \
DRY_RUN=0 OVERWRITE=1 GPU_ID=1 \
KERNELS="gated" SEEDS="7" \
NUM_TRAIN_TASKS=2 NUM_EVAL_TASKS=2 MAX_STEPS=15 \
bash scripts/run_alfworld_server.sh
```

Smoke 成功标准：无 OOM；train_metrics.csv 中 avg_trajectory_group_size=2；不是所有更新都长期保持 zero_advantage_frac=1；seen/OOD eval 行都存在；actual_task_id 不是 game.tw-pddl。

## 正式 ALFWorld 对比

```bash
BASE_CONFIG=configs/alfworld_gated_benchmark.json \
OUTPUT_ROOT=runs/alfworld_fair \
DRY_RUN=0 OVERWRITE=1 GPU_ID=1 \
KERNELS="grpo local recency evidence gated" \
SEEDS="7 13 21" \
NUM_TRAIN_TASKS=500 NUM_EVAL_TASKS=9999 MAX_STEPS=50 \
bash scripts/run_alfworld_server.sh
```

local 是 latest-step 工程基线，不是 StepPO 官方复现。与 STEP、StepPO、HiPER、DPEPO、Tool Orchestration 比较时，应运行其官方代码或严格复现其公开协议；本仓库只复用相同 base model 和预算进行横向表格汇总。

## 其他真实数据集

environment.name=external 时，配置 environment.task_manifest 和 environment.factory。manifest 支持 JSON 或 JSONL；每条任务至少包含 task_id、split，可选 task_type。factory 使用 module:function 形式，接收 split、task、task_id、seed，返回具有 reset()、step(action) 和文本 action space 的环境。

ScienceWorld 适合验证多种科学任务和 variation 泛化；WebShop 适合稀疏终局奖励与网页动作；函数/工具 benchmark 适合非局部工具返回和 orchestration reward。新增数据集时保持固定任务 manifest、相同模型、相同 seed、相同 rollout/token 预算，并分别报告 native 与 async-perturbed 结果。

## 结果文件

- eval_metrics.csv：每个 update、每个 eval split 一行，含动态任务类别列。
- train_metrics.csv：policy loss、ratio、两类 entropy、优势退化和 attribution 诊断。
- eval_traces.jsonl：固定 OOD 任务的动作轨迹、raw/shaping reward 和真实 game id。
- credit_assignments.jsonl：每个事件的路由、权重、entropy、effective steps 和 top margin。
- alfworld_summary.csv：每个 kernel/seed 的 seen/OOD 最终结果。

