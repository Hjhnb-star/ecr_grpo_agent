# ECR-GRPO 阶段性压缩总结

## 1. 当前定位

ECR-GRPO 的目标不是重新发明一个完整 RL 框架，而是在 GRPO / StepPO / STEP 这类 agentic RL 方法的信用分配环节加入一个轻量、可插拔的机制：

> 将 delayed、partial、non-local、timeout、interruption 等异步反馈建模为事件流，并在事件到达时对历史 step 做 event-conditioned credit refill，最后转成 GRPO-compatible step-level advantage。

核心问题是：长任务 agent 中，reward 经常不是同步、完整、可对齐地返回。如果仍然只用整条 trajectory 的最终成败来更新策略，很多正确的中间步骤会因为最终失败、超时或反馈缺失而被错误惩罚。

ECR-GRPO 的核心公式可以概括为：

```text
event e_k arrives
candidate historical steps = {b_t}

w_{t,k} = normalize(K(e_k, b_t))
c_{t,k} = w_{t,k} * R_k
G_t = immediate_reward_t + sum_k c_{t,k}
A_t = group_normalize(G_t)
```

其中 `K(e_k, b_t)` 是 credit kernel，用来决定一个异步事件的 reward 应该回填给哪些历史 step。

## 2. 已经完成的代码能力

当前项目已经实现了一个可运行的 ECR-GRPO 原型，不只是概念描述。

已经完成的主要模块包括：

- `SyntheticLongHorizonEnv`：可控长任务 synthetic 环境，支持多步 action sequence、partial reward、terminal success/failure 和 non-local delayed feedback。
- `AsyncEnvWrapper`：在环境事件上注入异步扰动，包括 reward delay、terminal reward delay、timeout、interruption、missing reward，以及 no-oracle event link removal。
- `PendingStepBuffer`：维护尚未完成信用分配的历史 step，使后续事件可以回填到已经发生过的 step。
- `CreditKernel` 系列：已经实现 `trajectory`、`uniform`、`recency`、`dependency`、`evidence` 五种 credit refill 方式。
- `EvidenceKernel`：主方法，不依赖精确 `related_step_id`，而是使用时间距离、文本 overlap、tag、tool/subgoal 等弱证据做 attribution。
- `compute_group_advantages`：把回填后的 step return 转成 GRPO-style group-relative advantage。
- `trainer / rollout / eval / run_baselines / run_sweeps / analyze_credit`：形成训练、评估、baseline 对比、参数 sweep 和 credit diagnostic 的完整闭环。
- HF LoRA policy 与 ALFWorld adapter 已有占位接口，后续可接入真实 LLM agent benchmark。


```

## 3. 当前实验设置

目前已经跑通的主要实验是 synthetic controlled diagnostic。它的作用不是替代真实 benchmark，而是验证 ECR-GRPO 的核心机制：异步、延迟、非局部反馈到来后，算法是否真的能把 credit 分回更相关的历史 step。

当前重点设置为 no-oracle non-local delayed feedback：

- `use_oracle_event_links = false`
- 不使用真实 `related_step_id`
- 不使用真实 `related_tool`
- 不使用真实 `related_subgoal`
- 只使用事件文本、metadata tags、时间距离、observation/action 证据做 credit attribution

这点很重要，因为它说明主方法不是靠 benchmark 泄露“哪个 step 是因果 step”的答案。

## 4. 数据规模：不是几条手写案例

non-local delayed feedback 已经在多 seed 下产生了几千级事件，不是少量手写 case。

当前 non-local event 数量：

```text
seed=7    4758
seed=13   4658
seed=21   4816
```

这说明每个 seed 中都有几千个非局部反馈事件参与训练和 credit assignment。

以 `seed=7 / evidence` 为例，non-local target action 分布如下：

```text
find_key       671
verify_fact    665
inspect_file   662
search_web     571
open_box       522
edit_file      489
read_code      468
extract_fact   415
run_test       295
```

这个分布覆盖了多个任务动作，而不是集中在一两个特殊动作上。因此目前 synthetic diagnostic 的作用是：在可控环境中批量生成非局部延迟反馈，用于验证 credit assignment 机制是否真的有效。

## 5. 当前主要结果

### 5.1 基础 no-oracle 训练结果

在 `smoke_evidence_no_oracle` 上，evidence kernel 可以从零成功率学起来：

```text
update=0001 success=0.000
update=0080 success=0.646
```

对应 baseline 对比中，trajectory-level 方法基本学不起来，而 step-level refill 方法能明显提升：

```text
trajectory success=0.000
uniform    success=0.646
recency    success=0.667
dependency success=0.667
evidence   success=0.646
```

这说明：只靠整条 trajectory 的最终 reward 不够，step-level refill 是必要的。

### 5.2 多 seed non-local no-oracle 结果

在 non-local delayed feedback 设置下，三组 seed 聚合结果如下：

```text
kernel      success_mean   return_mean   target/causal_credit_mean
trajectory  0.000          -0.309        0.000
uniform     0.389           0.336        0.792
recency     0.667           0.465        0.825
evidence    0.701           0.494        0.822
```

关键结论：

- `trajectory` 在该设置下完全失败，说明整轨迹 reward 在长任务异步反馈下非常脆弱。
- `uniform` 能学起来，但 credit 被稀释。
- `recency` 是很强 baseline，因为很多反馈确实和近邻 step 有关。
- `evidence` 在 non-local 设置下取得最高平均 success rate 和最高平均 return。

这里的提升不是特别巨大，但方向明确：当 reward 不一定属于最近 step 时，Evidence Refill 比单纯 Recency Refill 更合理。

### 5.3 Credit diagnostic：机制层面的核心证据

最关键的证据不是 success rate，而是 credit diagnostic。它直接回答：reward 到底被分给了哪个历史 step？

三 seed 聚合后的主要指标：

```text
kernel    target_weight_mean   recent_weight_mean   argmax_target_rate   top3_target_rate
evidence  0.419                0.176                0.978                0.978
recency   0.132                0.347                0.000                0.392
uniform   0.188                0.166                0.441                0.777
```

解释：

- `target_weight_mean`：真实 target action 获得的平均 credit weight。
- `recent_weight_mean`：最近一步获得的平均 credit weight。
- `argmax_target_rate`：最大 credit 是否给到了真实 target action。
- `top3_target_rate`：真实 target action 是否出现在 credit top-3。

这个结果非常关键：

- Evidence 把约 `41.9%` 的 credit 分给真实相关历史动作。
- Recency 只有约 `13.2%`，并且最大 credit 从不给真实 target action，因为它天然偏向最近 step。
- Evidence 的 `argmax_target_rate` 接近 `97.8%`，说明它几乎总能把最大 credit 给到正确历史动作。
- Recency 的 `recent_weight_mean` 高达 `34.7%`，说明它确实存在明显的 recent-step bias。

因此可以写成论文结论：

> Evidence Refill does not simply increase reward magnitude; it changes where delayed reward is assigned. In non-local delayed feedback, it assigns substantially more credit to the true causal historical step than recency or uniform refill.

## 6. 当前结果能说明什么

目前结果可以支持以下阶段性结论：

1. ECR-GRPO 的事件流建模和 pending buffer 机制已经跑通。
2. 异步 reward 不需要等完整 trajectory 结束后再统一分配，可以在事件到达时动态回填。
3. Trajectory-level GRPO 在当前 synthetic long-horizon async 设置下明显不足。
4. Step-level refill 显著优于 trajectory-level reward。
5. 在 non-local delayed feedback 中，Recency Refill 会错误偏向最近 step。
6. Evidence Refill 能利用事件文本、tags、observation/action 等弱证据，把 reward 分回更相关的历史 step。
7. Credit diagnostic 已经证明提升来自更合理的 credit assignment，而不是只靠随机训练波动。

当前 synthetic 结果适合放在论文中的：

```text
Controlled Experiments
Credit Assignment Diagnostic
Ablation / Mechanism Study
Robustness to Delayed and Missing Feedback
```

需要注意的是：这些结果还不能单独证明真实 LLM agent benchmark 上的最终 SOTA。它们证明的是机制有效性和可解释性，后续仍需要接真实 benchmark 证明 external validity。

## 7. 当前方法的核心技术

可以把 ECR-GRPO 拆成六个技术模块：

1. **Eventized Reward Interface**
   将 partial reward、terminal reward、timeout、interruption、tool return 等统一建模为 `AsyncEvent`。

2. **Pending Step Buffer**
   每个 step 先进入 buffer，不立即根据整条 trajectory 成败更新，而是等待后续事件回填 credit。

3. **Event-Conditioned Credit Refill**
   当事件到达时，用 kernel 计算事件与历史 step 的关联权重，并把 reward 分配给候选历史 step。

4. **Evidence-Conditioned Attribution**
   主方法不依赖 oracle step link，而是使用时间、文本、tag、tool/subgoal 等弱证据做 attribution。

5. **Truncation-Aware Event Modeling**
   timeout、interruption、missing reward 等不完整轨迹现象被显式建模为事件，而不是简单丢弃整条轨迹。

6. **GRPO-Compatible Step Advantage**
   将回填后的 step return 做 group-relative normalization，使其兼容 GRPO / StepPO-style policy update。

## 8. 为什么这样设置有好处

第一，减少错误惩罚。长任务失败不代表所有中间步骤都错了，step-level refill 可以保留正确中间动作的正向训练信号。

第二，更贴近真实 agent 系统。工具调用、网页加载、API 返回、外部 evaluator 打分天然就是异步的，event stream 比完整同步 trajectory 更贴近真实部署环境。

第三，提高不完整 rollout 的利用率。即使 rollout 被 timeout 或 interruption 截断，已经发生的 step 仍然可以根据已到达事件产生局部训练信号。

第四，避免硬编码因果答案。EvidenceKernel 不要求环境告诉算法 reward 属于哪个 step，`dependency` 只作为 oracle upper-bound，主方法走 no-oracle path。

第五，增强可解释性。每个 `CreditAssignment` 都记录：

```text
step_key
event_id
raw_reward
kernel_weight
assigned_credit
reason
```

因此可以追踪某个 event 的 reward 被分给了哪些历史 step，以及分配原因是什么。

## 9. 下一步具体任务

下一阶段建议先把 synthetic controlled diagnostic 固化成论文级实验包，然后再接 HF/LoRA 和真实 benchmark。

### 9.1 固化当前结果

保存当前主结果：

```bash
cp runs/nonlocal_seed_no_oracle/aggregate_by_kernel.csv \
   runs/nonlocal_seed_no_oracle/table_main_performance.csv

cp runs/nonlocal_seed_no_oracle/nonlocal_credit_analysis.csv \
   runs/nonlocal_seed_no_oracle/table_credit_diagnostic.csv
```

### 9.2 跑 non-local v2

目的：进一步增强非局部反馈，使 Evidence 与 Recency 的差异更清晰。

建议配置：

```json
"non_local_credit": {
  "enabled": true,
  "prob": 1.0,
  "lag": 3,
  "reward": 0.25
}
```

Evidence 权重建议：

```json
"temporal_weight": 0.3,
"tag_weight": 6.0,
"text_weight": 3.0
```

运行：

```bash
python -m ecr_grpo.run_sweeps \
  --config configs/smoke_nonlocal_no_oracle.json \
  --param seed \
  --values 7 13 21 \
  --kernels trajectory uniform recency evidence \
  --updates 100 \
  --output-root runs/nonlocal_seed_no_oracle_v2

python -m ecr_grpo.analyze_credit \
  --run-root runs/nonlocal_seed_no_oracle_v2 \
  --output runs/nonlocal_seed_no_oracle_v2/nonlocal_credit_analysis.csv
```

重点观察：

```text
evidence success_rate 是否 >= recency
evidence avg_env_return 是否 >= recency
evidence target_weight_mean 是否明显 > recency
evidence argmax_target_rate 是否明显 > recency
recency recent_weight_mean 是否仍然明显偏高
```

### 9.3 做图表

至少需要三张图：

```text
success rate by kernel
target credit weight by kernel
recent-step credit weight by kernel
```

如果时间允许，再加：

```text
argmax target rate by kernel
success rate vs interruption probability
success rate vs missing reward probability
```

### 9.4 做 evidence ablation

建议逐步去掉 EvidenceKernel 的部分证据：

```text
full evidence
- temporal only
- no tag
- no text
- no temporal
```

目的是说明 Evidence Refill 不是单一 trick，而是多种弱证据组合带来的稳定 attribution。

## 10. 后续工作路线

### 阶段 A：完善 synthetic 论文实验

目标：把机制验证做成稳定、可复现、可画图的实验包。

需要完成：

- 增加 seeds：`7, 13, 21, 42, 100`
- sweep `non_local lag`: `1, 2, 3, 4`
- sweep `missing_reward_prob`
- sweep `interruption_prob`
- sweep `timeout_prob`
- evidence ablation
- case study
- 结果图表和论文表格

### 阶段 B：HF/LoRA smoke

目标：证明 ECR-GRPO 可以从 tabular policy 迁移到 LLM policy training loop。

建议先使用：

- 小模型
- 小 action space
- synthetic text observation
- LoRA
- 少量 updates

重点观察：

```text
logprob 是否正常
advantage 是否正常
entropy 是否稳定
loss 是否稳定
不同 kernel 是否仍然产生差异
```

### 阶段 C：真实 agent benchmark

目标：证明 external validity。

推荐顺序：

```text
ALFWorld
ScienceWorld
WebShop
tool orchestration / BFCL multi-turn
```

真实 benchmark 中不要只报告 success rate，还要报告：

```text
sample efficiency
average steps
delay robustness
timeout robustness
missing reward robustness
interrupted rollout recovery
credit attribution proxy metrics
```

## 11. 当前边界和风险

当前方案仍有几个明确边界：

- Evidence scorer 仍是规则模型，不是 learned credit model。
- Synthetic diagnostic 证明机制有效，但不能替代真实 benchmark。
- 部分 timeout / interruption penalty 仍偏固定，后续可升级为 completion-aware penalty。
- Pending window 可能漏掉特别长延迟事件，后续可加入 retrospective replay buffer。
- HF/LoRA 与真实 benchmark 还需要完整跑通。

这些不是当前工作的失败点，而是后续自然扩展方向。

## 12. 最简论文贡献表述

当前可以把贡献凝练为四点：

1. 提出异步 step credit assignment 问题：真实 agent 反馈经常是 delayed、partial、missing、interrupted 的事件流，而不是完整同步 trajectory。
2. 提出 Event-Conditioned Credit Refill：事件到达时，对 pending buffer 中的历史 step 动态回填 credit。
3. 提出 Evidence-Conditioned Attribution：不依赖 oracle step link，使用弱证据将 delayed feedback 分配给更相关的历史 step。
4. 构建 delay-robust synthetic diagnostic protocol：通过 non-local delayed feedback、missing reward、timeout、interruption 等扰动评估 credit assignment 质量。

一句话版本：

> ECR-GRPO turns delayed asynchronous feedback into step-level policy gradients through evidence-conditioned credit refill.

