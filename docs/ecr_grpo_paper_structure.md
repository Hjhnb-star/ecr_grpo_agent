# ECR-GRPO 论文结构与写作蓝图

## 0. 论文一句话

**ECR-GRPO 将长任务 agent 中延迟、部分、非局部、缺失和中断后的反馈建模为异步事件流，并在事件到达时对历史 step 做 evidence-conditioned credit refill，从而把 asynchronous feedback 转化为可解释、可训练的 step-level policy gradients。**

更论文式的表述：

> ECR-GRPO turns delayed asynchronous feedback into step-level policy gradients through evidence-conditioned credit refill.

## 1. 论文核心问题

长任务 LLM agent 的训练正在从单轮回答走向多步交互：工具调用、网页操作、软件环境、API 编排、科学实验环境、购物环境、检索-验证-执行任务等。在这些任务中，agent 的一个 episode 往往由多个 step 组成，每一步都有 observation、action、tool call、intermediate state 和后续反馈。

传统 RL 或 GRPO/PPO 类 agentic RL 方法往往把完整 trajectory 的最终成败当作主要训练信号。这个假设在短任务中可以工作，但在长任务 agent 中会暴露出两个根本问题：

1. **Feedback is delayed and asynchronous.** 工具调用、网页加载、API 返回、外部 evaluator、人工打分、环境状态变化都可能延迟到达，甚至在 episode 结束后才到达。
2. **Trajectory-level reward is too coarse.** 一个最终失败的 trajectory 中可能包含许多正确的中间步骤；如果整条轨迹被负向更新，正确中间动作会被错误惩罚。

因此，论文要回答的核心问题是：

> 当 reward 不再是同步、完整、立即对齐的 trajectory-level 标量，而是以 delayed / partial / non-local / interrupted event stream 形式到达时，agentic RL 应该如何将这些反馈转化为更准确的 step-level credit assignment？

这个问题不是单纯工程上的“异步系统实现”，而是一个训练信号建模问题：

```text
传统视角：
  trajectory -> final reward -> whole-trajectory update

本文视角：
  step sequence + asynchronous event stream
    -> event-to-step credit refill
    -> step-level return
    -> group-relative advantage
```

## 2. 为什么 asynchronous reward 是重要问题

### 2.1 真实 agent 环境天然异步

长任务 agent 不像标准同步 MDP 那样每一步 action 之后马上获得完整 reward。真实任务中的反馈往往有以下形态：

- **Tool return delay**：工具/API 调用需要等待返回，返回内容可能对应几步之前的决策。
- **Page/environment latency**：网页加载、环境状态变化、外部系统响应并不与当前 step 严格同步。
- **External evaluator delay**：最终答案正确性、任务完成度、人工评估可能在 episode 结束后才给出。
- **Partial reward stream**：中间奖励可能分批到达，不一定覆盖所有 step。
- **Missing feedback**：某些工具调用失败、日志缺失、评估器超时，导致部分 reward 永远不返回。
- **Timeout/interruption**：rollout 可能因为环境超时、工具错误、外部中断或 max steps 被截断。

因此，在真实 agent 训练中，feedback 更像一个 event stream，而不是每一步都同步返回的标量 reward。

### 2.2 长任务中的最终成败不能解释每个 step

长任务里，最终失败不代表每一步都错。一个失败 trajectory 可能包含：

```text
正确检索 -> 正确调用工具 -> 正确解析信息 -> 某一步参数错误 -> 最终失败
```

如果只根据最终失败惩罚整条 trajectory，前三个正确步骤也会被负向更新。这会导致：

- 策略学不到哪些中间动作是有价值的；
- 长任务训练方差变大；
- agent 倾向于回避本来正确但最终未成功的中间行为；
- failure trajectory 中的有用经验被浪费；
- timeout 或 interruption 产生的训练信号过粗。

这正是 trajectory-level reward 在长任务中的信用误分配问题。

### 2.3 Step-level 方法还没有充分处理异步反馈

近期 agentic RL 已经开始关注 step-level optimization。例如 STEP / StepPO / hierarchical RL / tool orchestration reward 等工作，都在尝试缓解 trajectory-level reward 过粗的问题。

但它们通常仍隐含一个前提：

```text
step 序列可以完整收集；
反馈可以同步或较稳定地对齐到某个 step；
episode 结束后可以统一计算 trajectory 或 step return。
```

真实环境中，这个前提不总成立。反馈可能：

- 延迟到后续 step 才出现；
- 指向较早历史 step，而不是最近 step；
- 没有精确 `related_step_id`；
- 被 timeout/interruption 打断；
- 只携带弱证据，例如 tool name、error text、observation_delta、subgoal tag；
- 在 rollout 结束后才到达。

因此，已有 step-level 方法解决了“粒度太粗”的问题，但仍没有充分解决“反馈异步且不可靠对齐”的问题。

### 2.4 异步反馈不是噪声，而是训练信号的真实形态

本文的关键立场是：

> Asynchronous feedback should not be treated as an implementation nuisance; it is the natural form of supervision in long-horizon agent environments.

换句话说，异步反馈不是训练系统的边角问题，而是长任务 agent 学习中非常核心的监督形态。如果算法不能处理 delayed / partial / interrupted feedback，就会在真实工具调用、网页交互和 API 编排场景下变得脆弱。

## 3. 核心 Insight

### 3.1 Insight 1：Reward 应该被事件化，而不是只作为 trajectory 结果

传统做法把 reward 看成 trajectory 结束后的单一结果：

```text
trajectory τ -> R(τ)
```

本文将反馈统一表示为事件流：

```text
e_k = {
  event_type,
  event_time,
  reward,
  observation_delta,
  terminal,
  metadata
}
```

事件可以包括：

- `partial_reward`
- `tool_return`
- `timeout`
- `interruption`
- `terminal_success`
- `terminal_failure`
- `missing_reward`
- `non_local_support`

这样，reward 不再是 episode 结束后的一个标量，而是一个随着环境运行逐步到达的 event stream。

### 3.2 Insight 2：历史 step 应该先进入 pending 状态，等待后续事件回填

在异步环境中，当前 step 的价值不一定能立即判断。一个 step 可能要等几步之后的 tool return、状态变化或 terminal event 才知道是否有用。

因此，每个 step 不应立即根据当前 reward 固定训练信号，而应先进入 pending buffer：

```text
b_t = {
  observation,
  action,
  step_id,
  env_time,
  metadata,
  filled_credit
}
```

后续事件到达时，再对 pending buffer 中的历史 step 做 credit refill。

这个 insight 将训练信号从：

```text
step 发生时立即决定 reward
```

转变为：

```text
step 先进入可回填状态；
后续事件到达时动态修正它的 credit。
```

### 3.3 Insight 3：Delayed feedback 不一定属于最近 step

一个很自然但错误的 baseline 是 recency：把 delayed reward 更多分给最近 step。但真实长任务中，反馈经常是 non-local 的。

例如：

```text
step 1: search relevant page
step 2: extract key fact
step 3: call verification tool
step 4: compose final answer
step 5: submit
event: final correctness confirmed
```

最终 correctness event 可能真正依赖 step 2 或 step 3，而不是最近的 step 5。

因此，异步 credit assignment 不能只依赖时间近邻。它需要利用事件与历史 step 之间的弱证据：

- action 文本；
- observation delta；
- tool name；
- function name；
- subgoal/tag；
- entity overlap；
- error message；
- optional trace id。

### 3.4 Insight 4：无 oracle 的 evidence attribution 是关键

如果环境直接告诉算法：

```text
event.related_step_id = 3
```

那 credit assignment 就变成了 oracle 对齐，研究价值有限。真实 agent 系统中，很多 event 并不天然携带精确因果 step 标注。

因此，本文主方法不依赖 `related_step_id`，而是使用更普遍可得的弱证据做 attribution：

```text
K(e_k, b_t) =
  temporal evidence
  + text overlap
  + tool / API match
  + tag / subgoal match
  + observation_delta match
  + optional trace evidence
```

`dependency` kernel 可以作为 oracle upper-bound baseline，但主方法应是 `evidence` kernel。

### 3.5 Insight 5：中断和 timeout 不是“废轨迹”，而是可分配的负向事件

长任务 rollout 中，很多 episode 不是自然成功/失败结束，而是因为：

- timeout；
- tool error；
- max steps；
- stuck loop；
- external interruption；
- missing feedback；
- environment failure。

如果直接丢弃这些 trajectory，会浪费大量已经发生的 step。如果简单把整条轨迹标成失败，又会粗暴惩罚所有步骤。

本文的 insight 是：这些异常也应该被建模为事件，并带有原因 metadata：

```text
event_type = timeout / interruption / tool_error
metadata = {
  truncation_reason,
  last_tool,
  progress,
  completed_subgoals,
  valid_step_ratio
}
```

然后将负向 credit 更局部地分配给相关历史 step。

## 4. 方法概览

ECR-GRPO 的核心机制由四个部分组成：

```text
Eventized Reward Interface
  -> Pending Step Buffer
  -> Evidence-Conditioned Credit Refill
  -> GRPO-compatible Step Advantage
```

整体流程：

```text
1. Agent 产生 step b_t
2. Step 进入 pending buffer
3. 环境反馈以 AsyncEvent e_k 的形式延迟或部分到达
4. Credit kernel 计算 e_k 与历史 step b_t 的关联权重 w_{t,k}
5. 事件 reward R_k 被回填为 c_{t,k} = w_{t,k} R_k
6. 每个 step 累积 filled_credit
7. 计算 step-level return G_t
8. 在 group 内归一化得到 GRPO-style advantage A_t
9. 更新 policy
```

## 5. 问题定义

给定一个长任务 rollout，agent 在 step `t` 产生：

```text
b_t = {
  task_id,
  episode_id,
  group_id,
  step_id,
  env_time,
  observation,
  action,
  old_logprob,
  action_space,
  metadata
}
```

环境反馈以异步事件形式到达：

```text
e_k = {
  task_id,
  episode_id,
  event_id,
  event_type,
  event_time,
  reward,
  observation_delta,
  terminal,
  metadata
}
```

目标是在事件 `e_k` 到达时，计算它对历史 step `b_t` 的信用分配：

```text
c_{t,k} = w_{t,k} R_k
```

其中：

- `R_k` 是事件 reward；
- `w_{t,k}` 是事件 `e_k` 分给历史 step `b_t` 的权重；
- `sum_t w_{t,k} = 1`。

step-level return 为：

```text
G_t = r_t_immediate + sum_k c_{t,k}
```

GRPO-style group-relative advantage 为：

```text
A_t = (G_t - mean(G_group)) / (std(G_group) + epsilon)
```

## 6. Evidence-Conditioned Credit Refill

核心 credit kernel 定义为：

```text
w_{t,k} = normalize(K(e_k, b_t))
```

`K(e_k, b_t)` 衡量事件和历史 step 的关联程度。

一个可解释的规则版本可以写成：

```text
K_evi(e_k, b_t) =
  alpha_time * exp(-lambda * time_distance(e_k, b_t))
  + alpha_text * text_overlap(e_k, b_t)
  + alpha_tool * tool_match(e_k, b_t)
  + alpha_tag  * tag_overlap(e_k, b_t)
  + alpha_delta * observation_delta_match(e_k, b_t)
```

然后归一化：

```text
w_{t,k} =
  K_evi(e_k, b_t) / sum_j K_evi(e_k, b_j)
```

后续可以扩展为可学习 scorer：

```text
K_theta(e_k, b_t) = softmax_t(f_theta(phi(e_k, b_t)))
```

但论文第一阶段可以先以规则 evidence scorer 作为主要实现，因为它：

- 可解释；
- 不依赖 oracle causal label；
- 便于做 credit diagnostic；
- 便于和 uniform / recency / dependency baselines 对比。

## 7. Baselines 与角色

论文中需要清楚区分不同 kernel 的角色：

```text
Trajectory:
  trajectory-level baseline，只在终局 reward 上更新。

Uniform:
  将事件 reward 平均分给候选历史 steps。

Recency:
  根据时间距离偏向最近 step，是强时间近邻 baseline。

Dependency:
  使用 related_step_id / tool / subgoal 等 oracle link，作为 upper-bound。

Evidence:
  主方法，不使用 oracle step link，只使用弱证据做 attribution。
```

论文表达中应强调：

> Dependency is an oracle upper-bound, while Evidence is the deployable no-oracle method.

## 8. 论文贡献草案

可以将贡献写成四到五点：

### Contribution 1: 异步 step credit assignment 问题

提出并形式化长任务 agent 中的 asynchronous step credit assignment 问题。指出真实 agent 反馈经常以 delayed / partial / non-local / interrupted event stream 形式到达，而现有 trajectory-level 或 step-level 方法大多仍依赖同步、完整、可对齐的反馈。

### Contribution 2: Event-Conditioned Credit Refill

提出事件条件信用回填机制，在异步事件到达时对 pending buffer 中的历史 step 动态分配 credit，将 delayed feedback 转化为 step-level return。

### Contribution 3: Evidence-Conditioned Attribution

提出无需 oracle step link 的 evidence-conditioned attribution，用时间、文本、工具、tag、observation_delta 等弱证据估计 event-step 关联，使方法能够适用于真实 agent 系统中没有精确因果标注的场景。

### Contribution 4: GRPO-compatible Algorithm

将回填后的 step return 接入 GRPO-style group-relative advantage，使方法能够作为轻量 credit assignment 模块接入现有 GRPO / StepPO / PPO-style agentic RL 框架。

### Contribution 5: Diagnostic Protocol

构建包含 delayed reward、missing feedback、timeout、interruption、non-local delayed feedback 的 controlled synthetic benchmark，并通过 credit diagnostic 直接验证 reward 被分配到了哪些历史 step。

## 9. 论文结构建议

## 9.1 Abstract

需要回答四件事：

1. 长任务 agent 中 feedback 经常 delayed / asynchronous / partial。
2. 现有 trajectory-level 和 step-level 方法容易误分配 credit。
3. ECR-GRPO 将反馈建模为 event stream，并用 evidence-conditioned refill 分配给历史 step。
4. 阶段 A 实验展示在 synthetic async benchmark 中提升 credit assignment quality、success rate 和 robustness；后续真实 benchmark 待补。

摘要草稿：

> Long-horizon LLM agents increasingly operate through multi-step tool use, web interaction, and API orchestration, where feedback is often delayed, partial, missing, or returned after interruptions. Existing trajectory-level RL methods assign a single outcome to an entire rollout, while recent step-level methods still often assume synchronous and alignable feedback. We propose ECR-GRPO, an event-conditioned credit refill mechanism that models environmental feedback as asynchronous events and dynamically assigns their rewards to pending historical steps using evidence-conditioned attribution. The resulting step-level returns are converted into GRPO-style group-relative advantages, making the method compatible with existing agentic RL pipelines. In controlled synthetic asynchronous benchmarks, ECR-GRPO improves non-local credit attribution and provides more robust learning under delayed and incomplete feedback.

## 9.2 Introduction

Introduction 推荐结构：

1. LLM agents 正在进入多步交互任务。
2. 这些任务中的 feedback 天然异步。
3. Trajectory-level reward 会错误惩罚正确中间步骤。
4. Step-level optimization 是趋势，但异步反馈仍未充分解决。
5. 本文提出将 feedback eventized，并在事件到达时做 credit refill。
6. 总结贡献。

关键段落要强调：

```text
The central challenge is not merely long horizon, but delayed and non-local supervision.
```

也就是说，长 horizon 本身不是唯一难点。真正关键的是：监督信号在时间上晚到、部分到达、可能属于较早 step，甚至在 rollout 中断后才出现。

## 9.3 Related Work

建议分成四类：

### Long-horizon agentic RL

覆盖 GRPO / PPO-style agent training、agentic RL、long-horizon decision making。

写作重点：

- 这些方法能训练多步 agent；
- 但通常仍依赖 trajectory-level outcome 或同步环境反馈。

### Step-level optimization

覆盖 STEP、StepPO、step-level MDP、hierarchical advantage estimation。

写作重点：

- 它们指出 trajectory-level reward 太粗；
- 但多数仍假设 step feedback 可对齐、rollout 可完整收集。

### Tool-use and API orchestration rewards

覆盖 tool calling、graduated reward、multi-step API benchmark。

写作重点：

- 工具调用天然提供中间反馈；
- 但真实 tool return 可能 delayed、missing、timeout，不一定给出 causal step link。

### Credit assignment under delayed feedback

覆盖 RL 中 delayed reward、eligibility trace、credit assignment、off-policy replay 等。

写作重点：

- 传统 delayed reward 方法为本文提供背景；
- 但 LLM agent 的 event-rich feedback 可以利用文本、工具、tag 等语义证据，这是本文的特殊性。

## 9.4 Problem Formulation

应包含：

- step record 定义；
- async event 定义；
- pending buffer；
- event-to-step credit；
- step return；
- group-relative advantage；
- no-oracle setting。

必须强调：

```text
The algorithm does not assume access to oracle related_step_id in the main setting.
```

## 9.5 Method

建议拆为五个小节：

### Eventized Reward Interface

说明如何把 partial reward、terminal reward、tool return、timeout、interruption 统一成 `AsyncEvent`。

### Pending Step Buffer

说明历史 step 为什么要 pending，以及何时 finalize。

### Credit Refill Kernels

说明 trajectory、uniform、recency、dependency、evidence 五种 kernel。

### Evidence-Conditioned Attribution

重点写主方法，包含公式、证据项和 no-oracle 设定。

### GRPO-compatible Step Advantage

说明如何从 filled credit 得到 `G_t` 和 `A_t`，并接入 GRPO-style update。

## 9.6 Experiments

阶段 A 的 synthetic 实验包已经完成，服务器端已通过 `scripts/summarize_results.py` 汇总出论文表格：

```text
runs/paper_tables/main_performance.csv
runs/paper_tables/credit_diagnostic.csv
runs/paper_tables/robustness_all.csv
runs/paper_tables/robustness_success_pivot.csv
runs/paper_tables/robustness_return_pivot.csv
runs/paper_tables/ablation_performance.csv
runs/paper_tables/ablation_diagnostic.csv
runs/paper_tables/summary.md
```

实验章节现在应作为“阶段 A 结果入稿”处理：主表写 success / return，诊断表写 credit attribution，鲁棒性表画曲线，消融表证明 evidence source 的贡献。真实 benchmark 仍作为后续 external validity，不在当前版本中过度承诺。

阶段 A 的核心数值结论如下：

- 主实验中 Evidence 的 success / return 为 `0.725 / 0.501`，高于 Recency 的 `0.675 / 0.474` 和 Uniform 的 `0.383 / 0.338`；Trajectory 为 `0.000 / -0.289`。
- v2 中 Evidence 的 success / return 为 `0.654 / 0.465`，Recency 为 `0.642 / 0.462`，性能差距较小，但 attribution 差距显著。
- 主实验中 Evidence 的 `target_weight_mean=0.418`、`argmax_target_rate=0.977`；Recency 的 `target_weight_mean=0.132`、`argmax_target_rate=0.000`。
- v2 中 Evidence 的 `target_weight_mean=0.491`、`argmax_target_rate=0.934`；Recency 的 `target_weight_mean=0.081`、`argmax_target_rate=0.000`。
- robustness 平均优势主要体现在 missing reward（`+0.094` success，`+0.084` return）和 timeout（`+0.078` success，`+0.048` return）。
- ablation 显示 performance 与 attribution 会分化：`no_tag` 的 success 最高但 `argmax_target_rate=0.014`，因此消融必须同时报告性能和诊断指标。

### RQ1: Step-level refill 是否优于 trajectory-level reward？

实验：

```text
trajectory vs uniform / recency / evidence
```

指标：

- success rate；
- average return；
- learning curve；
- average steps。

当前结论：

```text
trajectory-level reward 在长任务异步反馈下明显不足；
step-level refill 能显著改善训练信号。
```

当前结果已经支持该结论：trajectory 在主实验和 v2 中 success 均为 `0.000`，而 Uniform / Recency / Evidence 均能训练到正 return。

### RQ2: Evidence refill 是否能解决 non-local delayed feedback？

实验：

```text
uniform vs recency vs evidence vs dependency oracle
```

指标：

- target_weight_mean；
- recent_weight_mean；
- argmax_target_rate；
- target_top3_rate；
- target_credit_fraction；
- success rate；
- return。

当前结论：

```text
recency 偏向最近 step；
evidence 能把 delayed reward 分回真正相关的历史 step；
dependency 作为 oracle upper-bound。
```

当前结果已经支持该结论：主实验中 Evidence 的 target weight 是 Recency 的 `3.17x`，v2 中是 `6.03x`；Recency 的 argmax target rate 在两组实验中均为 `0.000`。

### RQ3: 异步扰动增强时是否更鲁棒？

sweep：

```text
delay_prob
terminal_reward_delay
timeout_prob
interruption_prob
missing_reward_prob
non_local_lag
trajectory_length
```

指标：

- success rate vs perturbation；
- return vs perturbation；
- robustness AUC；
- advantage variance。

当前结果显示 Evidence 在 missing reward 和 timeout 下最稳。delay 中 Evidence 在 4/5 个点上 success 不低于 Recency，但 `delay_prob=0.6` 时 Recency 略高；interruption 中 Evidence 的 return 全部不低于 Recency，但 success 在 `interruption_prob=0.1` 时略低。

### RQ4: Evidence scorer 的哪些证据项有效？

消融：

```text
without temporal evidence
without text overlap
without tag overlap
without tool match
without observation_delta
different max_pending_age
different weight settings
```

指标：

- success rate；
- target credit；
- argmax target；
- recent-step credit mass。

当前消融结果显示：temporal evidence 对最终性能很关键，text-only 明显不足，tag signal 对 argmax attribution 很强但可能导致结构偏置。`no_tag` success 最高但 argmax target rate 几乎崩溃，因此消融分析必须避免只按 success 排序。

### RQ5: 能否迁移到真实 agent benchmark？

当前已有最小 smoke：

- HF/LoRA synthetic smoke 已跑通，Evidence 在 update 10 达到 `success=0.625`，Recency 为 `0.375`；
- ALFWorld `eval_out_of_distribution` split 已能加载 134 games，并完成 dependency-kernel 端到端训练日志；
- ALFWorld smoke 中 success 从 `0.000 -> 0.250 -> 0.500` 后又回到 `0.000`，说明 adapter 和训练链路可用，但稳定性和统计显著性还不足；
- 当前 `entropy=0.000` 来自 HF policy 日志占位，不能作为熵塌缩证据。

正式 external validity 仍待完成：

- ALFWorld：需要 Evidence / Recency / Dependency 同预算多 seed 对比；
- ScienceWorld；
- WebShop；
- BFCL / tool orchestration。

当前版本可以写 feasibility smoke，不应过度承诺真实 benchmark 上的最终效果。

## 9.7 Analysis

建议包含：

### Credit diagnostic

展示 reward 到底分给了哪个 step，而不是只看最终性能。

图：

- target credit weight by kernel；
- recent-step credit weight by kernel；
- argmax target rate by kernel。

### Case study

选择一个 non-local delayed event：

```text
event arrives after step 5
true target is step 2 or step 3
recency assigns credit to step 5
evidence assigns credit to true target
```

### Failure analysis

分析 evidence kernel 失败的情况：

- 文本证据不足；
- 多个历史 step 共享同一 tag；
- delay 超过 pending window；
- event metadata 过少；
- negative event 的 attribution 不稳定。

## 9.8 Limitations

当前可写：

- Evidence scorer 仍是规则权重，不是 learned scorer；
- 阶段 A 仍是 synthetic controlled diagnostic，尚未扩展到真实 benchmark；
- pending window 对超长延迟反馈可能漏分；
- event metadata 的质量会影响 attribution；
- interruption penalty 目前较简单，后续需要 completion-aware penalty；
- HF/LoRA policy 仍处于 smoke test 阶段。

## 9.9 Conclusion

强调：

- feedback eventization；
- pending buffer；
- evidence-conditioned refill；
- GRPO-compatible step advantage；
- 异步长任务 agent 的信用分配问题。

收束句：

> ECR-GRPO reframes delayed and interrupted feedback as assignable training events, enabling long-horizon agents to learn from partial, asynchronous, and non-local supervision rather than waiting for a single trajectory-level outcome.

## 10. 图表规划

### Figure 1: Motivation

展示 trajectory-level reward 的问题：

```text
correct step -> correct step -> wrong step -> final failure
trajectory-level reward punishes all steps
```

### Figure 2: Asynchronous Event Stream

展示：

```text
steps: b1, b2, b3, b4, b5
events: delayed partial reward, tool return, timeout, terminal reward
```

强调 event 不一定与最近 step 对齐。

### Figure 3: ECR-GRPO Architecture

模块：

```text
Agent rollout
PendingStepBuffer
AsyncEvent stream
EvidenceCreditKernel
Step return
GRPO advantage
Policy update
```

### Figure 4: Credit Assignment Diagnostic

柱状图：

```text
target_weight_mean by kernel
recent_weight_mean by kernel
argmax_target_rate by kernel
```

### Figure 5: Robustness Curves

曲线：

```text
success rate vs delay_prob
success rate vs missing_reward_prob
success rate vs interruption_prob
```

## 11. 主表规划

### Table 1: Main Synthetic Performance

```text
Method                  Success ↑   Return ↑   Steps ↓   Robustness AUC ↑
Trajectory              ...
Uniform                 ...
Recency                 ...
Evidence, ours          ...
Dependency, oracle      ...
```

### Table 2: Credit Diagnostic

```text
Method                  Target Weight ↑   Recent Weight ↓   Argmax Target ↑   Top-3 Target ↑
Uniform                 ...
Recency                 ...
Evidence, ours          ...
Dependency, oracle      ...
```

### Table 3: Ablation

```text
Variant                 Success ↑   Target Credit ↑   Argmax Target ↑
Evidence full           ...
w/o temporal            ...
w/o text                ...
w/o tag                 ...
w/o tool                ...
```

### Table 4: Real Benchmark Placeholder

```text
Benchmark      Method        Success ↑   Return ↑   Robustness AUC ↑
ALFWorld       ...
ScienceWorld   ...
WebShop        ...
Tool benchmark ...
```

## 12. 写作时必须避免的误解

### 12.1 不要把贡献写成“实现了异步 wrapper”

异步 wrapper 只是实验工具。真正贡献是：

```text
event-conditioned credit assignment mechanism
```

### 12.2 不要让 dependency kernel 看起来像主方法

`dependency` 使用 oracle link，只能作为 upper-bound。主方法必须是 no-oracle evidence kernel。

### 12.3 不要只报告 success rate

必须报告 credit diagnostic。否则无法证明提升来自更好的 credit assignment。

### 12.4 不要过度承诺真实 benchmark

真实 benchmark 可以作为后续 external validity。当前论文早期版本应先把 synthetic controlled diagnostic 写扎实。

### 12.5 不要把问题说成普通 delayed reward

普通 delayed reward 不足以概括本文问题。本文强调的是：

```text
delayed + partial + non-local + missing + interrupted + weakly evidenced feedback
```

## 13. 当前论文主线版本

可以把论文主线压缩成下面这段：

> Long-horizon LLM agents receive supervision not as clean synchronous rewards, but as delayed, partial, missing, and sometimes post-interruption events. Existing trajectory-level optimization assigns a single outcome to an entire rollout, while step-level methods still often assume alignable feedback. ECR-GRPO models feedback as an asynchronous event stream and keeps recent steps in a pending buffer. When an event arrives, an evidence-conditioned kernel assigns its reward to relevant historical steps using weak signals such as temporal distance, text overlap, tool names, tags, and observation deltas. The accumulated credit forms step-level returns, which are normalized into GRPO-style group-relative advantages. This converts asynchronous feedback into actionable policy gradients and reduces the misattribution caused by coarse trajectory-level rewards.

## 14. 后续填充清单

阶段 A 已完成，后续需要补的是论文入稿材料与阶段 B/C 外部验证：

- 从 `main_performance.csv` 写 main performance table；
- 从 `credit_diagnostic.csv` 写 credit diagnostic table；
- 从 robustness pivot 表画 robustness curves；
- 从 ablation 表写 evidence ablation table；
- case study；
- seed 数量和置信区间；
- exact config；
- baseline 超参数；
- 真实 benchmark 接入情况；
- limitation 中与实验相关的具体观察；
- appendix 中的算法伪代码和更多诊断图。

## 15. Algorithm 伪代码占位

```text
Algorithm: ECR-GRPO

Input:
  policy pi
  async environment env
  credit kernel K
  pending buffer B

for each rollout group:
  for each episode:
    while not done:
      observe o_t
      sample action a_t ~ pi(. | o_t)
      execute action
      create StepRecord b_t
      B.add(b_t)

      ready_events = env.pop_events()
      for each event e_k in ready_events:
        candidates = B.related_steps(e_k)
        weights = normalize(K(e_k, candidates))
        for each candidate b_t:
          b_t.filled_credit += weights[t] * e_k.reward

      finalized_steps = B.finalize_ready()

    drain delayed events
    flush remaining steps

  compute G_t = immediate_reward_t + filled_credit_t
  compute group-relative A_t
  update policy with GRPO-style objective
```
