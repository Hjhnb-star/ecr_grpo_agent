# ECR-GRPO 方案构思：面向长任务与异步反馈的事件条件信用回填

## 一句话概括

**ECR-GRPO 将延迟、异步、部分到达的环境反馈转化为 step-level policy gradients：它不等待完整 trajectory 结束，而是在事件到达时对历史 step 进行 evidence-conditioned credit refill，再接入 GRPO 的 group-relative advantage 更新。**

## 1. 研究定位

本方案的目标不是重新发明一个完整 RL 框架，而是在 GRPO / StepPO / STEP 这类 agentic RL 方法的信用分配环节提出一个小而清晰的算法创新。

它专门处理长任务和异步反馈下的三个问题：

- reward 延迟到达，训练时无法立即知道某一步是否有效；
- rollout 可能 timeout、中断或缺失部分反馈，trajectory 不一定完整；
- trajectory-level reward 容易把失败轨迹中的正确中间步骤错误惩罚。

因此，ECR-GRPO 的核心问题不是“如何搭一个异步系统”，而是：

> 当 partial reward、terminal reward、tool return、timeout、interruption 等事件异步到达时，如何把这些反馈合理回填到已经发生的历史 step，并重新计算 step-level advantage？

## 2. 研究背景

LLM agent 正在从单轮问答走向多步交互任务，例如网页操作、软件操作、工具调用、科学实验环境、购物环境和多 API 编排。这类任务通常具有以下特点：

- 决策链长，单个任务可能包含十几步甚至几十步动作；
- reward 稀疏，往往任务结束后才知道最终成功或失败；
- 中间步骤可能是正确的，但最终任务仍可能失败；
- 工具调用、网页加载、API 返回天然存在延迟、失败和 timeout；
- rollout 可能被外部条件中断，导致 trajectory 不完整。

传统 PPO / GRPO 类方法通常把一整条 trajectory 当作训练样本。如果最终失败，整条轨迹或其中大量 token/action 都可能被负向更新。这在长任务中会造成明显的信用误分配：很多正确的中间动作会因为最终失败而被错误惩罚。

近期 agentic RL 工作已经开始向 step-level optimization、hierarchical RL 和 parallel exploration 发展。例如：

- STEP 关注 success-rate-aware sampling 与 step-level optimization，指出 trajectory-level optimization 会惩罚失败轨迹中的正确中间动作；
- StepPO 主张把 agentic RL 从 token-level MDP 推进到 step-level MDP；
- HiPER 通过 planner-executor 分层和 hierarchical advantage estimation 缓解长任务信用分配问题；
- DPEPO 通过并行探索和 diversity reward 改善探索效率；
- 工具编排类工作通过 graduated rewards 给多步 API 调用提供更细粒度反馈。

这些工作共同说明：agentic RL 的训练粒度正在从整条 trajectory 走向 step-level。但它们大多仍默认一个隐含前提：step transition 或 trajectory 最终是同步、完整、可对齐的。

## 3. 现有方案的不足

现有方法虽然已经关注 step-level credit assignment，但对异步反馈仍处理不足。

### 3.1 STEP / StepPO

STEP / StepPO 解决了 trajectory-level reward 过粗的问题，将优化粒度推进到 step-level。但它们通常仍假设 step 序列可以被完整收集，且反馈可以较稳定地映射到对应 step。

### 3.2 HiPER / STEP-HRL / CoDA

这类方法通过分层 planner-executor、任务分解或上下文压缩缓解长 horizon 问题。它们重点解决的是“如何组织长任务决策”，而不是“异步事件到来后如何回填历史 credit”。

### 3.3 DPEPO

DPEPO 强调并行探索和 diversity reward，能提高探索效率，但它并不直接建模 delayed reward 到达后如何重新分配历史 step credit。

### 3.4 Tool orchestration graduated reward

工具编排类 reward 能提供更密集的工具调用反馈，但通常依赖较明确的工具执行链和同步反馈。真实系统中，一个 API 返回、网页状态变化或 timeout 事件不一定天然携带精确的因果 step 标注。

因此仍存在一个关键空缺：

> 当 reward 是 delayed、partial、asynchronous 的，甚至 rollout 被中断时，agentic RL 应该如何给历史 step 进行可泛化的信用回填？

这就是 ECR-GRPO 要解决的问题。

## 4. 问题定义

给定一个长任务 agent rollout，agent 在时刻 `t` 产生 step：

```text
b_t = {
  task_id,
  episode_id,
  step_id,
  observation,
  action,
  old_logprob,
  action_space,
  timestamp,
  status,
  metadata
}
```

环境反馈不一定同步返回，而是以异步事件形式到达：

```text
e_k = {
  event_type,
  event_time,
  reward_signal,
  terminal_status,
  observation_delta,
  metadata
}
```

事件可以是：

- `partial_reward`：中间反馈；
- `tool_return`：工具/API 返回；
- `timeout`：超时；
- `terminal_success`：任务成功；
- `terminal_failure`：任务失败；
- `interruption`：rollout 中断。

目标是在事件 `e_k` 到达时，对 pending buffer 中的历史 step `b_t` 进行 credit refill：

```text
c_{t,k} = w_{t,k} * R_k
```

其中 `R_k` 是事件 reward，`w_{t,k}` 是事件 `e_k` 分配给历史 step `b_t` 的信用权重。随后根据累计回填信用计算 step-level return：

```text
G_t = r_t_immediate + sum_k c_{t,k}
```

再在 rollout group 内计算 GRPO-style relative advantage：

```text
A_t = (G_t - mean(G_group)) / (std(G_group) + epsilon)
```

最后使用常规 GRPO clipped objective 更新策略。

## 5. 核心思路：Event-Conditioned Credit Refill

ECR-GRPO 不等待完整 trajectory 结束后统一计算 advantage，而是维护一个 step-level pending buffer：

1. 每个 agent step 产生后进入 pending buffer；
2. 当 partial reward、terminal reward、tool return、timeout、interruption 等事件到达时，算法计算该事件与 buffer 中历史 step 的关联权重；
3. 将事件 reward 按权重回填给历史 step；
4. 当 step terminal、expired 或 episode flush 时，将其取出用于 GRPO 更新。

关键公式是：

```text
w_{t,k} = normalize(K(e_k, b_t))
c_{t,k} = w_{t,k} * R_k
G_t = r_t_immediate + sum_k c_{t,k}
A_t = group_normalize(G_t)
```

其中 `K(e_k, b_t)` 是 credit kernel，用来估计事件和历史 step 的关联程度。

## 6. 从“硬编码依赖”到“通用证据归因”

原始版本中，`Dependency-Aware Refill` 依赖 `related_step_id`、`related_tool`、`related_subgoal` 等字段。这在 synthetic benchmark 中很方便，但在真实场景中容易被质疑为：环境已经告诉了算法 reward 应该归因给哪个 step。

为了解决这个问题，优化后的方案将 credit kernel 分成两类：

### 6.1 Oracle / upper-bound kernel

`Dependency-Aware Refill` 保留为 oracle baseline 或 upper-bound：

```text
K_dep(e_k, b_t) =
  exp(-lambda * distance(e_k, b_t))
  * bonus(exact_step_match, tool_match, subgoal_match)
```

它用于回答：“如果环境提供较强因果链接，ECR-GRPO 的上限表现如何？”

但它不应作为主算法贡献。

### 6.2 主方法：Evidence-Conditioned Credit Refill

主方法不要求环境提供精确 `related_step_id`。它只使用真实 agent 系统中更容易获得的弱证据：

- 时间证据：事件发生时间与历史 step 的距离；
- 文本证据：action、observation、observation_delta、event_type 的 token overlap；
- 工具证据：工具名、API 名、函数名、DOM selector、文件路径等；
- subgoal/tag 证据：planner subgoal、检索 query、任务阶段标签；
- 可选 trace 证据：如果系统天然有 request id、tool call id，可以作为额外证据，但不是必需条件。

定义一个通用 evidence feature：

```text
phi(e_k, b_t) = [
  temporal_distance,
  action_event_text_overlap,
  observation_delta_overlap,
  tool_name_match,
  subgoal_tag_overlap,
  optional_trace_match
]
```

第一版可以使用规则 scorer：

```text
K_evi(e_k, b_t) =
  alpha_time * exp(-lambda * distance)
  + alpha_text * text_overlap(e_k, b_t)
  + alpha_tag  * tag_overlap(e_k, b_t)
  + alpha_tool * tool_match(e_k, b_t)
  + alpha_trace * optional_trace_match(e_k, b_t)
```

然后归一化：

```text
w_{t,k} = K_evi(e_k, b_t) / sum_j K_evi(e_k, b_j)
```

这样，ECR-GRPO 不再假设 benchmark 告诉算法“这个 reward 属于哪个 step”。相反，它把 event-step attribution 明确建模为一个可替换模块。

后续版本可以将规则 scorer 升级为可学习 scorer：

```text
K_theta(e_k, b_t) = softmax_t(f_theta(phi(e_k, b_t)))
```

其中 `f_theta` 可以是线性模型、小型 MLP、cross-attention scorer，或由 LLM/embedding encoder 产生的相似度模型。

## 7. ECR-GRPO 算法流程

### 7.1 Rollout collection

对每个任务采样一个 rollout group：

```text
for task in tasks:
  for rollout in group:
    obs = env.reset(task)
    for t in max_steps:
      action, old_logprob = policy.act(obs)
      next_obs, reward, done, info = env.step(action)
      buffer.add(step_record)
      events = env.pop_ready_events()
      for event in events:
        buffer.assign_event(event, credit_kernel)
      finalized_steps += buffer.finalize_ready()
```

### 7.2 Credit refill

当事件 `e_k` 到达：

```text
candidates = buffer.related_steps(e_k)
scores = [K(e_k, b_t) for b_t in candidates]
weights = normalize(scores)
for b_t, w_t in zip(candidates, weights):
  b_t.filled_credit += w_t * e_k.reward
```

### 7.3 Advantage computation

对 finalized steps 计算 group-relative advantage：

```text
G_t = b_t.immediate_reward + b_t.filled_credit
A_t = (G_t - mean(G_group)) / (std(G_group) + epsilon)
```

### 7.4 Policy update

使用 GRPO / PPO-style clipped objective：

```text
L(theta) =
  - E_t [
      min(
        rho_t(theta) * A_t,
        clip(rho_t(theta), 1-eps, 1+eps) * A_t
      )
    ]
```

其中：

```text
rho_t(theta) = pi_theta(a_t | o_t) / pi_old(a_t | o_t)
```

## 8. 与现有方法的区别

### 相比 GRPO

GRPO 通常从完整 trajectory 或完整 response group 中计算 relative reward。ECR-GRPO 将 reward 来源从完整 outcome 扩展为异步 event stream，使 delayed feedback 可以被转化为 step-level advantage。

### 相比 STEP

STEP 强调 step-level optimization 和 success-rate-aware sampling。ECR-GRPO 关注的是 step feedback 不同步时，如何把 delayed/partial/terminal events 回填到 pending steps。

### 相比 StepPO

StepPO 提出 agentic RL 应从 token-level MDP 转向 step-level MDP。ECR-GRPO 进一步处理 step-level MDP 中 feedback 异步、缺失和中断的问题。

### 相比 HiPER / STEP-HRL

HiPER / STEP-HRL 通过分层规划缓解长任务信用分配和上下文问题。ECR-GRPO 不要求引入完整 planner-executor 架构，工程上更轻量，可以作为 GRPO / StepPO 的可插拔 credit assignment 模块。

### 相比 DPEPO

DPEPO 主攻 parallel exploration 和 diversity reward。ECR-GRPO 主攻 delayed event 到达后的 historical step credit assignment。

### 相比 Tool Orchestration graduated rewards

工具编排 reward 主要设计更细粒度奖励项。ECR-GRPO 不只关心奖励项是什么，还显式建模 reward/event 如何动态分配给历史 step。

## 9. 核心贡献

### C1：提出异步 step credit assignment 问题

指出现有 step-level / hierarchical agentic RL 仍然依赖同步、完整、可对齐的 trajectory，而真实 agent 反馈经常以 delayed event stream 形式到达。

### C2：提出 Event-Conditioned Credit Refill

设计一种事件条件信用回填机制，将 delayed、partial、terminal、timeout、interruption feedback 分配给历史 step，减少 trajectory-level reward 导致的错误惩罚。

### C3：提出 Evidence-Conditioned Attribution

将事件与历史 step 的关联从硬编码 `related_step_id` 升级为通用 evidence attribution。主方法可在没有 oracle step link 的情况下运行，只依赖时间、文本、工具、tag、trace 等弱证据。

### C4：提出 ECR-GRPO 算法

将 credit refill 接入 GRPO，使 agent 可以在 step-level pending buffer 上进行更新，并兼容 GRPO / StepPO / agentic RL 训练框架。

### C5：构建 delay-robust long-horizon benchmark protocol

通过在 synthetic、ALFWorld、ScienceWorld、WebShop 或 tool orchestration benchmark 上注入 delay、timeout、interruption、missing reward，系统评估方法对异步反馈和长任务的鲁棒性。

## 10. 实验设计

### 10.1 第一阶段：Synthetic async benchmark

第一阶段使用可控 synthetic async wrapper，优点是可以明确知道真实 causal step，从而计算 credit assignment error。

控制变量：

- delay probability；
- max delay steps；
- timeout probability；
- interruption probability；
- missing reward probability；
- trajectory length；
- distractor action ratio。

核心观察：

> 随着 delay、timeout、missing reward、trajectory length 增大，ECR-GRPO 的 success rate 是否比 baseline 下降更慢？

### 10.2 第二阶段：真实 agent benchmark

推荐 benchmark：

- ALFWorld：长任务明显，agent RL 常用；
- ScienceWorld：适合长 horizon 和科学实验式探索；
- WebShop：比 WebArena 更轻量，适合购物决策；
- ComplexFuncBench / BFCL multi-turn：适合工具编排和多 API 调用。

### 10.3 Baselines

建议对比：

- Trajectory-level GRPO：只用最终 trajectory reward；
- Uniform refill：事件 reward 平均分给 pending steps；
- Recency refill：越近的 step 得到越多 credit；
- Dependency-aware refill：使用 oracle link 的 upper-bound；
- Evidence-conditioned refill：无 oracle link 的主方法；
- StepPO-style step reward：如果 benchmark 能提供 step reward，可作为 step-level baseline。

### 10.4 Metrics

核心指标：

- success rate；
- sample efficiency；
- average return；
- reward delay robustness；
- timeout robustness；
- interrupted rollout recovery；
- credit assignment error，synthetic 环境可计算；
- positive credit mass on causal steps，synthetic 环境可计算；
- policy update stability，例如 ratio、entropy、advantage variance。

## 11. 当前实现对应关系

当前代码已经实现了最小可行版本：

- `StepRecord / AsyncEvent / CreditAssignment`：核心类型系统；
- `PendingStepBuffer`：pending step 生命周期管理；
- `Trajectory / Uniform / Recency / Dependency / Evidence` kernels：信用回填策略；
- `AsyncEnvWrapper`：注入 delay、timeout、interruption、missing reward；
- `use_oracle_event_links=false`：关闭 oracle 因果链接，测试通用 attribution；
- `compute_group_advantages`：GRPO-style group-relative advantage；
- synthetic environment：可控因果任务；
- ALFWorld adapter：真实长任务接口；
- tabular policy：轻量验证；
- HF LoRA policy：面向 LLM policy 的占位实现；
- train/eval/baseline runner：完整训练和评估流程。

推荐在论文或报告中把 kernel 定位写清楚：

- `dependency`：oracle / upper-bound baseline；
- `evidence`：主方法；
- `uniform / recency / trajectory`：基础 baseline。

## 12. 风险与改进方向

### 12.1 Evidence scorer 仍是规则模型

当前 evidence scorer 使用手动权重。它已经避免了精确 causal link 硬编码，但仍不是可学习 credit model。后续可以升级为：

- linear scorer；
- MLP scorer；
- attention-based event-step scorer；
- embedding similarity scorer；
- LLM-assisted attribution scorer。

### 12.2 Expired steps 的追溯修正

当前 pending buffer 使用 `max_pending_age` 控制过期。如果事件延迟超过 pending window，信用可能无法回填。后续可以加入 replay memory：

```text
active pending buffer + retrospective replay buffer
```

新事件到达时不仅更新 active steps，也允许对近期 expired steps 进行低权重追溯修正。

### 12.3 跨 episode 信用共享

当前 credit refill 主要在 episode 内完成。后续可以加入 hindsight credit sharing：如果多个 episode 中相似 action/subgoal 得到类似反馈，可以更新 shared attribution prior。

### 12.4 更真实的异步环境

第一阶段 synthetic benchmark 适合验证曲线，但最终需要在 ALFWorld、ScienceWorld、WebShop 或 tool benchmark 上验证泛化能力。

## 13. 最终目标

这篇工作的最终目标可以定义为：

> 提出一种轻量、可插拔、无需精确 oracle step link 的异步信用分配机制，使现有 GRPO / StepPO 类 agentic RL 方法能够更稳定地训练长任务、延迟奖励和异步工具调用场景下的 LLM agents。

更凝练的论文式表述：

> ECR-GRPO turns delayed asynchronous feedback into step-level policy gradients through evidence-conditioned credit refill.

## 14. 参考工作

- STEP: https://arxiv.org/abs/2511.13091
- StepPO: https://arxiv.org/abs/2604.18401
- HiPER: https://arxiv.org/abs/2602.16165
- DPEPO: https://arxiv.org/abs/2604.24320
- Tool Orchestration: https://arxiv.org/abs/2603.24709
