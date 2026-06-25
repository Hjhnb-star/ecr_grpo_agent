# ECR-GRPO 项目报告：面向长任务与异步反馈的事件条件信用回填

## 一句话概括

**ECR-GRPO 将延迟、异步、部分到达、甚至中断后的环境反馈转化为 step-level policy gradients：它不再等待完整 trajectory 结束，而是在事件到达时对历史 step 进行 evidence-conditioned credit refill，再接入 GRPO 的 group-relative advantage 更新。**

这项工作的核心不是“搭一个异步系统”，而是解决一个更具体的问题：

> 当 partial reward、terminal reward、tool return、timeout、interruption 等反馈以事件流形式到达时，如何把它们合理回填到已经发生的历史 step，并重新计算 step-level advantage？

## 1. 研究定位

本项目的目标不是重新发明一个完整 RL 框架，而是在 GRPO / StepPO / STEP 这类 agentic RL 方法的信用分配环节提出一个小而清晰的算法创新。它专门处理长任务和异步反馈下的三个问题：

- reward 延迟到达，训练时无法立即知道某一步是否有效；
- rollout 可能 timeout、中断或缺失部分反馈，trajectory 不一定完整；
- trajectory-level reward 容易把失败轨迹中的正确中间步骤错误惩罚。

因此，ECR-GRPO 的研究对象可以概括为：

> 在长任务 agent 训练中，如何把 delayed / partial / non-local / interrupted feedback 转换成更准确、更可解释的 step-level credit assignment。

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

## 4. 已完成工作

当前代码已经不是单纯想法，而是一个可运行的 ECR-GRPO 原型。项目中已经完成了以下部分：

- synthetic long-horizon agent 环境；
- async wrapper，可以模拟 reward delay、timeout、interruption、missing reward、terminal reward delay；
- step-level pending buffer；
- 多种 credit refill kernel：`trajectory`、`uniform`、`recency`、`dependency`、`evidence`；
- no-oracle setting：不使用真实 `related_step_id`、`related_tool`、`related_subgoal`；
- non-local delayed feedback：reward 不直接奖励当前 step，而是延迟确认前面某个历史动作；
- 多 seed sweep：阶段 A 已在服务器完成，覆盖主实验、non-local v2、robustness sweep 与 evidence ablation；
- credit attribution diagnostic：不仅观察 success rate，还分析 reward 最终分到了哪个历史 step；
- 论文表格汇总脚本：`scripts/summarize_results.py` 已可从 `runs` 汇总生成 `runs/paper_tables/main_performance.csv`、`credit_diagnostic.csv`、`robustness_all.csv`、`robustness_success_pivot.csv`、`robustness_return_pivot.csv`、`ablation_performance.csv`、`ablation_diagnostic.csv` 和 `summary.md`；
- tabular policy 训练闭环；
- HF LoRA policy 占位实现；
- ALFWorld adapter 占位接口；
- train / eval / baseline / sweep / credit analysis runner。

对应代码主要位于：

- `src/ecr_grpo/types.py`：`StepRecord`、`AsyncEvent`、`CreditAssignment` 等核心数据结构；
- `src/ecr_grpo/envs/synthetic.py`：可控长任务 synthetic 环境与 non-local feedback；
- `src/ecr_grpo/envs/async_wrapper.py`：延迟、丢失、timeout、中断事件注入；
- `src/ecr_grpo/buffers.py`：`PendingStepBuffer`；
- `src/ecr_grpo/credit_kernels.py`：多种信用回填 kernel；
- `src/ecr_grpo/attribution.py`：Evidence attribution scorer；
- `src/ecr_grpo/rollout.py`：rollout 采样与事件回填流程；
- `src/ecr_grpo/advantages.py`：GRPO-style group-relative advantage；
- `src/ecr_grpo/trainer.py`：训练、评估、鲁棒性 sweep；
- `src/ecr_grpo/analyze_credit.py`：non-local credit diagnostic。

## 5. 问题定义

给定一个长任务 agent rollout，agent 在时刻 `t` 产生 step：

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

环境反馈不一定同步返回，而是以异步事件形式到达：

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

最后使用 GRPO / PPO-style clipped objective 或当前 tabular policy update 更新策略。

## 6. 项目的实现逻辑

当前实现可以理解为一个从“事件产生”到“策略更新”的闭环。

### 6.1 Base environment 产生任务事件

`SyntheticLongHorizonEnv` 维护一个目标动作序列。agent 每一步选择一个 action：

- 如果 action 等于当前 expected action，环境前进一步，并产生正向 `partial_reward`；
- 如果 action 错误且不是 `wait`，环境产生负向 `partial_reward`；
- 如果完成全部序列，环境产生 `terminal_success`；
- 如果达到最大步数还没完成，环境产生 `terminal_failure`；
- 如果开启 `non_local_credit`，环境会在后续 step 产生一个指向更早历史动作的 delayed support event。

这使 synthetic 环境既能提供可控训练任务，也能提供真实 causal step，用于事后诊断 credit assignment 是否正确。

### 6.2 AsyncEnvWrapper 扰动事件流

`AsyncEnvWrapper` 不改变 base env 的任务规则，而是在 base env 产生事件后，对事件流做扰动：

- `delay_prob`：部分 event 不立即返回，而是延迟若干 step；
- `max_delay_steps`：控制最大延迟步数；
- `terminal_reward_delay`：终局 reward 可以延迟到任务结束之后才到；
- `timeout_prob`：中途插入 timeout event，并给负 reward；
- `interruption_prob`：直接截断 rollout，并插入 interruption event；
- `missing_reward_prob`：让部分非终局 reward 丢失，模拟工具/API 没返回或日志缺失；
- `use_oracle_event_links=false`：移除 `related_step_id`、`related_tool`、`related_subgoal`，模拟真实环境没有精确因果标注的情况。

也就是说，轨迹不完整不是简单“少跑几步”，而是模拟真实 agent 训练中更麻烦的情况：

```text
step 已经发生
reward 没有马上来
有些 reward 延迟来
有些 reward 丢了
有些 timeout 插进来
有些 rollout 被中断
terminal reward 可能事后才到
```

### 6.3 Rollout 过程中持续消费 ready events

在 `collect_rollout_group` 中，每个 step 的流程是：

```text
action = policy.act(obs)
next_obs, reward, done, info = env.step(action)
buffer.add_step(step_record)

ready_events = env.pop_events()
for event in ready_events:
  buffer.assign_event(event, kernel)

finalized_steps += buffer.finalize_ready(env.current_time)
```

episode 结束后，再调用：

```text
drained_events = env.drain_events()
```

把延迟队列里剩下的事件全部取出并回填。这样，ECR-GRPO 可以同时处理“运行中到达的异步反馈”和“episode 结束后才到达的 terminal / delayed feedback”。

### 6.4 PendingStepBuffer 管理历史 step 生命周期

`PendingStepBuffer` 维护尚未最终完成信用分配的历史 step：

- `add_step`：把新 step 放入 buffer；
- `related_steps`：筛选与当前事件同 task、同 episode、发生时间早于事件 source time 的候选 step；
- `assign_event`：调用 credit kernel 计算权重，并把 `event.reward * weight` 累加到 `step.filled_credit`；
- `finalize_ready`：terminal 或超过 `max_pending_age` 的 step 被取出用于训练；
- `flush_episode`：episode 结束时把剩余 step 全部取出。

这里的关键是：step 不再只能等 trajectory 结束后一次性赋值，而是可以随着事件流到达被多次局部修正。

### 6.5 Credit kernel 决定事件如何分配给历史 step

当前实现了五类 kernel：

- `trajectory`：只在 terminal event 到达时把 reward 分给轨迹，用作 trajectory-level baseline；
- `uniform`：把 event reward 平均分给候选 pending steps；
- `recency`：越近的 step 得到越高权重，是很强的时间近邻 baseline；
- `dependency`：使用 `related_step_id`、tool、subgoal 等 oracle link，适合作为 upper-bound；
- `evidence`：主方法，不依赖精确 oracle link，而是使用时间、文本、tag、工具名、observation_delta 等弱证据做 attribution。

报告和论文中应明确：

> `dependency` 是 oracle / upper-bound baseline，`evidence` 才是主方法。

这样可以避免“benchmark 已经告诉算法 reward 属于哪个 step”的质疑。

### 6.6 Group-relative advantage 接入 GRPO

每个 step 的 return estimate 为：

```text
G_t = immediate_reward + filled_credit
```

当前实现中 `immediate_reward` 默认设为 0，让训练信号主要来自异步事件回填。随后 `compute_group_advantages` 在同一个 rollout group 内做归一化：

```text
A_t = (G_t - mean(G_group)) / (std(G_group) + epsilon)
```

这保留了 GRPO 的 group-relative 思想，但 reward 来源不再局限于完整 trajectory outcome，而是来自 event-conditioned step credit refill。

## 7. 核心技术

从论文表达上，本项目可以拆成六个核心技术模块。这样写会比单纯说“异步 wrapper + credit kernel”更清楚，也更容易对应实验设计。

### 7.0 技术模块总览

**Eventized Reward Interface.** 将环境反馈统一表示为异步事件流，包括 `partial_reward`、`tool_return`、`timeout`、`terminal_success`、`terminal_failure`、`interruption`。这样 reward 不再被看作 trajectory 结束后的单一标量，而是被建模为一个随时间到达的 event stream。

**Pending Step Buffer.** 每个 step 先进入 buffer，不立即根据整条 trajectory 成败更新，而是等待后续事件进行 credit refill。这样即使 trajectory 尚未结束、被截断，或者后续 reward 延迟到达，历史 step 仍然保留可被重新分配信用的机会。

**Evidence-Conditioned Credit Kernel.** 根据事件和历史 step 的时间距离、文本证据、工具证据、subgoal/tag 证据计算 credit 权重，避免依赖硬编码的 `related_step_id`。`dependency` kernel 只作为 oracle upper-bound，`evidence` kernel 才是主方法。

**Truncation-Aware Event Modeling.** 对不完整轨迹显式记录截断原因，例如 timeout、外部中断、工具错误、stuck、max_steps，并将其作为 event metadata 参与 credit 分配。这样不完整轨迹不会被简单丢弃，而是被转化为带原因的训练信号。

**Dynamic Interruption Penalty.** 中断惩罚不应长期使用固定值，而可以根据任务完成度、已完成 subgoal 数、历史有效 step 比例动态调整。当前代码已有固定 interruption / timeout penalty，后续优化可以把它升级为 completion-aware penalty，使负向 credit 更精确。

**GRPO-compatible Step Advantage.** 将回填后的 step return 做 group-relative normalization，接入标准 GRPO / StepPO-style policy update。这样 ECR-GRPO 不需要重写整个 RL 框架，而是作为 credit assignment 模块接入现有训练范式。

### 7.1 Event-Conditioned Credit Refill

ECR-GRPO 的核心技术是事件条件信用回填：

```text
w_{t,k} = normalize(K(e_k, b_t))
c_{t,k} = w_{t,k} * R_k
G_t = r_t_immediate + sum_k c_{t,k}
A_t = group_normalize(G_t)
```

其中 `K(e_k, b_t)` 是 credit kernel，用来估计事件和历史 step 的关联程度。它使训练信号从：

```text
整条 trajectory 成功 / 失败
```

变成：

```text
异步事件到达后，对历史 step 做局部、动态、可解释的信用分配
```

### 7.2 Evidence-Conditioned Attribution

主方法不要求环境提供精确 `related_step_id`。它只使用真实 agent 系统中更容易获得的弱证据：

- 时间证据：事件 source time 与历史 step 的距离；
- 文本证据：action、observation、observation_delta、event_type 的 token overlap；
- 工具证据：工具名、API 名、函数名、DOM selector、文件路径等；
- subgoal/tag 证据：planner subgoal、检索 query、任务阶段标签；
- 可选 trace 证据：如果系统天然有 request id、tool call id，可以作为额外证据，但不是必需条件。

当前规则 scorer 的形式可以写成：

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

后续版本可以将规则 scorer 升级为可学习 scorer：

```text
K_theta(e_k, b_t) = softmax_t(f_theta(phi(e_k, b_t)))
```

其中 `f_theta` 可以是线性模型、小型 MLP、cross-attention scorer，或由 embedding / LLM encoder 产生的相似度模型。

### 7.3 Non-local delayed feedback

仅仅证明“最近一步应该得 reward”是不够的，因为 recency baseline 在这种设定下天然很强。项目中加入 `non_local_credit` 的目的，是模拟更真实的反馈模式：

```text
当前事件不是确认最近 step，而是延迟确认更早的某个历史动作。
```

例如，在第 5 步收到一个 support event，但它真正支持的是第 2 步的动作。此时：

- `recency` 会偏向最近 step；
- `uniform` 会稀释 credit；
- `evidence` 可以利用 event 文本、tags、observation_delta 等证据，把更大 credit 分给真正相关的历史 step。

这正是 Evidence Refill 相比 Recency Refill 的技术优势所在。

### 7.4 Credit attribution diagnostic

项目不只看 success rate，还分析 credit 分配机制本身是否正确。`analyze_credit.py` 会对 non-local support event 做诊断：

- `target_weight_mean`：分给真实 target action 的平均权重；
- `recent_weight_mean`：分给最近 step 的平均权重；
- `argmax_target_rate`：最大 credit 是否给到了真实 target step；
- `target_top3_rate`：真实 target step 是否出现在 top-3 credit step 中；
- `target_credit_fraction`：真实 target step 获得的 credit 占比。

这使实验结论不只是“模型跑分更好”，而是能说明：

> Evidence Refill 改变了 delayed reward 被分配的位置，并且更接近真实 causal step。

### 7.5 Truncation-aware credit signal

长任务 agent 中的失败不只有 `terminal_failure` 一种形式。很多 rollout 是因为 timeout、工具错误、stuck、max_steps 或外部中断而不完整结束。如果把这些情况全部当作普通失败，会再次回到 trajectory-level 粗惩罚的问题。

因此后续优化中，截断事件需要携带更具体的 metadata：

```text
event_type = interruption / timeout / terminal_failure
metadata = {
  truncation_reason,
  progress,
  completed_subgoals,
  valid_step_ratio,
  last_tool,
  last_observation_delta
}
```

credit kernel 可以利用这些信息区分不同失败类型。例如：

- `timeout` 更可能惩罚最近的工具调用或等待动作；
- `tool_error` 更可能惩罚相关 tool/API 参数 step；
- `max_steps` 更可能惩罚低效循环或 stuck 片段；
- `external_interruption` 不应过度惩罚已经完成的有效中间步骤。

这一点可以作为 ECR-GRPO 相比普通“截断轨迹丢弃”或“整条轨迹负奖励”的重要扩展。

### 7.6 Dynamic Interruption Penalty

当前实现中 timeout / interruption penalty 还是固定值，例如 `timeout_penalty=-0.25`、interruption reward 为负值。后续可以改成动态惩罚：

```text
penalty = base_penalty
          * (1 - progress_ratio)
          * (1 - valid_step_ratio)
          * reason_scale(truncation_reason)
```

其中：

- `progress_ratio` 表示任务完成度；
- `valid_step_ratio` 表示历史 step 中有效动作比例；
- `reason_scale` 区分 timeout、tool_error、stuck、external_interruption 等原因。

这样可以避免一种不合理情况：一个已经完成大部分 subgoal 的 rollout 因外部中断被严重惩罚，而一个几乎没有进展且反复 stuck 的 rollout 只受到同样大小的惩罚。动态惩罚能让异常事件产生更局部、更公平的负向 credit。

## 8. 异步与中断任务的模拟方式

异步和中断模拟由 `AsyncEnvWrapper` 完成。它的设计原则是：不改变任务本身，只改变反馈到达方式。

### 8.1 Reward delay

base env 产生 event 后，wrapper 根据 `delay_prob` 和 `max_delay_steps` 决定是否延迟。延迟事件进入优先队列，只有当 `due_time <= current_time` 时才会被 `pop_events()` 返回。

这模拟工具返回、网页加载、API 响应、外部 evaluator 延迟打分等情况。

### 8.2 Terminal reward delay

`terminal_reward_delay` 允许终局成功/失败 reward 在 episode 结束后才到达。rollout 结束后通过 `drain_events()` 处理剩余事件。

这模拟真实系统里“任务已经结束，但评估结果稍后才返回”的情况。

### 8.3 Timeout

`timeout_prob` 会在 rollout 中途插入一个 `timeout` event，并给负 reward，例如 `-0.25`。它不一定终止 episode，但会给 pending steps 注入一个负向异步反馈。

这模拟工具调用超时、网页长时间无响应、API 请求失败等情况。

### 8.4 Interruption

`interruption_prob` 会直接把 rollout 截断，并插入 `interruption` terminal event。即使 episode 没有完整结束，前面已经发生的 step 仍可根据 interruption event 和已到达的 partial events 产生训练信号。

这模拟真实 agent 训练中常见的外部中断、资源限制、执行崩溃或任务被迫停止。

### 8.5 Missing reward

`missing_reward_prob` 会让部分非终局 reward 丢失。这样可以测试算法在反馈不完整时是否仍能稳定训练。

这模拟工具日志缺失、evaluator 漏评、观测状态没有及时记录等情况。

### 8.6 No-oracle setting

当 `use_oracle_event_links=false` 时，wrapper 会移除：

```text
related_step_id
related_tool
related_subgoal
```

但保留更弱、更真实的 evidence，例如：

```text
event_time
source_time
delay
observation_delta
metadata.tags
```

这使主方法不能直接读取“正确答案”，必须根据弱证据进行 attribution。

## 9. 当前实验结果与阶段性结论

当前阶段可以定位为：

> controlled diagnostic experiment：阶段 A 已完成，用于验证 ECR-GRPO 的信用分配机制本身有效，而不是宣称已经在真实 benchmark 上达到最终 SOTA。

阶段 A 的服务器实验已经从“能跑通”推进到“可汇总为论文表格”的状态。当前结果包由 `scripts/summarize_results.py` 统一整理，核心产物包括：

- `runs/paper_tables/main_performance.csv`：主实验与 non-local v2 的 success、return、credit mass、positive credit、entropy；
- `runs/paper_tables/credit_diagnostic.csv`：non-local event 的 attribution 质量，包括 target weight、target credit fraction、argmax target rate、top-3 target rate；
- `runs/paper_tables/robustness_all.csv` 与两个 pivot 表：不同 delay、missing reward、interruption、timeout 条件下的鲁棒性曲线；
- `runs/paper_tables/ablation_performance.csv` 与 `ablation_diagnostic.csv`：证据项消融后的性能与 attribution 变化；
- `runs/paper_tables/summary.md`：自动生成的论文解读草稿。

### 9.1 主实验：non-local no-oracle

5 seed 主实验结果为：

```text
trajectory success = 0.000, return = -0.289
uniform    success = 0.383, return =  0.338
recency    success = 0.675, return =  0.474
evidence   success = 0.725, return =  0.501
dependency success = 0.675, return =  0.474
```

这组结果支持两点：第一，trajectory-level final reward 在该异步长任务设置下完全无法训练起来；第二，no-oracle 的 Evidence Refill 相比 Recency Refill 有 +0.050 success 和 +0.027 return 的提升，相比 Uniform Refill 有 +0.342 success 和 +0.162 return 的提升。

### 9.2 更强 non-local v2

v2 设置下，非局部反馈更强，整体任务更难：

```text
trajectory success = 0.000, return = -0.271
uniform    success = 0.396, return =  0.354
recency    success = 0.642, return =  0.462
evidence   success = 0.654, return =  0.465
dependency success = 0.642, return =  0.462
```

这里 Evidence 对 Recency 的最终性能优势较小（+0.013 success，+0.003 return），但仍明显优于 Uniform。更重要的是，v2 的 credit diagnostic 显示 Evidence 的归因质量显著更强，这说明最终性能差距低估了 attribution 层面的改进。

### 9.3 Credit diagnostic

机制诊断是阶段 A 最关键的证据。主实验中：

```text
Evidence target_weight_mean = 0.418
Recency  target_weight_mean = 0.132
Uniform  target_weight_mean = 0.186

Evidence argmax_target_rate = 0.977
Recency  argmax_target_rate = 0.000
Uniform  argmax_target_rate = 0.431
```

v2 中差距更大：

```text
Evidence target_weight_mean = 0.491
Recency  target_weight_mean = 0.081
Uniform  target_weight_mean = 0.137

Evidence argmax_target_rate = 0.934
Recency  argmax_target_rate = 0.000
Uniform  argmax_target_rate = 0.463
```

也就是说，Evidence 在主实验中给真实相关 step 的权重大约是 Recency 的 3.17 倍，在 v2 中大约是 Recency 的 6.03 倍。同时，Evidence 几乎总能把最大 credit 给到真正相关的历史 step，而 Recency 的 argmax target rate 为 0，因为它系统性偏向最近 step。

### 9.4 Robustness sweep

鲁棒性实验覆盖 delay、missing reward、interruption 和 timeout。平均来看，Evidence 相比 Recency 的优势为：

```text
delay:          +0.038 success, +0.035 return
missing reward: +0.094 success, +0.084 return
interruption:   +0.021 success, +0.028 return
timeout:        +0.078 success, +0.048 return
```

其中 missing reward 和 timeout 是最稳定的优势场景。delay 下 Evidence 在 4/5 个点上 success 不低于 Recency，但在 `delay_prob=0.6` 时 Recency 略高；interruption 下在 `interruption_prob=0.1` 时 Recency 的 success 略高，但 Evidence 的 return 仍更高。因此论文中不应写成“Evidence 在所有扰动下都严格优于 Recency”，而应写成“Evidence 在时间近邻不可靠、反馈缺失或异常事件增强时更稳”。

### 9.5 Evidence ablation

消融结果显示 Evidence 的性能和归因质量会发生分化：

```text
no_tag      success = 0.743, return = 0.507, argmax_target = 0.014
no_text     success = 0.722, return = 0.497, argmax_target = 0.973
tag_only    success = 0.590, return = 0.436, argmax_target = 0.972
no_temporal success = 0.472, return = 0.343, argmax_target = 0.960
text_only   success = 0.368, return = 0.253, argmax_target = 0.690
```

这说明不能简单声称“去掉任一证据项都会降低 success”。更准确的表述是：不同证据源承担不同角色。temporal evidence 对最终性能很关键；text-only 明显不足；tag signal 对 argmax attribution 很强，但单独使用时 recent weight 也很高，容易形成过强的结构偏置；`no_tag` 虽然短期 success 最高，但 argmax_target_rate 几乎崩溃，说明它可能依赖更分散的 credit 而不是准确定位 causal step。

### 9.6 当前核心结论

目前可以形成一个清晰的论文逻辑：

- trajectory-level GRPO 在长任务、异步反馈下会错误惩罚整条轨迹；
- step-level refill 能缓解这个问题；
- 但简单 recency refill 在 non-local delayed feedback 下会错误偏向最近步骤；
- Evidence-conditioned refill 能根据事件证据，把 reward 分回真正相关的历史 step；
- 阶段 A 的主实验、robustness sweep 和 ablation 共同用于证明：Evidence 的优势不只是最终 success / return，而是 credit attribution 质量、非局部反馈鲁棒性和证据项贡献三者一致。

一句话总结：

> ECR-GRPO 的核心优势不是“异步系统”，而是把 delayed / partial / non-local feedback 转换成更准确的 step-level advantage。

### 9.7 阶段 B/C preliminary smoke

阶段 B 已经跑通 HF/LoRA synthetic smoke，并比较了 Evidence 与 Recency：

```text
Evidence:
update=0001 success=0.250 credit_causal=0.610
update=0005 success=0.250 credit_causal=1.000
update=0010 success=0.625 credit_causal=0.769

Recency:
update=0001 success=0.500 credit_causal=0.843
update=0005 success=0.250 credit_causal=0.853
update=0010 success=0.375 credit_causal=0.908
```

这说明 HF/LoRA 训练闭环已经能跑，并且不同 credit kernel 会产生可观察差异。到 update 10，Evidence 的 success 高于 Recency（0.625 vs 0.375），但 Recency 的 `credit_causal` 更高（0.908 vs 0.769）。因此这组结果应写成 HF/LoRA feasibility smoke，而不是正式性能结论。

阶段 C 已经跑通 ALFWorld `eval_out_of_distribution` split。日志显示共有 134 games，当前使用 dependency kernel：

```text
update=0001 success=0.000 credit_causal=0.995
update=0005 success=0.000 credit_causal=1.000
update=0010 success=0.250 credit_causal=0.957
update=0015 success=0.500 credit_causal=0.945
update=0020 success=0.000 credit_causal=1.000
```

这说明真实 benchmark adapter、事件生成、pending buffer、credit refill 和 policy update 已经能端到端跑通。但 success 波动很大，update 20 回到 0.000，说明当前仍是 small-scale smoke，不足以证明 external validity。下一步应固定 evaluation protocol、扩大任务数/seed，并补 Evidence vs Recency vs Dependency 的同预算对比。

注意：当前 HF/LoRA policy 的 `entropy` 日志为 0.000，是因为 `HFLoraPolicy.update()` 目前固定返回 `entropy: 0.0`，不能据此判断策略已经熵塌缩。后续需要单独实现 token-level 或 action-level entropy 统计。

## 10. 最终想达到的效果

ECR-GRPO 最终希望把训练信号从：

```text
整条 trajectory 成功 / 失败
```

推进到：

```text
异步事件到达后，对历史 step 做局部、动态、可解释的信用分配。
```

具体希望达到以下效果：

- 正确中间步骤即使在最终失败 trajectory 中，也能得到正 credit；
- 真正导致 timeout、错误工具调用或失败的附近步骤得到负 credit；
- 终局成功 reward 可以按 evidence 回填给相关历史步骤；
- rollout 中断时，不完全丢掉已经发生的训练信号；
- 缺失部分 reward 时，训练仍然比 trajectory-level baseline 更稳定；
- 在 non-local feedback 场景中，Evidence Refill 能比 Recency Refill 更准确地找到真正相关的历史 step；
- 在真实 agent benchmark 中，表现为更高 success rate、更好 sample efficiency、更强 delay / timeout / missing feedback robustness。

## 11. 通过什么方式展示效果

这里的“展示效果”主要包括两层：第一层是和其他 baseline 在 benchmark 指标上做直接对比，证明任务表现确实提升；第二层是通过 credit diagnostic 解释为什么提升，证明优势来自更准确的异步信用分配，而不是随机波动或调参偶然性。

### 11.1 Baseline 对比主表

主表应直接比较 ECR-GRPO 与其他 baseline 在同一 benchmark、同一训练预算、同一异步扰动设置下的指标。推荐 baseline 包括：

```text
Trajectory-level GRPO
Uniform Refill
Recency Refill
Dependency-aware Refill, oracle upper-bound
Evidence-conditioned Refill, ours
StepPO-style step reward, if available
```

主表建议报告：

- success rate；
- average return；
- average steps；
- sample efficiency；
- final performance across seeds；
- robustness AUC；
- credit assignment quality, synthetic only。

表格可以写成：

```text
Method                         Success ↑   Return ↑   Steps ↓   Robustness AUC ↑   Target Credit ↑
Trajectory-level GRPO           ...         ...        ...       ...                ...
Uniform Refill                  ...         ...        ...       ...                ...
Recency Refill                  ...         ...        ...       ...                ...
Evidence Refill, ours           ...         ...        ...       ...                ...
Dependency Refill, oracle        ...         ...        ...       ...                ...
```

在 synthetic benchmark 中，`Target Credit`、`argmax_target_rate` 这类指标可以进入主表或附表；在真实 benchmark 中，如果没有真实 causal label，则改用 proxy attribution metrics。

### 11.2 Benchmark 指标怎么比较

如果是在真实 benchmark 上展示，核心比较对象就是 benchmark 指标本身。例如：

- ALFWorld：task success rate、average steps to success、sample efficiency；
- ScienceWorld：score、task completion、steps / actions used；
- WebShop：purchase success、reward score、search/click efficiency；
- tool orchestration / BFCL multi-turn：exact match、tool-call success、argument correctness、multi-turn completion。

对比方式建议统一为：

```text
same benchmark
same policy backbone
same training budget
same async perturbation protocol
different credit assignment method
```

这样可以明确说明：提升来自 ECR-GRPO 的 credit assignment，而不是模型大小、训练轮数或环境设置不同。

### 11.3 鲁棒性曲线

分别 sweep：

```text
delay_prob
timeout_prob
interruption_prob
missing_reward_prob
terminal_reward_delay
trajectory length
non_local lag
```

观察随着异步程度变强，ECR-GRPO 是否比 trajectory baseline 下降更慢。

推荐图：

- success rate vs delay probability；
- success rate vs missing reward probability；
- success rate vs interruption probability；
- average return vs non-local lag。

鲁棒性曲线用于展示：当反馈越来越延迟、缺失或容易中断时，ECR-GRPO 是否比 trajectory-level GRPO、uniform refill、recency refill 下降更慢。

### 11.4 Learning curve / sample efficiency

除了最终指标，还应展示训练曲线：

- success rate vs updates；
- average return vs environment steps；
- success rate vs number of sampled trajectories；
- advantage variance / entropy vs updates。

如果 ECR-GRPO 的优势成立，理想现象是：在相同训练预算下更快达到同等 success rate，或者在同等训练步数下达到更高 success rate。

### 11.5 Credit diagnostic 图

至少需要三张图：

- target credit weight by kernel；
- recent-step credit weight by kernel；
- argmax target rate by kernel。

这三张图用于证明：

> Evidence Refill does not simply improve reward magnitude; it changes where delayed reward is assigned.

### 11.6 Ablation study

对 EvidenceKernel 做消融：

- 去掉 temporal 分量；
- 去掉 text overlap；
- 去掉 tag overlap；
- 去掉 tool/subgoal signal；
- 调整 `max_pending_age`；
- 调整 `lambda`、`tag_weight`、`text_weight`、`temporal_weight`。

目的是说明 Evidence attribution 不是单一 trick，而是多个弱证据组合带来的稳定提升。

### 11.7 可解释 case study

选一个 non-local event，展示：

```text
event: non_local_support:verify_fact:confirmed_after_step_5

历史 steps:
step 1: search_web
step 2: extract_fact
step 3: verify_fact
step 4: submit_code
step 5: answer

recency top step: step 5
evidence top step: step 3
true target step: step 3
```

这类 case study 能直观说明为什么 Evidence 比 Recency 更适合非局部延迟反馈。

### 11.8 最终展示逻辑

最终报告或论文可以按下面顺序展示：

1. Synthetic controlled benchmark：证明机制有效，能计算真实 causal credit。
2. Robustness sweep：证明异步、缺失、中断增强时，我们比 baseline 更稳。
3. Credit diagnostic：证明提升来自 credit 分配更准。
4. Ablation：证明 evidence、truncation metadata、pending window 等模块各有贡献。
5. Real benchmark：证明方法在 ALFWorld / ScienceWorld / WebShop / tool benchmark 上有 external validity。

也就是说，最终不是只比较一个 benchmark success rate，而是用 benchmark 指标证明任务提升，用 diagnostic 指标证明机制正确，用 robustness 曲线证明异步场景优势。

## 12. 真实 benchmark 如何体现技术与优势

真实 benchmark 的作用不是替代 synthetic diagnostic，而是证明 external validity：

> 在真实长任务 agent 环境中，ECR-GRPO 是否仍然提升 success rate、sample efficiency 和 delay robustness。

推荐顺序：

1. ALFWorld；
2. ScienceWorld；
3. WebShop；
4. tool orchestration / BFCL multi-turn。

### 12.1 接入方式

真实 benchmark 接入时，不需要重写 ECR-GRPO 主逻辑，只需要把环境输出转换为统一的 `AsyncEvent`：

```text
benchmark observation/action/result
  -> base env info/events
  -> AsyncEnvWrapper delay/timeout/missing/interruption
  -> PendingStepBuffer assign_event
  -> credit kernel
  -> group-relative advantage
  -> policy update
```

对于 ALFWorld / ScienceWorld：

- 每个环境动作作为一个 step；
- 环境状态变化、subgoal 完成、最终成功/失败转成 event；
- timeout / interruption 由 wrapper 注入；
- 如果环境没有精确 causal step，则使用 no-oracle evidence path。

对于 WebShop：

- 搜索、点击、查看商品、选择商品等动作作为 step；
- 页面加载、商品匹配、购买成功/失败转成 event；
- 延迟页面反馈和 missing feedback 可由 wrapper 注入。

对于 BFCL multi-turn / tool orchestration：

- 每次 tool call 或 API call 作为 step；
- tool return、参数错误、执行失败、最终 answer correctness 转成 event；
- tool name、function name、argument key、error text 可作为 evidence。

### 12.2 如何体现 ECR-GRPO 的优势

真实 benchmark 上不要只报告最终 success rate，而要设计能体现异步信用分配优势的 protocol：

**第一，做同步环境和异步扰动环境对比。**

在原始 benchmark 上先跑 baseline，再注入：

```text
delay_prob
timeout_prob
missing_reward_prob
interruption_prob
terminal_reward_delay
```

如果 ECR-GRPO 的优势来自异步 credit assignment，那么它应该在扰动增强时比 trajectory-level GRPO 掉得更慢。

**第二，做 no-oracle 设置。**

真实 benchmark 中默认不使用精确 `related_step_id`，只使用 observation/action/tool/tag/text 等弱证据。这样可以证明方法不是靠 benchmark 泄露因果答案。

**第三，把 DependencyKernel 作为 upper-bound。**

如果某些 benchmark 能提供精确 tool call id 或 subgoal id，可以额外跑 dependency kernel，但只作为 oracle upper-bound。理想结果是：

```text
trajectory < uniform / recency < evidence <= dependency upper-bound
```

这能说明 Evidence 已经接近 oracle link 的效果，但不依赖 oracle link。

**第四，强调 non-local feedback 场景。**

在真实任务中，很多反馈天然不是最近一步导致的。例如：

- WebShop 的最终购买成功可能依赖很早的搜索 query；
- ALFWorld 的最终状态成功可能依赖早期拿起某个物品；
- tool benchmark 的最终 answer correctness 可能依赖前几轮 API 参数选择；
- ScienceWorld 的实验成功可能依赖早期选择的实验对象或操作顺序。

这些场景是 Evidence Refill 展示优势的关键。

**第五，保留机制诊断。**

真实 benchmark 不一定有真实 causal step label，但仍可使用 proxy diagnostic：

- credit 是否集中到含有同一 tool/function/entity/tag 的历史 step；
- credit 是否过度集中到最近 step；
- 成功 episode 中的 positive credit 是否更集中到关键 action；
- interruption / timeout 的 negative credit 是否分配给相关失败动作；
- attribution entropy 是否随证据增强而下降。

### 12.3 真实 benchmark 的展示指标

真实 benchmark 建议报告：

- success rate；
- average return；
- sample efficiency；
- average steps to success；
- robustness AUC across delay / timeout / missing reward；
- interrupted rollout recovery；
- policy update stability：advantage variance、entropy、ratio / logprob stats；
- attribution proxy metrics：tool/tag/entity match credit mass、recent-step credit mass。

这样可以从三个层面证明方案优势：

1. 任务层面：成功率更高；
2. 训练层面：样本效率和稳定性更好；
3. 机制层面：credit 分配更接近相关历史动作。

## 13. 为什么这样设计

### 13.1 减少错误惩罚

长任务里最终失败不代表每一步都错。trajectory-level reward 会把很多正确中间步骤一起惩罚，导致策略学不到哪些动作其实值得保留。Event-conditioned refill 能把 reward 更细地分配到 step，从而减少错误惩罚。

### 13.2 更贴近真实异步 agent 环境

工具调用、网页加载、API 返回、外部 evaluator 打分本来就不是同步的。如果算法只能处理完整同步 trajectory，到了真实 agent 系统会很脆。ECR-GRPO 显式把反馈建模为 event stream，更贴近真实系统。

### 13.3 提高训练样本利用率

即使 rollout 被中断，前面已经发生的 step 仍然可以根据 interruption、timeout、partial reward 等事件产生训练信号。这比直接丢弃不完整 trajectory 更高效。

### 13.4 避免硬编码因果答案

主方法 EvidenceKernel 不要求环境告诉算法 reward 属于哪个 step，而是用时间、文本、工具名、tag、observation_delta 等弱证据做 attribution。DependencyKernel 保留为 oracle upper-bound，而不是主算法贡献。

### 13.5 兼容现有 GRPO / StepPO 训练框架

ECR-GRPO 不要求重写整个 RL 系统。它主要替换信用分配模块：

```text
trajectory reward -> event-conditioned step return
```

后续仍可以接入 GRPO-style relative advantage、PPO clipped objective、StepPO-style step MDP 或 LLM LoRA policy update。

### 13.6 让算法更可解释

每个 `CreditAssignment` 都记录：

```text
step_key
event_id
raw_reward
kernel_weight
assigned_credit
reason
```

因此可以追踪某个 event 的 reward 被分给了哪些历史 step，以及分配原因是什么。这对论文诊断、debug 和真实 agent 失败分析都很重要。

## 14. 当前风险与改进方向

### 14.1 Evidence scorer 仍是规则模型

当前 evidence scorer 使用手动权重。它已经避免了精确 causal link 硬编码，但仍不是可学习 credit model。后续可以升级为：

- linear scorer；
- MLP scorer；
- attention-based event-step scorer；
- embedding similarity scorer；
- LLM-assisted attribution scorer。

### 14.2 Pending window 可能漏掉长延迟事件

当前 `PendingStepBuffer` 使用 `max_pending_age` 控制过期。如果事件延迟超过 pending window，信用可能无法回填。后续可以加入：

```text
active pending buffer + retrospective replay buffer
```

新事件到达时不仅更新 active steps，也允许对近期 expired steps 进行低权重追溯修正。

### 14.3 Synthetic 结果已形成阶段 A 实验包，下一步是入稿与作图

阶段 A 已在服务器完成，并通过 `scripts/summarize_results.py` 汇总为 `runs/paper_tables` 下的一组论文表格。当前不再需要优先证明 synthetic 机制“能跑”，而是需要把结果整理为：

- 主性能表：展示 trajectory、uniform、recency、evidence、dependency 的 success / return 差异；
- credit diagnostic 表：展示 Evidence 是否把更多 reward 分配给真正相关历史 step；
- robustness 曲线：展示 delay、missing reward、interruption、timeout 增强时各 kernel 的下降趋势；
- evidence ablation 表：展示 tag、text、temporal 等证据源对性能和 attribution 的贡献；
- case study：选取一个 non-local delayed event，展示 evidence 如何将 credit 回填到历史 causal step。

主要风险也随之变化：现在风险不再是 synthetic 实验缺失，而是论文表达是否过度外推。阶段 A 只能证明 controlled synthetic diagnostic 下的机制有效性；真实 agent benchmark 仍应作为 external validity 另行验证。

### 14.4 HF/LoRA smoke 需要作为第二阶段

HF/LoRA 阶段的目的不是立刻刷大 benchmark，而是证明：

> ECR-GRPO 可以从 tabular policy 迁移到 LLM policy training loop。

建议先做：

- 小模型；
- 小 action space；
- synthetic text observation；
- LoRA；
- 少量 updates；
- 检查 logprob、advantage、entropy、loss 是否稳定。

## 15. 后续规划

### 阶段 A：synthetic 论文实验包

状态：已在服务器完成，并已通过 `scripts/summarize_results.py` 生成论文汇总表。

剩余工作不是继续大规模补跑，而是入稿整理：

- 将服务器 `runs/paper_tables` 同步到本地或论文工程；
- 根据 `main_performance.csv` 写主实验表；
- 根据 `credit_diagnostic.csv` 写机制诊断表；
- 根据 robustness pivot 表画 success / return 曲线；
- 根据 ablation 表写证据项贡献分析；
- 从 `credit_assignments.jsonl` 中挑选 1-2 个可解释 case study。

### 阶段 B：接小模型 HF/LoRA smoke

目标：证明训练闭环可迁移到 LLM policy。

需要观察：

- LoRA update 是否稳定；
- logprob 是否正常；
- advantage 是否正常；
- 不同 kernel 是否仍产生差异；
- 小规模 synthetic text task 上是否保持 evidence 优势。

### 阶段 C：接真实 agent benchmark

推荐顺序：

1. ALFWorld；
2. ScienceWorld；
3. WebShop；
4. tool orchestration / BFCL multi-turn。

目标：证明 external validity，即在真实长任务 agent 环境中，ECR-GRPO 是否仍然提升 success rate、sample efficiency 和 delay robustness。

## 16. 核心贡献总结

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

### C6：提出 truncation-aware event modeling

将 timeout、外部中断、工具错误、stuck、max_steps 等不完整轨迹原因显式建模为事件 metadata，而不是简单丢弃或统一视为失败，使异常事件能够转化为局部可分配的负向 credit。

### C7：提出 completion-aware interruption penalty 方向

进一步将固定 interruption penalty 扩展为动态惩罚，根据任务完成度、已完成 subgoal 数和历史有效 step 比例调整负向 reward，使不完整轨迹中的有效步骤仍然可以保留正向训练信号。

综合起来，论文贡献可以凝练为：

> 我们不是简单并行化 reward 计算，而是将异步反馈建模为 event stream，并提出一种 truncation-aware、evidence-conditioned 的信用回填机制，使不完整轨迹中的有效步骤仍然可以产生正向训练信号，同时将 timeout、interruption、tool failure 等异常事件转化为局部可分配的负向 credit。

## 17. 最终目标

这篇工作的最终目标可以定义为：

> 提出一种轻量、可插拔、无需精确 oracle step link 的异步信用分配机制，使现有 GRPO / StepPO 类 agentic RL 方法能够更稳定地训练长任务、延迟奖励和异步工具调用场景下的 LLM agents。

更凝练的论文式表述：

> ECR-GRPO turns delayed asynchronous feedback into step-level policy gradients through evidence-conditioned credit refill.

## 18. 参考工作

- STEP: https://arxiv.org/abs/2511.13091
- StepPO: https://arxiv.org/abs/2604.18401
- HiPER: https://arxiv.org/abs/2602.16165
- DPEPO: https://arxiv.org/abs/2604.24320
- Tool Orchestration: https://arxiv.org/abs/2603.24709
