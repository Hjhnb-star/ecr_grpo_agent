# ECR-GRPO 阶段性研究汇报

## 0. 汇报摘要

本项目关注长任务 LLM agent 训练中的异步信用分配问题。真实 agent 环境中的反馈往往不是同步、完整、立即对齐的 reward，而是以延迟、部分、缺失、非局部或中断后的事件形式到达。例如工具返回、网页状态变化、API 错误、timeout、interruption 和 terminal reward 都可能在若干 step 之后才出现，也不一定天然携带精确的因果 step 标注。

ECR-GRPO 的核心思路是：将这些反馈统一建模为 asynchronous event stream，并在事件到达时对 pending buffer 中的历史 step 做 evidence-conditioned credit refill，最后把回填后的 step-level return 转换为 GRPO-compatible group-relative advantage。换句话说，ECR-GRPO 试图将 delayed asynchronous feedback 转化为可解释、可训练的 step-level policy gradients。

当前项目已经完成 synthetic controlled benchmark、异步扰动 wrapper、pending step buffer、多种 credit refill kernel、no-oracle evidence attribution、credit diagnostic、主实验/鲁棒性/消融结果汇总脚本，以及 HF/LoRA 和 ALFWorld adapter 的初步接口。阶段 A 的 synthetic 结果已经能支持机制有效性；后续需要进一步完成小模型 HF/LoRA smoke，并逐步接入真实 agent benchmark 验证 external validity。

## 1. 研究问题

长任务 LLM agent 正在从单轮问答走向多步交互任务，例如工具调用、网页操作、软件环境控制、科学实验环境、购物环境和多 API 编排。这类任务有几个共同特点：

- 决策链长，单个 episode 可能包含十几步甚至几十步 action；
- reward 稀疏，最终成功或失败往往在任务结束后才知道；
- 中间步骤可能是正确的，但最终任务仍可能失败；
- 工具调用、网页加载、API 返回和外部 evaluator 天然存在延迟、失败和 timeout；
- rollout 可能因为 timeout、外部中断或环境异常而不完整。

传统 PPO / GRPO 类方法通常把一整条 trajectory 的最终结果作为主要训练信号。如果最终失败，整条轨迹或其中大量 token/action 都可能被负向更新。这在长任务中会造成明显的信用误分配：一些正确的中间动作会因为最终失败而被错误惩罚。

因此，本项目要解决的问题可以概括为：

> 当 partial reward、terminal reward、tool return、timeout、interruption 等反馈以 delayed / partial / non-local / interrupted event stream 形式到达时，如何将这些反馈合理回填到已经发生的历史 step，并形成更准确、更可解释的 step-level advantage？

这个问题不是单纯的异步工程实现，而是一个训练信号建模问题。

## 2. 方法概述

ECR-GRPO 的整体流程可以拆成四个核心模块：

1. **Eventized Reward Interface**
   将环境反馈统一表示为 `AsyncEvent`，包括 `partial_reward`、`tool_return`、`timeout`、`interruption`、`terminal_success`、`terminal_failure` 等。

2. **Pending Step Buffer**
   每个历史 step 先进入 pending buffer，不立即根据整条 trajectory 的最终成败固定训练信号，而是等待后续事件到达后动态回填 credit。

3. **Evidence-Conditioned Credit Refill**
   当事件到达时，credit kernel 根据事件和历史 step 之间的关联程度计算权重，将事件 reward 分配给候选历史 step。主方法 `evidence` 不依赖 oracle `related_step_id`，而是使用时间距离、文本 overlap、tag、tool/subgoal 和 observation delta 等弱证据做 attribution。

4. **GRPO-Compatible Step Advantage**
   每个 step 的累计回填信用形成 step-level return，再在 rollout group 内做 group-relative normalization，得到 GRPO-style advantage，用于策略更新。

核心公式为：

```text
w_{t,k} = normalize(K(e_k, b_t))
c_{t,k} = w_{t,k} * R_k
G_t = r_t_immediate + sum_k c_{t,k}
A_t = (G_t - mean(G_group)) / (std(G_group) + epsilon)
```

其中 `e_k` 是异步事件，`b_t` 是历史 step，`R_k` 是事件 reward，`K(e_k, b_t)` 是 credit kernel。

## 3. 与已有方法的区别

已有 STEP、StepPO、HiPER、DPEPO 和工具编排 reward 等工作已经开始关注 step-level optimization 或长任务探索问题，但大多仍默认 step transition 或 trajectory 最终是同步、完整、可对齐的。

ECR-GRPO 的区别在于：

- 不把 reward 只看作完整 trajectory 结束后的标量结果，而是建模为随时间到达的事件流；
- 不要求反馈立即对应当前 step，而允许事件延迟到达并回填历史 step；
- 不要求环境提供精确 `related_step_id`，主方法走 no-oracle evidence attribution；
- 不把 timeout / interruption 简单视为废轨迹，而是显式建模为可分配的训练事件；
- 不重写整个 RL 框架，而是作为 credit assignment 模块接入 GRPO / StepPO / PPO-style 更新。

因此，本工作的核心贡献不是“实现了异步 wrapper”，而是提出一种面向长任务 agent 的 event-conditioned step credit assignment 机制。

## 4. 已完成工作

当前项目已经形成一个可运行的 ECR-GRPO 原型，主要完成内容如下：

- `SyntheticLongHorizonEnv`：可控长任务 synthetic 环境，支持多步 action sequence、partial reward、terminal success/failure 和 non-local delayed feedback；
- `AsyncEnvWrapper`：支持 reward delay、terminal reward delay、timeout、interruption、missing reward，以及 no-oracle event link removal；
- `PendingStepBuffer`：维护尚未完成信用分配的历史 step，使后续事件可以回填到已经发生过的 step；
- `CreditKernel` 系列：实现 `trajectory`、`uniform`、`recency`、`dependency`、`evidence` 五种 credit refill 方式；
- `EvidenceKernel`：主方法，不依赖精确 `related_step_id`，而是使用时间、文本、tag、tool/subgoal 等弱证据做 attribution；
- `compute_group_advantages`：将回填后的 step return 转成 GRPO-style group-relative advantage；
- 训练、评估、baseline 对比、参数 sweep、credit analysis runner 已经跑通；
- 阶段 A 已在服务器完成多 seed synthetic 实验，并通过 `scripts/summarize_results.py` 生成论文表格；
- HF LoRA policy 和 ALFWorld adapter 已有初步接口，可作为后续真实 LLM agent benchmark 接入基础。

## 5. 阶段性实验结果

当前主要实验是 synthetic controlled diagnostic。它的目的不是替代真实 benchmark，而是在可控环境中验证 ECR-GRPO 的核心机制：异步、延迟、非局部反馈到达后，算法是否能把 credit 分配回更相关的历史 step。

### 5.1 主性能结果

在 no-oracle non-local delayed feedback 设置下，阶段 A 聚合结果显示：

```text
Method       Success   Return
Trajectory   0.000    -0.289
Uniform      0.383     0.338
Recency      0.675     0.474
Evidence     0.725     0.501
```

可以看到，trajectory-level reward 在当前长任务异步设置下基本学不起来，而 step-level refill 方法可以显著改善训练信号。其中 Evidence Refill 在 success rate 和 return 上高于 Uniform 和 Recency。

### 5.2 Credit diagnostic 结果

比最终 success rate 更关键的是 credit diagnostic，因为它直接回答 delayed reward 最终被分给了哪个历史 step。

主实验中：

```text
Method      Target Weight   Recent Weight   Argmax Target Rate
Evidence      0.418            0.176              0.977
Recency       0.132            0.347              0.000
Uniform       0.188            0.166              0.441
```

这说明 Evidence Refill 不只是增加 reward magnitude，而是改变 delayed reward 的分配位置。在 non-local delayed feedback 中，Recency 会明显偏向最近 step，而 Evidence 能把更高权重分给真正相关的历史 step。

### 5.3 鲁棒性与消融观察

鲁棒性 sweep 显示，Evidence 的优势主要体现在 missing reward 和 timeout 场景中：当反馈缺失或异常中断增强时，Evidence 相比 Recency 和 trajectory baseline 更稳定。

消融结果也提示一个重要现象：性能指标和 attribution 指标可能分化。例如某些变体在 success 上表现不差，但 `argmax_target_rate` 明显下降。因此后续论文中不能只报告 success rate，必须同时报告 credit diagnostic，才能证明方法确实改善了信用分配机制。

## 6. 当前可以支持的结论

基于阶段 A 的结果，目前可以较稳妥地支持以下结论：

1. ECR-GRPO 的 event stream、pending buffer、credit refill 和 GRPO-style advantage 闭环已经跑通。
2. 在长任务异步反馈设置下，trajectory-level GRPO 明显不足。
3. Step-level credit refill 能显著改善训练信号。
4. 在 non-local delayed feedback 中，Recency Refill 存在明显 recent-step bias。
5. Evidence Refill 能利用弱证据把 delayed reward 分配给更相关的历史 step。
6. Credit diagnostic 证明提升来自更合理的 credit assignment，而不仅是随机训练波动。

需要注意的是，当前 synthetic controlled diagnostic 证明的是机制有效性和可解释性，还不能单独证明真实 LLM agent benchmark 上的最终效果。真实 benchmark 仍需要作为后续 external validity 单独验证。

## 7. 当前风险与边界

当前方案仍有几个明确边界：

- Evidence scorer 仍是规则模型，不是 learned credit model；
- synthetic diagnostic 不能替代真实 agent benchmark；
- pending window 可能漏掉特别长延迟事件；
- timeout / interruption penalty 目前仍较简单，后续需要 completion-aware penalty；
- HF/LoRA policy 仍处于 smoke test 阶段；
- ALFWorld 等真实 benchmark adapter 虽有初步接口，但还需要多 seed、同预算、同扰动协议下的正式对比。

这些问题不影响阶段 A 的机制结论，但需要在论文表达中避免过度外推。

## 8. 后续计划

### 阶段 A：整理 synthetic 论文实验包

当前状态：已在服务器完成，并已通过 `scripts/summarize_results.py` 生成论文汇总表。

下一步工作：

- 将服务器 `runs/paper_tables` 同步到本地或论文工程；
- 根据 `main_performance.csv` 写主实验表；
- 根据 `credit_diagnostic.csv` 写机制诊断表；
- 根据 robustness pivot 表画 success / return 曲线；
- 根据 ablation 表写 evidence source 贡献分析；
- 从 `credit_assignments.jsonl` 中挑选 1-2 个可解释 case study。

### 阶段 B：接小模型 HF/LoRA smoke

目标是证明 ECR-GRPO 可以从 tabular policy 迁移到 LLM policy training loop。

需要重点观察：

- LoRA update 是否稳定；
- logprob 是否正常；
- advantage 是否正常；
- loss / entropy / ratio 等训练统计是否稳定；
- 不同 kernel 是否仍产生差异；
- 小规模 synthetic text task 上是否保持 evidence 优势。

### 阶段 C：接真实 agent benchmark

目标是证明 external validity，即在真实长任务 agent 环境中，ECR-GRPO 是否仍然提升 success rate、sample efficiency 和 delay robustness。

推荐顺序：

1. ALFWorld；
2. ScienceWorld；
3. WebShop；
4. tool orchestration / BFCL multi-turn。

真实 benchmark 中建议统一比较：

```text
same benchmark
same policy backbone
same training budget
same async perturbation protocol
different credit assignment method
```

这样可以尽量保证提升来自 ECR-GRPO 的 credit assignment，而不是模型大小、训练轮数或环境设置差异。


## 9. 一句话总结

ECR-GRPO 的核心目标是提出一种轻量、可插拔、无需精确 oracle step link 的异步信用分配机制，使现有 GRPO / StepPO 类 agentic RL 方法能够更稳定地训练长任务、延迟奖励和异步工具调用场景下的 LLM agents。

