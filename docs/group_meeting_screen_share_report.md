# ECR-GRPO 组会汇报说明文稿

## 0. 一句话概括

本工作的核心目标不是提出一个新的强化学习优化器，而是提出一个可以无缝接入标准 GRPO 流水线的异步信用回填模块。

更具体地说，ECR-GRPO 解决的是长程 LLM Agent 训练中 reward 进入 GRPO 之前的一个关键问题：当反馈是延迟到达、缺失、中断、timeout 或者与早期步骤相关的非局部事件时，应该如何把这些反馈分配给真正应该被训练的历史步骤。

因此，本文的定位是：

> ECR-GRPO is a reward-construction plug-in for GRPO, not a replacement optimizer.

也就是说，我们不修改 GRPO 的核心更新逻辑。ECR-GRPO 只负责把异步事件反馈转成 GRPO 可以消费的 step-level 或 trajectory-level reward，后续的 group-relative advantage、ratio clipping 和 policy update 都保持标准 GRPO。

---

## 1. 论文整体框架和逻辑

这一部分需要向老师说明：论文不是简单介绍一个工程模块，而是按照“问题提出 - 方法边界 - 形式化定义 - 算法设计 - 实验验证 - 讨论局限”的逻辑展开。整体结构可以分为八个部分。

### 1.1 Introduction：提出核心问题和论文定位

Introduction 主要回答三个问题。

第一，为什么长程 LLM Agent 需要重新考虑 reward assignment。

LLM Agent 的交互通常不是单步输入输出，而是多轮动作序列：观察环境、选择文本动作、调用工具、等待工具返回、继续推理或执行下一个动作。在这个过程中，反馈经常不是同步到达的。比如：

1. 工具调用的结果可能几步之后才返回。
2. evaluator 可能只在 episode 结束后给出整体成功或失败。
3. 某个错误可能由早期动作导致，但失败信号在后面才出现。
4. timeout、interruption 或 missing feedback 会导致部分反馈缺失。
5. 某个晚到的 positive event 可能支持的是早期正确动作，而不是最近一步。

因此，长程 Agent 的训练问题不仅是“如何用 GRPO 更新策略”，还包括“晚到的反馈应该归因给哪一个历史步骤”。

第二，为什么现有 GRPO 流水线没有解决这个问题。

GRPO 解决的是 policy optimization，它假设每个 sample 已经有了 reward，然后做 group-relative advantage、ratio clipping 和 policy update。但是 GRPO 本身不定义异步 event 如何转成 reward。换句话说，GRPO 可以消费 reward，但不负责构造 reward。

第三，本文的定位是什么。

论文在 Introduction 中明确强调：ECR-GRPO 不是替代 GRPO 的新优化器，而是 GRPO 前面的 reward-construction plug-in。它的作用是把 delayed、missing、non-local 的 event feedback 转换成 GRPO 可消费的 step return 或 trajectory return。

### 1.2 Related Work：说明本文和已有工作的区别

Related Work 主要分成三条线。

第一条线是 PPO / GRPO / policy optimization。

这部分说明 GRPO 是当前 LLM reasoning 和 agent training 中常用的优化框架，但本文不修改 GRPO 的优化目标，而是补充 GRPO 前面的 reward construction。

第二条线是 process reward 和 step-level supervision。

这些工作强调中间步骤监督的重要性，但它们通常假设 step-level reward 已经被对齐好了。本文的问题更前置：如果反馈是异步到达的，那么 step-level reward 本身怎么来。

第三条线是 long-horizon agent 和 delayed reward。

长程 Agent benchmark，比如 ALFWorld、WebShop，天然存在稀疏反馈、多步依赖和延迟反馈。传统 delayed reward 方法更偏 RL temporal credit，而本文强调 LLM Agent trace 中还有语义证据，比如动作文本、工具名、metadata tag 和 observation change。

这一部分的逻辑作用是把本文和已有工作区分开：别人主要解决“怎么优化 reward”，我们解决“异步反馈如何先变成可靠 reward”。

### 1.3 Problem Setting：形式化异步反馈和回填目标

Problem Setting 负责把问题定义清楚。

论文先定义 step record：

\[
b_t = (\text{task}, \text{episode}, \text{group}, t, o_t, a_t, \ell_t, \mathcal{A}_t, m_t)
\]

它表示第 \(t\) 步的历史动作信息，包括 observation、action、old log probability、candidate action set 和 metadata。

然后定义 asynchronous event：

\[
e_k = (\text{task}, \text{episode}, k, \tau_k, R_k, \Delta o_k, y_k, \mu_k)
\]

它表示晚到或异步到达的反馈，包括 event time、event reward、event type、observation delta 和 event metadata。

接着定义核心目标：当 event \(e_k\) 到达时，需要在 pending historical steps 中计算分配权重 \(w_{t,k}\)，并把 event reward 回填为：

\[
c_{t,k}=w_{t,k}R_k
\]

最终 step return 是：

\[
G_t = r_t + \sum_k c_{t,k}
\]

这里 \(r_t\) 是同步即时奖励，\(c_{t,k}\) 是异步 event 回填信用，两者不重复。

这一部分的作用是告诉老师：本文不是笼统地说“做 credit assignment”，而是把异步事件、候选历史步骤、回填权重和最终 GRPO reward 都形式化定义了。

### 1.4 Method：提出 ECR-GRPO 的核心机制

Method 是论文主体，分成四个模块。

第一个模块是 Pending Step Buffer。

每个动作执行后不会立刻从训练视野中消失，而是进入 pending buffer。event 到达时，只在同一个 task/episode 中、时间早于 event、仍然没有过期的步骤里做 credit assignment。

第二个模块是 Evidence-conditioned Attribution。

这个模块负责回答“event 更应该分配给哪个历史 step”。它不是只看最近一步，而是综合时间距离、文本证据、工具匹配、metadata tag 和 observation delta，为每个 event-step pair 计算 affinity score，再归一化成 credit weight。

第三个模块是 Event-gated Routing。

不同 event 不应该用同一种归因规则。短延迟的局部反馈更适合 recency，长延迟的非局部反馈更适合 evidence，terminal event 使用两者混合。这样可以避免 pure evidence 破坏局部 next-action learning，也避免 pure recency 错分非局部反馈。

第四个模块是 GRPO-compatible Adapter。

ECR 只负责生成 reward。adapter 把回填后的 \(G_t\) 或 trajectory return 组织成标准 GRPO sample，然后交给不变的 GRPO pipeline。

### 1.5 Experiments：验证 reward construction 是否真的有效

实验部分不是单纯报告 success rate，而是围绕几个问题设计。

第一，任务效果是否提升。

比较 terminal、uniform、recency、evidence、gated 等 reward construction，在相同 GRPO optimizer 下看 success rate、return 和 learning AUC。

第二，信用分配是否更准确。

在 controlled benchmark 中，我们知道 delayed event 对应的真实 target step。训练时不使用这个 oracle，只在评估时计算 target weight、recent weight、argmax target rate 等 credit metrics。

第三，方法是否依赖 oracle metadata。

去掉 `related_step_id`、`related_tool`、`related_subgoal` 等字段，只让 evidence kernel 使用 public trace evidence。

第四，gating 是否必要。

比较 pure recency、pure evidence 和 gated evidence，证明 gated 方法能同时保留局部学习和非局部归因。

第五，能否迁移到 LLM policy 和真实 benchmark。

用 LoRA-fine-tuned language model 做 candidate-action scoring，并进一步在 ALFWorld 这类真实长程文本动作环境上接入 async wrapper。

### 1.6 Results and Discussion：解释为什么有效

Results 不只是说哪个方法分数高，而是解释机制。

如果 Evidence 的 target weight 更高、recent weight 更低，说明它不是简单增加 reward，而是改变了 delayed reward 的分配位置。

如果 Gated 在 LLM policy 上比 pure evidence 更稳定，说明局部反馈和非局部反馈确实需要不同 credit rule。

Discussion 进一步强调：ECR-GRPO 和 step-level optimization 是互补关系。step-level optimization 解决“用 step reward 怎么优化”，ECR-GRPO 解决“step reward 怎么从异步 event 中构造出来”。

### 1.7 Limitations：主动说明边界

论文最后说明方法的限制。

1. Evidence kernel 依赖 trace 中存在可恢复的弱证据。
2. 真实 benchmark 不一定有 ground-truth credit label。
3. 门控阈值、pending window、terminal mixing coefficient 需要敏感性分析。
4. 当前 scorer 是透明规则式的，表达能力有限。
5. 本方法解决 credit assignment，不直接解决 exploration、base model capability 或 action space 设计问题。

### 1.8 整体逻辑总结

整篇论文的逻辑链条可以总结为：

1. 长程 Agent 中 feedback 经常异步到达。
2. 标准 GRPO 需要 reward，但不负责异步 reward construction。
3. 如果 delayed feedback 被广播到整条轨迹或简单分给最近一步，会产生错误训练信号。
4. ECR-GRPO 把 feedback 建模成 AsyncEvent，把动作保存在 PendingStepBuffer 中。
5. Credit kernel 根据 public trace evidence 计算 event-step affinity，把 event reward 回填到相关历史步骤。
6. 回填后的 reward 交给标准 GRPO，optimizer 不变。
7. 实验从 task performance 和 credit localization 两个层面验证：提升来自更准确的 reward assignment，而不是换了优化器。

---

## 2. 方案的 insight、创新点和定位

### 2.1 核心 insight

本文最核心的 insight 是：

> 在长程 Agent 强化学习中，训练不稳定或效果差，很多时候不是因为 GRPO 优化器本身不够好，而是 reward 在进入 GRPO 之前已经被错误分配了。

举例来说，一个 episode 最后失败，不代表所有步骤都错。前面可能有正确的检索、正确的工具调用、正确的中间推理，只是最后格式错误或某一步决策失败。如果把 terminal failure 广播到整条轨迹，就会错误惩罚这些有价值的步骤。

反过来，一个晚到的 positive event 可能对应早期某个工具调用。如果简单使用 recency heuristic，就会把奖励分给最近一步，而真正起作用的早期动作没有被强化。

所以，本文认为长程 Agent 训练需要一个独立的 credit assignment layer，专门负责把异步反馈变成更可靠的 GRPO reward。

### 2.2 方法定位：GRPO 的前置增强模块，而不是新 RL 算法

这个定位非常重要。我们不把 ECR-GRPO 写成“一个新的 RL optimizer”，原因有三点：

1. 我们没有修改 GRPO 的目标函数、advantage normalization 或 policy update。
2. 如果声称是新 RL 算法，会被要求证明收敛性、无偏性、单调改进等理论性质，这不是本文贡献的重点。
3. 实际落地上，插件式 reward construction 更容易接入现有 GRPO 训练系统。

因此，论文强调：

1. ECR-GRPO 改的是 reward construction。
2. 标准 GRPO 继续负责 policy optimization。
3. 所有实验中，optimizer 固定为 GRPO，只比较不同 reward source。

这使得贡献边界更清楚，也更符合 Agent 系统型工作的评价习惯。

### 2.3 创新点

本文的创新点可以概括为四个层次。

第一，提出异步事件视角。

我们不再把反馈简单看成同步 reward，而是统一建模为 AsyncEvent。事件可以包括 terminal success、terminal failure、partial reward、tool return、timeout、interruption、non-local support 等。

第二，提出 pending step buffer 和 credit refill。

每个 action 执行后都会形成一个 StepRecord，并进入 pending buffer。当异步事件到达时，只在同一个 task/episode 中、发生在 event 之前、仍然 eligible 的步骤中分配 credit。

第三，提出 no-oracle evidence attribution。

训练时不使用 `related_step_id`、`related_tool`、`related_subgoal` 这类 oracle causal link。credit kernel 只能使用真实可获得的 public trace evidence，比如：

1. 时间距离。
2. action text 和 event text 的相似度。
3. tool name 或 API metadata。
4. public tag。
5. observation delta。

这样可以避免方法变成 benchmark-specific 的人工规则。

第四，提出 event-gated routing。

不同类型的反馈不应该强行用同一种 credit rule。局部短延迟反馈适合 recency，非局部长延迟反馈适合 evidence，terminal event 可以用混合权重。这样可以同时保留局部学习信号和非局部归因能力。

---

## 3. 方案实现逻辑和核心细节

### 3.1 整体训练流水线

完整流程可以拆成五步。

第一步，采样 rollout group。

同一个 task 或 prompt 下采样多条 trajectories，保持 GRPO 所需的 group 结构。

第二步，记录 step。

每执行一个动作，系统保存一个 StepRecord，包括 observation、action、old logprob、candidate action space、step metadata、immediate reward 等。

第三步，接收 async event。

环境反馈被包装成 AsyncEvent，包括 event time、reward、event type、observation delta 和 metadata。event 不一定立刻到达，也不一定和最近一步相关。

第四步，做 ECR credit refill。

当 event 到达时，pending buffer 先选出 eligible historical steps。然后 credit kernel 计算每个 step 的权重 \(w_{t,k}\)，并回填：

\[
c_{t,k}=w_{t,k}R_k
\]

最终每个 step 的 return 是：

\[
G_t = r_t + \sum_k c_{t,k}
\]

其中 \(r_t\) 是同步即时奖励，\(c_{t,k}\) 是异步事件回填信用，二者不重复计算。

第五步，交给标准 GRPO。

GRPO adapter 把回填后的 reward 转成标准 GRPO sample，包含 group id、prompt、completion 和 reward。后续流程保持标准 GRPO：

1. group-relative advantage normalization。
2. clipped ratio objective。
3. entropy regularization。
4. policy update。

### 3.2 Evidence kernel 的细节：如何判断 event 应该分配给哪个历史 step

Evidence kernel 的核心任务是：当一个异步 event 到达时，在 pending buffer 里找到最相关的历史 step，并给这些 step 分配不同的 credit weight。

它不是直接做硬分类，也不是只选一个 step，而是对每一个 event-step pair 计算一个相关性分数，然后把分数归一化成分配权重。

#### 3.2.1 第一步：确定候选历史步骤

当 event \(e_k\) 到达时，系统不会在所有历史步骤里查找，而是先用 pending buffer 缩小候选集合。

候选步骤集合记为：

\[
\mathcal{B}(e_k)
\]

一个 step 要进入这个集合，需要满足几个条件：

1. 和 event 属于同一个 task。
2. 和 event 属于同一个 episode。
3. step 的环境时间早于 event 的到达时间。
4. step 还没有因为 terminal finalization 或 pending window expiration 被移出 buffer。

这样做的作用是先保证基本因果顺序：event 不能反向分配给未来动作，也不能跨 episode 分配给无关动作。

#### 3.2.2 第二步：为每个 event-step pair 提取证据

对于候选集合中的每个历史 step \(b_t\)，Evidence kernel 会比较 event \(e_k\) 和 step \(b_t\) 之间的多种证据。

第一类证据是时间证据。

时间距离定义为 event time 和 step time 的差：

\[
d_{t,k} = \tau_k - t
\]

一般来说，距离越近，相关性越高。但这里时间只是一个弱信号，不会直接决定最终归因。原因是很多非局部 delayed event 本来就应该指向更早的 step。

第二类证据是文本证据。

LLM Agent 的动作通常是文本动作，比如 `open fridge`、`take apple`、`search product`、`call weather API`。Event 的 observation delta 或反馈文本中也可能出现实体、工具名、目标对象或错误描述。

Evidence kernel 会比较：

1. action text 中的关键词。
2. event observation delta 中的关键词。
3. event metadata 中的文本字段。
4. step metadata 中的文本字段。

如果 event 里出现的对象、工具或子目标和某个历史 action 更匹配，那么这个 step 的 evidence score 会更高。

第三类证据是工具或 API 匹配。

在工具调用型 Agent 中，event 很可能来自某个工具返回。例如 event 表示 search API 返回了结果，那么之前调用 search API 的 step 就比最近的普通文本动作更相关。

因此 kernel 会检查：

1. event metadata 中是否有 tool name。
2. step metadata 中是否记录了 tool name。
3. 二者是否一致或部分匹配。

如果匹配，就给该 step 增加 tool-based score。

第四类证据是 public tag 匹配。

有些环境或 wrapper 会记录非 oracle 的公共标签，比如 subtask name、object tag、room name、operation type 等。这些 tag 不直接告诉我们 related step id，但可以提供弱归因线索。

例如 event 的 tag 是 `inventory`，某个历史 step 也带有 `inventory` 或相关对象标签，那么该 step 更可能与 event 有关。

第五类证据是 observation delta 匹配。

Event 往往携带环境变化，比如：

1. 某个物体被拿起。
2. 某个门被打开。
3. 工具返回了某个实体的信息。
4. 当前状态发生了与早期动作相关的变化。

Kernel 会比较这个变化和历史 action / observation 是否对应。如果某个 step 的 action 可以解释当前 delta，这个 step 的分数会增加。

#### 3.2.3 第三步：计算 event-step affinity score

对每个候选 step，Evidence kernel 会把上述证据组合成一个 affinity score：

\[
K(e_k,b_t)
=
\alpha_{\mathrm{time}}s_{\mathrm{time}}
+ \alpha_{\mathrm{text}}s_{\mathrm{text}}
+ \alpha_{\mathrm{tool}}s_{\mathrm{tool}}
+ \alpha_{\mathrm{tag}}s_{\mathrm{tag}}
+ \alpha_{\Delta}s_{\Delta}
\]

这里每一项的含义是：

1. \(s_{\mathrm{time}}\)：时间接近程度。
2. \(s_{\mathrm{text}}\)：event 文本和 action 文本的语义或词面重叠。
3. \(s_{\mathrm{tool}}\)：工具名、API 名或调用类型是否匹配。
4. \(s_{\mathrm{tag}}\)：公共 metadata tag 是否匹配。
5. \(s_{\Delta}\)：observation delta 是否能由该 step 解释。

对应的 \(\alpha\) 是每类证据的权重，用来控制不同证据的重要性。

这一步的关键是：kernel 不需要知道真实 causal step id。它只使用训练时真实可见的 public trace evidence，因此是 no-oracle 的。

#### 3.2.4 第四步：把 affinity score 转成 credit weight

算出每个候选 step 的分数后，需要把分数变成 event reward 的分配比例。

通常用 softmax 或归一化：

\[
w_{t,k} =
\frac{\exp(K(e_k,b_t)/T)}
{\sum_{j}\exp(K(e_k,b_j)/T)}
\]

其中 \(T\) 是 temperature，用来控制分配是更尖锐还是更平滑。

如果 temperature 较低，reward 会更集中地给最高分 step。

如果 temperature 较高，reward 会更分散地给多个相关 step。

最后满足：

\[
\sum_{t\in \mathcal{B}(e_k)} w_{t,k}=1
\]

然后 event reward \(R_k\) 被分配为：

\[
c_{t,k}=w_{t,k}R_k
\]

这就是“异步事件信用回填”的实际计算过程。

#### 3.2.5 第五步：记录 attribution diagnostics

为了证明方法真的做了正确归因，kernel 不只输出权重，还会记录诊断信息。

包括：

1. 每个 step 的 assigned weight。
2. attribution entropy，表示分配是否过于分散。
3. top margin，表示最高权重和次高权重之间的差距。
4. selected route，表示 event 是走 recency、evidence 还是 mixture。
5. explanation string，说明这个分配主要来自哪些证据。

这些诊断非常重要，因为论文不能只说 success rate 变高，还要证明 success rate 变高的原因是 delayed reward 被分配到了更合理的历史步骤。

#### 3.2.6 一个直观例子

假设 Agent 有以下历史步骤：

1. Step 1：`search for the red key`
2. Step 2：`open the drawer`
3. Step 3：`move to the hallway`

几步之后，一个 event 到达：

> tool result: red key location found in drawer

如果使用 recency，这个 event 可能会被分给 Step 3，因为 Step 3 最近。

但 Evidence kernel 会发现：

1. event 中有 `red key` 和 `drawer`。
2. Step 1 中有 `red key`。
3. Step 2 中有 `drawer`。
4. Step 3 只是最近，但文本和工具反馈不相关。

因此它会把更高权重分给 Step 1 和 Step 2，而不是盲目分给 Step 3。

这个例子体现了本文的核心思想：长程 Agent 的 delayed feedback 往往带有语义线索，credit assignment 应该利用这些线索，而不是只依赖时间最近性。

### 3.3 Gated routing 的细节

Gating 的动机是：pure evidence 不一定总是最好。

对于 local partial feedback，例如“上一步动作错误”或“当前子目标完成”，最有用的训练信号通常就是最近一步。如果把这类局部反馈分散给多个语义相关步骤，可能会削弱 next-action learning。

对于 non-local support event，recency 反而容易错，因为事件虽然现在才到，但真正相关的是更早的动作。

因此，我们用 event delay 做形式化门控：

\[
\Delta_k = \tau_k - \max\{t \mid b_t \in \mathcal{B}(e_k)\}
\]

门控规则是：

1. 短延迟、非终端事件：走 Recency。
2. 长延迟、非终端事件：走 Evidence。
3. Terminal event：走 Recency 和 Evidence 的混合。

这样做的好处是规则可复现、可消融，也避免了“人工手动分类事件”的质疑。

---

## 4. 实验设计：每组实验验证什么

### 4.1 主实验：不同 reward construction 的任务性能对比

目的：验证 ECR 回填后的 reward 是否能提升训练效果。

比较方法：

1. GRPO-Terminal：只使用 terminal reward。
2. GRPO-Uniform：把 terminal reward 均匀分给所有步骤。
3. GRPO-Recency：根据时间距离分配 event credit。
4. GRPO-Evidence：用 no-oracle evidence 做归因。
5. GRPO-Gated：短延迟走 recency，长延迟走 evidence，terminal 走混合。

需要观察的指标：

1. success rate。
2. average return。
3. learning AUC。
4. steps to success。
5. final success 和 peak success。

这个实验回答：在 optimizer 完全相同的情况下，更好的 reward construction 是否带来更好的 policy learning。

### 4.2 Credit localization 实验

目的：验证性能提升是否真的来自更准确的信用分配，而不是 reward magnitude 或偶然调参。

在 controlled synthetic benchmark 中，我们可以知道 delayed event 真正对应哪个 target step。但训练时不把这个 oracle link 给 credit kernel，只用于 evaluation。

指标包括：

1. target weight：分给真实目标步骤的平均权重。
2. recent weight：分给最近一步的平均权重。
3. argmax target rate：权重最大的步骤是否是真实目标步骤。
4. top-3 target rate：真实目标步骤是否出现在权重 top-3 中。
5. attribution entropy：分配是否过于分散。
6. top margin：最高权重和次高权重的差距。

这个实验回答：ECR-GRPO 是否真的把 delayed reward 分配给了更合理的历史步骤。

### 4.3 No-oracle 实验

目的：证明方法不是依赖 benchmark 提供的人工 causal link。

训练时去掉以下字段：

1. `related_step_id`
2. `related_tool`
3. `related_subgoal`

Evidence kernel 只能用 public trace evidence。Dependency-aware kernel 可以作为 oracle upper bound，但不作为 deployable method。

这个实验回答：在真实部署更接近的条件下，方法是否仍然有效。

### 4.4 Gating 消融实验

目的：证明 event-gated routing 是必要的。

比较：

1. pure recency。
2. pure evidence。
3. gated evidence。

预期现象是：

1. Recency 对局部反馈较好，但非局部 delayed event 容易分错。
2. Evidence 对非局部 delayed event 更准，但可能削弱局部 next-action learning。
3. Gated 同时保留局部学习能力和非局部归因能力。

这个实验回答：为什么不能只用一个统一 credit kernel。

### 4.5 超参与鲁棒性实验

目的：回应方法是否依赖特定超参。

建议做以下敏感性分析：

1. delay threshold \(\delta\)：例如 1、2、3、5。
2. terminal mixing coefficient \(\alpha\)。
3. pending window size。
4. event delay length。
5. event drop rate。
6. timeout/interruption rate。

这个实验回答：方法在不同异步强度和反馈缺失条件下是否稳定。

---

## 5. 真实 benchmark 上如何落地

### 5.1 推荐真实 benchmark：ALFWorld

ALFWorld 适合作为第一阶段真实 benchmark，原因是：

1. 它是长程文本动作环境。
2. 每一步有 admissible actions，适合 candidate-action scoring。
3. 奖励通常比较 sparse，符合长程 credit assignment 问题。
4. episode 中存在多步依赖，终局成功往往取决于早期动作是否正确。

### 5.2 ALFWorld 实验实现方式

真实 benchmark 中不改变环境本身的任务目标，而是在环境外加 async wrapper。

具体流程：

1. Agent 根据 observation 从 admissible actions 中选择动作。
2. 每一步动作保存成 StepRecord。
3. 原始环境反馈、terminal success、failure、timeout 等被转成 AsyncEvent。
4. async wrapper 可以人为加入 delay、drop、interruption，用来构造异步压力测试。
5. ECR credit kernel 对 pending steps 做信用回填。
6. GRPO adapter 输出标准 GRPO samples。
7. 后续用相同 GRPO optimizer 训练。

### 5.3 ALFWorld 上的对比设置

为了保证公平，所有方法保持相同：

1. 相同 base model。
2. 相同 LoRA 设置。
3. 相同 rollout budget。
4. 相同 random seeds。
5. 相同 candidate action space。
6. 相同 GRPO optimizer。

唯一变化是 reward construction：

1. terminal GRPO。
2. uniform refill。
3. recency refill。
4. evidence refill。
5. gated ECR refill。

### 5.4 ALFWorld 上需要看的指标

任务指标：

1. final success rate。
2. peak success rate。
3. average return。
4. average episode length。
5. timeout rate。
6. failure rate。

训练效率指标：

1. learning AUC。
2. steps to reach certain success threshold。
3. variance across seeds。

信用诊断指标：

1. target weight，如果 async perturbation 中可构造 pseudo target。
2. recent weight。
3. attribution entropy。
4. top margin。
5. event route distribution。

真实 benchmark 的核心结论应该是：在不改变 GRPO optimizer 的前提下，gated ECR reward source 能在异步扰动下保持或提升任务成功率，并且比 recency 更能处理非局部 delayed feedback。

---

## 6. 当前方案的不足

### 6.1 Evidence kernel 仍然依赖 trace evidence 的质量

如果 event text 很短、metadata 很少、工具名缺失，或者多个动作高度相似，Evidence kernel 就很难区分到底哪个历史步骤更相关。

也就是说，方法不是凭空恢复因果关系，它依赖 trace 中存在可利用的弱证据。

### 6.2 真实 benchmark 缺少 ground-truth credit label

在 synthetic benchmark 中可以知道真实 target step，因此可以直接评估 credit localization。

但在真实环境中，通常没有明确的 causal step label。因此真实 benchmark 更容易验证 task performance，而 credit correctness 需要通过 async perturbation、人工构造事件或少量标注来辅助分析。

### 6.3 Gating 有超参，需要敏感性实验

门控里的 delay threshold \(\delta\)、terminal mixing coefficient \(\alpha\)、pending window size 都可能影响结果。

因此论文中必须通过 ablation 和 sensitivity analysis 说明方法不是靠某一个精调超参成立。

### 6.4 当前 scorer 是规则式的，表达能力有限

规则式 Evidence kernel 的优点是透明、可解释、容易审计。

缺点是它不能捕捉更复杂的语义关系，例如隐式因果、长距离任务依赖或多步工具链关系。

未来可以考虑 learned credit scorer，但这会引入新的问题：可解释性下降、训练数据来源不清楚、可能重新引入 oracle bias。

### 6.5 本方法不直接解决探索和模型能力问题

ECR-GRPO 解决的是 feedback assignment。它不能单独解决：

1. policy exploration 不足。
2. base model 不会执行某类动作。
3. 长上下文记忆不稳定。
4. action space 设计不合理。

因此实验设计中要避免把所有性能问题都归因于 credit assignment。

---

## 7. 汇报时可以强调的结论

这篇工作的核心价值可以总结成三句话。

第一，ECR-GRPO 把长程 Agent 中晚到、缺失、非局部的反馈显式建模为 asynchronous event stream。

第二，它通过 pending step buffer、evidence attribution 和 event-gated routing，把这些事件转换成标准 GRPO 可以消费的 reward。

第三，它不替代 GRPO，而是补上 GRPO 在异步 Agent 训练场景中缺失的 reward construction 前置环节，因此具有更清晰的贡献边界和更好的工程可接入性。

最终想证明的是：

> 在长程 LLM Agent 中，优化器可以保持标准 GRPO，但 reward 进入优化器之前必须经过异步信用回填；否则 delayed feedback 会被错误广播、错误归因，导致训练信号噪声变大。
