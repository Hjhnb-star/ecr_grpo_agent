# Stage A Result Interpretation

本文件基于阶段 A 汇总结果解读论文证据。结果来源为 `scripts/summarize_results.py` 生成的 `runs/paper_tables` 系列表格，对应当前已有文件：

- `main_performance.csv`
- `credit_diagnostic.csv`
- `robustness_all.csv`
- `robustness_success_pivot.csv`
- `robustness_return_pivot.csv`
- `ablation_performance.csv`
- `ablation_diagnostic.csv`
- `summary.md`

## 1. 总体结论

阶段 A 已经可以支撑一个清晰的 controlled diagnostic conclusion：

> ECR-GRPO 的主要贡献不是简单让异步反馈“能被训练”，而是把 delayed / partial / non-local feedback 转换成更准确、更可解释的 step-level credit assignment。

最终写论文时要区分两个层面的证据：

- 性能层面：Evidence Refill 在 success / return 上通常优于 Recency 和 Uniform；
- 机制层面：Evidence Refill 在 target weight、argmax target rate 等 attribution 指标上显著优于 Recency，这比最终性能差距更关键。

## 2. Main Performance

### 2.1 nonlocal_seed_no_oracle_main

5 seed 主实验结果：

```text
trajectory success = 0.000, return = -0.289
uniform    success = 0.383, return =  0.338
recency    success = 0.675, return =  0.474
evidence   success = 0.725, return =  0.501
dependency success = 0.675, return =  0.474
```

可支撑的论文结论：

- trajectory-level reward 在该异步长任务下完全无法训练起来；
- step-level refill 明显优于 trajectory-level final reward；
- Evidence 比 Recency 高 `+0.050` success、`+0.027` return；
- Evidence 比 Uniform 高 `+0.342` success、`+0.162` return。

这组可以作为论文主表的核心结果。

### 2.2 nonlocal_seed_no_oracle_v2

v2 结果：

```text
trajectory success = 0.000, return = -0.271
uniform    success = 0.396, return =  0.354
recency    success = 0.642, return =  0.462
evidence   success = 0.654, return =  0.465
dependency success = 0.642, return =  0.462
```

可支撑的论文结论：

- v2 中 Evidence 对 Recency 的最终性能优势较小，success 只高 `+0.013`，return 只高 `+0.003`；
- 但 Evidence 仍明显优于 Uniform；
- v2 更适合作为“机制诊断强证据”，而不是单纯性能大幅领先的证据。

论文写法上应避免夸大 v2 的最终 performance gap，应强调 attribution gap。

## 3. Credit Diagnostic

Credit diagnostic 是阶段 A 最关键的机制证据。

### 3.1 主实验诊断

```text
Evidence target_weight_mean = 0.418
Recency  target_weight_mean = 0.132
Uniform  target_weight_mean = 0.186

Evidence recent_weight_mean = 0.175
Recency  recent_weight_mean = 0.347

Evidence argmax_target_rate = 0.977
Recency  argmax_target_rate = 0.000
Uniform  argmax_target_rate = 0.431
```

解释：

- Evidence 给真实 target step 的权重大约是 Recency 的 `3.17x`；
- Recency 给最近 step 的权重更高，说明它系统性偏向最近 step；
- Evidence 几乎总能把最大 credit 给到真实 target step；
- Recency 的 argmax target rate 为 0，说明它在 non-local delayed feedback 下几乎必然误归因。

### 3.2 v2 诊断

```text
Evidence target_weight_mean = 0.491
Recency  target_weight_mean = 0.081
Uniform  target_weight_mean = 0.137

Evidence recent_weight_mean = 0.118
Recency  recent_weight_mean = 0.325

Evidence argmax_target_rate = 0.934
Recency  argmax_target_rate = 0.000
Uniform  argmax_target_rate = 0.463
```

解释：

- v2 中 Evidence 给 target step 的权重大约是 Recency 的 `6.03x`；
- v2 的最终 success gap 不大，但 attribution gap 非常大；
- 这正好支撑论文主张：Evidence 的核心价值在于修正 delayed reward 的归因位置，而不仅是提高一个 scalar reward。

推荐论文句子：

```text
The performance gap between evidence and recency is moderate, but the attribution gap is large: evidence assigns 3.17x to 6.03x more target-step weight and achieves over 0.93 argmax-target rate, while recency assigns its maximum credit to the true non-local target in none of the analyzed events.
```

## 4. Robustness

平均来看，Evidence 相比 Recency 的优势为：

```text
delay:          +0.038 success, +0.035 return
missing reward: +0.094 success, +0.084 return
interruption:   +0.021 success, +0.028 return
timeout:        +0.078 success, +0.048 return
```

### 4.1 Delay

Evidence 在 4/5 个 delay sweep 点上 success 不低于 Recency：

```text
delay_prob=0.0: +0.021 success, +0.019 return
delay_prob=0.2: +0.063 success, +0.044 return
delay_prob=0.4: +0.104 success, +0.067 return
delay_prob=0.6: -0.042 success, -0.004 return
delay_prob=0.8: +0.042 success, +0.049 return
```

解释：纯 delay 下 Recency 是强 baseline，因为“近的 step 更相关”在很多事件中仍然成立。论文中不应写 Evidence 在 delay 下严格支配 Recency。

### 4.2 Missing Reward

Evidence 在 4/4 个 missing reward 点上 success 不低于 Recency：

```text
missing=0.0: +0.000 success, -0.011 return
missing=0.2: +0.125 success, +0.074 return
missing=0.4: +0.146 success, +0.083 return
missing=0.6: +0.104 success, +0.191 return
```

解释：这是阶段 A 中最适合强调的 robustness 结果。随着 reward 缺失增强，Evidence 的优势更明显，说明 evidence-conditioned refill 能更好利用残缺反馈。

### 4.3 Interruption

```text
interruption=0.00: +0.000 success, +0.005 return
interruption=0.05: +0.104 success, +0.048 return
interruption=0.10: -0.021 success, +0.008 return
interruption=0.20: +0.000 success, +0.051 return
```

解释：interruption 下 success 有一个点 Recency 略高，但 Evidence 的 return 在所有点都不低于 Recency。论文中可以说 Evidence improves return robustness under interruption, while success gains are mixed.

### 4.4 Timeout

Evidence 在 4/4 个 timeout 点上 success 和 return 都高于 Recency：

```text
timeout=0.00: +0.083 success, +0.049 return
timeout=0.05: +0.125 success, +0.054 return
timeout=0.10: +0.083 success, +0.067 return
timeout=0.20: +0.021 success, +0.020 return
```

解释：timeout 是另一个强结论场景，适合和 missing reward 一起作为 robustness 主图。

## 5. Ablation

消融结果不能简单写成“去掉任一 evidence source 都会降低 success”，因为结果呈现出性能和 attribution 的分化。

```text
variant     success  return  target_weight  argmax_target  recent_weight
no_tag      0.743    0.507   0.190          0.014          0.284
no_text     0.722    0.497   0.451          0.973          0.181
tag_only    0.590    0.436   0.834          0.972          0.731
no_temporal 0.472    0.343   0.572          0.960          0.093
text_only   0.368    0.253   0.290          0.690          0.216
```

### 5.1 关键观察

- `no_tag` 的 success / return 最高，但 argmax target rate 几乎崩溃到 `0.014`，说明最终性能可以和准确 argmax attribution 分离；
- `no_text` 基本保持完整 Evidence 的 attribution 能力，说明 text overlap 不是当前 synthetic 设定中的主要证据源；
- `tag_only` 的 target weight 很高，但 recent weight 也极高，说明 tag signal 很强但可能过度集中或过度绑定结构模式；
- `no_temporal` 的 attribution 仍强，但 success 明显下降，说明 temporal evidence 对训练稳定性和最终性能很重要；
- `text_only` 表现最弱，说明单独文本证据不足以完成可靠 attribution。

### 5.2 论文表述建议

推荐写成：

```text
Ablation reveals a separation between task performance and attribution sharpness. Temporal evidence is important for final performance, tag evidence strongly controls argmax attribution, and text-only evidence is insufficient. Therefore, evidence-conditioned refill should be evaluated jointly by performance and credit diagnostic metrics rather than success alone.
```

## 6. 论文图表建议

- 主表：`main_performance.csv`，列出 success、return、positive credit fraction；
- 诊断表：`credit_diagnostic.csv`，列出 target weight、recent weight、argmax target rate；
- 鲁棒性图：优先画 missing reward 和 timeout，再画 delay 和 interruption；
- 消融图：一张 performance bar，一张 attribution bar，避免只看 success；
- case study：从 `credit_assignments.jsonl` 选一个 non-local event，对比 Recency top step 与 Evidence top step。

## 7. 当前论文边界

可以强写：

- trajectory-level reward 在该 synthetic async benchmark 下明显失败；
- step-level refill 是必要的；
- Evidence 在 non-local delayed feedback 下显著改善 attribution；
- Evidence 在 missing reward 和 timeout 下鲁棒性更稳；
- ablation 显示 success 和 attribution 需要同时报告。

需要谨慎写：

- 不能说 Evidence 在所有设置下严格优于 Recency；
- 不能说当前结果已经证明真实 LLM agent benchmark 上的 SOTA；
- 不能把 `dependency` 写成主方法，它在当前结果里和 Recency 数值相同，更适合作为 oracle / diagnostic baseline；
- 不能只用 success 解释 ablation，因为 `no_tag` 的 success 高但 attribution 明显坏。

