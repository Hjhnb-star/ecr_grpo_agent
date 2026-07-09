# ECR-GRPO Literature Coverage Matrix

This matrix is a human-readable companion to `.agents/cross_index.json`. The automatic CoPaper cross-index was built successfully, but its exact-string matcher reported low storyline coverage because paper summaries and storyline terms use different headings. This file records the substantive coverage used for writing.

## Coverage by Claim

| Paper claim / section need | Seed papers | How they support the ECR-GRPO story | Remaining gap |
|---|---|---|---|
| GRPO is a practical critic-free policy optimization backbone for reasoning models | `shao2024deepseekmath`, `deepseekai2025deepseekr1`, `zhou2026demystifyinggrpo`, `schulman2017ppo` | Establishes PPO/GRPO lineage, group-relative advantages, and modern RL-for-LLM reasoning context. | These papers assume rewards can be assigned at outcome/process level; they do not solve asynchronous event-to-step credit refill. |
| Delayed reward credit assignment is a known RL bottleneck | `arjonamedina2018rudder` | Provides the closest classical delayed-reward baseline: return decomposition and reward redistribution. | RUDDER assumes complete trajectories and learned return decomposition, not partial, missing, interrupted, or weakly evidenced event streams. |
| Step-level supervision improves reasoning when labels are available | `lightman2023verify`, `cobbe2021verifiers` | Supports the value of denser step-level supervision and verifier/process-reward signals. | These works depend on curated step labels or verifier outputs; ECR-GRPO targets cases where feedback arrives late, sparsely, or only indirectly. |
| Agent RL needs credit assignment across tool and environment interactions | `luo2025agentlightning` | Provides the closest agent-training comparison point, including hierarchical credit assignment for agent trajectories. | Agent Lightning does not explicitly model event-conditioned refill for delayed, non-local, weak evidence attached to pending historical steps. |
| Interactive language agents need benchmarks with long-horizon action traces | `shridhar2020alfworld`, `yao2022webshop` | Supplies external-validity environments for embodied/text and web-interaction agents. | Existing benchmarks usually provide task outcome feedback, not the asynchronous feedback stream ECR-GRPO studies. |
| Reasoning/action agents use feedback at inference time | `yao2022react`, `shinn2023reflexion`, `zhou2023lats` | Motivates the importance of feedback, reflection, and planning in long-horizon agents. | These methods primarily improve prompting/inference or verbal memory, not policy-gradient credit refill under delayed event supervision. |

## Related-Work Positioning

ECR-GRPO should be framed as a credit-assignment layer on top of GRPO, not as a new standalone RL optimizer. Its novelty is the event-conditioned credit refill mechanism: feedback events are matched to unresolved historical steps using time, action, text, and tool evidence, then converted into step-level returns for GRPO-style updates.

The strongest contrast set is:

1. GRPO/PPO papers: strong optimizer, weak asynchronous credit model.
2. RUDDER: strong delayed-reward redistribution, weak support for partial and event-stream feedback.
3. Process-supervision papers: strong step labels, weak support when labels are delayed or missing.
4. Agent Lightning: strong agent-RL framing, weaker emphasis on weak-evidence event-to-step refill.
5. ReAct/Reflexion/LATS: strong interactive agent behavior, mostly inference-time or memory-based rather than training-time credit assignment.

## Citation Keys to Use in `paper.md`

- GRPO / PPO: `shao2024deepseekmath`, `deepseekai2025deepseekr1`, `zhou2026demystifyinggrpo`, `schulman2017ppo`
- Delayed credit assignment: `arjonamedina2018rudder`
- Process supervision / verifiers: `lightman2023verify`, `cobbe2021verifiers`
- Agent RL: `luo2025agentlightning`
- Benchmarks and agent settings: `shridhar2020alfworld`, `yao2022webshop`
- Inference-time agent feedback/planning: `yao2022react`, `shinn2023reflexion`, `zhou2023lats`

## Submission Risk

Before a final AAAI-27 submission, rerun automated literature search with a Semantic Scholar API key and add missing 2025-2026 work on LLM-agent RL, trajectory-level reward modeling, and credit assignment. The current seed set is enough for framework drafting, but not enough for a final novelty claim.
