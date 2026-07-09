# ECR-GRPO Storyline

## One-Sentence Thesis

ECR-GRPO turns delayed, partial, missing, interrupted, and non-local feedback in long-horizon LLM agents into step-level policy gradients by treating supervision as asynchronous events and refilling credit to pending historical steps.

## Core Problem

Long-horizon LLM agents interact with tools, web pages, APIs, codebases, and simulated environments over many steps. Their feedback is rarely a clean synchronous reward for the current action. Tool returns can arrive late, evaluators may score after rollout, partial rewards may be missing, and a later event may describe the value of an earlier action rather than the latest step.

Central question:

> How should agentic RL assign credit when feedback arrives as a delayed, weakly evidenced event stream rather than as immediate step rewards or a single final trajectory outcome?

This is the paper's main framing. The key difficulty is not only long horizon and not only delayed reward. The difficult setting is:

```text
delayed + partial + weakly evidenced + non-local + sometimes interrupted feedback
```

## Why It Matters

Trajectory-level GRPO/PPO-style training can punish useful intermediate actions when a later step fails. Recent step-level methods reduce the granularity problem, but they often still assume that feedback is complete, available, or alignable to steps. Real agent systems violate that assumption through tool latency, API failure, page delay, timeout, interruption, missing logs, and non-local evaluator feedback.

Asynchronous credit assignment is therefore a training-signal modeling problem, not only an implementation issue.

## Shared Limitation Of Existing Approaches

Trajectory-level RL is too coarse. Uniform refill is diffuse. Recency refill is a strong baseline but is biased toward the latest step. Step-level methods such as STEP or StepPO-style optimization improve the action granularity, but they do not by themselves solve event-to-step alignment when feedback arrives late and lacks oracle causal links. Dependency-based refill can be an oracle upper bound when exact related-step links are available, but it is not deployable when real environments provide only weak evidence.

The missing piece is a no-oracle mechanism for delayed, partial, weakly evidenced, and non-local feedback.

## Key Insight

Feedback should be modeled as assignable events. Each step remains pending until later feedback arrives. When an event arrives, an evidence-conditioned kernel assigns its reward to relevant historical steps using weak signals such as temporal distance, action text, observation deltas, tool names, tags, and public trace metadata.

This changes reward processing from:

```text
trajectory -> final scalar reward -> whole-rollout update
```

to:

```text
step records + asynchronous event stream
-> event-to-step credit refill
-> step-level returns
-> GRPO-style group-relative advantages
```

## Method Overview

1. Eventized Reward Interface: represent partial reward, terminal reward, timeout, interruption, tool return, and non-local support as `AsyncEvent` objects. Missing feedback is modeled by dropping or withholding events, not as its own reward event type.
2. Pending Step Buffer: keep recent `StepRecord` objects eligible for later refill.
3. Evidence-Conditioned Credit Refill: compute event-to-step weights and assign event reward to historical steps.
4. GRPO-Compatible Step Advantage: convert filled step credit into group-relative advantages for policy updates.
5. Gated Evidence: keep local feedback sharp with recency-like credit while routing non-local feedback to evidence attribution.

Core equations:

```text
w_{t,k} = normalize(K(e_k, b_t))
c_{t,k} = w_{t,k} R_k
G_t = r_t + sum_k c_{t,k}
A_t = (G_t - mean(G_group)) / (std(G_group) + epsilon)
```

## Contributions

1. Formulate asynchronous step credit assignment for long-horizon LLM agents.
2. Propose event-conditioned credit refill over pending historical steps.
3. Introduce no-oracle evidence-conditioned attribution using weak event-step evidence rather than exact causal step labels.
4. Integrate refilled step returns into GRPO-style group-relative advantages.
5. Evaluate task performance, robustness, and attribution quality across controlled synthetic diagnostics, HF/LoRA transfer experiments, and ALFWorld-style real benchmark protocols.

## Research Questions

RQ1: Does event-conditioned step refill improve over trajectory-level reward under asynchronous feedback?

RQ2: Does evidence-conditioned refill assign delayed non-local feedback to the true relevant historical step better than recency or uniform baselines?

RQ3: Is the method robust when feedback is delayed, missing, interrupted, or affected by timeout?

RQ4: Which evidence sources matter for attribution and performance?

RQ5: In HF/LoRA policy learning, is event-gated routing needed to preserve local learning while improving non-local attribution?

RQ6: On ALFWorld, does ECR-GRPO improve or preserve real-task learning under the same policy, budget, and async perturbation protocol?

## Experiment Ladder

### Stage A: Controlled Synthetic Diagnostic

Purpose: prove the credit assignment mechanism under conditions where true target steps are known for post-hoc analysis.

Required variants:

- diagnostic synthetic: structured environment for credit attribution analysis
- no-hint or weak-hint synthetic: removes direct expected-action hints for stronger task-performance claims

Core comparisons:

- GRPO-Trajectory
- Trajectory-Uniform
- Uniform Refill
- Recency Refill
- Evidence Refill
- Gated Evidence
- Dependency Oracle

### Stage B: HF/LoRA Synthetic Transfer

Purpose: show that ECR-GRPO transfers from tabular diagnostics to LLM policy training.

Core comparisons:

- Recency
- Evidence
- Gated Evidence
- GRPO-Trajectory if compute allows

Main message:

```text
Pure Evidence improves non-local attribution but can dilute local learning.
Gated Evidence preserves local learning while correcting non-local credit.
```

### Stage C: ALFWorld External Validity

Purpose: test whether event-conditioned credit remains useful beyond synthetic environments.

Protocol:

1. Native ALFWorld sanity: train and evaluate under the ordinary environment interface, with optional shaping.
2. Async-perturbed ALFWorld: inject delayed terminal feedback, missing partial feedback, timeout events, and interruption events through the same `AsyncEvent` wrapper.

Core comparisons:

- GRPO-Trajectory
- Step-GRPO / StepPO-style local baseline
- Recency Refill
- Evidence Refill
- Gated Evidence
- Dependency Oracle as diagnostic upper bound

Boundary:

Do not claim broad ALFWorld SOTA. The claim is external validity for asynchronous credit assignment under matched budget and matched perturbation protocol.

## Current Evidence

Stage A synthetic controlled experiments:

- Main no-oracle setting: Evidence reaches success/return 0.725/0.501, above Recency at 0.675/0.474 and Uniform at 0.383/0.338, while Trajectory fails at 0.000/-0.289.
- Credit diagnostic: Evidence gives more credit to the true non-local target step than Recency, with target weight around 0.418 vs 0.132 and argmax-target rate around 0.977 vs 0.000.
- Robustness: Evidence is especially stronger under missing reward and timeout perturbations.
- Ablation: performance and attribution can diverge, so both task metrics and credit diagnostics must be reported.

Stage B HF/LoRA synthetic experiments:

- Pure Evidence improves non-local attribution but can dilute local next-action learning.
- Gated Evidence routes local feedback to recency-like sharp credit and non-local feedback to evidence attribution.
- In the fair lag-2 comparison, Gated matches Recency final success while improving logged mean success and non-local target credit.

Stage C ALFWorld:

- Current role: planned external-validity experiment.
- Minimum acceptable evidence: same policy, same seeds, same budget, same async protocol, varying only the credit assignment method.
- Main success condition: Gated Evidence matches or improves GRPO/Step-GRPO success while showing better robustness or proxy attribution under async perturbation.

## Claim Boundary

The strongest current claim is a controlled diagnostic claim:

> ECR-GRPO improves asynchronous credit assignment quality and can improve learning under controlled long-horizon feedback perturbations.

The stronger AAAI claim requires Stage C evidence:

> ECR-GRPO remains useful on real long-horizon agent benchmarks when feedback is delayed, missing, interrupted, or non-local.

Do not claim real-benchmark SOTA unless full multi-seed fair comparisons on ALFWorld, ScienceWorld, WebShop, or tool-orchestration benchmarks are completed.

## Figures And Tables

Figure 1: Motivation example showing useful early steps punished by final trajectory failure.

Figure 2: Asynchronous event stream over pending historical steps.

Figure 3: ECR-GRPO architecture: rollout, pending buffer, event stream, credit kernel, step return, GRPO update.

Figure 4: Credit diagnostic comparing target weight, recent weight, and argmax-target rate.

Figure 5: Robustness curves for missing reward, timeout, delay, and interruption.

Figure 6: ALFWorld native vs async-perturbed comparison.

Table 1: Main synthetic performance with mean +/- std.

Table 2: Credit diagnostic.

Table 3: Robustness AUC.

Table 4: Evidence and gated-evidence ablations.

Table 5: HF/LoRA synthetic fair comparison.

Table 6: ALFWorld fair comparison against GRPO and Step-GRPO/StepPO-style baselines.

## Baseline Definitions

GRPO-Trajectory: episode return with group-relative trajectory advantage.

Trajectory-Uniform: terminal reward uniformly assigned across episode steps, still optimized with trajectory advantage.

Step-GRPO / StepPO-style local baseline: step-level update using immediate/local/shaped rewards without asynchronous credit refill.

Uniform Refill: arriving event reward split equally over eligible pending steps.

Recency Refill: arriving event reward decays by temporal distance.

Evidence Refill: no-oracle event-to-step attribution using weak public evidence.

Gated Evidence: local feedback uses recency-like assignment; non-local feedback uses evidence attribution.

Dependency Oracle: exact-link upper bound, not deployable and not the main method.

## AAAI-27 Critical Next Steps

1. Create no-hint or weak-hint synthetic configs and rerun the main synthetic table with at least five seeds.
2. Add Gated Evidence to the main synthetic result table.
3. Produce robustness AUC tables for delay, missing feedback, timeout, interruption, and non-local lag.
4. Finish evidence-source and gated-routing ablations.
5. Rerun HF/LoRA with candidate-action scoring, non-zero entropy diagnostics, and at least three seeds.
6. Run ALFWorld native and async-perturbed fair comparisons against GRPO and Step-GRPO/StepPO-style baselines.
7. Pick one or two credit-assignment case studies from `credit_assignments.jsonl`.
8. Generate publication-ready tables from reproducible scripts rather than hand-copied numbers.

## Writing Warnings

- Do not present Dependency as the main method; it is an oracle upper bound.
- Do not call missing reward an event type; describe it as dropped or withheld feedback.
- Do not claim real-benchmark SOTA without full fair multi-seed evidence.
- Do not overclaim pure Evidence; emphasize Gated Evidence when HF/LoRA results support it.
- Always report credit diagnostics alongside success rate.
- Clearly separate controlled diagnostics, HF/LoRA transfer, and ALFWorld external validity.
