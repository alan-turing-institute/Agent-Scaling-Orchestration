# Plan: an orchestrator that chooses real agents per task

Agreed in a design session on 2026-10-19, after the E2 selection × topology sweeps. Terms in
**bold** are defined in [`CONTEXT.md`](../CONTEXT.md). Decisions that are hard to reverse have
ADRs in [`docs/adr/`](adr/). The research behind the pool and the baselines is in
[`docs/notes/specialists.md`](notes/specialists.md) and
[`docs/notes/orchestration-baselines.md`](notes/orchestration-baselines.md).

Work is tracked in GitHub issue #31, which links one issue per work item (#9–#35).

---

## 1. Why change course

**Selecting personas does not beat picking them at random.** E2 crossed three selection arms
(random, orchestrator without memory, orchestrator with continual memory) with four topologies
(vote, debate, centralized, synthesis), on the 699-question tagged set, 5 folds.

| Agent model | Held-out accuracy, all 12 cells | Best selection gain over random | Survives Holm? |
|---|---|---|---|
| Qwen3.5-0.8B | 49.4–54.9% | a few points, some cells negative | no |
| Qwen3.5-4B | 87.7–90.4% | +1.4 (no-memory debate) | no (all p_holm = 1.00) |

The team-of-one follow-up (all 50 personas alone, plus orchestrator repeats) gave the same answer.

**Personas on one model are mostly noise.**

- At 0.8B, identical temperature-0 requests give the same answer string only ~49% of the time,
  and correctness flips on 29% of them. Persona explains under 1% of the variance (~0.7% above a
  permutation null). What effect there is tracks verbosity (r = −0.70 with truncation).
- At 4B, flips fall to 8%. One persona is broken (Elegant_Minimalist, 0% on gsm8k).
  Out-of-sample best-persona selection is worth about +2.5 points.

**On one model, letting agents interact rarely beats outvoting them.** Qwen3.5-4B, 2,097 held-out
answers per topology:

| Topology | ≥1 first-round agent right | Team right | Fixed (no first-round agent right) | Lost (first-round majority right) | Tokens / question |
|---|---|---|---|---|---|
| vote | 95.6% | 89.1% | 0 | 0 | 4.5k |
| debate | 95.5% | 89.6% | 3 | 26 | 23.4k |
| centralized | 93.9% | 88.3% | 5 | 40 | 7.2k |
| synthesis | 93.9% | 88.8% | 1 | 40 | 7.0k |

**The headroom is in the gap between "someone was right" and "the team was right"** (96% vs 89%),
and in cost. Exploiting it needs agents that genuinely differ in capability and cost.

**What the anchor paper does and does not give us.** Kim et al., *Towards a Science of Scaling
Agent Systems* (arXiv:2512.08296):

- It evaluates agentic benchmarks only, and excludes MMLU, GSM8K, MATH, GPQA and AIME as
  non-agentic. It uses frozen frontier models and does no agent selection or routing.
- Its robust finding is capability saturation: adding agents hurts once single-agent accuracy is
  above ~45%.
- Its efficiency framing (success per 1k tokens, matched-token comparisons) and its "measure the
  single agent, then decide whether to delegate" rule carry over. Its numbers do not.
- Its future work names role-specialised selection and capability-aware routing, which is this
  plan.

## 2. Goal

For each **task**, an **orchestrator** either gives a **direct answer** or delegates to a
**configuration** (agents, team size, **topology**). It weighs each agent's accuracy against its
**cost**. Success means a better **efficiency** curve (accuracy against cost) than the baselines in §6.

Scope comes in two steps:

- **(a)** Single questions, per-task selection. This plan.
- **(c)** Multi-step tool tasks whose sub-steps can go to different agents. Not yet designed (§9).

We do not train or fine-tune anything ourselves ([ADR-0003](adr/0003-no-training-of-our-own.md)).

## 3. The pool

Issues: #10 pool, #11 thinking, #14 serving, #15 gates, #16 personas, #30 vLLM upgrade, #32 more base
models.

An **agent** is checkpoint + **thinking mode** + prompt (+ tools in step (c))
([ADR-0001](adr/0001-agents-are-checkpoints-not-personas.md)).

| Base | Thinking off | Thinking on | Specialists (pending gates) |
|---|---|---|---|
| Qwen3.5-0.8B | ✓ | ✓ | — |
| Qwen3.5-4B | ✓ | ✓ | RL-MATH; Med-Qwen (text-only wrapper) |
| Gemma-4-E4B | ✓ | ✓ if supported | ShiningValiant3 (science) |
| Ministral-3-3B | Instruct | Reasoning-2512 (Mistral's sibling) | — |
| Qwen3.6-35B-A3B (NVFP4) | ✓ | ✓ | BigBang-v1 → step (c) |
| Llama-3.1-8B (+ Llama-3.2-3B) | ✓ (no thinking mode) | — | HuatuoGPT-o1-8B, Fino1-8B, R1-Distill-Llama-8B |
| Granite-4.2-8B | ✓ | ✓ (also low effort) | — |
| gpt-oss-20b | low effort | medium / high effort | — |
| Nemotron-Nano-9B-v2 | ✓ | ✓ | — |
| Qwen2.5-7B (specialist host) | ✓ | — | Fin-R1, HuatuoGPT-o1-7B |

The last five rows are new families (Q36, #32; survey in `docs/notes/base-models.md`), so that agents
differ in family and architecture as well as size. Use ungated NVIDIA/Red Hat Llama copies unless
our token has Meta access. gpt-oss sets reasoning by effort level (Harmony format), not `<think>`
tags. Not added: Mistral-7B v0.1, which would only host Saul-7B (law). This roughly brings the pool
to ~28 agents and makes the answer matrix ~40% larger; it fits only with the second Spark.

- **Personas.** A persona joins as a separate agent only if it beats the same model with the
  **neutral prompt**, on training folds, by more than the run-to-run noise. At most 2 personas per
  model. Candidates are shortlisted on the **development split** (§5).
- **Thinking** may run to the model's context limit (32k). There is no separate thinking budget;
  cost absorbs it. Report per agent how often the ceiling is hit.
- **Sampling (Q40).** No more temperature 0: every v2 agent uses its vendor's recommended sampling
  (for Qwen3.5/3.6, "general tasks": thinking on T 1.0, top_p 0.95; thinking off T 0.7, top_p 0.8;
  both top_k 20, presence_penalty 1.5), and never below T 0.1. Greedy decoding made the small models
  loop, and repeats and agreement-based escalation need sampling anyway. Results are not compared
  with the temperature-0 runs; every result set states its settings.
- **Specialists** must load in our vLLM container and pass a gate before joining the pool: a
  paired comparison of at least 200 dev-split items against their own base. How each was trained
  is recorded in `docs/notes/specialists.md`. Keep looking for more.
- **vLLM 0.21 limitation.** Our image (`nvcr.io/nvidia/vllm:26.05.post1-py3`) is vLLM 0.21.0 and
  cannot load text-only Qwen3.5/3.6 checkpoints (added in 0.27). Load those through a local wrapper
  (their weights plus the base model's config and tokenizer). Upgrading the image is deferred,
  because it reopens the stability work.

## 4. The orchestrator

Issues: #24 orchestrator, #25 continual memory, #17 topologies.

- **One call per task.** It returns either a direct answer or a configuration: the agents, the
  topology and, for a sequential topology, the whole sequence: how many steps and which pool
  agent takes each. No template is fixed in advance, and there are no separate role prompts
  (Q34). An agent behaves the same in a sequence as anywhere else, and later steps get the
  earlier steps' output through one fixed **handoff** instruction.
- **Topology menu:** single agent, vote, cascade, sequential (composed by the orchestrator) and
  synthesis. Debate is dropped (5× vote's tokens for +0.5 points at 4B), and so is centralized
  (it behaves like synthesis).
- **Decide first or solve first (Q37).** Two ways of running it, both main arms, compared at
  matched cost: **decide first** (the one call above) and **solve first** (it answers, then decides
  whether to escalate). Endpoints: always answer alone, always delegate. See X4 in §10.
- **It sees an agent card for each agent:** what the agent is, its accuracy on the training folds
  by question source, and its average cost. Ablations:
  - names only;
  - cards plus **continual memory**, which now records cost as well as accuracy.
- **Cost budgets.** It is run at several stated budgets per task to trace its curve. Its own calls,
  including a direct answer, count against the budget. Budgets are soft: the curve plots what was
  actually spent, and how well it keeps to budget is reported.
- **Orchestrator models:**
  - Qwen3.6-35B with thinking on (strong, good direct answers);
  - Qwen3.5-4B with thinking off (cheap);
  - possibly Qwen3.5-4B with thinking on.

  Each is also an ordinary pool agent, so a bare-model baseline exists for it.

## 5. Benchmarks and splits

Issues: #9 PR #7, #13 splits.

- **Questions:** the existing easy set (699 questions: gsm8k, arc, hellaswag, truthfulqa,
  winogrande, professional medicine, formal logic) plus the PR #7 hard sets:

  | Hard set | Questions |
  |---|---|
  | GPQA-Diamond | 198 |
  | MMLU-Pro hard | 200 |
  | MATH-500 levels 4–5 | 262 |
  | AIME | 120 |

  PR #7 is smoke-tested on a real model, then merged, before anything here runs.
- **Development split:** ~15%, stratified by source. It is used only for the persona shortlist,
  the specialist gates and placing the cost budgets.
- **Evaluation:** 5-fold cross-validation on the rest, folds stratified by source. Agent cards and
  every learned baseline are fit on training folds only.

## 6. Baselines

Issues: #21 simple, #22 cascade, #23 routers, #27 later.

All are re-run on our pool; no published numbers transfer.

1. **Random configuration** at sizes 1–3.
2. **Bare model**, for each base model and each orchestrator model.
3. **Best single agent**, chosen by cross-validation and also in hindsight. Plus **self-consistency**
   of the 35B with thinking on, with the number of samples swept to bracket the orchestrator's cost.
   Plus a static table of the best agent per benchmark.
4. **Cascade**, small → large, escalating when repeated answers disagree, with thresholds fit on
   training folds (Dekoninck et al. cascade routing, arXiv:2410.10347).
5. **Learned routers:** a kNN utility router and Avengers cluster routing (arXiv:2505.19797), both
   trained on our folds.
6. **Later:** Symbolic-MoE (arXiv:2503.05641), then MasRouter (arXiv:2502.11133) if the
   orchestrator clears 1–5.

There is no "best fixed configuration" baseline (Q30). It would be a floor rather than a ceiling,
and the best vote team need not be the best sequential team. The **best single agent** is the fixed
baseline, the same under every topology; the **oracle** is the ceiling.

## 7. Evaluation and cost

Issues: #12 cost, #19 answer matrix, #20 replay and oracle, #26 two-box serving.

- **Answer matrix:** every agent × every question × k repeats. Initial plan: k = 3 for
  thinking-off agents and k = 1 for thinking-on, adding repeats for agents the orchestrator picks.
- **Replay:** vote, cascade and single-agent configurations are scored from the matrix without new
  calls ([ADR-0002](adr/0002-replay-for-non-interacting-topologies.md)). Interacting and sequential
  topologies run live.
- **Cost** is measured three ways, reported side by side with no headline measure:
  - tokens (thinking included);
  - compute (tokens × active parameters);
  - wall-clock.

  The orchestrator's own calls count.
- **The oracle** (the cheapest correct agent per task, known in hindsight) and the best single
  agent are reported first. They show how much room the orchestrator has.
- **Compute.** One pass of ~20 agents over ~1,500 questions with uncapped thinking is roughly 100M
  tokens. That is 3–4 days on one Spark at the ~360 tok/s seen in the 4B E2 sweep. The full answer
  matrix and orchestrator runs **wait for a second DGX Spark**. Until then this box does code,
  smoke tests and dev-split work.
- **Statistics:** pooled accuracy with Wilson intervals; exact McNemar paired tests,
  Holm-corrected within families, as in E2.

## 8. Decisions settled after the first draft

Issue #29.

- **Q30:** no best fixed configuration; the best single agent is the fixed baseline.
- **Q31:** sequential topologies are composed per task by the orchestrator: the length and the agents
  are chosen from the pool, with no fixed solve → review → revise template.
- **Q34:** roles come from the pool too. There is no role library for sequences; an agent keeps its
  own prompt in every topology, and later steps see earlier output through one fixed handoff.
- **Q32:** no separate test of fixed sequential pairs before building (#18 closed), since no
  sequence is fixed. Instead, add more base model families to the pool (#32).
- **Q33:** order of work on one box: PR #7 → pool → specialist smoke tests → dev-split work → agent
  cards and orchestrator prompt. The answer matrix and orchestrator runs go to the second Spark.

- **Q35:** personas enter the pool only through the solo test (beat the neutral prompt answering
  alone). A second route, for personas that are good at checking another agent's answer, can be
  added if the orchestrator's sequences show checking steps help.
- **Q36:** add Llama-3.1-8B (+ 3.2-3B), Granite-4.2-8B, gpt-oss-20b, Nemotron-Nano-9B-v2, and
  Qwen2.5-7B as a host for finance and medicine specialists (§3).

- **Q37:** decide first and solve first are both main arms, compared at matched cost, with the
  "always answer alone" and "always delegate" endpoints.
- **Q38:** after solving first, the orchestrator chooses among pass on, second opinion and review
  per task. Each is a configuration whose first member is the orchestrator's own answer.
- **Q39:** the main escalation trigger is the orchestrator's own judgement after answering. Rules
  fit on training folds (cut-off or missing answer, sample disagreement, stated confidence) are
  baselines; the oracle gate is the ceiling.
- **Q40:** vendor-recommended sampling for every agent, never temperature 0 (at least 0.1). No
  comparison with earlier temperature-0 runs is needed, as long as each result set states its
  settings.
- **Q41:** run X0 now on one box (started 2026-10-21).
- **Q42:** X5 covers the agentic pool at whole-task level, WorkBench included. Sub-step delegation
  stays in step (c).
- **Q43:** X1 tests four prompt groups answering alone, under the Q35 gate.

Still open: the wording of the handoff (#17), settled when it is implemented.

## 9. Not yet designed

Issues: #28 step (c), #30 vLLM upgrade.

- **Step (c), multi-step tool tasks.** BigBang-v1 and the tool-calling specialists belong here.
  Tool-call serve flags were added on 2026-10-19. Whole-task escalation on the agentic pool is
  already in X5 (§10); what is left for (c) is handing sub-steps to different agents.
- **Upgrading vLLM to ≥0.27,** as its own stability test, if more text-only specialists prove
  worth having.

## 10. Experiments on the harder benchmarks

Agreed 2026-10-21 (Q37–Q43). What exists so far, bare model, temperature 0, 4,096-token answers:

| Pool | Qwen3.5-0.8B | Qwen3.5-4B |
|---|---|---|
| Hard, 780 questions, 5 folds | 19.6% (53% cut off) | 60.8% (33% cut off) |
| Agentic, 350 tasks, 5 folds | 6.6% (32k context) | 48.3% (131k context) |

**One idea links most of them.** When the orchestrator answers alone, that answer is a row of the
answer matrix like any other agent's. So solve first, followed by pass on, second opinion or a
cascade, can be scored by replay. Only the orchestrator's decision to escalate, and review steps,
need live calls. A 4-agent matrix (4B and 35B, thinking off and on) is enough to start, on one box.

| # | Question | Arms | Needs | Issue |
|---|---|---|---|---|
| **X0 Headroom** | How often is a stronger or thinking agent right where a cheaper one is wrong? How far is the best single agent from the oracle? | Bare 4B and 35B, thinking off and on, hard pool, 5 folds, 30,720-token answers, vendor sampling; 4B thinking off sampled 3 times | Running since 2026-10-21 (branch `experiments/x0-headroom`) | #33 |
| **X1 Prompts** | Do prompts matter on hard questions, where personas did not on easy ones? | Neutral, source-matched expert, shortlisted personas, and a commit-to-an-answer prompt aimed at cut-offs and loops; answering alone, 4B thinking off and on, k = 3, paired tests; winners join the pool under the Q35 gate | Existing persona path | #16 |
| **X2 Models and specialists** | Does a specialist or another family beat its base on matching sources? | RL-MATH on MATH/AIME, ShiningValiant3 on GPQA, Med-Qwen on medicine, new families against the 4B | #10, #14 | #15 |
| **X3 Per-task selection** | Does choosing per task beat the best single agent and random at equal cost? | Orchestrator v2, decide first, on the hard sources, by replay | #10, #19–#24 | #24 |
| **X4 Solve first** | Does answering first and escalating only when needed beat deciding first or always delegating? | Below | New orchestrator mode | #34 |
| **X5 Agentic, whole task** | Does solve first work on tool tasks? | The 4B runs the task; on failure, or when it judges itself unsure, it retries with the 35B or hands the 35B its transcript. BrowseComp-Plus, Finance-Agent, Plancraft, WorkBench | Tool-calling screen of the 35B | #35 |

**X4 in detail.**

- Orchestrators: 4B thinking off first (the cheap solver), then 4B thinking on and 35B. Help can
  be the same model with thinking on, the 35B, or a specialist.
- Help (Q38): the orchestrator chooses pass on, second opinion or review per task.
- Triggers (Q39): its own judgement (main arm); baselines fit on training folds: cut-off or missing
  answer, disagreement among its k samples (the cascade baseline, #22), stated confidence. Ceiling:
  the oracle gate. Endpoints: never escalate (its bare model) and always escalate.
- Gain = wrong answers escalated and fixed − right answers escalated and broken. Cost = its own
  solve + escalation rate × the help's cost. Report escalation rate, how well the trigger separates
  its wrong answers from its right ones (AUROC), fixes and breaks by help form, and the
  accuracy-against-cost curve against decide first and the cascade.
- With a 35B orchestrator, solve first pays only if some helper fixes its mistakes. X0 and X2 count
  how often any agent is right where the 35B with thinking on is wrong, before this is built.

