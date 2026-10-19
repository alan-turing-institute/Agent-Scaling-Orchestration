# Plan: an orchestrator that chooses real agents per task

Agreed in a design session on 2026-10-19, after the E2 selection × topology sweeps. Terms in
**bold** are defined in [`CONTEXT.md`](../CONTEXT.md). Decisions that are hard to reverse have
ADRs in [`docs/adr/`](adr/). The research behind the pool and the baselines is in
[`docs/notes/specialists.md`](notes/specialists.md) and
[`docs/notes/orchestration-baselines.md`](notes/orchestration-baselines.md).

Work is tracked in GitHub issue #31, which links one issue per work item (#9–#30).

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

Issues: #10 pool, #11 thinking, #14 serving, #15 gates, #16 personas, #30 vLLM upgrade.

An **agent** is checkpoint + **thinking mode** + prompt (+ tools in step (c))
([ADR-0001](adr/0001-agents-are-checkpoints-not-personas.md)).

| Base | Thinking off | Thinking on | Specialists (pending gates) |
|---|---|---|---|
| Qwen3.5-0.8B | ✓ | ✓ | — |
| Qwen3.5-4B | ✓ | ✓ | RL-MATH; Med-Qwen (text-only wrapper) |
| Gemma-4-E4B | ✓ | ✓ if supported | ShiningValiant3 (science) |
| Ministral-3-3B | Instruct | Reasoning-2512 (Mistral's sibling) | — |
| Qwen3.6-35B-A3B (NVFP4) | ✓ | ✓ | BigBang-v1 → step (c) |

- **Personas.** A persona joins as a separate agent only if it beats the same model with the
  **neutral prompt**, on training folds, by more than the run-to-run noise. At most 2 personas per
  model. Candidates are shortlisted on the **development split** (§5).
- **Thinking** may run to the model's context limit (32k). There is no separate thinking budget;
  cost absorbs it. Report per agent how often the ceiling is hit.
- **Specialists** must load in our vLLM container and pass a gate before joining the pool: a
  paired comparison of at least 200 dev-split items against their own base. How each was trained
  is recorded in `docs/notes/specialists.md`. Keep looking for more.
- **vLLM 0.21 limitation.** Our image (`nvcr.io/nvidia/vllm:26.05.post1-py3`) is vLLM 0.21.0 and
  cannot load text-only Qwen3.5/3.6 checkpoints (added in 0.27). Load those through a local wrapper
  (their weights plus the base model's config and tokenizer). Upgrading the image is deferred,
  because it reopens the stability work.

## 4. The orchestrator

Issues: #24 orchestrator, #25 continual memory, #17 topologies.

- **One call per task.** It returns either a direct answer, or a configuration: 1–3 agents, a
  topology and, for sequential topologies, which agent takes which role. A later variant answers
  first and then decides whether to delegate, its own answer joining the team.
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

The exhaustive "best fixed configuration" baseline was dropped as too costly. Whether it comes back
for the replayable topologies is open (§8, Q30).

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

## 8. Open decisions

Issue: #29 (and #18 for Q32).

Proposed answers are in italics; none is settled yet.

- **Q30 – Best fixed configuration.** It is a floor, not a ceiling; the oracle is the ceiling. The
  best vote team need not be the best sequential team.
  *Proposed:* drop it, and let the best single agent be the fixed baseline that is the same under
  every topology.
- **Q31 – Topology menu.**
  *Proposed:* single agent (including direct answer), vote, cascade, sequential (solve → review →
  revise, or solve → check, with roles assigned by the orchestrator) and synthesis. Drop debate
  (5× vote's tokens for +0.5 points at 4B) and centralized (behaves like synthesis).
- **Q32 – Test sequential topologies before building on them.**
  *Proposed:* on the dev split, run a few mixed-model pairs live (for example, 4B solver → 35B
  thinking reviewer) against each agent alone. Measure fixes, breaks and tokens. Keep sequential
  topologies on the menu only if some pair fixes clearly more than it breaks, at lower cost than its
  strongest member alone.
- **Q33 – Order of work on one box.**
  *Proposed:* PR #7 → pool refactor → specialist smoke tests → dev-split work → agent cards and
  orchestrator prompt. When the second Spark arrives, decide which models live on which box, since
  a live mixed team needs all its members served at once.

## 9. Not yet designed

Issues: #28 step (c), #30 vLLM upgrade.

- **Step (c), multi-step tool tasks.** BigBang-v1 and the tool-calling specialists belong here.
  Tool-call serve flags were added on 2026-10-19.
- **Upgrading vLLM to ≥0.27,** as its own stability test, if more text-only specialists prove
  worth having.
