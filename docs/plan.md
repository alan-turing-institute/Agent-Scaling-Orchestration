# Plan

The one current plan: where the work stands, what the paper claims, and the experiments still to run.
The checklist of code changes and runs is [`todo.md`](todo.md); keep both up to date.

Written 2026-11-13. It replaces `docs/orchestration-v2-plan.md`, `docs/paper-plan.md`, the root
`plan.md` (harness clean-up, done) and `docs/tom-agent-plan.md` (not pursued); they remain in git history.

Terms in **bold** are defined in [`CONTEXT.md`](../CONTEXT.md). Hard-to-reverse decisions are ADRs in [`adr/`](adr/).
X1's full design (personas, specialists, teams, settings) is [`x1-plan.md`](x1-plan.md). Results are on the results page ("Does Orchestrator Memory Help?").

Literature notes, each with sources:
- [`notes/paper-prior-work.md`](notes/paper-prior-work.md): closest work and what is new;
- [`notes/paper-evaluation-norms.md`](notes/paper-evaluation-norms.md): baselines, cost, statistics, benchmark protocols;
- [`notes/paper-ablations.md`](notes/paper-ablations.md): ablations in comparable papers;
- [`notes/orchestration-baselines.md`](notes/orchestration-baselines.md): routers, cascades and MAS methods.

---

## 1. Where the work stands

| Line | What | Status | Result in one line |
|---|---|---|---|
| E1–E2, team of one | LLM orchestrator picks persona teams (easy pool, 699 Qs) | Done | Selection ≈ random; debate/centralized/synthesis ≈ vote at 2–5× tokens |
| X0 | Bare 4B and 35B, thinking off/on, hard (780) and agentic (350) pools | Done | Oracle over 4 agents +5 (hard) / +12 (agentic) above the best single agent |
| X1 | 10 fixed expert teams (personas + specialists), 35B orchestrator reads summaries | Done (5 folds × 3 samples, T8 rerun) | Orchestrator beats the vote 10/10 and the best worker 7/10 |
| L2 | The 35B alone, thinking off, X1 settings | Done | Team beats it on maths, physics and 3 tool tasks; n.s. elsewhere; ~10× tokens |
| Frontier | Claude Sonnet via headless Claude Code, fold 0, 1 sample | Done; **not to be rerun** (usage) | Beats every open configuration on all 8 benchmarks with far fewer output tokens |
| X4 | Solve first, then escalate (35B thinking off answers first) | Live calls done 2026-11-14; fold-fitted rules and results tab pending | Its own judgement plus a 35B-on review: 83.6% → 85.4% at 1.6× compute; always passing on to the 35B on reaches 90.1% at 7.5× |
| Delegation | Division of labour on tool tasks (includes the old X5) | Not started | — |

### The matched-compute check that sets the framing

X0 already has the 35B with **thinking on at the long limit**, which is the orchestrator model's best solo setting. The limit is 81,920 tokens on hard questions; tool tasks run at 131k context. The table compares it with the orchestrator + team on the same questions. The orchestrator is averaged over 3 samples and the 35B has 1. The CI is a paired 95% interval over questions.

| Team | Qs | Orch. + team | 35B thinking on | Δ | 95% CI | Team tokens | 35B-on tokens |
|---|---|---|---|---|---|---|---|
| Maths | 382 | 94.4% | 97.6% | −3.2 | [−4.9, −1.5] | 29,394 | 18,238 |
| Physics | 136 | 90.9% | 91.2% | −0.2 | [−2.2, +1.7] | 13,149 | 13,765 |
| Chemistry | 143 | 82.1% | 84.6% | −2.6 | [−7.5, +2.4] | 25,683 | 23,644 |
| Biology | 19 | 66.7% | 63.2% | +3.5 | [−10.5, +17.5] | 14,855 | 12,026 |
| Law | 50 | 62.0% | 66.0% | −4.0 | [−13.1, +5.1] | 12,884 | 5,890 |
| Engineering | 50 | 80.7% | 80.0% | +0.7 | [−7.4, +8.7] | 29,295 | 19,162 |
| BrowseComp-Plus | 100 | 58.7% | 48.0% | **+10.7** | [+3.9, +17.4] | 40,776 | 16,514 |
| Finance-Agent | 50 | 54.7% | 70.0% | **−15.3** | [−27.5, −3.2] | 31,692 | 5,139 |
| WorkBench | 100 | 90.3% | 76.0% | **+14.3** | [+7.8, +20.9] | 4,769 | 1,397 |
| Plancraft | 100 | 69.3% | 65.0% | +4.3 | [−1.9, +10.5] | 12,280 | 2,220 |

- **On reasoning questions the team adds nothing over letting the 35B think, at more tokens; on maths it is worse.**
- **The team's gains survive only on WorkBench and BrowseComp-Plus.** On those the orchestrator re-runs the task with the workers' reports.
- **On Finance-Agent the team loses clearly.**
- **L2's thinking-off comparison overstated the team's value.**

## 2. The paper

**Novelty.**
- Nobody has published exactly X1: fixed per-domain teams of persona workers plus community fine-tunes, an untrained larger orchestrator, reasoning and tool benchmarks, and an orchestrator-alone control.
- But most single findings are known or expected:
  - persona and topology nulls;
  - aggregator > vote (Mixture-of-Agents and successors);
  - frontier > all (Kim et al.).
- The contested question is whether reading a team of weaker models helps a stronger model. The literature disagrees (MoA vs 2610.15897 vs Kim et al.), so a careful, compute-matched answer is publishable either way.

**Framing A (recommended).**
> Across six reasoning and four tool domains, a 35B model reading a fixed team of small specialists beats its own thinking-off answer on half the domains. Against its own thinking-on answer at similar tokens, it gains only on tool tasks where its own attempt is weak. Persona prompts add nothing. A frontier model alone beats every open configuration at a fraction of the tokens.

It answers "when does it help?" by domain, with the mechanism ablations (§4) and the division-of-labour experiment (§5).

**Framing B (fallback).** A cost-accounted reality check on training-free orchestration of small open models:
- persona selection ≈ random;
- topology ≈ vote;
- specialist teams recover part of the oracle headroom, but no more than letting the orchestrator think;
- one frontier model beats all of them.

This suits NeurIPS Datasets & Benchmarks or an ACL Findings evidence paper.

Choose between A and B once the matched-compute baselines (§3) are in.

## 3. Baselines

All run on our pool; no published numbers transfer.

| Baseline | Protocol | Status |
|---|---|---|
| Team majority vote | Plurality of workers' answers, seeded tie-break | Have |
| Best worker | Chosen per benchmark **on the training folds**; hindsight only as an upper bound | Hindsight only |
| 35B alone, thinking off | X1 settings | Have (L2) |
| 35B alone, thinking on, long limit | X0 settings | Have, 1 sample |
| 35B self-consistency maj@k, thinking off and on | k swept to bracket the team's tokens. Estimate maj@k by averaging over subsets of a larger sample pool. | Missing |
| Universal Self-Consistency / judge-pick | The 35B picks one worker's answer from the same summaries | Missing |
| Self-MoA | The 35B orchestrator reads 4 of its own samples instead of the team's | Missing |
| Cascade and kNN router over the pool | Thresholds and routers fit on the training folds | Missing |
| Symbolic-MoE (optional) | The closest published training-free method on our hard benchmarks: per-question expert recruitment plus an aggregator | Missing; live runs |
| Frontier | Sonnet fold 0; state model id, effort, harness, date; compare in $ | Have (subset) |
| Oracle | Any worker right | Have |

**Matching rules.**
- Count every call: workers, orchestrator, judge and retries.
- Choose one primary cost axis before seeing results.
- Prefer accuracy–cost curves to single points.

## 4. Ablations

Most are replays over saved stage-1 worker answers: only the orchestrator is rerun. Listed cheapest first. The ids (A1–A10, B1–B10) are from `notes/paper-ablations.md` §4, which gives the motivation for each.

1. **Analysis only.**
   - A1: gain by vote margin.
   - A10: question-clustered paired tests and Pareto plots.
   - B1: worker error correlation.
   - B2: gain against difficulty.
   - B6: which worker the orchestrator copies.
   - K\* (the predecessor paper's diversity measure) of each team against its gain: links the paper to its predecessor.
2. **Orchestrator calls only.**
   - A3: blank, other-question and all-wrong inputs. This is the "is it just re-solving?" control; no prior paper ran it.
   - A4: input order. Workers are always shown personas first and specialists last, so position is confounded with type; this one is required.
   - A6: a 4B orchestrator (L1).
   - A7: leave-one-out, personas-only and specialists-only, and the strongest member removed.
   - A8: personas replaced by neutral 4B samples.
   - B3: select vs synthesise.
   - B5: orchestrator thinking on.
   - A5: answer only, summary, or full response (L4, needs a 131k serve).
   - L10: the orchestrator picks each specialist's thinking mode per task (stage 1 has both modes).
3. **New generation.**
   - The 35B samples for maj@k and Self-MoA.
   - L3: solo runs of the specialists' other bases (Gemma-E4B, Ministral-3B-Instruct), so each fine-tune is compared with its own base.
   - L9: larger thinking cap for the specialists.
   - B7: orchestrator episodes with no reports, the re-run control for the tool-task gains.
   - B4: adversarial and weak members.
   - B9: extra rounds (low priority).

## 5. Division of labour: one agent's output as another's input

**Where.** Only on the tool pool, which decomposes naturally:
- search then reason (BrowseComp-Plus);
- gather filings then compute (Finance-Agent);
- plan then act (WorkBench, Plancraft).

Hard questions are atomic: a chain there is solve → review → revise, which E2 found ≈ vote. Kim et al. find centralized delegation +80.8% on Finance-Agent but −39 to −70% on Plancraft, so mapping where it helps is itself a result.

**Code that exists.**
- `team_config.delegated`: a hub with a `delegate(worker, instruction)` tool, whose workers act on the hub's environment.
- `team_config.pipeline`: stages, each reading the earlier ones.
- Neither has been run with real specialists.

**Arms.** Fold 0 first as a pilot, then 5 folds.

| Arm | Set-up | Tests |
|---|---|---|
| D2 | The 35B manages and delegates sub-steps to the team's specialists (e.g. Occamy for WorkBench actions, Agents-A1 or BigBang for search) | Division of labour vs parallel answering, at matched tokens |
| D3 | A small specialist drives and can `ask_expert` the 35B, or hands the 35B its transcript on failure (the old X5) | The cheap direction: most tokens on the small model |
| D4 | Fixed chain: the 35B plans → specialist executes → the 35B checks | Fixed vs chosen decomposition |
| C1 | D2 with the 35B delegating to itself | Heterogeneity vs extra turns |
| C2 | D2 with a random worker per call | Whether choosing *which* agent matters |

**Pilot first.** Measure per-step skill:
- give each agent a correct plan and score its execution;
- score each agent's plan with a fixed executor.

If no worker beats the 35B at any sub-step, delegation can only add cost. That would be a reportable negative, and it saves the full runs.

**Measures.**
- Success.
- Tokens per agent.
- Number and timing of delegations.
- The stage that introduced each failure.

**Serving.** The 35B on one Spark, the specialist on the other. Finance-Agent on one box only (rate locks).

## 6. Escalation: solve first (X4)

**X4 Solve first.** In progress in `-x4`. The orchestrator answers alone, then escalates.
- Helpers: the same model thinking on, the 35B, or a specialist.
- Form: pass on, second opinion or review (Q38).
- Trigger: its own judgement (main arm). Baselines fit on the training folds: cut-off or missing answer, sample disagreement, stated confidence. The oracle gate is the ceiling; the endpoints are never and always escalate (Q39).
- Gain = fixed − broken. Report:
  - escalation rate;
  - trigger AUROC;
  - fixes and breaks by help form;
  - accuracy against cost vs the cascade and the matched-compute baselines.
- Scoring: replay from the X0 matrix, with only judgement and review calls made live.

**Results so far** (780 hard questions; compute = tokens × active parameters, per question):

| Policy | Accuracy | Compute | Fixed / broke |
|---|---|---|---|
| Never escalate (the 35B thinking off) | 83.6% | 8,109 | — |
| Own judgement → review by the 35B thinking on | 85.4% | 13,307 | +14 / −0 |
| Own judgement → pass on to the 35B thinking on | 86.7% | 24,429 | +29 / −5 |
| Always pass on to the 35B thinking on | 90.1% | 60,806 | — |
| Ceiling: escalate exactly the wrong answers, pass on | 92.9% | — | — |
| Ceiling: escalate exactly the wrong answers, review | 87.2% | — | — |

- **Its own judgement escalates at about the right rate:** 15.8%, against 16.4% wrong. It always chose review by the 35B thinking on.
- **Reviewers are anchored.** The 35B thinking on changed 39 of 780 answers (+28 / −2), the 4B thinking on 42 (+19 / −11) and the 4B thinking off 21 (+10 / −4).
- **Stated confidence is no use as a trigger.** Its AUROC is 0.78, but nearly every answer says 95 or 100.
- **Still to do:** thresholds and forms fit on the training folds, and a results tab.
- **Code:** branch `experiments/x4-solve-first`, not pushed yet. Data: `data-claude/x4-solve-first/`.

## 7. Evaluation rules

- **Statistics.**
  - The question is the unit: average a question's samples, then use a paired bootstrap clustered by question, with Holm correction within a family.
  - Folds are not a variance estimate.
  - State power: tool-benchmark comparisons detect only gaps of about 9–18 points.
- **Judge.** BrowseComp-Plus and Finance-Agent are graded by the 35B, which is also the orchestrator and a baseline. Regrade with an independent judge, or report agreement on a human-labelled subset.
- **Leakage.** Teams and specialists must be shown to be fixed from model cards and dev or smoke-test questions only.
- **Contamination.** AIME (all 120) and MATH-500 predate Qwen3.5/3.6, and some specialists may have trained on them. Report AIME by year, add a post-release slice, and argue the comparisons are within-model.
- **Protocol deviations table.**
  - BrowseComp-Plus: our retriever; comparable only to the paper's BM25 column.
  - Finance-Agent: 50 public questions with Wikipedia/SEC search, not Vals' private set.
  - WorkBench: also report the side-effect rate.
  - Plancraft: list its deviations too.
  - No side-by-side comparison with leaderboard numbers.
- **Cost.** Report side by side, with no single headline measure:
  - tokens, prompt and completion, thinking included;
  - list-price $ (prices in `notes/paper-evaluation-norms.md` §2);
  - active-parameter FLOPs;
  - wall-clock.

  They rank our models differently: by FLOPs the 35B-A3B is cheaper per token than the dense 4B, by price about 6× dearer. The orchestrator's own calls count.
- **Frontier arm.** State the model id (`claude-sonnet-5-5`), effort, harness and date. Compare it in $, since tokenizers and hidden thinking differ.

## 8. Settled decisions still in force

- An **agent** is checkpoint + thinking mode + prompt (+ tools) ([ADR-0001](adr/0001-agents-are-checkpoints-not-personas.md)).
- Non-interacting topologies are scored by **replay** ([ADR-0002](adr/0002-replay-for-non-interacting-topologies.md)).
- No training of our own ([ADR-0003](adr/0003-no-training-of-our-own.md)).
- **Sampling (Q40).** Vendor-recommended, never below T 0.1. Each result set states its settings.
- **Splits.** X0's 5 folds, stratified by source, split seed 0. Every learned baseline, and the CV-chosen best worker, is fit on the training folds only.
- **Answer matrix.** Every agent × question × k. k = 3 for thinking-off agents; thinking-on agents have 1 sample so far.
- **No more pool building.** The pool is X1's agents plus the 35B; new models only as baselines (e.g. an open frontier-class model). Our vLLM image is 0.21: text-only Qwen3.5/3.6 checkpoints need a wrapper or vLLM ≥ 0.27.
- **Debate** is dropped (5× vote's tokens for +0.5 points).
- **Web search.** Finance-Agent reaches only SEC, Wikipedia and Wikidata, through machine-wide rate locks. Run it on one box at a time.
- **Frontier runs.** No further Claude CLI runs without the user's go-ahead: fold 0 took four 5-hour usage windows.

## 9. Compute and order of work

- **GPU, two Sparks.**
  - The 35B samples for maj@k and Self-MoA take several days per box.
  - Each orchestrator-call ablation is about one stage-2 pass, roughly half a day per box.
  - The independent judge is one regrade pass.
  - The delegation pilot needs two models served at once.
- **Claude usage.**
  - Further Sonnet coverage needs a decision.
  - The hard set costs about $1.60 a fold; the tool set about $70 a fold, almost all of it BrowseComp-Plus.
  - A local open frontier-class model (e.g. gpt-oss-120b, which fits one Spark) is an alternative bridge.
- **Serving note.** The 35B orchestrator server froze silently 6 times during long thinking-on reviews in X4, with prefix caching on. Relaunching it with `--no-enable-prefix-caching` was proposed and awaits the user's decision.
- **Order.**
  1. Analysis-only items and the statistics.
  2. The 35B maj@k / Self-MoA samples (decides framing A or B).
  3. Orchestrator-call ablations.
  4. The delegation pilot.
  5. The independent judge.
  6. Frontier coverage, if approved.

## 10. Earlier plans: what was kept

Triaged 2026-11-13. The sources are the plan page "Orchestration Meets Agent Scaling" (E1–E14, October), the v2 plan (X2–X5, pool, agent cards) and X1's later list (L1–L10).

| Kept as | From |
|---|---|
| A5 handoff format | E12, L4 |
| A6 4B orchestrator (a weak model checking strong ones; D3 is the reverse) | E14, L1 |
| A7 subsets and team size | E9 (n > 1) |
| X4, and D3 | E4, E13, X4, X5 |
| Delegation D4 (tool tasks only) | E10 |
| Cascade and kNN router baselines | E1, #22, #23 |
| K\* analysis | E7 |
| L3, L9, L10 | X1 later list |

**Done:**
- E2 (selection × topology);
- E8 (X1 teams mix families);
- E9 at n = 1;
- X2 (inside X1 stage 1);
- the persona gate (personas add nothing).

**Dropped:**
- **E3, E5, E6, E11, X3, agent cards, continual memory and cost budgets.** They all improve a selector that learns which agents to pick, and choosing agents did no better than random.
- **E10 on single questions.** Chains there were no better than voting in E2.
- **Pool expansion.** Llama, Granite, Nemotron, Qwen2.5 hosts and the second wave of specialists (L5).
- **The ToM partner, and L6–L8.**

