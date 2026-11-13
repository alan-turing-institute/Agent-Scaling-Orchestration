# Ablations in comparable papers, and the ablation plan for ours

Compiled 2026-11-13 from arXiv full text (HTML), mostly the latest versions. Section, table and figure numbers refer to the URL given.

**Notation.**
- **†** marks an item not verified in the full text: an abstract or figure we did not read, or an inference.
- **OpenReview:** forum pages and the per-forum API return a browser challenge to our tools.
  - MoA reviews and Self-MoA author replies came from the OpenReview search API (`api2.openreview.net/notes/search`).
  - Every other "reviewer asks" item is †. Where marked "inferred", it is guessed from what a later arXiv version added.

"Replay" means the run needs only new orchestrator (stage-2) calls over saved stage-1 worker answers
(`data-claude/x1-teams/stage1/…`, ADR 0002). On tool tasks the orchestrator runs a fresh episode, so "replay"
there still means live orchestrator episodes, but no new worker runs.

---

## 1. Catalogue: what each paper varied

### 1.1 Aggregator / mixture papers

| Paper | Varied → finding | Where |
|---|---|---|
| **MoA** [2406.04692](https://arxiv.org/html/2406.04692); ICLR'25 camera-ready [pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/5434be94e82c54327bb9dcaf7fca52b6-Paper-Conference.pdf) | **Seeing others' answers:** all 6 LLMs improve when shown other models' answers ("collaborativeness"). | §2.1 Fig 1 |
| | **Number of proposers n**, mixed models vs one model sampled n times: mixed 47.8/58.8/58.0/61.3 vs same model 47.8/54.5/56.1/56.7 (n=1/2/3/6). Mixed wins at every n>1. | Tab 3 (camera-ready) |
| | **Aggregator model**, proposers fixed: 61.3 (Qwen-110B) down to 41.5 (dbrx). WizardLM is the best proposer but a weak aggregator. | Tab 4 |
| | **One proposer** (MoA-Lite-Single) 64.7 vs six proposers 74.1 vs aggregator alone 63.3. | Tab 2 |
| | **Select vs synthesise:** an LLM ranker that picks one answer is clearly below synthesis. | Fig 5a (camera-ready) |
| | **Copy analysis:** BLEU, TF-IDF and Levenshtein overlap with each proposal correlates with that proposal's quality. The aggregator overlaps most with its own model's proposal (BLEU 0.40–0.51). | Fig 5b; App G Tab 12 |
| | **Layers:** MATH 0.500/0.570/0.576 for 1/2/3 layers; gains flatten after the first aggregation. | App E Tab 10 |
| | **Versus other methods, and cost:** beats MAD and ReConcile at 1–2 rounds (Tab 5). Cost–quality Pareto in TFLOPs (Fig 6). Optimised proposer set: +2.7 LC (App F). | Tab 5; Fig 6; App F |
| **Self-MoA** [2502.00674](https://arxiv.org/html/2502.00674) (TMLR'26) | **Self vs mixed:** 6 samples of the best model beat 6 different models: AlpacaEval +6.6 LC; beats all 13 mixes on MMLU/CRUX/MATH. | Tab 1, 3 |
| | **Quality vs diversity regression** (Vendi score): the quality coefficient beats the diversity coefficient on every dataset (MATH 4.72 vs 2.84, R²≈0.7–0.8). | §4 Tab 4 |
| | **When mixing wins:** a mixed-task workload (+0.35) and two models of equal quality (Self-MoA still slightly ahead). | Tab 6–7 |
| | **Samples and members:** 6→30 samples is non-monotonic. Adding a weak second model hurts (−1.5). | §5 Fig 3 |
| | **Selection vs synthesis:** USC-style selection is 5.5 below Self-MoA. | App B.2 Tab 12 |
| | **Aggregator strength:** a GPT-4o aggregator keeps the ordering. | Tab 10 |
| **Symbolic-/Skill-MoE** [2503.05641](https://arxiv.org/html/2503.05641) (ICML'26) | **Aggregator choice:** random 52.3/48.9, instance-adaptive 57.1/58.0, task-specific 63.7/57.8 (MMLU-Pro/GPQA). Good solvers are not good aggregators. | §5.2 Tab 3; App E–F |
| | **Experts × aggregator:** random experts with a good aggregator 31.8; recruited experts with MV 53.5; recruited with the good aggregator 57.8 (GPQA). A strong aggregator cannot rescue weak experts. | Tab 4 |
| | **Aggregator vs MV, experts fixed:** +2.8 to +4.4 for the aggregator. | App M Tab 14 |
| | **Team size:** top-3 52.9 vs top-5 47.7; adding weaker members hurts. | Tab 5 |
| | **Discussion rounds** before aggregation help only weak aggregators. | Tab 7 |
| | **Failure modes:** all experts wrong in 29–33% of cases; the aggregator picks a wrong answer in 19–20% of errors. | App L |
| **LLM-Blender** [2306.02561](https://arxiv.org/html/2306.02561v3) | **Best model per example:** the best model is best on only 21% of examples. | Fig 1 |
| | **Ranker variants:** PairRanker beats the alternatives; rank-then-fuse top-3 beats rank-only. | Tab 2 |
| | **Order swap** in pairwise comparison: same verdict over 90% of the time. | App C.5 |
| | **Ceiling and gap:** oracle-selection ceiling; ranker accuracy grows with the rank gap. | Tab 2; Fig 7 |
| **USC** [2311.17311](https://arxiv.org/html/2311.17311) | **Versus SC:** USC ≈ SC (GSM8K 90.2 vs 90.4). | Tab 1–4 |
| | **Response order** (5 shuffles): 89.7±0.3, little effect. | §4.3 Tab 5 |
| | **Number of samples k:** peaks at 8 and drops at 16. | Fig 3 |
| | **Agreement with SC** is higher than either method's accuracy. | §4.4 Fig 4–5 |
| | **Oracle best-of-8:** a large gap remains (MATH 57.2 vs 37.4). | App A |
| **AggLM** [2509.06870](https://arxiv.org/html/2509.06870) | **Trained vs prompted aggregator, vs MV** (k=8): 50.0 vs 45.9 (AIME25). A prompted, untrained aggregator gets 44.9. | Tab 1 |
| | **Gain by majority size:** largest when the majority is small (correct answer in a minority); ties MV when the majority is large. | Fig 4 |
| | **k:** AggLM at k=8 beats MV at k=16. | Fig 3 |
| | **Transfer:** gains carry over to stronger (8B) and thinking-off workers. | Tab 2–3 |
| | **Controls and cost:** the solver fine-tuned on the same data does not improve (Tab 6). The aggregator costs about ⅓ of a solver call (Tab 7). | Tab 6–7 |
| **RSA** [2509.26626](https://arxiv.org/html/2509.26626) | **Recursive vs one-shot self-aggregation vs MV:** 73.2 vs 56.5 vs 68.3 (AIME25, Qwen3-4B). MV still wins on SuperGPQA. | Tab 1 |
| | **Subset size K:** most of the gain is from K=1 to 2; little after 3 (long context). | Fig 6 |
| | **Diversity** collapses after the first aggregation step. | App C |
| **Generative Self-Aggregation** [2503.04104](https://arxiv.org/html/2503.04104) | **N:** gain peaks at N=5 for the 8B model (context-bound). **Versus alternatives:** beats select-one and ≈ SC (89.6 vs 88.5). | Fig 2; Tab 3 |
| **SC-MoA / "Beyond Consensus"** [2605.29116](https://arxiv.org/html/2605.29116) (preprint) | **Input format:** answer-only input and majority-only proposals both fall to MV. Minority-only proposals keep the gain (73.2 vs 72.2). Shuffling steps inside a trace has no effect. | † tables unnumbered in our read |
| | **Aggregator alone:** GPQA −14.1 pp. | — |
| | **Fixed vs broke:** 81 fixed vs 4 broken (n=575). Still positive under unanimity (BBH 16 vs 5). | Tab 31 |
| | **Proposers and aggregator size:** N peaks at 7. An 8B aggregator costs 2.5 pp. | — |
| | **Diversity source:** persona vs other diversity sources ≈ equal. | — |
| **Bias Control & Injection** [2610.21290](https://arxiv.org/html/2610.21290) (Oct 2026) | **Recover / keep vs MV:** recovers 53.6% of MV-wrong items and keeps 98.4% of MV-right ones. Baselines recover 11–46% and keep 89–96%. | Fig 6 |
| | **Showing a rival answer twice** flips 56% of incorrect but only 19% of correct decisions. | Tab 4 |
| | **Order swap** as an instability signal. | Fig 9b |
| | **Style:** style normalisation gives +2.1 pp; model identity is recoverable from style 94% of the time. | Fig 9a; Tab 9 |
| **"This Is Your Doge"** [2503.05856](https://arxiv.org/html/2503.05856) | **One deceptive proposer** in a 3-3-1 MoA: QuALITY 92.8→44.3. | §4 |
| | **Dose–response:** 0/1/2/3 deceptive proposers give AlpacaEval LC 48.3/43.8/38.1/24.7. | Tab 3 |
| | **Deception type and strength:** opposers do more harm than promoters (Tab 1); stronger deceivers do more harm (Tab 4). | Tab 1, 4 |
| | **Aggregator size** (8B/70B/405B) protects on QuALITY (0.72/0.88/0.92) but not on AlpacaEval. | Fig 8 |
| | **Defences** (dropout or clustering) recover 0.44→0.82. | Tab 5 |

### 1.2 Orchestrator / MAS papers

| Paper | Varied → finding | Where |
|---|---|---|
| **Kim et al.** [2512.08296](https://arxiv.org/html/2512.08296) | **Architectures:** 5 architectures × 6 agentic benchmarks (ours: BrowseComp-Plus, Finance-Agent, Plancraft, WorkBench). Centralized gives +80.8% on Finance and −39 to −70% on Plancraft; the mean over all MAS is −0.3%. | §4.2 Fig 2 |
| | **Token matching:** about 4.8k tokens per trial. | §4.1 |
| | **Capability saturation:** coordination stops helping once the single agent is above ~45%. This is the only effect that survives clustered SEs and Holm correction. | §4.3–4.5 |
| | **Error amplification** relative to a single agent: Centralized 4.4×, Independent 17.2×. Verification absorbs 22.7% of errors. | Tab 5 |
| | **Number of agents** n∈{1,3,5,7,9}: Centralized flattens past 5. | Fig 5 |
| | **Heterogeneity:** strong sub-agents matter more than a strong orchestrator. A heterogeneous Centralized team averages −12.6 pp vs a strong homogeneous team. | Fig 4; App F Tab 12 |
| | **Cost:** successes per 1k tokens: single agent 67.7 vs Centralized 21.5. | Fig 3 |
| **Illusion of MAS advantage** [2606.13003](https://arxiv.org/html/2606.13003) | **Automatic MAS vs CoT-SC (5 samples):** six automatic MAS (DyLAN, MAS-Zero, ADAS, AFlow, MaAS, MAS-Orchestra) lose to CoT-SC at about 10× the cost. | Fig 2 Tab 4 |
| | **Backbone strength:** MAS helps only strong backbones (a "competency floor"). | §3.2 |
| | **Conformity:** DyLAN agents agree unanimously at once 70–90% of the time. | §4 |
| | **Roles:** all-assistant team 54.4 vs task experts 53.4. | App E.1 |
| | **Position bias:** the MAS-Zero verifier picks the first block over 45% of the time. | Fig 6, 9 |
| | **Routing ignores difficulty.** | Tab 1–2 |
| **Debate or Vote** [2508.17536](https://arxiv.org/html/2508.17536v2) (NeurIPS'25) | **Debate vs vote:** MV 0.769 vs best MAD 0.738. Centralized MAD is worst (0.66). More rounds hurt. | Tab 1 |
| | **Number of agents N:** 1→5 gains read as an ensemble effect. | Fig 3 |
| | **Martingale test:** mean agent accuracy stays flat across rounds. | Fig 4; App E |
| | **Majority-keeping variants:** MAD-Conformist and MAD-Follower ≥ MV, both beating vanilla MAD. | §5 Tab 2 |
| | **Scale, personas, extractor:** results hold at 32B and with personas. The answer extractor flips the conclusion. | Tab 3–4; App F Tab 9 |
| **More Agents** [2402.05120](https://arxiv.org/html/2402.05120) (TMLR) | **Number of samples (1–40):** gains of +6–24%; Llama-13B with 40 samples beats Llama-70B with 1. | §5.1 |
| | **Combined with other methods** (CoT, SPP, debate, reflection): voting alone gets the best average rank. | Tab 3–4 |
| | **Cost** grows linearly with the number of samples. | §5.5 |
| | **Gain vs difficulty** (synthetic axes): rises, then falls. | §6 Fig 6 |
| | **Significance:** ANOVA over sizes. | App B.2 |
| **MasRouter** [2502.11133](https://arxiv.org/html/2502.11133) (ACL'25) | **Component ablation**, each router replaced by random: the LLM router matters most (−2.1/−4.3), then collaboration mode, then roles. Dropping the cost term gives +42–54% cost. | Tab 3 |
| | **Max agents:** 2→6 helps; 6→10 gives a marginal gain at 1.5× cost. | Fig 5 |
| | **Pareto:** on the front vs AFlow. | Fig 3 |
| | **Unseen LLM:** added without retraining, picked 12–27% of the time. | Fig 4 |
| **Puppeteer** [2505.19591](https://arxiv.org/html/2505.19591) (NeurIPS'25) | **Pool:** heterogeneous pool vs one model: +1.8–2.8. | Tab 1 |
| | **Training:** before vs after RL. | Tab 1 |
| | **Cost weight λ** and **width × depth** sweeps. | Fig 3, 7 |
| | **Final answer:** decided by MV, not by an aggregator. | — |
| **MAS-Zero** [2505.14996](https://arxiv.org/html/2505.14996) | **Per-component removal:** removing the selector costs the most (41.97→20.27). | Tab 6 |
| | **Oracle verifier:** GPQA ~95%, i.e. headroom in selection. | Fig 4B |
| | **Meta-agent × worker swap:** worker strength sets the ceiling (o3-mini workers + 4o meta 78.5 vs 4o workers + o3-mini meta 48.4). | Tab 5 |
| | **Weaker meta-agents** fail to follow instructions or write valid code. | §4.2 |
| **DyLAN** [2310.02170](https://arxiv.org/html/2310.02170) | **Team size:** 3 selected of 7 beats all 7, with 53–68% fewer calls. | Fig 3 |
| | **Selection score:** agent importance 73.6 vs human prior 66.7, random 64.8, all 7 agents 63.5. | Tab 13 |
| | **Importance vs Shapley:** importance tracks Shapley in-domain. | Tab 15 |
| | **Domain-expert prompts on MATH:** no significant gain. | §4.1 |
| **AgentVerse** [2308.10848](https://arxiv.org/html/2308.10848) | **Solo vs Group:** Group ≥ Solo with GPT-4; Group < Solo on 2 of 3 tasks with GPT-3.5. | Tab 1 |
| | **Swayed agents:** about 10% of MGSM errors come from a correct agent swayed by a wrong peer. | §3.1 |
| | **Number of experts:** capped at 2 for maths (an observation, not a sweep). | App A |

### 1.3 Conformity and correlation evidence

- **Talk Isn't Always Cheap** [2509.05396](https://arxiv.org/html/2509.05396):
  - debate lowers accuracy in 21 of 30 team setups (Tab 1);
  - correct→incorrect flips exceed incorrect→correct flips (Figs 2–3)†;
  - a correct agent flips most when it is alone (Fig 4)†.
- **Conformity frontier** [2608.11247](https://arxiv.org/abs/2608.11247)†: across 23 open models, a unanimous wrong majority overturns 22.8% (MMLU) and 54.8% (GPQA) of correct answers. Every fix that helps a model keep a correct answer makes it worse at adopting a correct peer answer.
- **Correlated errors** [2506.07962](https://arxiv.org/html/2506.07962):
  - when two models are both wrong, they agree 0.42–0.60, against chance of 0.12–0.33;
  - same company and same architecture each raise this agreement;
  - more accurate models share more errors (Tab 1).
- **MAST** [2503.13657](https://arxiv.org/abs/2503.13657)†: a 14-mode failure taxonomy for MAS traces, usable for coding our tool-task failures.

## 2. Coverage by ablation axis

| Axis | Run by | Not run by / gap |
|---|---|---|
| Number of proposers or agents | MoA Tab 3, Symbolic-MoE Tab 5, DyLAN Fig 3, Kim Fig 5, MasRouter Fig 5, USC Fig 3, RSA Fig 6, SC-MoA, More Agents | — |
| Aggregator / orchestrator strength | MoA Tab 4, Symbolic-MoE Tab 3, MAS-Zero Tab 5, Kim Fig 4, Doge Fig 8, SC-MoA (8B) | Results disagree: Doge says size helps only on QuALITY; Kim and MAS-Zero say worker strength matters more |
| Diversity vs quality | Self-MoA §4, MoA Tab 3, Illusion E.1, SC-MoA, DyLAN §4.1 | Personas add little (Illusion, DyLAN, SC-MoA) |
| Same model sampled vs mixed models | MoA Tab 3, Self-MoA Tab 1/3, RSA, GSA | — |
| Layers / rounds | MoA App E, Debate-or-Vote Tab 1, Symbolic-MoE Tab 7, Puppeteer Fig 7 | More than one aggregation step rarely helps |
| Select vs synthesise | MoA Fig 5a, Self-MoA Tab 12, LLM-Blender Tab 2, GSA Tab 3 | — |
| Aggregator alone / no proposals | MoA Tab 2 (aggregator alone), SC-MoA (−14 pp), AgentVerse (Solo) | **Blank, shuffled-from-another-question and wrong-proposal controls: none found** |
| Input format (answer / trace / summary) | SC-MoA (answer-only ≈ MV) | **Short summaries: none found** |
| Order / position | USC Tab 5, LLM-Blender App C.5, Illusion Fig 6/9, Bias-Control Fig 9b | Not in MoA, Self-MoA or Symbolic-MoE |
| Majority following, fixed/broke | AggLM Fig 4, SC-MoA Tab 31, Bias-Control Fig 6/Tab 4, Debate-or-Vote Tab 2, Talk-isn't-cheap | Few papers report counts by vote margin |
| Adversarial / weak members | Doge Tab 1/3/4, Self-MoA §5, Symbolic-MoE Tab 5 | — |
| Leave-one-out / strongest member removed | DyLAN Shapley Tab 15 (approximate) | **Leave-one-out of the strongest member: none found** |
| Token / cost matching | Kim §4.1 (strict), Illusion, More Agents, MoA Fig 6, MasRouter Fig 3, Self-MoA (forward passes) | Self-MoA, Symbolic-MoE and MoA are not strictly token-matched |
| Error correlation | Correlated-errors 2506.07962, RSA App C, Kim (redundancy) | Rare inside MAS papers |
| Gain vs difficulty / baseline accuracy | Kim §4.3 (~45%), More Agents §6 | — |

## 3. What reviewers asked

- **MoA (ICLR'25, scores 8/8/8/6; forum `h0ZfDIrj7T`, reviews via the search API):**
  - single-proposer results across all benchmarks (answered: Tab 2);
  - an ensemble/MAD baseline (answered: Tab 5);
  - reasoning benchmarks MMLU/BBH/MATH (answered: Tab 9);
  - **"does the aggregator just copy?" and per-proposer contribution** (answered: App G Tab 12);
  - a guideline for choosing proposers and layers (answered: App F);
  - TFLOPs as an unfair cost and latency proxy (not answered with data);
  - human evaluation (not added†);
  - specialist (maths/science) proposers.
  - The AC summary flags proposer selection, budget analysis and human evaluation.
- **Self-MoA (TMLR'26; author replies via the search API, reviews themselves not retrieved†):**
  - fairness and tuning of the Mixed-MoA baseline, and run-to-run noise;
  - aggregator sensitivity, answered with a new AIME-24 run using a GPT-5-mini aggregator†;
  - other sources of within-model diversity;
  - cost in FLOPs.
- **Inferred from version diffs†:**
  - Symbolic-MoE v4: expert quality separated from aggregation (Tab 14), failure modes, larger models;
  - MAS-Zero v3: per-step ablation, oracle verifier, meta/worker swap, standard deviations;
  - Kim v3: clustered SEs and Holm correction, held-out frontier models, heterogeneous teams;
  - Debate-or-Vote: extractor robustness (App F), the 32B model.
- **Recurring themes:**
  1. Cost fairness: matched tokens, not calls or TFLOPs.
  2. Is the aggregator just copying or re-solving?
  3. How should the team be chosen, and what does each member contribute?
  4. Does the result hold for a stronger aggregator or other models?
  5. Variance across seeds and significance tests.

## 4. Ablations for our paper

Our setup relevant to these ablations:
- **Input order:** the orchestrator sees "Agent i (solver)" in fixed order, **personas first and specialists last** (`x1_hub.team_workers`). Position is therefore confounded with worker type.
- **Already measured:** orchestrator vs MV, best worker and oracle; fixed/broke vs the vote (`hub_transition`, `copied`); bare 35B (L2); Sonnet alone (1 fold); `--modes` on/off/headline.

### 4.1 Must-have

| # | Vary | Tests | Motivated by | Cost |
|---|---|---|---|---|
| A1 | **Gain by vote margin.** Bin questions by vote pattern (4-0, 3-1, 2-2, 2-1-1, all differ, majority wrong). In each bin report fixed/broke, copied rate, and how often a correct minority answer is recovered. | Does it follow the majority? Where does the gain come from? | AggLM Fig 4; SC-MoA Tab 31; Bias-Control Fig 6; Debate-or-Vote Tab 2 | **Analysis only** (fields exist) |
| A2 | **Matched-cost baselines.** Best single worker, chosen in hindsight and by cross-validation. Self-consistency vote of the best worker and of the 35B, with n set so tokens match the team plus orchestrator. Tokens are reported, not calls. | Is the team worth its tokens? | Kim §4.1; Illusion Fig 2; More Agents; LLMRouterBench (see `orchestration-baselines.md`) | Replay from the X0 matrix where k samples exist. **The 35B with n>1 needs new 35B samples** (35B only) |
| A3 | **Blank and wrong inputs for the orchestrator.** (a) Same prompt with summaries and answers blanked. (b) Workers' summaries taken from another question. (c) Every answer replaced by one wrong option, summary kept. (d) The correct minority answer duplicated so it is shown twice. | Is it re-solving, reading the content, or counting votes? Separates "orchestrator = bare 35B" from real use of the team. | SC-MoA (aggregator alone −14 pp); Bias-Control Tab 4 (frequency injection); Doge; **blank/shuffled controls are a gap in the literature** | **Replay** (built from saved inputs) |
| A4 | **Order / position.** Reverse the order and use 2 random permutations per question; also put the specialists first. Report the answer-change rate and pick rate by position and by worker type. | Position bias, and whether it explains any specialist preference | USC Tab 5; LLM-Blender C.5; Illusion Fig 6/9; Bias-Control Fig 9b | **Replay** |
| A5 | **Handoff format (L4 + answer-only).** Inputs: answer only; summary + answer (headline); last N tokens; full response. Orchestrator served at 131k. Report the summary fallback rate per worker. | How much reasoning the orchestrator needs; whether summaries lose information | SC-MoA (answer-only ≈ MV); MoA App G; **summaries are a gap** | **Replay** (stage-1 full responses saved). 131k serve |
| A6 | **Orchestrator strength (L1 + upward).** 4B thinking off (L1) and 35B (headline); optionally Sonnet as orchestrator on the Sonnet fold. Compare each with its own bare model. | Does the gain come from the orchestrator's own ability? Is the weak-orchestrator gain negative? | MoA Tab 4; Symbolic-MoE Tab 3/14; MAS-Zero Tab 5; Kim Fig 4; Doge Fig 8 | **Replay**. Sonnet costs API spend |
| A7 | **Leave-one-out and subsets.** Drop each worker; personas only; specialists only; all subsets of size 1–3. Gives a team-size curve and Shapley contribution per worker, and specifically the strongest worker removed. | Which members carry the team? Do weak members hurt? | MoA Tab 3; Symbolic-MoE Tab 5; DyLAN Tab 13/15; Self-MoA §5; **strongest-removed is a gap** | **Replay**. 15 subsets × folds × k: sample a subset, or use only size 3 plus singles |
| A8 | **Personas vs neutral samples.** Replace the 2–3 personas with the X0 neutral 4B thinking-off samples s0–s2, keeping the specialists. Also an all-neutral team (Self-MoA analogue on the 4B). | Do approach personas add anything over plain resampling? | MoA Tab 3; Self-MoA Tab 1/3; Illusion E.1; DyLAN §4.1; Debate-or-Vote Tab 4 | **Replay** (X0 rows share folds and question keys) |
| A9 | **Thinking modes (L9, L10).** Already planned. `--modes on/off/headline` is the proposer-compute axis. L10 lets the orchestrator choose each specialist's mode, reported against token cost. L9's larger-cap rerun is the only part that needs new worker runs. | Does more proposer compute help, and is the tradeoff worth it? | Kim (token budget); MoA (proposer quality); Self-MoA (quality > diversity) | `--modes` and L10: **replay**. L9 rerun: **new worker runs** |
| A10 | **Significance and variance.** Paired tests per question across folds and k (`compare_runs.py`, McNemar, Holm); seed spread of the orchestrator's k runs; Pareto plot of accuracy vs tokens (L2). | Reviewer theme 5 | Kim §4.5; More Agents App B.2; MAS-Zero App E | Analysis only |

### 4.2 Nice-to-have

| # | Vary | Tests | Motivated by | Cost |
|---|---|---|---|---|
| B1 | Error correlation between workers (pairwise wrong-answer agreement; persona–persona vs persona–specialist), plus a quality-vs-diversity regression over A7 subsets | Why the vote or orchestrator gains (or does not) | 2506.07962; Self-MoA §4; Kim (redundancy) | Analysis only |
| B2 | Gain vs difficulty: bin by the best worker's or the bare 35B's accuracy per source or tag | The capability-saturation pattern | Kim §4.3; More Agents §6; Illusion "competency floor" | Analysis only |
| B3 | Select vs synthesise: the orchestrator must pick one worker's answer (USC-style). Also a "keep the majority unless confident" variant. | Value of synthesis over selection | MoA Fig 5a; Self-MoA Tab 12; Debate-or-Vote MAD-Conformist | Replay |
| B4 | Adversarial or weak member: one worker replaced by confidently wrong text (opposer and promoter), or by Qwen3.5-0.8B answers. Dose–response over 1–2 bad members. | Robustness; whether the orchestrator detects bad advice | Doge Tab 1/3/4; Self-MoA §5 | Replay for constructed text; 0.8B answers need new runs at X1 settings |
| B5 | Orchestrator thinking on | Whether reasoning reduces conformity and broke rate | Conformity frontier†; MAS-Zero (meta strength) | Replay |
| B6 | Copy and attribution: which worker the final answer matches, by worker type; summary-text overlap | Does it defer to specialists? Ties to A4. | MoA App G Tab 12 | Analysis only |
| B7 | Tool tasks: orchestrator episode with no reports (= orchestrator alone in the same harness) and with reports from another task; MAST coding of failures | Re-solving check on agentic tasks | Kim (centralized on Plancraft −); MAST† | Orchestrator episodes only (live environment; Finance-Agent rate limits) |
| B8 | Self-aggregation of the 35B: the orchestrator reads 4 of its own samples instead of the team's | Team vs Self-MoA of the orchestrator | Self-MoA; GSA; RSA | **New 35B samples** (if X0 has 1 per question) |
| B9 | Second aggregation round or debate between workers | Layers/rounds | MoA App E; Debate-or-Vote; Symbolic-MoE Tab 7 | **New worker runs**. Literature says it rarely helps: low priority. |
| B10 | Style anonymisation: rewrite worker summaries into one style | Whether the orchestrator recognises specialists by style | Bias-Control Tab 9, Fig 9a | Replay plus one rewrite pass |

### 4.3 Where the planned items fit

- **L1 = A6.**
- **L2 = A2 + A10** (Pareto).
- **L4 = A5.**
- **L9 and L10 = A9.**
- **L3** (paired bases for specialists) supports A8 and B1: specialist vs its base.
- **X4 / X5** (solve first, escalation) are a separate paper axis. A3's blank-input arm is the "orchestrator alone in the same prompt" control that X4 also needs.

### 4.4 Sequencing (all replay unless noted)

1. **Analysis only:** A1, A10, B1, B2, B6.
2. **Orchestrator calls only:** A3, A4, A8, A6 (L1), A7 (size-3 and singles first), B3, B5, then A5 (131k serve).
3. **New generation:** A2's 35B samples; B8; L9's larger-cap rerun; B4's 0.8B runs; B9 last.
