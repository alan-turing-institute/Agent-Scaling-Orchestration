# Closest prior work for a paper on our results, and what is new

Searched 2026-11-13 (arXiv, OpenReview/ICLR/NeurIPS pages, Nature MI). Extends
[`orchestration-baselines.md`](orchestration-baselines.md) (routers, cascades, MAS baselines, verified
2026-10-19), which is not repeated here. "Abs" means checked against the arXiv abstract page;
"text" means checked against the paper's HTML. **†** = not verified against the paper itself (memory,
search snippet or secondary write-up); check before citing.

---

## A. Closest prior work, by claim

### A(i) A larger aggregator over heterogeneous small or specialist models

| Paper | Set-up | Benchmarks | Headline | Check |
|---|---|---|---|---|
| **Mixture-of-Agents**, Wang et al., 2406.04692 (ICLR'25) | 6 open proposers (Qwen1.5-110B/72B, WizardLM-8x22B, Llama-3-70B, Mixtral-8x22B, dbrx); Qwen1.5-110B aggregator, 3 layers | AlpacaEval 2.0, MT-Bench, FLASK, MATH | Open-only MoA 65.1% LC vs GPT-4o 57.5% (AlpacaEval). MATH 0.50 → 0.58 over layers. **"Collaborativeness": a model answers better when shown other models' outputs, "even when the auxiliary responses … are of lower quality"** | text |
| **Self-MoA**, Li et al., 2502.00674 | Aggregate repeated samples of the single best model, compared with mixing models | AlpacaEval 2.0, MMLU, CRUX, MATH | Self-MoA beats MoA by 6.6% (AlpacaEval) and 3.8% on average. MoA is "quite sensitive to quality"; mixing lowers average quality | abs |
| **Symbolic/Skill-MoE**, Chen et al., 2503.05641 (ICML'26) | Per-instance skill-based recruitment from 16 open ~7B experts, plus an aggregator | **MMLU-Pro, GPQA, AIME**, MedMCQA | +8.15 points average over the best multi-agent baseline; 1 GPU vs 4 | abs |
| **X-MAS**, Ye et al., 2505.16997 | X-MAS-Bench: 27 LLMs (20 chatbots incl. domain fine-tunes such as Qwen2.5-Math and ChemDFM; 7 reasoners), 5 domains, 21 test sets, 5 MAS functions (QA, revise, aggregation, planning, evaluation†) | MATH, AIME, science, medicine, finance, coding | Heterogeneous MAS beats homogeneous: up to +8.4% MATH, up to +47% AIME (chatbot+reasoner). Agent selection is manual | abs (+ function list †) |
| **S-DAG**, Dong et al., 2511.06727 (AAAI'26) | GNN finds subjects per query → most expert model per subject → DAG of models | Multi-subject MMLU-Pro, GPQA, MedMCQA | Beats task-level model selection and MAS baselines (numbers not in abstract) | abs |
| **MoDEM**, Simonds et al., 2410.07490 | BERT router → domain-expert fine-tunes (health, maths, science) | various† | Beats general models of comparable size | abs |
| **LLM-Blender**, Jiang et al., 2306.02561 (ACL'23)† | Pairwise ranker plus generative fuser over 11 open LLMs | MixInstruct | Fused output beats each member† | † |
| **Universal Self-Consistency**, Chen et al., 2311.17311† | The LLM itself selects the most consistent of its own samples | GSM8K, code, summarisation | ≈ SC on maths; works on free-form† | † |
| **AggLM**, Zhao et al., 2509.06870 | RL-trained aggregator over candidate solutions | Maths (AIME etc.†) | Beats majority vote and reward-model selection; generalises to stronger generators | abs (numbers †) |
| **AggAgent**, Lee et al., 2604.11753 (COLM'26) | An aggregation *agent* inspects parallel agentic rollouts with tools | 6 search / deep-research tasks; GLM-4.7, Qwen3.5, MiniMax-M2.5 | +5.3 points average over prior aggregation methods (+10.3 on deep research) | abs |
| **Co-failure ceiling**, Chen, 2606.27288 | 67 models from 21 providers; routing, voting, cascade, fusion, MoA, Self-MoA | Open maths, code, GPQA-D free-response | Combining "rarely beats the best single model" without a strong per-query routing signal. Low-correlation ensembles beat Self-MoA at matched quality | abs |
| **"When Agents Disagree"**, Maryanskyy et al., 2603.20324 (Appl. Sci.) | Generate-then-select vs MoA synthesis | 42 tasks, judge win-rate | Judge **selection** beats **synthesis** by +0.63 win rate. Diversity helps only above a selector-quality threshold | abs |
| **Beyond Consensus**, Fadnavis et al., 2605.29116 | Trace-level aggregation; perturbed single-model traces | Science, competition maths, code | One model with perturbed traces beats heterogeneous pools; "aggregate traces, not answers" | abs |
| **Complementary-MoA**, Zhang et al., 2605.24048 | Pick proposers by complementarity with each other *and the summariser* | n/a in abs | Complementarity beats pure accuracy or diversity | abs |
| **LLM-ensemble survey**, Chen et al., 2502.18036 (IJCAI'26) | Taxonomy: before / during / after inference | — | Our method is "ensemble-after-inference, generative fusion" | abs |

### A(ii) Strong model reading weaker models' answers vs the strong model alone (our L2)

| Paper | Finding | Direction vs ours |
|---|---|---|
| **"Review Rarely Beats the Stronger Model"**, Zhang & Shrestha, 2610.15897 | 12 open models (3B–72B; Qwen3-4/8/32B, Qwen2.5, Llama-3, Mistral). 48 pairs, 96 review directions. MATH, MMLU-Pro, BigCodeBench, ReClor/FOLIO/GPQA-D. Review beats the **stronger model alone in only 9/96 directions, all in maths and knowledge**. Strong reviewer −3.0 points on average. Cause: **anchoring** on the generator's answer. Costs 2.6–2.8× the strong model. A logistic **selector** is +1.2 over the stronger model | **Closest and contrary.** One weaker generator; ours has 4 workers incl. specialists, and a gain in 5/10 domains |
| **Kim et al.**, 2512.08296 / *Nature MI* 8:1157 (2026), DOI 10.1038/s42256-026-01268-y | Heterogeneous centralized set-ups (Table 12, BrowseComp-Plus): GPT-5 orchestrator + GPT-5-nano sub-agents 0.19 (−0.15 vs homogeneous GPT-5). Across 13 mixes, centralized mixes −12.6 points vs strong-homogeneous; "model mixing does not bypass the capability-saturation threshold" (~45% single-agent accuracy) | **Contrary on BrowseComp-Plus** (ours: +21 over the 35B alone). Different baseline: strong *MAS*, not strong single |
| **MoA "collaborativeness"**, 2406.04692 | Aggregator improves even with lower-quality inputs (AlpacaEval) | Supports ours |
| **Self-MoA**, 2502.00674 | The strong model's *own* samples beat mixing in weaker models | Predicts our gain would vanish against a 35B Self-MoA at matched tokens: **the missing control** |
| **"Easier to Mislead Than to Correct"**, 2606.01637† | Wrong peer agreement flips correct answers more than correct peers fix wrong ones | Mechanism for our losses in chemistry, biology, law and engineering |
| **Tran & Kiela**, 2604.02460 | Single agent ≥ MAS at matched thinking tokens (Qwen3, R1-Distill, Gemini 2.5). MAS gains come from unaccounted compute | Our +5 to +26 comes with ~10× tokens |
| **Can Small Agents Collaborate…**, Żywot et al., 2601.11327 | Qwen3 1.7B–32B. Small orchestrator + sub-agents beat larger single agents on tool tasks. **Orchestrator capacity matters more than sub-agent capacity** | Consistent with ours (35B orchestrator, 4B workers) |
| **Decoupling knowledge and reasoning**, 2507.18178† | Reasoning helps on maths/physics; on knowledge-heavy domains it adds noise | A candidate explanation for our domain split |

### A(iii) Selection ≈ random; personas do not help

| Paper | Finding |
|---|---|
| **Zheng et al.**, 2311.10054 (EMNLP-F'24) | 162 personas, 4 open families, 2,410 MMLU questions: no persona improves accuracy; best-persona selection is hard; effects inconsistent across questions (v1 claimed the opposite) |
| **Prompting Science Report 4**, Basil et al., 2512.05858 | 6 models, GPQA-D + MMLU-Pro: expert personas (in-domain, off-domain, low-knowledge) do not reliably change accuracy; domain match does not matter (except Gemini 2.0 Flash) |
| **Beyond Symmetric Agents**, Ferreira et al., 2609.35875 (ICLR'27 sub.) | 23 small open models, 5,500+ runs: **"persona tax"** (personas lower accuracy). Temperature and model diversity also fail. Mixed-model teams lose to votes over their own roster. Debate ties or loses to budget-matched SC at 3.4× tokens |
| **When does persona prompting help?**, 2605.29420 | Personas reshape style (depth vs clarity), not capability |
| **Diversity audit**, Kim, 2607.20768 | 31,900 subsets of 30 LLMs on MMLU-Pro: oracle gain always > 0, yet **vote beats the best member in ~10%** of size-3 subsets. Diversity metrics ≈ capability (ρ ≈ 0.99) |
| **LLMRouterBench**, 2601.07206 (ACL-F'26) | Includes a Random Router. Several routers fail to beat Best-Single; the gap to the oracle comes from "model-recall failures" |
| **Yang et al.**, 2602.03794 (ICLR'26), *the predecessor paper of this repo's code* | 2 diverse agents ≈ 16 homogeneous. Diversity (models, prompts, tools) drives MAS scaling. **Our result 1 partly contradicts its prompt-diversity claim at small scale** |
| AgentInit, 2509.19236 (EMNLP-F'25) | Reports gains over predefined strategies (units unclear in abs); no clean random-team control found |

No paper found that tests **an LLM orchestrator with scoreboard memory choosing persona teams against random teams under Holm correction**. The null itself is new, but in light of the rows above it is expected.

### A(iv) Multi-agent systems vs a single strong model at matched cost

| Paper | Finding |
|---|---|
| Kim et al., 2512.08296 / Nature MI | 3 families (GPT-5 nano/mini/full; Gemini 2.0F/2.5F/2.5P; Sonnet 3.7/4/4.5), 6 agentic benchmarks (incl. **BrowseComp-Plus, Finance-Agent, PlanCraft, WorkBench**), ~4.8k tokens matched. Centralized vs single: Finance-Agent +80.8%, BrowseComp-Plus +0.2%, PlanCraft −50.3%, WorkBench −1.2%. Saturation above ~45% single-agent accuracy |
| Illusion of MA Advantage, Jwalapuram et al., 2606.13003 (NeurIPS'26 poster) | Auto-designed MAS < CoT-SC at up to 10× cost, incl. BrowseComp-Plus |
| Debate or Vote, Choi et al., 2508.17536 (NeurIPS'25 spotlight) | Majority vote explains most MAD gains; debate is a martingale |
| Stop Overvaluing MAD, Zhang et al., 2502.08788 | 5 MAD methods × 9 benchmarks × 4 models: MAD often < CoT/SC. **Model heterogeneity is a "universal antidote"** |
| Are More LLM Calls All You Need?, Chen et al., 2403.02419 | Vote accuracy is non-monotonic in calls (helps easy, hurts hard questions) |
| Tran & Kiela, 2604.02460 | See A(ii) |
| Equal inference cost, 2609.04217† | ALFWorld: team 0.769 vs single 0.754, p = 0.80, at 1.8× calls |
| OneFlow, Xu et al., 2601.12307 | One agent simulates homogeneous (and an auto-optimised heterogeneous) workflow at equal accuracy and lower cost |
| Beyond the Strongest LLM, Tian et al., 2509.23537 | Orchestrating GPT-5, Gemini 2.5 Pro, Grok 4, Sonnet 4 "matches or exceeds" the strongest single model (GPQA-D, IFEval, MuSR) |

### A(v) Small or specialist open models vs frontier

| Paper | Claim |
|---|---|
| **SLMs are the Future of Agentic AI**, Belcak et al. (NVIDIA), 2506.02153 | Position paper: SLMs are sufficient and more economical for most agentic calls; heterogeneous systems. **No experiments** |
| **ToolOrchestra**, Su et al.† (NVIDIA), 2511.21689 | RL-trained 8B orchestrator calls tools, specialist models (e.g. Qwen2.5-Math/Coder†) and frontier LLMs. HLE 37.1 vs GPT-5 35.1 at 2.5× efficiency; FRAMES 76.3; ~30% of GPT-5's cost†. **Its pool includes frontier models**, so it does not show that open specialists alone suffice |
| **Conductor**, Sakana, 2512.04388 (ICLR'26) | RL-trained 7B (Qwen2.5-7B†) orchestrates **frontier** workers (GPT-5, Sonnet 4, Gemini 2.5 Pro); beats every worker; GPQA-D 87.5%, LCB 83.9%† |
| **Fugu**, Sakana, 2606.21228 | Product orchestrator family over an agent team; SOTA claims on GPQA-D, HLE, SWE-Pro (no numbers in abs) |
| **Learnware of LMs**, 2505.13425† | ~100 task-specific 8B models (finance, health, maths) can beat a single large model on task-specific queries |
| **Small Model as Master Orchestrator**, 2604.17009 | Lightweight RL orchestrator (ParaManager), agent-as-tool (no numbers in abs) |

**Gap:** every 2025–26 "small orchestrator beats frontier" result either **trains** the orchestrator with RL or puts **frontier models in the worker pool**. None found that tests a training-free team of **open community fine-tunes** against a frontier model at token-level cost on the same questions.

### A(vi) Near-duplicates of X1

Nothing found that does exactly X1 (fixed per-domain teams of persona workers + HF community fine-tunes, a larger untrained orchestrator, reasoning **and** agentic benchmarks, with an orchestrator-alone control). The nearest, in order:

1. **Symbolic-MoE** (2503.05641): open experts, our hard benchmarks, an aggregator. Recruits per instance, not a fixed team; no agentic tasks; no aggregator-alone control†.
2. **X-MAS** (2505.16997): heterogeneous domain fine-tunes per MAS function; same domains. Gains reported vs homogeneous MAS, not vs the aggregator alone.
3. **S-DAG** (2511.06727): subject-routed expert models on MMLU-Pro / GPQA.
4. **Kim et al.** (2512.08296): same 4 agentic benchmarks, centralized topology, heterogeneous appendix. Frontier models only.
5. SOLVE-Med† (ten 1B medical fine-tunes + router + orchestrator; seen only in an HF Papers listing, no arXiv ID verified).

---

## B. Our results 1–5 against the literature

| # | Result | Status | Prior work | How a reviewer frames it |
|---|---|---|---|---|
| 1 | LLM orchestrator (± memory) choosing persona teams on one small model ≈ random; debate / centralized / synthesis ≈ vote at 2–5× tokens; personas ≈ noise | **Known in parts; the selection null is new but expected** | Personas: 2311.10054, 2512.05858, 2609.35875. Topology ≈ vote: 2508.17536, 2502.08788, 2609.35875, 2606.13003. Selection rarely realises oracle gain: 2607.20768, 2601.07206 | "If arms differ by <1% variance, no selector can beat random: tautological." Worth a section as a **negative replication** of 2602.03794's prompt-diversity claim, not a headline |
| 2 | Oracle over 4 agents is +5 points (hard) / +12 (agentic) above best single | **Known concept**; numbers new for these pools | Oracle routers: RouterBench, LLMRouterBench 2601.07206; co-failure ceiling 2606.27288; oracle gain always > 0 but rarely realised, 2607.20768 | Motivation only. Report it with selection-valid CIs (2608.08265) |
| 3 | Orchestrator > team vote on 10/10 teams, > best worker on 7/10 (AIME 86 vs 74 vs 71; BrowseComp-Plus 59 vs 33) | **Partially known** | MoA, Symbolic-MoE, X-MAS, AggLM, AggAgent all show aggregator > vote. New: fixed per-domain teams with community fine-tunes, and **agentic re-run aggregation** on BrowseComp-Plus / WorkBench / Plancraft / Finance-Agent | "The aggregator is ~9× larger than the workers, so beating their vote is unsurprising: compare it with the aggregator alone." That is result 4 |
| 4 | 35B orchestrator + team vs 35B alone: clear gains on maths, physics, BrowseComp-Plus, WorkBench, Plancraft; n.s. on chemistry, biology, law, Finance-Agent; −2.7 on engineering; ~10× tokens | **Most novel, but contested and confounded** | Supporting: MoA collaborativeness, 2601.11327. Contrary: 2610.15897 (review beats the stronger model in 9/96 directions), Kim et al. heterogeneous centralized −12.6 points, Self-MoA. Confound: matched compute (2604.02460, 2606.13003) | "Is it the team or the 10× compute? Run the 35B with Self-MoA / SC / two attempts at matched tokens." Also: "is the domain split reasoning vs knowledge, or headroom (Kim's ~45% saturation)?" |
| 5 | Claude Sonnet alone beats orchestrator+team on all 8 benchmarks (89.8 vs 81.5 hard; 85.7 vs 70.0 tools), ~37× fewer output tokens | **Expected; consistent with prior work** | Kim et al. Nature MI ("capable models outgrow collaboration"); 2606.27288; RL orchestrators beat frontier only with frontier workers (ToolOrchestra, Conductor) | Useful as an honest ceiling. Weak as is: fold 0, 1 sample, Sonnet version unstated. Needs all folds, a $ / latency / energy axis, and at least one open frontier-class model |

**Fidelity risks a reviewer will raise on 3–4:**
- **Contamination:** RL-MATH-4B and other community fine-tunes may have trained on MATH or AIME.
- **Thinking off:** the orchestrator runs thinking off while some workers think. Partly defused by X0: the 35B scores 83.0% thinking off vs 75.0% thinking on.
- **Re-run, not aggregation:** on tool tasks the orchestrator re-runs the episode, so it gets a second attempt. The control is the 35B with 2 attempts plus self-selection.

---

## C. Candidate contribution statements

**C1. Aggregation vs the aggregator alone, by domain and at matched compute (strongest; builds on result 4).**
> "Across ten domains (six reasoning, four agentic), a mid-size open model that reads answers from a team of small specialist models beats its own solo answer only where [multi-step reasoning or tool use / single-model accuracy is below X], and this gain [survives / does not survive] matching compute with the aggregator's own samples."

Evidence needed:
- 35B Self-MoA and SC@k, plus two-attempt re-runs on agentic tasks, at matched tokens.
- Per-domain paired tests with Holm correction.
- Regression of the gain on 35B-alone accuracy (Kim's saturation) vs domain type.
- Fixed / broke transition counts, to show anchoring (2610.15897) vs correction.
- L1 (4B orchestrator) to separate aggregator size from team value.

Positions against 2610.15897, Self-MoA and Kim et al.'s heterogeneous appendix, which disagree with each other. A careful matched-compute answer is publishable either way.

**C2. Cost-accounted reality check of training-free open-model orchestration (benchmark / negative-results framing; results 1, 3, 5).**
> "Training-free orchestration of small open models (persona selection, topology, fixed specialist teams with a larger aggregator) recovers part of a 5–12-point oracle headroom, but stays 8–16 points below a single frontier model at ~37× its output tokens."

Evidence needed:
- All 5 folds and 3 samples for the frontier arm, with the exact model ID.
- At least one more frontier model and one open frontier-class model (e.g. a large open MoE) as a bridge.
- $ and wall-clock / energy on the local box vs API.
- Pareto plots; the shortlist baselines from `orchestration-baselines.md` §4 (SC at matched cost, cascade, kNN router) on the same axes.

Fits a NeurIPS Datasets & Benchmarks or ACL Findings "position + evidence" paper. It answers 2506.02153's untested claim with data.

**C3. Do community fine-tunes add complementary signal beyond personas on one base? (narrower; results 1 and 3).**
> "In fixed teams, the aggregator's gain over vote and solo comes from heterogeneous fine-tuned specialists, not from persona prompts on a shared base (persona-only teams ≈ noise)."

Evidence needed:
- Ablate stage 2 over saved stage-1 answers: personas-only, specialists-only, k samples of the base 4B.
- Paired base-vs-fine-tune solo rows (L3).
- Error-correlation (co-failure β, 2606.27288) of specialists vs personas.
- Contamination checks for each specialist.

Replay-only, so cheap. It tests the claim of 2502.08788 and 2602.03794 that heterogeneity is the lever, with community checkpoints rather than different vendor families.
