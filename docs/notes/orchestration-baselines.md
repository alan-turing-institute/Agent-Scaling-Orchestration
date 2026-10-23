# Published baselines for a per-question orchestrator (routing / cascade / MAS search)

Verified with arXiv and GitHub on 2026-10-19. Rows marked † come from memory plus abstracts and were only partly checked against the full text. "Replay" means it can be scored offline from our answer matrix, which holds every agent × question × k repeats.

## 1. Anchor paper: it shares no benchmarks with ours

**Kim et al., "Towards a Science of Scaling Agent Systems", arXiv:2512.08296.**
- **What it covers:** 5 architectures (Single, Independent, Centralized, Decentralized, Hybrid), 260 configurations, GPT-5 / Gemini-2.x / Claude-Sonnet families.
- **Benchmarks:** BrowseComp-Plus, Finance-Agent, PlanCraft, WorkBench, SWE-bench Verified, Terminal-Bench. The paper explicitly excludes MMLU/GSM8K/MATH/GPQA/AIME as "non-agentic".
- **Cost:** tokens, matched at about 4.8k per trial. It also reports success per 1k tokens and USD.
- **Selection:** a scaling-law predictor picks the best architecture for 87% of held-out configurations.
- **Code:** no public repo found.
- **Verdict:** we cannot reuse its numbers. We can borrow two things: matched-token comparisons and "a measured single-agent baseline decides whether to delegate". Its capability-saturation finding is the hypothesis our orchestrator tests.

## 2. Single-model routers and cascades

| Paper (arXiv) | Decides | Overlapping benchmarks | Cost metric | Code | Training | Replay with our pool? |
|---|---|---|---|---|---|---|
| **FrugalGPT** (2305.05176) | Cascade: model order, per-stage accept thresholds, up to 3 stages | None (HEADLINES, OVERRULING, CoQA, AGNEWS, SciQ) | USD | github.com/stanford-futuredata/FrugalGPT | DistilBERT answer scorer plus threshold search; hundreds to thousands of labelled (q, a, correct) | **Yes.** Swap the scorer for vote-agreement or a small probe. Cost becomes our tokens or compute. |
| **AutoMix** (2310.12963) | 2–3-model cascade driven by few-shot self-verification and a POMDP | None (QASPER, QuALITY, CoQA, MuTual, DiplomatQA) | IBC (Δperf/Δcost) | github.com/automix-llm/automix | POMDP fit on about 50+ labelled examples | **Partly.** Self-verification needs one extra call per small-model answer, which is cheap to add once. Vote-entropy is a replayable proxy. |
| **Hybrid LLM** (2404.14618)† | Binary small/large router from predicted quality gap | MixInstruct (no overlap) | Fraction of calls to the large model | Microsoft code not confirmed | Fine-tuned DeBERTa on about 10k queries | **Yes,** as a binary special case of a learned router. Not worth a separate baseline. |
| **RouteLLM** (2406.18665) | Binary strong/weak router (matrix factorisation, BERT, causal-LLM, similarity-weighted) | **MMLU, GSM8K**, MT-Bench | % calls to the strong model, then USD | github.com/lm-sys/RouteLLM | Arena preference data (80k battles) plus augmentation | **Partly.** Pairwise only. The MF and SW-ranking variants can be retrained on our win-matrix. The Arena-pretrained checkpoints do not cover our models. |
| **Cascade routing**, Dekoninck et al. (2410.10347)† | Unified routing and cascading, provably optimal under quality and cost estimates | **MMLU, GSM8K**, RouterBench, SWE-Bench | Cost-quality AUC | github.com/eth-sri/cascade-routing | Needs quality and cost estimators (linear or logistic on features) | **Yes.** It is the principled form of what we want, and replay is native: it was evaluated on precomputed answers. |
| **RouterDC** (2409.19886) | Single model per query (dual contrastive encoder) | **MMLU, GSM8K**, ARC-C, CMMLU, HumanEval; OOD PreAlgebra | Accuracy only | github.com/shuhao02/RouterDC | mDeBERTa encoder trained on a few thousand labelled queries | **Yes.** Accuracy-only, so a cost term must be added. |
| **EmbedLLM** (2410.02223) | Model embeddings (matrix factorisation of correctness), then linear router | Leaderboard-style (MMLU, GSM8K, etc.)† | Accuracy | github.com/richardzhuang0412/EmbedLLM† | Correctness matrix of many models | **Yes,** but designed for 100+ models. With our ~15 the gain over kNN is likely small. |
| **GraphRouter** (2410.03834) | Model choice via a task–query–LLM GNN | GSM8K, MATH† plus others | Reward = perf − λ·cost | github.com/ulab-uiuc/GraphRouter | GNN on interaction graph | **Yes,** but heavy for our size. |
| **Router-R1** (2506.09033)† | LLM router doing multi-round route-and-aggregate, RL-trained | NQ, TriviaQA, HotpotQA, etc. (no overlap) | Cost reward | github.com/ulab-uiuc/Router-R1 | PPO/GRPO on a 3B router | **No replay;** needs a heavy RL run. Skip. |
| **Avengers** (2505.19797) | Embed → k-means (K=64) → per-cluster best model → self-consistency vote (10 samples) | **AIME, MATH-500, MMLU-Pro, GPQA**, BBH, MedQA, etc. | Accuracy (the Pro variant adds a perf–cost score) | github.com/ZhangYiqun018/Avengers, …/AvengersPro (2508.12631) | None beyond a 70/30 split for cluster scoring | **Best fit.** Closest published analogue to us: ~7B open models, our hard benchmarks, training-free, fully replayable. |
| **kNN router**, Li (2505.12601) | Single model per query; utility = kNN-mean(score) − λ·kNN-mean(cost) | Open LLM Leaderboard v2 (**GPQA, MMLU-Pro, MATH-L5**), RouterBench | λ-weighted cost | Not found | None (k=10–100, BERT embeddings) | **Yes, trivially.** Its main claim is that kNN matches or beats learned routers. |

**Routing benchmarks for reference.**
- **RouterBench** (2403.12031): 405k precomputed outputs, MMLU/GSM8K/MBPP, 11 models. Released at github.com/withmartian/routerbench.
- **RouterEval** (2503.10657): 200M records from more than 8,500 open-LLM-leaderboard models, including MMLU-Pro, GPQA, MATH-L5, GSM8K, MMLU.
- **LLMRouterBench** (2601.07206): 21 datasets including AIME, 33 models (20 of them ~7B open), 10 routers, at github.com/ynulihao/LLMRouterBench. Key result: most routers, including OpenRouter, do not reliably beat **Best-Single-in-hindsight**.
- **LLMRouter** (2608.06867): a library of more than 16 routers.

Two recent sceptical papers support simple baselines:
- "Most of the LLM routing gap is task type" (2608.23023): a static task-type → model table captures most of the gain.
- "Opportunity is not realizability" (2608.08265): use selection-valid confidence intervals when comparing against the best fixed model.

## 3. Multi-agent, team and topology methods

| Paper (arXiv) | Decides | Overlapping benchmarks | Cost | Code | Training | Verdict |
|---|---|---|---|---|---|---|
| **Self-consistency** (2203.11171) / **More Agents** (Agent Forest, 2402.05120) | n samples → majority vote | **GSM8K, MATH, MMLU** (More Agents) | n× calls | github.com/MoreAgentsIsAllYouNeed/AgentForest | None | **Fully replayable** from k repeats. Mandatory. |
| **Debate or Vote**, Choi et al. (2508.17536) | Debate vs. majority vote | 7 NLP benchmarks incl. GSM8K / MMLU-type† | Rounds / tokens | arXiv | None | Shows that voting explains most of debate's gain. This is the justification for a vote-at-matched-cost baseline. |
| **Mixture-of-Agents** (2406.04692)† / **Self-MoA** (2502.00674) | Proposer layer plus aggregator. Self-MoA aggregates samples of the single best model. | MoA: AlpacaEval/MT-Bench. Self-MoA: **MMLU, MATH**, CRUX | Calls | github.com/togethercomputer/MoA | None | ≈ our "synthesis" topology. Needs one live aggregator call per question over cached proposals: cheap and semi-replayable. |
| **Symbolic/Skill-MoE** (2503.05641) | Per-instance recruits k experts from 16 open ~7B models by inferred skills, then an aggregator | **MMLU-Pro, GPQA, AIME**, MedMCQA | GPU-time | github.com/dinobby/Symbolic-MoE | Gradient-free: skill profiles built from a small validation set | **Strong candidate.** Open models, our benchmarks, decides team composition. Needs live aggregator calls. |
| **DyLAN** (2310.02170)† | Agent importance scores → team selection, dynamic debate | **MMLU, MATH**, HumanEval | Calls | github.com/SALT-NLP/DyLAN | Unsupervised importance scoring | Feasible but live-only, built on GPT-3.5. Medium effort. |
| **GPTSwarm** (2402.16823)† / **AgentPrune** (2410.02506)† / G-Designer | Edge-optimised agent graphs | **MMLU, GSM8K**, MultiArith, AQuA, HumanEval | Tokens, USD | github.com/metauto-ai/GPTSwarm, github.com/yanweiyue/AgentPrune | RL/REINFORCE over graph edges, per task | Topology is fixed per task, not per query. Weak fit, live-only. |
| **AFlow** (2410.10762)† / **ADAS** (2408.08435)† | Search over workflow code per task (MCTS / meta-agent) | AFlow: **GSM8K, MATH**. ADAS: **MMLU, GPQA**, MGSM, DROP | USD | MetaGPT repo; github.com/ShengranHu/ADAS | Search needs a strong optimizer LLM (GPT-4o / Claude) and hundreds of executions | **Per-task, not per-query.** Needs a strong proprietary meta-LLM, or Qwen3.6-35B as a weak stand-in. Not recommended. |
| **MacNet** (2406.07155)† | Fixed DAG topologies with up to 1000 agents | **MMLU**, HumanEval, SRDD | Tokens | ChatDev `macnet` branch | None | A topology study, not a selector. Skip. |
| **MasRouter** (2502.11133, ACL'25) | Per-query collaboration mode (CoT / Reflection / SC / Debate / MacNet) → roles → LLM per agent | **MMLU, GSM8K, MATH** (519 stratified), HumanEval, MBPP | USD tokens | github.com/yanweiyue/masrouter | Policy-gradient with **online MAS execution**. Small: K = 5–10 iterations, about $1.4–3.6 of API spend. Pool: gpt-4o-mini, claude-3.5-haiku, gemini-1.5-flash, llama-3.1-70b, deepseek-v3. | **The closest published analogue to our decision space.** The `.env` takes one URL/KEY pair, so it needs patching to multiple vLLM endpoints and our model list. Training needs live runs of debate etc., so it is not replayable; vote and single-agent arms are. Medium–high effort, but the most defensible "published orchestrator" baseline. |
| **MaAS** (2502.04180, ICML'25 oral) | Per-query sample from an "agentic supernet" of operators (CoT, SC, debate, reflexion, ensemble, …) | **GSM8K, MATH**, HumanEval, MBPP, MultiArith, GAIA | USD, tokens | github.com/bingreeky/MaAS | Supernet trained by sampled execution, built on MetaGPT, GPT-4o-mini only | Chooses **topology/operators, not models**. Single backbone. Swapping the backbone to vLLM is feasible via the OpenAI-compatible config. Live-only training. |
| **Puppeteer**, Dang et al. (2505.19591) | RL-trained orchestrator sequences agents step by step | **MMLU-Pro**, GSM-Hard, SRDD, CommonGen-Hard | Tokens | github.com/OpenBMB/ChatDev (branch `puppeteer`) | RL on the orchestrator | Interesting, but RL plus live rollouts. Second-tier. |
| **MAS-Zero** (2505.14996) | Inference-time meta-agent designs the MAS per query | **AIME24, GPQA-Diamond**, SWE | Calls | github.com/SalesforceAIResearch/MAS-Zero | None (zero supervision), but needs a ≥32B meta-agent | Expensive per query. Possible with Qwen3.6-35B as meta-agent. Second-tier. |
| DAAO (2509.11079, WWW'26), Uno-Orchestra (2605.05007), ATLAS (2606.01667) | Difficulty-aware workflow and router (VAE); RL (model, primitive) dispatch; test-time allocation | Six benchmarks for DAAO; MATH-500/AIME for Uno-Orchestra; GPQA-D for ATLAS | — | None found | Learned / RL | Too new or no code. Cite only. |

**Counter-evidence.** "The Illusion of Multi-Agent Advantage" (2606.13003, Jwalapuram et al.) finds that auto-designed MAS consistently underperform **CoT-SC** at up to 10× the cost. LLMRouterBench reaches the same conclusion for routers. Both argue our headline comparison must be against SC and best-single at **matched cost**.

## 4. Recommended shortlist

1. **(a) Best-single and SC@matched-cost (trivial but strong). Cost: none; pure replay.**
   - Best fixed model in hindsight, plus a CV-selected version, following LLMRouterBench and 2608.08265.
   - Majority vote over n samples of the strongest model (Qwen3.6-35B, thinking on), with n swept so the orchestrator's token and compute cost is bracketed (Agent Forest / CoT-SC).
   - A per-benchmark static table: the best (model, n) per source dataset, after 2608.23023.
2. **(b) Cascade (FrugalGPT / AutoMix, implemented through Dekoninck's cascade-routing). Cost: replay only, no live calls.**
   - Order the models by cost, for example 0.8B → 4B → E4B → 35B-A3B, then the thinking variants.
   - Accept or escalate on a confidence signal available in replay: agreement among the k repeats (vote margin).
   - Fit thresholds on training folds under a cost budget.
   - Report the cost-accuracy AUC.
3. **(c) Learned router (kNN-utility router, plus Avengers cluster routing with voting). Cost: an embedding pass only.**
   - Embed questions with a small embedding model.
   - kNN (k = 10–100) predicts each configuration's accuracy and cost.
   - Pick the configuration maximising accuracy − λ·cost.
   - Train only on our folds; it is fully replayable.
   - Avengers (K-means, per-cluster best model, SC vote) is the published variant evaluated on GPQA, MMLU-Pro, MATH-500 and AIME with ~7B open models, so its numbers give an external sanity check.
   - Optionally add RouterDC or the RouteLLM matrix-factorisation router if a parametric comparison is wanted. The literature says they rarely beat kNN.
4. **(d) Published MAS orchestrator: MasRouter first, Symbolic-MoE as the cheaper alternative. Cost: live runs.**
   - **MasRouter** has exactly our decision space (mode + roles + model). It is trainable for a few dollars-equivalent of tokens on our train fold. It needs a patch for multi-endpoint vLLM and live execution of debate and reflection arms. Of the four (a)–(d) baselines it is the only one that needs live runs.
   - **Symbolic-MoE** is gradient-free, uses open-model pools on our hard benchmarks, and decides team composition plus aggregation. It needs only a validation-set skill profile and live aggregator calls.
   - Skip MaAS/AFlow/ADAS. They optimise per task, are tied to GPT-4o-mini/MetaGPT, and would need a strong proprietary meta-LLM to be faithful.

**Honest caveats.**
- No published method uses our exact pool. All numbers have to be regenerated on our pool; none can be copied.
- Kim et al. does not share our benchmarks.
- MasRouter, MaAS, AFlow, ADAS, DyLAN, GPTSwarm and AgentPrune were all tuned on GPT-3.5/4o-mini-class APIs. Reproducing them with 0.8B–4B agents may understate them, so state this as a fidelity limitation.
- RL methods (Router-R1, Puppeteer, Uno-Orchestra) need training runs we should not budget for.
