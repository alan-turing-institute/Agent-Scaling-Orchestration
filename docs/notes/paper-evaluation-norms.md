# What reviewers expect from an orchestration / multi-agent / routing paper: baselines, cost, statistics, benchmarks

Researched 2026-11-13. This note builds on [orchestration-baselines.md](orchestration-baselines.md), which covers the routers, cascades and MAS methods themselves; this one covers *how* to compare against them. Every claim carries a source. Claims marked † were not checked against the primary text in this session; they come from memory, abstracts or secondary summaries.

**OpenReview limitation.** The OpenReview API sits behind a bot challenge. Every forum fetch returned `ChallengeRequiredError` (tried on the MoA ICLR 2025 forum `h0ZfDIrj7T`), so no review text was read. Evidence for "what reviewers ask" therefore comes from:
- published critiques of the field;
- what accepted 2025–2026 papers include;
- venue checklists.

## 0. Gaps in our current design that a reviewer is most likely to flag

| # | Gap | Why a reviewer will ask | Fix |
|---|---|---|---|
| 1 | No **35B self-consistency at matched tokens** and no **35B thinking-on at a matched thinking budget**. We have only "35B alone, thinking off". | The main criticism of MAS papers is that their gains come from extra test-time compute. CoT-SC at a fraction of the cost beats auto-designed MAS ([2606.13003](https://arxiv.org/abs/2606.13003)). MAD loses to SC at equal responses ([2310.01798](https://arxiv.org/abs/2310.01798), [2502.08788](https://arxiv.org/abs/2502.08788)). A single agent matches MAS at equal thinking tokens ([2604.02460](https://arxiv.org/abs/2604.02460)). | Sweep k for maj@k of the 35B (off and on) and its thinking budget, so the orchestrator's total tokens are bracketed (§1). |
| 2 | No **USC / best-of-N-with-judge over the same worker answers**. | An orchestrator that reads worker answers can be just a consensus picker. USC is exactly that ([2311.17311](https://arxiv.org/abs/2311.17311)), and Self-MoA beats mixed MoA ([2502.00674](https://arxiv.org/abs/2502.00674)). | Run the 35B with a USC prompt and with a "pick the best" judge prompt on the same summaries. Report how often the orchestrator's answer is none of the workers' answers. |
| 3 | Best worker is chosen **in hindsight** only. | Hindsight selection is an oracle, not a baseline. Most routers fail to beat best-single ([2601.07206](https://arxiv.org/abs/2601.07206)), and comparisons with the best fixed model need selection-valid CIs ([2608.08265](https://arxiv.org/abs/2608.08265)†). | Select on the training folds per benchmark, evaluate on the test fold, and keep hindsight as an upper bound. |
| 4 | **Team composition per domain**: how was it chosen? | If it was chosen with test-fold results, it leaks. Kapoor et al. list missing holdouts as a main flaw of agent evaluation ([2407.01502](https://arxiv.org/abs/2407.01502)). | Show that teams were fixed from the dev split or training folds only. |
| 5 | **Wilson intervals on pooled samples** (3 samples × n questions) | The three samples of one question are correlated. Treating them as 3n independent draws understates the SE; Miller recommends clustered SEs ([2411.00640](https://arxiv.org/abs/2411.00640)). | Use the question as the unit: average the samples per question, then bootstrap over questions (§3). |
| 6 | **The judge is the 35B**, which is also the orchestrator and the 35B-alone baseline. | LLM evaluators favour their own generations ([2404.13076](https://arxiv.org/abs/2404.13076), NeurIPS 2024). The official judges differ: GPT-4.1 in the BrowseComp-Plus paper, Qwen3-32B in its repo script, GPT-5.2 mode-of-3 at Vals. | Regrade with an independent judge, or report agreement with a human-labelled subset (Kim et al. report κ 0.87–0.91 per benchmark, [2512.08296](https://arxiv.org/abs/2512.08296)). |
| 7 | **Sonnet on 1 fold**, run through the Claude Code CLI | A reader cannot tell harness effects from model effects. | Run every fold, or report it as a subset with its own CI. State the model id (`claude-sonnet-5-5`), the effort level and the harness, and cost it in $ (its tokenizer differs, §2). |
| 8 | **Benchmarks are subsets** with a non-standard harness (BM25, Wikipedia search, 50 public Finance-Agent questions, …) | ARR B6 asks for split statistics even for standard benchmarks ([ARR](https://aclrollingreview.org/responsibleNLPresearch/)). Published numbers will not transfer. | Add a "deviations from the official protocol" table (§4) and avoid side-by-side comparisons with leaderboard numbers. |
| 9 | **Contamination**: all 120 AIME problems (2022–2025) and MATH-500 predate Qwen3.5/3.6. | AIME 2024 shows contamination ([2505.23281](https://arxiv.org/abs/2505.23281)). Qwen2.5 regenerates MATH-500 problems from 60% prefixes ([2507.10532](https://arxiv.org/abs/2507.10532)). | Report AIME by year. Add a post-release slice (AIME 2026 / HMMT Feb 2026, already reported on the Qwen3.6 card, so only partly clean). Argue that the relative comparisons are within-model. |
| 10 | Cost given in tokens only. | Kapoor et al. ask for input/output tokens **plus $** ([2407.01502](https://arxiv.org/abs/2407.01502)). HAL plots accuracy against both $ and tokens ([2510.11977](https://arxiv.org/abs/2510.11977)). FLOPs and $ rank our 4B and 35B in opposite order (§2). | Report tokens (prompt and completion), list-price $, active-param FLOPs and wall-clock, and draw Pareto plots. |

## 1. Baselines: protocol and how to match compute

**Evidence that reviewers expect compute-matched simple baselines**

- **"The Illusion of Multi-Agent Advantage"** ([2606.13003](https://arxiv.org/abs/2606.13003)).
  - Six automatic MAS (DyLAN, MAS-Zero, ADAS, AFlow, MaAS, MAS-Orchestra) are compared with CoT-SC using a 5-sample majority vote.
  - The MAS cost about 10× CoT-SC, and CoT-SC often reaches higher accuracy at under 10% of their cost.
  - Results are averaged over 3 runs; costs are in $ and include search overhead.
  - Quote: prior comparisons "rarely control for inference budgets such as number of LLM calls, total cost, retries, or sampled paths."
- **"Stop Overvaluing Multi-Agent Debate"** ([2502.08788](https://arxiv.org/abs/2502.08788), position paper): 5 MAD methods, 9 benchmarks, 4 models. MAD often fails to beat CoT/SC despite more compute, and the field shows "weak baselines, and inconsistent setups".
- **Huang et al., ICLR 2024** ([2310.01798](https://arxiv.org/abs/2310.01798)): at an equal number of responses, debate underperforms majority voting.
- **MacNet, ICLR 2025** (Qian et al., [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/66a026c0d17040889b50f0dfa650e5e0-Paper-Conference.pdf)): includes "equalizing the number of LLM calls … majority voting … or best-of-N" baselines. Accepted MAS papers ship these.
- **Tran & Kiela** ([2604.02460](https://arxiv.org/abs/2604.02460), preprint): a single agent matches or beats MAS on multi-hop QA at a fixed reasoning-token budget. It flags that API budget control (Gemini) is approximate.
- **Wunderlich et al.** ([2605.01566](https://arxiv.org/abs/2605.01566), ACL 2026 SRW): at an equal budget, debate beats SC by +1.3 pp and MoA by +2.7 pp on MMLU-Pro/BBH. Gains are far smaller once compute is matched, but not zero.
- **Kim et al.** ([2512.08296](https://arxiv.org/abs/2512.08296)): matched total reasoning tokens across architectures, with a mean of about 4,800 per trial.

**Baselines table**

| Baseline | Protocol | How to match compute | Our status |
|---|---|---|---|
| **Majority vote of the team** | Plurality of the workers' final answers, with a seeded tie-break | Costs exactly the workers' tokens, so the orchestrator's extra cost is its own call | Have it |
| **Self-consistency (maj@k) of the 35B** | k samples at vendor sampling, plurality vote ([2203.11171](https://arxiv.org/abs/2203.11171)). Estimate maj@k by averaging over random size-k subsets of a larger pool of K ≥ 2k samples, not from one draw.† | Sweep k ∈ {1, 2, 4, 8, …} and plot accuracy against mean total tokens per question. Read off accuracy at the orchestrator system's token count (workers plus orchestrator). Do this for thinking off **and** on. | Missing. The answer matrix has k = 3; the 35B needs more samples. |
| **35B thinking-on at a matched budget** | Same model, thinking on, with a thinking cap (`THINKING_TOKEN_BUDGET`, served with `--reasoning-config`). Budget forcing as in s1 ([2501.19393](https://arxiv.org/abs/2501.19393)†). | Set the cap so prompt plus completion roughly equals the system total, and also sweep it. | Missing at a matched budget. X0 has thinking-on runs at 30,720 tokens. |
| **Universal Self-Consistency** | The LLM is shown n candidate responses and asked to pick the one most consistent with the others ([2311.17311](https://arxiv.org/abs/2311.17311)). It matches SC on maths without needing parsable answers. | Same candidates (the workers' summaries) and the same aggregator (35B, thinking off), so the cost equals ours. **The cleanest ablation of "does the orchestrator add reasoning?"** | Missing |
| **Best-of-N with an LLM judge or verifier** | The judge scores each of N candidates (pointwise) or runs pairwise comparisons, and the top one is returned. Resampling with imperfect verifiers plateaus, and the best N is often under 10 ([2411.17501](https://arxiv.org/abs/2411.17501)). | N = team size and judge = 35B, so the cost is about ours. Count the judge's tokens. | Missing |
| **Self-MoA / MoA** | MoA: layers of proposers plus an aggregator. The original used 3 layers × 6 open proposers with Qwen1.5-110B as aggregator, and MoA-Lite used 2 layers ([2406.04692](https://arxiv.org/abs/2406.04692), ICLR 2025). Self-MoA aggregates samples of the single best model and beats mixed MoA by 6.6% on AlpacaEval and 3.8% averaged over MMLU/CRUX/MATH ([2502.00674](https://arxiv.org/abs/2502.00674)). | Our system **is** a 1-layer MoA with a 35B aggregator. Self-MoA means 35B samples → 35B aggregator at the same token total. | Must position against MoA explicitly. Self-MoA is missing. |
| **Best single worker, chosen by CV** | Pick the best agent per benchmark on the training folds and score it on the held-out fold. Hindsight best is reported as an upper bound only. | Its own cost; plot it on the Pareto chart. | Have hindsight only |
| **35B alone** (thinking off) | As run | As run | Have it |
| **Frontier API** | Same prompts. State the model id, effort or thinking setting, harness and date. Kim et al. cover GPT-5 / Gemini / Claude families ([2512.08296](https://arxiv.org/abs/2512.08296)). | Compare in $ only (§2). | Sonnet 5.5, 1 fold |
| **Routers** (kNN, RouteLLM-MF) | Train on our folds. RouteLLM metrics: PGR = (r(router) − r(weak)) / (r(strong) − r(weak)), APGR = mean PGR over 10 cost levels, CPT(x%) = minimum strong-call share for PGR = x%, plus a random router ([2406.18665](https://arxiv.org/abs/2406.18665)). | Cost is measured as the share of calls to the strong model, or as $. Compare curves with the random router. | Planned (#23) |
| **Cascades** | Escalate on vote disagreement, with thresholds fit on training folds. Kapoor et al. use "escalation" (cheap → expensive on failure) as a simple agent baseline ([2407.01502](https://arxiv.org/abs/2407.01502)). The cascade-routing quality-cost AUC is in [2410.10347](https://arxiv.org/abs/2410.10347). | Accuracy–cost curve; AIQ = area under the non-decreasing convex hull, divided by the cost range ([2403.12031](https://arxiv.org/abs/2403.12031)) | Planned (#22) |
| **Oracle** (any worker right) | Ceiling | — | Have it |

**Matching rules worth stating in the paper**
- Count every call: workers, orchestrator, judge and retries. The orchestrator's prompt includes the workers' summaries, so its prompt tokens are large.
- Match on one primary axis, chosen before seeing results, and show the others in the appendix.
- Prefer curves to single points. Kapoor et al. take the convex hull, since any two agents can be mixed at random ([2407.01502](https://arxiv.org/abs/2407.01502)).

## 2. Cost reporting

**Norms**

| Source | What they report |
|---|---|
| Kapoor et al., "AI Agents That Matter" ([2407.01502](https://arxiv.org/abs/2407.01502)) | Accuracy–cost Pareto frontier, constrained to be convex, with costs in $. Downstream evaluations "should include input/output token counts in addition to dollar costs" so readers can reprice. Active-parameter proxies mislead: Mixtral 8x7B looked as cheap as Llama-2-13B but cost about 2× on Anyscale. |
| HAL, ICLR 2026 ([2510.11977](https://arxiv.org/abs/2510.11977), [site](https://hal.cs.princeton.edu/)) | 21,730 rollouts, 9 models, 9 benchmarks, about $40k. Pareto frontiers on both $ and tokens. The authors note that provider prices change and frontier positions move.† |
| RouterBench ([2403.12031](https://arxiv.org/abs/2403.12031)) | Open models are costed at **Together AI's** prices ("use their cost as reference"). AIQ is computed over the cost-quality convex hull. |
| MoA ([2406.04692](https://arxiv.org/abs/2406.04692)) | $ from provider list prices: Together AI for open models, dated 2024-05-22. TFLOPs serve as a latency proxy, taking the max over parallel proposers per layer and summing over layers. |
| Kim et al. ([2512.08296](https://arxiv.org/abs/2512.08296)) | Matched reasoning tokens. Also success per 1k tokens and USD (see the existing note). |
| Snell et al. ([2408.03314](https://arxiv.org/abs/2408.03314)) | FLOPs-matched comparison of test-time compute against a 14× larger model. The FLOPs formula was not checked.† |

**Recommendation for us.** Report four axes side by side; plan §7 already lists three of them.
1. **Tokens**: prompt and completion, with thinking counted inside completion. Only within the Qwen family; tokenizers differ across vendors.
2. **List-price $**: one provider and one date, plus a sensitivity check with the cheapest and median OpenRouter providers.
3. **FLOPs** ≈ 2 × N_active × tokens.†
4. **Wall-clock** on the Spark at fixed concurrency.

**FLOPs and $ disagree for our models.**
- Qwen3.6-35B-A3B has about 3B active parameters (the "A3B")†, fewer than Qwen3.5-4B's 4B (dense). By FLOPs per token, the 35B is the cheaper model.
- By list price the 35B costs 6.3× more per output token at DeepInfra ($0.95 against $0.15).
- On a bandwidth-bound Spark, NVFP4 decode speed decides wall-clock (`../vllm/qwen3.6/README.md`).
- This is Kapoor's Mixtral point again. Report all axes so the conclusion does not depend on which one is picked.

**List prices (USD per 1M tokens, fetched 2026-11-13)**

| Model | Input | Output | Cache read | Source |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 0.05–0.25 (DeepInfra 0.10) | 0.576–1.80 (DeepInfra 0.95) | — | [OpenRouter providers](https://openrouter.ai/qwen/qwen3.6-35b-a3b/providers): 10 providers, cheapest Darkbloom 0.05/0.70, Prime Intellect 0.063/0.576; CoreWeave, AkashML and Parasail 0.10/0.89–0.90; SiliconFlow 0.24/1.80 |
| Qwen3-30B-A3B-Instruct-2507 (older proxy) | 0.09–0.10 | 0.30 | — | [OpenRouter providers](https://openrouter.ai/qwen/qwen3-30b-a3b-instruct-2507/providers) |
| Qwen3.5-4B (reasoning) | 0.03 | 0.15 | — | DeepInfra FP8, the only provider listed by [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-5-4b/providers). Fireworks non-reasoning at 0.20/0.20 per [pricepertoken](https://pricepertoken.com/pricing-page/model/qwen-qwen3.5-4b)†. Not on OpenRouter. |
| Claude Sonnet 5.5 (`claude-sonnet-5-5`) | 2.00 | 10.00 | 0.10 (5-min cache write 2.50) | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing); Batch API 1.00/5.00 |
| Claude Sonnet 5 | 2.00 | 10.00 | 0.20 | same; the planned rise to $3/$15 on 2026-09-01 "will not occur" |
| Claude Sonnet 4.6 / 4.5 | 3.00 | 15.00 | 0.30 | same |
| Gemma-E4B, Ministral-3B, fine-tuned specialists | not looked up | | | Price fine-tunes at their base model's rate, and say so† |

**Pricing caveats**
- Claude 4.7-and-later tokenizers produce about 30% more tokens for the same text than Sonnet 4.6 and earlier ([pricing page](https://platform.claude.com/docs/en/about-claude/pricing)). Sonnet's thinking is billed but not shown. Compare Sonnet to Qwen in **$**, not in tokens.
- Claude Code adds its own system prompt, tools and cache traffic. Report its cost as "harness included", or rerun through the Messages API.
- Fix one provider: DeepInfra lists both Qwen3.5-4B and Qwen3.6-35B-A3B. Date-stamp the prices and give a ± range.

## 3. Statistics

**Guidance papers**

| Source | Recommendation |
|---|---|
| Miller, "Adding Error Bars to Evals" ([2411.00640](https://arxiv.org/abs/2411.00640)) | (1) SE of the mean via the CLT. (2) **Clustered SEs** when questions are related. (3) Reduce variance by resampling answers: pick K samples per question so sampling noise is small next to question-to-question variance. (4) **Run inference on question-level paired differences**, not on the two summary scores. (5) **Power analysis** before running. Multiple comparisons are not covered. |
| Bowyer et al., ICML 2025 spotlight ([2503.01747](https://arxiv.org/abs/2503.01747)) | CLT intervals undercover below a few hundred items. Use Beta/Bayesian or exact intervals instead (library: `sambowyer/bayes_evals`). This applies to Finance-Agent (50), AIME per year (30) and the 100-task agentic sets. |
| Hochlehnert et al., "A Sober Look" ([2504.07086](https://arxiv.org/abs/2504.07086)) | Seed std of Pass@1 is 5–15 pp; on AIME'24 one question moves Pass@1 by about 3 pp. v2 text: "For small benchmarks (e.g., AIME'24), run evaluations with at least 30 random seeds" (its own runs used 10). Tune temperature and top_p per model, then fix them. Use ≥32k max tokens to avoid truncation. Report hardware, since cluster-to-cluster gaps reach about 8 pp. Use the chat template. Use robust answer extraction with equivalence checking. |
| Madaan et al. ([2406.10229](https://arxiv.org/abs/2406.10229)) | Treat seed variance as the yardstick: a claimed gain should exceed it.† |
| Dror et al., ACL 2018 ([P18-1128](https://aclanthology.org/P18-1128/); appendix [1809.01448](https://arxiv.org/abs/1809.01448)) | Significance testing in NLP is often skipped or wrong. Gives a protocol for choosing the test: paired bootstrap or permutation for accuracy-like metrics, McNemar for paired binary outcomes.† |
| Dror et al., TACL, replicability analysis | Correct for multiple datasets when claiming "wins on k of n benchmarks".† |
| Card et al., EMNLP 2020 ([2010.06595](https://arxiv.org/abs/2010.06595)) | Underpowered comparisons are common in NLP, and most comparisons on small test sets are underpowered. Notebooks: `dallascard/NLP-power-analysis`. |
| Bengio & Grandvalet, JMLR 2004 ([v5](https://jmlr.org/papers/v5/grandvalet04a.html)) | No unbiased estimator of the variance of K-fold CV exists, and naive across-fold variance underestimates it. **Do not use fold-to-fold spread as the CI.** |
| NeurIPS 2026 checklist item 7 ([guide](https://neurips.cc/public/guides/PaperChecklist)) | State: the factors of variability captured; how bars were computed and under what assumptions; SD or SEM; 1σ or other ("preferably report a 2-sigma error bar than state that they have a 96% CI" when normality is unverified). No out-of-range bars. |
| ARR C3 ([checklist](https://aclrollingreview.org/responsibleNLPresearch/)) | State whether a number is a single run, a max over seeds, or a mean. Report validation results alongside test. |

**Run-level noise (cite when disclosing serving nondeterminism)**
- Batch size, GPU count and GPU type change outputs. A reasoning model varies by up to 9% in accuracy under bf16 greedy decoding ([2506.09501](https://arxiv.org/abs/2506.09501)).
- The cause is lack of batch invariance. Batch-invariant vLLM kernels make outputs identical at about 1.6–2× the time ([Thinking Machines, 2025-09-10](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).
- We measured about 20% agent-answer flips at `--eval_workers 5` (CLAUDE.md, "Serving noise"). Report this as the noise floor.

**Recommended protocol for our design**
- **Unit of analysis = distinct question.**
  - Average the 3 samples per question to a per-question accuracy. Each arm's score is the mean of these.
  - The CI is a nonparametric bootstrap over questions, stratified by benchmark. Equivalently, a CLT SE clustered by question.
  - For n ≤ ~100, also give exact or Beta intervals (Bowyer).
  - Wilson on 3n pooled draws is anti-conservative.
- **Comparisons are paired.**
  - Use the per-question difference of the per-question means, with a paired bootstrap CI.
  - Test with the sign-flip permutation test (`scripts/compare_runs.py`).
  - McNemar applies only to a single sample per question.
- **Folds** are for fitting learned baselines and selecting the CV best worker. Pool the out-of-fold predictions over all questions and test on the pool. Report per-fold numbers only as descriptive spread (Bengio & Grandvalet).
- **Multiple comparisons.** Pre-declare one primary comparison per pool, e.g. orchestrator against the 35B at matched tokens on the hard pool. Holm-correct within each family (baselines × benchmarks). Report CIs for everything else.
- **Factors of variability to name** (NeurIPS item 7):
  - question sampling (bootstrap);
  - generation sampling (3 samples at vendor temperature);
  - serving nondeterminism, not controlled: report the measured flip rate;
  - judge stochasticity on BrowseComp-Plus and Finance-Agent: grade 3 times and take the mode, as Vals does.

**Approximate minimum detectable paired difference.** Assumptions: 80% power, α = 0.05 two-sided, normal approximation, MDE ≈ 2.8·√(d/n) with d = discordant-pair rate, one sample per question. Averaging three samples shrinks this somewhat.

| n (set) | d = 10% | d = 20% | d = 30% |
|---|---|---|---|
| 50 (Finance-Agent) | 12.5 pp | 17.7 pp | 21.7 pp |
| 100 (BrowseComp-Plus / WorkBench / Plancraft) | 8.9 | 12.5 | 15.3 |
| 120 (AIME) | 8.1 | 11.4 | 14.0 |
| 198–200 (GPQA-D, MMLU-Pro subset) | 6.3 | 8.9 | 10.9 |
| 262 (MATH-500 L4–5) | 5.5 | 7.7 | 9.5 |
| 350 (agentic pool pooled) | 4.7 | 6.7 | 8.2 |
| 780 (hard pool pooled) | 3.2 | 4.5 | 5.5 |

Per-benchmark claims on the agentic sets can only detect differences of about 10 pp or more. Make the headline claims on the pooled sets, and treat per-benchmark results as exploratory, or run all 690 WorkBench and 580 Plancraft test tasks.

## 4. Benchmarks: official protocol, our deviations, reference numbers

**Hard pool**

| Benchmark | Official / common protocol | Ours | Contamination | Reference numbers | What a reviewer will ask for |
|---|---|---|---|---|---|
| **AIME** (120 = 2022–24 from `AI-MO/aimo-validation-aime`, plus 2025 from `math-ai/aime25`; `src/benchmarks/aime.py`) | Integer answers, avg@k pass@1. MathArena uses 4 samples per problem and reports the mean ([2505.23281](https://arxiv.org/abs/2505.23281)). Qwen evaluates with 81,920 max output tokens on competition maths ([card](https://huggingface.co/Qwen/Qwen3.5-4B)). | 3 samples, vendor sampling; output caps in X0/X1 below 81,920 | MathArena: strong signs of AIME 2024 contamination; HMMT shows less ([2505.23281](https://arxiv.org/abs/2505.23281)). Every 2022–2025 problem predates Qwen3.5 (large models Feb 2026, small 4B/0.8B 2 Mar 2026, [source](https://artificialanalysis.ai/articles/qwen3-5-small-models)†). | Qwen3.6-35B-A3B **AIME26 92.7**, HMMT Feb 26 83.6 (thinking) ([card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)). NVFP4 against bf16: AIME 2025 88.8 vs 89.2 ([NVIDIA card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)). | Results by year. A post-cutoff slice. The truncation (cut-off) rate per arm. Why thinking is off. |
| **MATH-500 L4–5** (loader: 262 problems; the brief says 264, so check) | Boxed answer, symbolic equivalence; the full 500 from PRM800K | Levels 4–5 only | Qwen2.5-Math-7B regenerates the remaining 40% of MATH-500 problems at 54.6% exact match from 60% prefixes ([2507.10532](https://arxiv.org/abs/2507.10532)). The test has not been run on Qwen3.5/3.6. | MATH-500 is near ceiling for strong models; Pass@1 seed std is about 0.3 for a 1.5B distill ([2504.07086](https://arxiv.org/abs/2504.07086)). | The subset rule. A robust equivalence checker (Sober Look). Ideally the partial-prompt contamination test on our 4B and 35B. |
| **GPQA-Diamond** (198) | 4-choice MCQ; options shuffled; zero-shot CoT† | Options shuffled once, seeded by record id (`gpqa_diamond.py`) | Gated dataset with a canary string†. Epoch estimates at least 90% of items are valid ([Epoch](https://epochai.substack.com/p/gpqa-diamond-whats-left)). | Qwen3.6-35B-A3B **86.0**, Qwen3.5-4B **76.2** (thinking; cards). NVFP4 84.8 vs bf16 84.9 (NVIDIA card). | Several option orders, or a position-bias check. Per-question CIs, since n = 198 is small. |
| **MMLU-Pro** (200 = 50 each of law, engineering, physics, chemistry) | 12k questions, 10 options, CoT. Leaderboard at [HF](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro). 5-shot CoT, with a random-guess fallback when extraction fails.† ([repo](https://github.com/TIGER-AI-Lab/MMLU-Pro)) | 4 categories picked as "the subjects models score lowest on", 0-shot† | Released 2024, within every model's training window | Qwen3.6-35B-A3B **85.2**, Qwen3.5-4B **79.1** (full set, thinking; cards). NVFP4 85.0 vs bf16 85.6 (NVIDIA card). | State that the categories were fixed before any results. Do not compare to the full-set numbers. Count how many answers failed to parse, by arm. |

**Agentic pool** (all four are the Kim et al. subsets, [2512.08296](https://arxiv.org/abs/2512.08296). Kim et al.'s single-agent means: Finance-Agent 0.349, WorkBench 0.629, BrowseComp-Plus 0.318, PlanCraft 0.568)

| Benchmark | Official protocol | Ours (`src/benchmarks/*.py`) | Reference numbers | What a reviewer will ask for |
|---|---|---|---|---|
| **BrowseComp-Plus** ([2508.06600](https://arxiv.org/abs/2508.06600), [repo](https://github.com/texttron/BrowseComp-Plus)) | 830 queries, fixed corpus of 100,195 docs. Retriever: BM25 (Pyserini) or Qwen3-Embedding-0.6B/4B/8B. Top k = 5, each result cut to its first 512 tokens. `get_document` only in an ablation. Paper judge: GPT-4.1; the repo's script uses Qwen3-32B. Metrics: accuracy, evidence recall, search calls, calibration error. | 100 of Kim et al.'s sampled queries (10 per category). `bm25s` BM25 with a Snowball stemmer, not Pyserini. k = 5 at about 2,000 chars rather than 512 tokens. `get_document` clipped to 12,000 chars. 30 steps. **Judge: the 35B.** | Paper Table 1, **BM25 / Qwen3-Embed-8B**: Qwen3-32B 3.49 / 10.36; gpt-oss-20B-high 21.08 / 34.58; gpt-oss-120B-high 28.67 / 42.89; Sonnet 4 14.34 / 36.75; Opus 4 15.54 / 36.14; o3 49.28 / 63.49; GPT-5 55.90 / 70.12 | Compare with the paper's **BM25** column only. Report evidence recall and search calls. Use the official judge prompt and an independent judge. List the retriever deviations. |
| **Finance-Agent** ([2508.00828](https://arxiv.org/abs/2508.00828), [Vals](https://www.vals.ai/benchmarks/finance_agent), [repo](https://github.com/vals-ai/finance-agent)) | 537 questions; leaderboard on the **private 337-question test set**. Tools: EDGAR via sec-api, Tavily web search, ParseHTML, RetrieveInformation (an LLM call), submit. v1.1 judge: GPT-5.2, mode of 3 gradings. Final-answer accuracy. | 50 public questions. EDGAR full-text search instead of sec-api. Wikipedia/Wikidata in place of Tavily. No RetrieveInformation LLM. Graded rubric line by line. Judge: the 35B. EDGAR capped at 2025-04-07, but Wikipedia results can postdate it. | Vals v1.1 (2026-06-04): Sonnet 4.6 **63.33%** ($1.44/test); GPT-5 52.15%; Qwen 3.5 Flash 45.64%; Qwen 3.6 Plus 54.63%; o3 46.8% in the original paper. **Not comparable** to our subset. | Exact-binomial or Bayesian CIs (n = 50). Judge agreement with human labels on a sample. Tool deviations. Note that the HTTP cache freezes the web state. |
| **WorkBench** ([2405.00823](https://arxiv.org/abs/2405.00823), [repo](https://github.com/olly-styles/WorkBench)) | 690 tasks, outcome-centric: compare sandbox state after the run. Also count side effects (wrong writes). 2026 "Revisited" study: 24 models on ground-truth v2. | Upstream tools, data and grader at a pinned commit. 100 of Kim et al.'s tasks on current (v2) wording. 20 steps. | Revisited, v2, all 690 ([model_results.json](https://github.com/olly-styles/WorkBench/blob/main/retro/data/model_results.json)): Sonnet 4.6 **609/690 = 88.3%** (61 side effects); GPT-5 84.3%; GPT-5.5 94.9%; Qwen3.5-Flash 70.7%; Haiku 4.5 74.8%. Original GPT-4 ReAct: 43%. | Report the **side-effect rate** as well as accuracy. Consider all 690 tasks, since n = 100 is underpowered. |
| **Plancraft** ([2412.21033](https://arxiv.org/abs/2412.21033)) | Splits: 1,145 train / 570 val / 580 test; small eval subsets of 110 val / 117 test with 20 impossible each. Metrics: success rate (impossible tasks excluded), Impossible-F1, Action Efficiency against the planner, tokens. 30 steps, temperature 0.6, 5 generations averaged. | First 100 test examples in file order (41 hard, 30 easy, 15 impossible, 14 medium). Success counts an impossible task when `impossible` is called. Text-only. 30 steps. | Paper Table 2, text, all tools: Llama-3.3-70B 0.65; Gemma-3-27B 0.48; gpt-4o-mini 0.20; Qwen3-30B-A3B 0.20. | Separate success and Impossible-F1 as the paper does. Keep in mind that file order is not a random sample. |

**General benchmark pitfalls**
- **Truncation.** Sober Look ties drops below 32k tokens to truncation. Report the cut-off rate per arm. A capped thinking-on baseline that is often cut off is a weak baseline ([2504.07086](https://arxiv.org/abs/2504.07086)).
- **Quantization.** The 35B is served as NVFP4. NVIDIA reports at most 0.6 pp loss on MMLU-Pro, GPQA and AIME 2025 ([card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)). Name the checkpoint and do not quote the bf16 card's numbers as ours.
- **Model-card numbers.** They are thinking-mode with 32k–81k max tokens and vendor sampling ([Qwen3.6 card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)). Our thinking-off, capped runs will sit below them; say why.

## 5. Reproducibility and checklist items

| Venue | Requirement relevant to us |
|---|---|
| **NeurIPS 2026** checklist ([guide](https://neurips.cc/public/guides/PaperChecklist)) | Missing checklist → desk reject; answering "No" with a justification is allowed. **Item 7**: error bars (see §3). **Item 8**: compute type, memory, time per run, total, and whether the project used more than reported (failed and preliminary runs). For us: GB10 DGX Spark ×2, 121 GB unified memory, vLLM image `nvcr.io/nvidia/vllm:26.05.post1-py3`, NVFP4, GPU-hours per arm. **Item 12**: licences and versions of the assets (GPQA gated, Finance-Agent, WorkBench commit, Plancraft 0.4.9). **Item 16**: declare LLM use when it is a core method component, which covers the orchestrator and the LLM judge. |
| **ICLR 2026** ([author guide](https://www.iclr.cc/Conferences/2026/AuthorGuide)) | A reproducibility statement is "strongly encouraged". It is a paragraph pointing to the relevant sections and does not count toward the page limit. Significant LLM use in ideation or writing goes in a separate section, and not disclosing it "can lead to desk rejection". |
| **ARR** (ACL/EMNLP; [checklist](https://aclrollingreview.org/responsibleNLPresearch/)) | **B1/B2**: cite and license artifacts. **B6**: split statistics, even for standard benchmarks, which matters because ours are subsets. **C1**: parameters and total compute, including all experiments. **C2**: hyperparameters, including the prompt search. **C3**: single run, max or mean. **C4**: package versions, including vLLM. **E1**: use of AI assistants. A "No" needs a justification. |
| **LLM-judge disclosure** (norm, from the papers above) | Name the judge model and version, give the prompt verbatim, say how many gradings and how they are aggregated (Vals takes the mode of 3), and report agreement with humans (Kim et al. report κ). Avoid a judge from the same family as a system under test, or quantify self-preference ([2404.13076](https://arxiv.org/abs/2404.13076)). |
| **Release** (Sober Look, [2504.07086](https://arxiv.org/abs/2504.07086)) | Code, prompts, every model output and a container. For us: the answer matrix, the orchestrator transcripts, the HTTP cache for Finance-Agent, and the BM25 index build script. |
