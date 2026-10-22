# Plan: X1 + X2 as one fixed centralized team

Started in a design session on 2026-10-22, while X0 runs. X1 (prompts) and X2 (models and
specialists) from [§10 of the v2 plan](orchestration-v2-plan.md#10-experiments-on-the-harder-benchmarks)
become one experiment: a **fixed team** of personas and **specialists** whose **hub** reads their
answers and gives the final one. Terms in **bold** are defined in [`CONTEXT.md`](../CONTEXT.md). How
each specialist was trained is in [`notes/specialists.md`](notes/specialists.md) and
[`notes/base-models.md`](notes/base-models.md). Issues: #16 (personas), #15 (specialist gates),
#14 (serving), #10 (agent registry), #32 (new base families).

Status: inventory written; Q6–Q26 settled 2026-10-22; design still being settled (§6).

---

## 1. Settled so far

- **A fixed team, centralized topology.** Workers answer alone; the hub reads all their answers and
  gives the team's answer (`team_config.centralized`). Nothing is chosen per task; that is X3.
- **One run gives everything.** A worker that reads nothing *is* answering solo, so the same run
  yields each persona's and specialist's solo answer (X1's and X2's comparisons against the neutral
  base), the majority vote over workers (by **replay**, if needed), and the hub's answer. No separate
  gate runs first.
- **Questions:** the hard pool (780) and the agentic pool (350).
- **Where:** develop on a branch off `main` on sparkie12, push it, and pull it on the second Spark
  (spark-6695), which runs the experiment. sparkie12 is busy with X0 until about 23 Oct.
- **On agentic tasks** each worker runs its own episode in a fresh environment, and the hub starts a
  fresh episode with the workers' attempts as advice (`runner.py:140-146`). The hub sees each
  worker's final message and final answer or outcome, not its full transcript (Q10).
- **One fixed team per source** (Q6): each source (and MMLU-Pro / GPQA subject) gets its own team of
  matching experts and specialists, fixed before the run.
- **Hub** (Q7): Qwen3.6-35B, thinking off, neutral prompt. A 4B (thinking off) hub is a later
  ablation.
- **Persona workers** (Q8) run on Qwen3.5-4B, thinking off. Each specialist runs in the thinking mode
  it was trained for.
- **Settings** (Q9) match X0: 30,720-token answers, vendor sampling, X0's 5 folds and question keys,
  so every worker's solo answer is a new row of the X0 answer matrix. k = 3 samples per worker; the
  hub runs once per sample.
- **Record everything** (Q10, Q11): every worker's answer, correctness and token counts (thinking
  included) are kept, as are the hub's, so solo performance is reported per worker. Results are
  reported raw first: per-worker, majority vote, hub. Comparisons (hub vs best single worker per
  source vs bare 35B, against cost) come afterwards.
- **Team composition** (Q13, Q20, Q22, Q23): **4 workers per team**: 2 expert personas and 2
  specialists. Where fewer than 2 specialists fit a source, 3 expert personas and 1 specialist.
  No neutral worker, no commit-to-an-answer prompt, no shortlisted bank personas. The neutral
  baselines come from X0 (4B thinking off, samples s0–s2; 35B thinking off).
- **No matching specialist** (Q25): the slot goes to the best general-purpose or adjacent-domain
  specialist on a served base, labelled "generalist" in results (law, GPQA biology, Plancraft;
  engineering if needed). Candidates: `notes/model-cards.md` Part C.
- **Teams per subject, shared across sources** (Q26): 10 teams: maths (AIME + MATH-500), physics
  (GPQA + MMLU-Pro), chemistry (GPQA + MMLU-Pro), biology (GPQA), law, engineering (MMLU-Pro), and
  BrowseComp-Plus, Finance-Agent, WorkBench, Plancraft. Results are still reported per source.
- **Agentic neutral baseline** (Q24): X0 will add a neutral 4B run on the agentic pool; that is
  left to the sessions running X0, not built here.
- **Thinking budget** (Q21): `thinking_token_budget` 26,624 inside X0's 30,720-token answer cap,
  leaving 4,096 for the answer. Record per worker how often the budget is exhausted (vLLM forced the
  end of thinking) and how often the answer is still cut off, in case the budget must rise.
- **Summary instruction** (Q22): every worker's prompt gets, before the format line: *Before your
  final answer, write "Summary:" followed by at most five sentences giving the key steps of your
  reasoning.* The hub gets that block plus the final answer. When a worker leaves the block out, the
  hub gets the last 600 characters instead; report this **fallback rate** (the share of a worker's
  answers without a usable summary) per worker.
- **Specialists** (Q14, Q18): chosen by the user from the reviewed list (`notes/model-cards.md`),
  for the hard pool and for tool use on the agentic pool.
- **Specialists' other bases** (Q15): Gemma-E4B and Ministral-Instruct solo rows are deferred until
  no experiment is running.
- **Thinking** (Q16): cap thinking per request in vLLM (`thinking_token_budget`, which needs
  `--reasoning-config` on the server) wherever a model supports it; where it cannot, give that model
  a larger answer cap instead.
- **Handoff** (Q17): the hub reads each worker's summary plus its answer (Q22), not the last 600
  characters.
- **Staged run** (Q19): stage 1 serves one model at a time and collects every worker's answers;
  stage 2 serves only the hub and runs it over the saved answers.
- **Expert prompts** (Q12) are new, one or two per source or subject, written the same way and in
  the same style as the bank's personas (a "You are …" first line, then "Your approach:" bullets).

## 2. The datasets

Hard pool, 780 questions (`data-claude/tagged_hard`):

| Source | n | Subjects |
|---|---|---|
| GPQA-Diamond | 198 | Chemistry 93 (organic 72), physics 86 (quantum 25, high-energy 14, astro 13, …), biology 19 |
| MMLU-Pro hard | 200 | Law 50, engineering 50, physics 50, chemistry 50 |
| AIME 2022–25 | 120 | 30 per year |
| MATH-500 L4–5 | 262 | Algebra 60, intermediate algebra 59, prealgebra 39, number theory 31, counting & probability 25, precalculus 25, geometry 23 |

Agentic pool, 350 tasks (`data-claude/tagged_agentic`):

| Source | n | Breakdown |
|---|---|---|
| Plancraft | 100 | Easy 30, medium 14, hard 41, impossible 15 |
| WorkBench | 100 | Analytics, calendar, CRM, email 17 each; multi-domain, project management 16 each |
| BrowseComp-Plus | 100 | 10 categories × 10 |
| Finance-Agent | 50 | 9 question types (numerical reasoning 8, retrieval 18, beat-or-miss 7, …) |

There is **no medicine and no finance QA** in either pool. Medicine exists only in the easy set's
`pro_medicine`; finance only as the agentic Finance-Agent.

## 3. Personas

### 3.1 What exists

All 50 personas are in `chosen_persona_bank()` (`src/personas.py:39-616`). Each was written for one
easy-set source and runs at temperature 0, which v2's sampling rule (Q40) no longer allows.

| Set | Personas | Written for |
|---|---|---|
| GSM8K | Conservative_Verifier, Creative_Explorer, Rigorous_Formalist, Intuitive_Estimator, Systematic_Decomposer | Grade-school maths |
| Medicine | Cautious_Diagnostician, Evidence_Based_Analyst, Holistic_Clinician, Pattern_Recognition_Expert, Systematic_Reviewer | MMLU professional medicine |
| Formal logic | Logical_Formalist, Commonsense_Reasoner, Critical_Analyzer, Eliminative_Reasoner, Analogical_Thinker | MMLU formal logic |
| Coding | Defensive_Coder, Elegant_Minimalist (broken: 0% at 4B), Algorithm_Optimizer, Test_Driven_Developer, Creative_Problem_Solver | HumanEval, MBPP |
| TruthfulQA | Decomposition_Planner, Causal_Mechanism_Reasoner, Time_Place_Category_Checker, Scope_Quantifier_Reader, Contradiction_Sanity_Checker | TruthfulQA |
| Truth-checking extras | Consensus_Fact_Checker, Imitative_Falsehood_Spotter, Rhetoric_and_Absolutes_Filter, Causality_and_Mechanism_Auditor, Core_Claim_MinAssumption | (no paper set) |
| ARC | Science_Exam_Solver, Concept_to_Option_Matcher, Elimination_Specialist, Diagramless_Physicist, Careful_Reader | Grade-school science |
| PIQA | Action_Sequence_Simulator, Mechanics_Stability_Analyst, Material_Interaction_Reasoner, Human_Factors_Controller, Robustness_Comparator | Physical commonsense |
| Winogrande | Coreference_Formalist, Scenario_Simulator, Contrastive_Tester, Bias_Resistance_Checker, Elimination_Based_Solver | Pronoun resolution |
| Default | Careful_Analyst, Broad_Thinker, Practical_Reasoner, Detail_Oriented, Intuitive_Judge | Fallback, HellaSwag |

On the easy set, personas on one model differed only by run-to-run noise (E2, team-of-one; ADR-0001).

### 3.2 Per source: what exists and what is missing

Every hard and agentic module has `PERSONA_SET = []`. The only existing link is the "proxy" mapping
(`proxy_persona_source`, used by `paper_persona_baseline.py --proxy`).

| Source / subject | Existing proxy personas | Source-matched expert (to write) |
|---|---|---|
| GPQA chemistry | none | Organic / general chemist |
| GPQA physics | none | Physicist (quantum, high-energy, astro) |
| GPQA biology | none | Molecular biologist / geneticist |
| MMLU-Pro law | Formal logic set | Lawyer (US professional law) |
| MMLU-Pro engineering | none | Electrical / mechanical engineer |
| MMLU-Pro physics | none | Physicist |
| MMLU-Pro chemistry | none | Chemist |
| AIME | GSM8K set | Competition mathematician |
| MATH-500 L4–5 | GSM8K set | Competition mathematician |
| Plancraft | none | (task prompt already says "You are crafting in Minecraft…") |
| WorkBench | none | Office / workplace assistant |
| BrowseComp-Plus | none | Web research analyst |
| Finance-Agent | none | Financial analyst (task prompt already says "You are a financial agent…") |

### 3.3 The neutral prompt and other planned prompts

- **Neutral prompt** today, runner path: system "You are a helpful assistant." (`runner.py:39`),
  user = question + the scorer's format suffix (`roles.py:70-74`). Agentic: the same system message,
  user = `env.task_prompt()`.
- **Commit-to-an-answer** prompt: planned in the original X1, dropped (Q20–Q23); the thinking
  budget handles cut-offs instead.
- The hub's role text is `ROLES["hub"]` (`roles.py`), appended after its persona, if any.

## 4. Specialists

### 4.1 Per source

| Source / subject | Specialists matching it | Base it must beat |
|---|---|---|
| AIME, MATH-500 | RL-MATH; Ministral-3B-Reasoning; R1-Distill-Llama-8B | Qwen3.5-4B; Ministral-3B-Instruct; Llama-3.1-8B |
| GPQA, MMLU-Pro physics/chemistry | ShiningValiant3; RL-MATH (claims GPQA); Ministral-3B-Reasoning (GPQA); BigBang-v1 (science post-train, no GPQA number) | Gemma-4-E4B; Qwen3.5-4B; Ministral-3B-Instruct; Qwen3.6-35B |
| MMLU-Pro engineering | none | — |
| MMLU-Pro law | Saul-7B (excluded: Mistral-7B v0.1 host) | — |
| GPQA biology | none (medical fine-tunes are the nearest) | — |
| BrowseComp-Plus, Finance-Agent, WorkBench | BigBang-v1 (BrowseComp 68→77) | Qwen3.6-35B |
| Finance-Agent | Fino1-8B, Fin-R1 (finance QA; tool calling unknown) | Llama-3.1-8B; Qwen2.5-7B |
| Plancraft | none | — |
| (no matching source) | Med-Qwen, HuatuoGPT-o1-8B/-7B | — |

### 4.2 Every candidate

**The full list for review is [`notes/model-cards.md`](notes/model-cards.md)**: a per-dataset index,
then Part A (domain fine-tunes, below) and Part B (28 tool-use and agentic fine-tunes for the
agentic pool, found 2026-10-22). Their Hugging Face cards, condensed, are in [`notes/model-cards.md`](notes/model-cards.md). None is downloaded or served on either box, and no smoke test or gate has run.

| Specialist (HF repo) | Base | Training | Published claims | Loads on vLLM 0.21 | Serve plan / problems |
|---|---|---|---|---|---|
| `wflying/Qwen3.5-4B-RL-MATH` | Qwen3.5-4B | GRPO RLVR, thinking off, 15k maths prompts | AIME25 38→59, GPQA-D 71→77 (unreproduced) | Yes | 4B script with handle swapped; MTP may degrade; long outputs |
| `mistralai/Ministral-3-3B-Reasoning-2512` | Ministral-3-3B-Base (sibling of Instruct) | Distil from Magistral, GRPO, ODPO; always thinks | AIME25 72.1, GPQA-D 53.4 | Probably | New script with `--reasoning-parser mistral`; `[THINK]` system prompt; trained to 80K reasoning |
| `ValiantLabs/gemma-4-E4B-it-ShiningValiant3` | Gemma-4-E4B-it | SFT on R1-0528 science traces | none | Probably | Gemma script with handle swapped; no MTP |
| `endless-frontier/BigBang-v1` (NVFP4: `TheShed/BigBang-v1-W4A16-NVFP4-MTP`) | Qwen3.6-35B-A3B | Post-train on ~10k synthetic frontier problems (science, coding, tool use) | BrowseComp 68→77; no MATH/GPQA numbers | Expected | NVFP4 script needs BF16 KV; ~23 GB; ≥128K context |
| `adi274903/Med-Qwen-3.5-4B-Instruct` | Qwen3.5-4B-**Base** | Biomed CPT + DoRA SFT | Weak, harness looks broken | **No** (text-only) | Local wrapper; `stop_token_ids` not in client |
| `FreedomIntelligence/HuatuoGPT-o1-8B` | Llama-3.1-8B-Instruct | Verifier-searched CoT SFT + PPO | MedQA 72.6 vs 58.7 | Yes | Needs a Llama host script (#32) |
| `FreedomIntelligence/HuatuoGPT-o1-7B` | Qwen2.5-7B-Instruct | Same recipe | — | Yes | Needs a Qwen2.5 script |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Llama-3.1-8B **base** | SFT on ~800k R1 samples; always thinks | MATH-500 89.1, AIME24 50.4 | Yes | Llama script; lineage exception |
| `TheFinAI/Fino1-8B` (gated, auto-approve) | Llama-3.1-8B-Instruct | SFT + RL on GPT-4o finance paths | FinQA, DocMath, TATQA | Yes | Llama script |
| `SUFE-AIFLM-Lab/Fin-R1` | Qwen2.5-7B-Instruct | SFT + GRPO on ~60k finance CoT | FinQA 76 vs 60 | Yes | Qwen2.5 script |
| `Equall/Saul-7B-Instruct-v1` | Mistral-7B v0.1 | Legal CPT + instruction tuning | LegalBench-Instruct | Yes | Excluded: weak base, law only |

Longer-tail candidates (Nemotron-Nano-8B/4B, Meditron3, Aloe, Med42, Qwen3.5-4B-Neo,
MathThink-Qwen-3.5-4B, occamy and other 35B agentic fine-tunes) are listed in the notes.

### 4.3 Base models needed as hosts

| Base | Cached (both boxes) | Serve script | Hosts |
|---|---|---|---|
| Qwen3.5-4B | Yes | `vllm/qwen3.5-4b/` | RL-MATH, personas, neutral |
| Gemma-4-E4B | Yes | `vllm/gemma4-e4b/` | ShiningValiant3 |
| Ministral-3-3B-Instruct | Yes | `vllm/ministral3-3b/` | (Reasoning is its sibling) |
| Qwen3.6-35B-A3B NVFP4 | Yes | `vllm/qwen3.6/run_docker_nvfp4.sh` | Hub; BigBang |
| Llama-3.1-8B-Instruct | No | none | HuatuoGPT-o1-8B, Fino1, R1-Distill |
| Qwen2.5-7B-Instruct | No | none | Fin-R1, HuatuoGPT-o1-7B |

The second Spark also has the `nvcr.io/nvidia/vllm:26.06-py3` image; its vLLM version is unchecked
and may matter for text-only checkpoints (#30).

### 4.4 Teams to run (for review)

Ten fixed teams (Q26), 4 workers each (Q23). Expert personas run on Qwen3.5-4B, thinking off; the
hub is Qwen3.6-35B, thinking off. Candidates are fine-tunes of a served base, from
`notes/model-cards.md`; the user picks the specialists. Sizes: NVFP4 / FP8 builds where they exist,
otherwise BF16 (35B fine-tunes ~67 GB BF16, served one at a time in stage 1).

| # | Team | Sources (n) | Shape | Expert personas (to write) | Specialist candidates (served base) | Picks |
|---|---|---|---|---|---|---|
| T1 | Maths | AIME 120, MATH-500 L4–5 262 | 2 + 2 | Competition mathematician; olympiad problem-solver | RL-MATH (4B); Ministral-3B-Reasoning; AgentMercury-4B (AIME claim, cred 1) | |
| T2 | Physics | GPQA 86, MMLU-Pro 50 | 2 + 2 | Theoretical physicist; experimental / applied physicist | ShiningValiant3 (E4B); RL-MATH (claims GPQA); Ministral-3B-Reasoning; BigBang (35B, NVFP4) | |
| T3 | Chemistry | GPQA 93, MMLU-Pro 50 | 2 + 2 | Organic chemist; physical / analytical chemist | ShiningValiant3; RL-MATH; Ministral-3B-Reasoning; BigBang | |
| T4 | Biology | GPQA 19 | 2 + 2 (adjacent) | Molecular biologist; geneticist | No exact biology tune. ShiningValiant3 (science traces incl. biology); BigBang (BioMystery-HS 45→58); Reasoning-Medical0.1-E4B (cred 1); Qwen3.5-4B-Neo (cred 1–2) | |
| T5 | Law | MMLU-Pro 50 | 3 + 1 | US lawyer / bar-exam expert; legal analyst (rules → facts); judge / jurisprudence | No credible law tune; generalist slot: BigBang (top); 0GM-1.0-35B-A3B (law flat, BF16 only); E4B legal-QA LoRA (cred 1) | |
| T6 | Engineering | MMLU-Pro 50 | 2 + 2 (adjacent) | Electrical engineer; engineering problem-solver (units, formulas) | No exact engineering tune. 0GM-1.0-35B-A3B (engineering 42.0→46.4 at 4k context, BF16 only); RL-MATH; DeepSeek-V4-Pro-Qwen3.5-4B (cred 1–2); Ministral-3B-Reasoning | |
| T7 | BrowseComp-Plus | 100 | 2 + 2 | Web research analyst; fact-checking investigator | Iris-mini (35B, BF16); XYZ-Aquila-mini (35B, community NVFP4); BigBang; T-Search (retriever only); Agents-A1-4B (FP8); AREX-Turbo (4B, custom loop) | |
| T8 | Finance-Agent | 50 | 2 + 2 | Financial analyst (SEC filings); equity research analyst | XYZ-Aquila-mini; Occamy-1.0 (35B, NVFP4); Fx-Work (35B, BF16) | |
| T9 | WorkBench | 100 | 2 + 2 | Office / executive assistant; operations analyst (CRM, analytics) | CompoWorld (35B, BF16); Occamy-1.0; Skill2Env (35B, BF16); CloudSurf-4B-FC (E4B, needs bridge); AgentMercury-4B (cred 1) | |
| T10 | Plancraft | 100 | 3 + 1 | Minecraft crafting planner; inventory / recipe checker; feasibility checker (when impossible) | No crafting tune; adjacent slot: R2S ALFWorld student (4B, cred 2, top); SkillGym-4B (cred 2); CompoWorld (35B); TextCraft LoRAs (cred 1) | |

## 5. Gaps

- No persona is written for any hard or agentic source.
- No specialist for MMLU-Pro engineering, GPQA biology or Plancraft; law only via the excluded Saul.
- Med-Qwen and the HuatuoGPT models match no source in either pool.
- The smoke-test spec in `notes/specialists.md` still says temperature 0 and a 4,096-token cap,
  which conflicts with Q40 and X0's 30,720-token answers.
- `src` has no per-agent thinking control beyond `chat_template_kwargs`, no `reasoning_effort`, no
  `stop_token_ids`, and no agent registry with specialists in it (#10, #11).

## 6. Later: ablations and deferred work

Everything flagged during the design session as "later" or "ablation". Nothing here blocks the
first run.

| # | Item | From | When |
|---|---|---|---|
| L1 | **4B hub ablation**: rerun stage 2 with Qwen3.5-4B (thinking off) as the hub over the same saved worker answers | Q7 | After the first run; stage 1 answers are reused |
| L2 | **Comparisons**: hub vs best single worker per source vs bare 35B vs majority vote (replay), on accuracy against cost | Q11 | After the raw results are reported |
| L3 | **Solo rows for specialists' other bases**: Gemma-4-E4B (for ShiningValiant3) and Ministral-3-3B-Instruct (for Ministral-R) on the same questions and settings, so each specialist has its paired base | Q15 | When no experiment is running |
| L4 | **Fuller handoff ablation**: hub served at 131k context, reading each worker's last N tokens (or full answer) instead of its summary | Q17 | After the first run; reuses stage 1 answers |
| L5 | **Second wave of specialists** needing new hosts (Llama-3.1-8B, Qwen2.5-7B: R1-Distill-Llama-8B, Fino1-8B, Fin-R1, …), if the user picks any | Q14 | After the Llama / Qwen2.5 serve scripts exist (#32) |
| L6 | **Text-only checkpoints** (Med-Qwen and other `Qwen3_5ForCausalLM` fine-tunes): local wrapper or vLLM ≥ 0.27; check the `26.06-py3` image on the second Spark | §4 | Only if one is picked (#30) |
| L7 | **Update the specialist smoke-test spec** in `notes/specialists.md` (still says temperature 0 and a 4,096-token cap) to vendor sampling and X0's caps | §5 | Before any specialist is smoke-tested |
| L8 | **Re-sync the second Spark** (`data-claude/`, env) from sparkie12 before the run, and copy results back after | Q4 | Before stage 1 / after each stage |

## 7. To build before the first run

- Expert prompts for every source and subject, in the bank's style (Q12), written into this plan
  for review before coding.
- A worker instruction to end with a short summary of its reasoning, and a `summary` handoff that
  passes it to the hub (Q17).
- Per-agent thinking budget: `THINKING_TOKEN_BUDGET` is one global environment variable today
  (`model/openai_compat.py:17-23`), and vLLM rejects the field on servers without
  `--reasoning-config` (#11).
- Stage 2 runner: feed saved worker answers to the hub without rerunning the workers (Q19).
- Serve scripts and smoke tests for the chosen specialists on the second Spark (#14), including
  whether `--reasoning-config` can cap Gemma-4's and Ministral's thinking (Q16).
- Per-source team configs of equal size (Q13).

## 8. Open decisions

- Specialist picks for the hard-pool teams (Q14) and agentic teams (Q18), from `notes/model-cards.md`.
