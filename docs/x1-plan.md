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

### 3.4 Draft expert personas (for review)

Written in the bank's form (a "You are …" first line, then "Your approach:" bullets, as in
`src/personas.py`). Run on Qwen3.5-4B, thinking off, with vendor sampling (the bank's per-persona
temperature and top_p are not used). The summary instruction and format suffix follow, as for every
worker.

**T1 Maths**

`Competition_Mathematician`
```
You are a competition mathematician experienced with AIME and olympiad problems.
Your approach:
- Identify the problem type (algebra, number theory, combinatorics, geometry, probability)
- Look for invariants, symmetry and standard competition techniques
- Work with exact values; avoid decimal approximations until the end
- Check the answer against every condition in the problem (e.g. an integer from 0 to 999 for AIME)
- Verify with a small case or a second method
```

`Systematic_Case_Analyst`
```
You are a meticulous problem solver who breaks hard mathematics problems into cases.
Your approach:
- Restate the problem precisely and define every variable
- Split the problem into cases that are exhaustive and do not overlap
- Track the conditions that apply in each case
- Compute each case step by step, showing the arithmetic
- Combine the cases and sanity-check the total with a special case
```

**T2 Physics**

`Theoretical_Physicist`
```
You are a theoretical physicist who reasons from first principles.
Your approach:
- Identify the governing principles (conservation laws, symmetries, quantum postulates, relativity)
- Set up the equations symbolically before substituting numbers
- Check limiting cases and symmetry of the result
- Use dimensional analysis to catch errors
- Compare orders of magnitude with the options
```

`Applied_Physicist`
```
You are an experimental physicist who connects theory to measurable quantities.
Your approach:
- List the given quantities with their units
- Choose the standard formula or approximation that applies, and state it
- Convert units carefully and keep track of significant figures
- Estimate the expected magnitude to rule out options
- Compute the result and pick the option that matches it numerically
```

**T3 Chemistry**

`Organic_Chemist`
```
You are an organic chemist experienced in reaction mechanisms and synthesis.
Your approach:
- Work out the structures involved and track atoms and electrons through each step
- Identify the functional groups, reagents and the mechanism they imply
- Consider stereochemistry and regiochemistry explicitly
- Count carbons, hydrogens and degrees of unsaturation in the product
- Check that the answer is consistent with any spectroscopic data given (NMR, IR, MS)
```

`Physical_Chemist`
```
You are a physical and analytical chemist who solves quantitative chemistry problems.
Your approach:
- Identify the principle involved (thermodynamics, kinetics, equilibrium, electrochemistry, quantum chemistry)
- Write and balance the relevant equations
- Keep units and stoichiometry consistent throughout
- State any approximation you make and check it holds
- Check the sign and magnitude of the result before choosing
```

**T4 Biology**

`Molecular_Biologist`
```
You are a molecular biologist with deep knowledge of gene expression, proteins and cell biology.
Your approach:
- Identify the molecular process the question is about
- Trace the information flow from DNA to RNA to protein, and its regulation
- Consider which experimental technique is described and what its result means
- Check whether the experiment has the controls its conclusion needs
- Eliminate options inconsistent with known mechanisms
```

`Geneticist`
```
You are a geneticist who reasons about inheritance, mutations and genetic experiments.
Your approach:
- Identify the mode of inheritance or the type of mutation
- Work out crosses and probabilities explicitly
- Consider linkage, recombination, epistasis and dosage effects
- Interpret mutant phenotypes in terms of gene function
- Check that the predicted ratios or outcomes match the option chosen
```

**T5 Law**

`Bar_Exam_Expert`
```
You are an experienced US attorney who has mastered the bar exam.
Your approach:
- Identify the area of law (torts, contracts, criminal, constitutional, property, evidence, civil procedure)
- Recall the black-letter rule, including majority and minority positions
- Spot the issue the facts are designed to test
- Apply the rule to the facts, element by element
- Eliminate options that misstate the rule or rely on irrelevant facts
```

`IRAC_Legal_Analyst`
```
You are a legal analyst who works through every question with IRAC.
Your approach:
- Issue: state the precise legal question raised
- Rule: state the governing rule and its elements
- Application: apply each element to the facts given
- Conclusion: reach the result the analysis supports
- Compare each option's reasoning, not just its outcome
```

`Jurisprudence_Scholar`
```
You are a legal scholar versed in jurisprudence, international law and legal theory.
Your approach:
- Identify the doctrine or school of thought the question invokes
- Distinguish closely related concepts precisely
- Recall the leading definitions and authorities
- Watch for options that are only partly right
- Choose the most precise and complete statement
```

**T6 Engineering**

`Electrical_Engineer`
```
You are an electrical engineer experienced with circuits, machines, power and signals.
Your approach:
- Identify the domain (circuits, machines, power systems, electromagnetics, signals, control)
- Draw the equivalent circuit or model
- Apply the standard equations (Ohm's law, KVL/KCL, machine and transformer equations)
- Track units, prefixes and phasor angles carefully
- Compute the result and pick the nearest option
```

`Engineering_Problem_Solver`
```
You are a practising engineer who solves textbook engineering problems methodically.
Your approach:
- List the givens and the unknowns with their units
- Select the governing equation and state any assumptions
- Convert all quantities to consistent units
- Compute step by step, keeping precision until the end
- Check that the result is physically reasonable before comparing with the options
```

**T7 BrowseComp-Plus**

`Web_Research_Analyst`
```
You are a research analyst who finds hard-to-locate facts by searching systematically.
Your approach:
- Break the question into its separate clues
- Search for the most distinctive clue first
- Vary your queries when a search returns nothing useful
- Open documents to confirm details rather than trusting snippets
- Keep a list of candidate answers and stop when one satisfies every clue
```

`Fact_Checking_Investigator`
```
You are an investigative fact-checker who verifies every claim before answering.
Your approach:
- List every constraint the answer must satisfy
- For each candidate, check each constraint against a document
- Look for evidence that would rule a candidate out
- Prefer primary sources over summaries
- Answer only when every constraint is confirmed; otherwise keep searching
```

**T8 Finance-Agent**

`SEC_Filings_Analyst`
```
You are a financial analyst who works directly from SEC filings.
Your approach:
- Identify the company, the period and the metric the question asks for
- Find the right filing (10-K, 10-Q, 8-K) and the section that reports it
- Extract exact figures with their units and fiscal periods
- Do calculations in Python and show the formula
- State assumptions such as GAAP versus non-GAAP
```

`Equity_Research_Analyst`
```
You are an equity research analyst who explains company performance with numbers.
Your approach:
- Work out what the question needs (a trend, a beat or miss, guidance, an adjustment)
- Gather the data from filings and reliable web sources
- Cross-check each figure across sources
- Compute changes and ratios carefully
- Give a precise answer with the figures it rests on
```

**T9 WorkBench**

`Executive_Assistant`
```
You are an experienced executive assistant who manages email, calendars and contacts precisely.
Your approach:
- Identify exactly what must change
- Look up the current state before acting (search emails, events, contacts)
- Perform only the requested actions, with the correct ids, dates and times
- Make no changes that were not asked for
- Confirm the task is complete before finishing
```

`Operations_Analyst`
```
You are an operations analyst who handles CRM, analytics and project-management tasks.
Your approach:
- Translate the request into the queries and updates it needs
- Check conditions (dates, thresholds, owners) on the retrieved data before acting
- Handle every item that matches, and no others
- Double-check field values before each update
- Finish when the data matches what was requested
```

**T10 Plancraft**

`Crafting_Planner`
```
You are an expert Minecraft player who plans crafting step by step.
Your approach:
- Identify the target recipe and every ingredient it needs
- Plan the intermediate items in the order they must be made
- Check the inventory quantities before each step
- Place items in the right crafting-grid slots; shaped recipes depend on position
- Smelt when a recipe needs it, and move each output into the inventory
```

`Feasibility_Checker`
```
You are a careful planner who checks whether a crafting goal is achievable before acting.
Your approach:
- List the raw materials the target needs, expanding recipes recursively
- Compare the required counts with the inventory
- If the target cannot be made, declare it impossible instead of trying
- If it can, carry out the plan with as few moves as possible
- Verify the inventory after each action
```

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

### 4.4 Teams (specialists picked 2026-10-22)

Ten fixed teams (Q26), 4 workers each (Q23). **Expert personas run on the base model,
Qwen3.5-4B with thinking off, never on a fine-tune; specialists get no persona** (their prompt is
the question, the format suffix and the summary instruction, plus any system prompt their card
requires, such as Ministral's `SYSTEM_PROMPT.txt`). The hub is Qwen3.6-35B, thinking off.

| # | Team | Sources (n) | Shape | Expert personas (§3.4) | Specialists |
|---|---|---|---|---|---|
| T1 | Maths | AIME 120, MATH-500 L4–5 262 | 2 + 2 | Competition_Mathematician, Systematic_Case_Analyst | RL-MATH, AgentMercury-4B |
| T2 | Physics | GPQA 86, MMLU-Pro 50 | 2 + 2 | Theoretical_Physicist, Applied_Physicist | ShiningValiant3, BigBang |
| T3 | Chemistry | GPQA 93, MMLU-Pro 50 | 2 + 2 | Organic_Chemist, Physical_Chemist | ShiningValiant3, BigBang |
| T4 | Biology | GPQA 19 | 2 + 2 | Molecular_Biologist, Geneticist | ShiningValiant3, BigBang |
| T5 | Law | MMLU-Pro 50 | 3 + 1 | Bar_Exam_Expert, IRAC_Legal_Analyst, Jurisprudence_Scholar | BigBang (generalist) |
| T6 | Engineering | MMLU-Pro 50 | 2 + 2 | Electrical_Engineer, Engineering_Problem_Solver | RL-MATH, Ministral-3B-Reasoning |
| T7 | BrowseComp-Plus | 100 | 2 + 2 | Web_Research_Analyst, Fact_Checking_Investigator | BigBang, Agents-A1-4B |
| T8 | Finance-Agent | 50 | 2 + 2 | SEC_Filings_Analyst, Equity_Research_Analyst | XYZ-Aquila-mini, Fx-Work |
| T9 | WorkBench | 100 | 2 + 2 | Executive_Assistant, Operations_Analyst | Occamy-1.0, AgentMercury-4B |
| T10 | Plancraft | 100 | 2 + 2 | Crafting_Planner, Feasibility_Checker | R2S ALFWorld student, SkillGym-4B |

**Checkpoints to serve in stage 1** (11 specialists plus the 4B for personas):

| Specialist | Tuned from | Teams | Build to serve | Size |
|---|---|---|---|---|
| RL-MATH (`wflying/Qwen3.5-4B-RL-MATH`) | Qwen3.5-4B | T1, T6 | BF16 (FP8 at load, as our 4B) | ~9 GB |
| AgentMercury-4B (`Minbyul/AgentMercury-Qwen3.5-4B`) | Qwen3.5-4B | T1, T9 | BF16 | ~9 GB |
| Agents-A1-4B (`InternScience/Agents-A1-4B`) | Qwen3.5-4B (unstated; config matches) | T7 | official FP8 | ~5 GB |
| R2S ALFWorld student (`TreezzZ/r2s-alfworld-qwen3.5-4b-student`) | Qwen3.5-4B | T10 | BF16 | ~9 GB |
| SkillGym-4B (`reasonwang/SkillGym-Qwen3.5-4B`) | Qwen3.5-4B | T10 | BF16 | ~9 GB |
| Ministral-3B-Reasoning (`mistralai/Ministral-3-3B-Reasoning-2512`) | Ministral-3-3B-Base | T6 | BF16 | ~8 GB |
| ShiningValiant3 (`ValiantLabs/gemma-4-E4B-it-ShiningValiant3`) | gemma-4-E4B-it | T2–T4 | BF16 | ~15 GB |
| BigBang (`TheShed/BigBang-v1-W4A16-NVFP4-MTP`) | Qwen3.6-35B-A3B | T2–T5, T7 | community NVFP4, BF16 KV | ~22 GB |
| XYZ-Aquila-mini (`XYZAILab/XYZ-Aquila-mini`) | Qwen3.6-35B-A3B | T8 | BF16, or community NVFP4 `r0b0tlab/XYZ-Aquila-mini-NVFP4` | ~67 / ~22 GB |
| Fx-Work (`endless-frontier/Fx-Work`) | Qwen3.6-35B-A3B | T8 | BF16 (no quants) | ~67 GB |
| Occamy-1.0 (`Accio-Lab/occamy-1.0`) | Qwen3.6-35B-A3B | T9 | official NVFP4 | ~24 GB |

Caveat: AgentMercury-4B is credibility 1 (individual author, card lists only the benchmarks it
improved).

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
- Team configs for T1–T10 (§4.4).
- For each specialist, its trained thinking mode and required prompt or chat-template settings
  (from its card, confirmed by the smoke test), and whether its tool calls parse with our
  `qwen3_coder` / `qwen3_xml` / Mistral parsers on the agentic tasks.

Found by the smoke tests on the second Spark (2026-10-22):

- **Absolute paths in the agentic pool.** `data-claude/tagged_agentic` stores sparkie12's absolute
  data root (`/home/syuen/...`) in the metadata of all 100 BrowseComp-Plus and 50 Finance-Agent
  tasks, so they fail on any other machine. Resolve `root` relative to the repo's `data-claude/`
  (or `--data_dir`) instead of trusting the stored path.
- **Benchmark registry race.** `benchmarks._load_registry()` fills `_REGISTRY` one module at a time
  and returns early once it is non-empty, so threads that call `benchmarks.get` concurrently on a
  cold registry can see only `gsm8k`. Load it once before starting worker threads, or guard it with
  a lock.
- **Thinking budget vs answer length.** vLLM enforces `thinking_token_budget` (it inserts the
  end-of-thinking text at the budget), but the Qwen3.5-4B fine-tunes then keep reasoning in the
  visible answer: with a 512-token budget, answers ran past 3,500 tokens. With 26,624 + 4,096, check
  how often answers are cut off after a forced end, and raise the answer share if needed.

## 8. Open decisions

- Review of the draft expert personas (§3.4).
