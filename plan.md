# Plan: cleaning up the harness and making benchmarks pluggable

Written 2026-09-25 while the fold-1 cross-validation sweep was running. Nothing in
here has been applied. It is ordered so that the parts which cannot change a
reported number come first, and everything that *can* change one is quarantined in
section 5.

The motivating requirement is to add harder benchmarks — the six agentic ones from
*Towards a Science of Scaling Agent Systems* (arXiv:2512.08296v3), and coding
benchmarks. Section 1 is the diagnosis of why that is hard today; sections 2–4 are
the work.

---

## 0. Constraints this plan is written under

- **A sweep is in flight.** `src/train_orchestrator.py` is running fold 1 of the
  cross-validation, arm `random`/`memory none`, 30 iterations. Do not touch
  `train_orchestrator.py`, `team_evaluation.py`, `holdout_evaluation.py`,
  `orchestration/orchestrator.py`, `summariser.py`, `splits.py` or
  `model/model_utils.py` until `data-claude/crossval/fold1/` has all its arms and
  `scripts/report_crossval.py` reads them. The loop imports these at start-up but
  Python does not re-read them, so an edit mid-run is *probably* inert — that is not
  a good enough reason to risk a multi-hour run.
- **Results are the only copy.** `.gitignore` excludes every result extension, so a
  refactor that changes an output schema orphans the runs already in
  `data-claude/`. Any schema change needs a reader that accepts both shapes, or a
  migration, or it is not worth doing.
- **Published numbers depend on the parsers.** The extraction quirks in section 5
  are load-bearing for the numbers in `docs/experiment-log.md` and for the paper's
  results. They get fixed behind a flag and a re-run, never silently.
- **One box.** DGX Spark, 121 GB unified memory, one vLLM container. Anything in
  section 4 that wants Docker-based evaluation competes with the serving container
  for that memory.

---

## 1. Diagnosis: what actually breaks when a new benchmark arrives

### 1.1 The core assumption

Every path in the repo assumes one shape:

```
question: str  →  one prompt  →  one response: str  →  regex  →  scalar answer  →  majority vote
```

That shape is spread across eight files as `if`/`elif` chains keyed on dataset
*name*, and nothing declares it. Adding a benchmark today means finding and editing
all of these:

| File | What is keyed on the dataset name |
|---|---|
| `src/data/data_utils.py:7` | `load_data` if/elif router, one branch per dataset |
| `src/evaluator.py:20,27` | `get_instruction_suffix` — two hardcoded dataset lists |
| `src/main.py:333` | picks `evaluate_gsm8k` vs `evaluate_mcq`, else `raise NotImplementedError` |
| `src/main.py:451,517` | writes a *different history record schema* for gsm8k than for MCQ |
| `src/model/model_utils.py:876+` | `_build_enhanced_personas`, one hand-written persona set per dataset |
| `src/team_evaluation.py:52,106,140,205,213` | the string `'gsm8k'` used as an *answer-type* sentinel |
| `scripts/bare_model_baseline.py:70,86,121` | the same sentinel again, independently |
| `src/tag_questions.py:21` | `SUPPORTED_DATASETS` allow-list |

### 1.2 The specific thing that will break on day one of a coding benchmark

`src/team_evaluation.py:41`, `_infer_answer_type`, decides how to score a question
by calling `float()` on the gold answer: numeric → gsm8k scoring, anything else →
**MCQ scoring**. A coding benchmark's gold answer is a string, so it will silently
take the MCQ branch and be scored by `evaluate_mcq`, which extracts the *second
character* of the last `{...}` group. It will not error. It will report a number.

The fix is already sitting in the data: `data-claude/tagged_dataset` carries a
`dataset` column (`{dataset, question, answer, tags}`). The answer type should be
looked up from that column through a registry, not sniffed from the answer's shape.

### 1.3 What the arXiv:2512.08296 benchmarks actually need

Checked against the PDF (v3, 8 Apr 2026), Table 1 and Appendix D. The six are
BrowseComp-Plus, Finance-Agent, Plancraft, WorkBench, SWE-bench Verified and
Terminal-Bench. The paper is explicit that GSM8K/MMLU-shaped items are *not* what it
measures — §3.2 requires sequential interdependence, partial observability and
trajectory length L > 3, and names GSM8K and MMLU as counter-examples.

So none of the six fit the `question → answer` shape, and they do not share a
grader either:

| Benchmark | Instances used | What "correct" means |
|---|---|---|
| BrowseComp-Plus | 100 | LLM judge, agent answer vs ground truth, with confidence |
| Finance-Agent | 50 | expert rubric per instance, judged free text |
| Plancraft | 100 | environment state — was the item crafted within the step limit |
| WorkBench | 100 | exact match on the executed function-call sequence |
| SWE-bench Verified | 20 (seed-42 shuffle of 500) | repository test suite passes against the patch |
| Terminal-Bench | 20 (first 20 of 86) | Docker environment, objective success criteria |

They also need things the harness has no concept of: a tool loop (7 tools for
SWE-bench, 2 for Terminal-Bench), an iteration cap per instance, and per-trajectory
logging — the paper's own metrics (coordination efficiency, error amplification,
redundancy, message density, turn count, token cost) are all trajectory-level.

**Consequence for this plan:** the work splits cleanly in two. A *scorer* seam
(section 3) which is cheap, unblocks coding benchmarks and short-answer benchmarks,
and is worth doing regardless. And a *trajectory* seam (section 4) which is a much
larger piece of work and should not be started until the scorer seam exists.

### 1.4 Token budget is not controllable today

`src/model/openai_compat.py:61` opens `complete()` with `max_tokens = 4096`,
overwriting the argument the caller passed. So `--max_new_tokens` and every
persona's `max_new_tokens` are silently ignored on every vLLM run — which is every
run. The paper matches compute across architectures at ≈4,800 reasoning tokens per
trial; that comparison cannot be reproduced until this line goes.

The same wrapper discards `resp.usage`, so no run in `data-claude/` records what it
cost.

---

## 2. Phase 0 — safe to do now, no behaviour change

These can land while a sweep runs, because nothing they touch is imported by the
loop, or because the change is provably inert.

1. **Delete dead code.**
   - `src/evaluator.py:69` `_evaluate_gsm8k` — never called from anywhere.
   - `src/data/base_ds.py` — `format_ds` is imported by all seven dataset modules and
     called by none. It references `args.reverse_landmark`, `args.synonym_replacement`,
     `args.random_deletion`, `args.word_level_shuffling`, `args.answer_level_shuffling`
     and `args.perturbation`, none of which any entry point defines, and it contains a
     live `import pdb;pdb.set_trace()` at line 31. It is a leftover from a
     perturbation study. Delete the module and the seven imports.
   - `src/model/model_utils.py:1268` — the second `elif args.data in ['truthfulqa']`
     branch is unreachable behind the one at :1229. Its five personas
     (`Core_Claim_MinAssumption`, `Imitative_Falsehood_Spotter`,
     `Consensus_Fact_Checker`, `Rhetoric_and_Absolutes_Filter`,
     `Causality_and_Mechanism_Auditor`) are all present in the bank and reachable from
     the orchestrator, so deleting the branch loses nothing. **But** decide first
     which of the two truthfulqa sets the paper baseline should use —
     `scripts/paper_persona_baseline.py` calls `_build_enhanced_personas` directly and
     currently gets the first set.
   - `src/main.py` flags parsed and never read: `--agent_selection`, `--alpha`,
     `--max_num_agents`, `--generate_first_round`, `--load_in_4bit`, `--split`,
     `--debug`. `--solver` is read only to build the output filename — it selects
     nothing. Either drop them or leave a one-line comment saying they are inert;
     what is not acceptable is a flag that looks like it does something.

2. **Write down the two contracts that are currently only in `CLAUDE.md`.**
   The agent-name format `{data}_{data_size}__{model}__{persona}__Agent{i}` and the
   history-record schema are parsed by `K_star_analysis/` and by the debate loop.
   Put a docstring on `src/main.py` naming both, so the next person does not learn it
   from a stack trace.

3. **`docs/` housekeeping.** `docs/experiment-log.md` notes that `--solver debate`
   is accepted and not implemented; make that a `parser.add_argument` help string in
   `train_orchestrator.py` too, or drop the choice.

---

## 3. Phase 1 — the benchmark seam (the main piece of work)

This is the change that makes a new benchmark one file instead of eight edits.

### 3.1 Target shape

```
src/benchmarks/
    __init__.py       # REGISTRY: name -> Benchmark; get(name), list_names()
    base.py           # Benchmark and Scorer protocols, Instance dataclass
    scorers.py        # numeric, mcq_letter, exact_match, judged, code_tests
    gsm8k.py  arc.py  hellaswag.py  truthfulqa.py  winogrande.py
    mmlu_pro_medicine.py  mmlu_formal_logic.py
```

`src/data/` moves here wholesale; each module grows the metadata that currently
lives in the if/elif chains.

```python
@dataclass
class Instance:
    question: str
    answer: Any
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)   # tests, rubric, repo snapshot...

class Benchmark(Protocol):
    name: str
    answer_type: str                  # key into SCORERS
    def load(self, cfg) -> list[Instance]: ...
    def instruction_suffix(self, style: str) -> str: ...   # style: plain | cot | bae
    def personas(self) -> list[str]: ...                   # names into the bank

class Scorer(Protocol):
    def extract(self, text: str) -> Prediction: ...        # value + parsed: bool
    def correct(self, pred: Prediction, gold) -> bool: ...
    def aggregate(self, preds: list[Prediction]) -> Prediction: ...
```

Three things this buys that the current code cannot express:

- **`Prediction.parsed`.** Today a response the parser could not read and a response
  that was wrong are the same thing: `""`, counted as incorrect. Separating them is
  what makes section 5's parser fixes measurable rather than a leap of faith.
- **`aggregate` on the scorer.** Majority vote over floats and majority vote over
  `(A)` are the same operation, which is why they are duplicated four times in
  `evaluator.py`. Majority vote over a *program* is not that operation at all — a
  coding scorer aggregates by running the tests and voting on pass/fail, or by
  picking one candidate. Putting aggregation on the scorer is what lets a coding
  benchmark in without touching the solver.
- **`instruction_suffix` on the benchmark.** `get_instruction_suffix`'s fallback
  branch asks for a *numeric* answer, which is why `team_evaluation.py:106` has to
  build a fake `args` with `data = 'arc'` to get the MCQ suffix, and why
  `bare_model_baseline.py:121` independently does the same thing. Both hacks go.

### 3.2 Answer type comes from the data, not from `float()`

Replace `_infer_answer_type` (`team_evaluation.py:41`) with a lookup on the tagged
dataset's existing `dataset` column through the registry. Two follow-ons:

- Add an explicit `answer_type` column in `src/tag_dataset.py` when the dataset is
  next rebuilt, so a benchmark whose name is not in the registry fails loudly at
  load rather than being scored as MCQ.
- `src/tag_questions.py:21` `SUPPORTED_DATASETS` becomes `benchmarks.list_names()`.

**Compatibility:** `data-claude/tagged_dataset` on disk has no `answer_type` column.
The loader must fall back to the registry lookup on `dataset` when the column is
absent, so the current sweeps' dataset keeps working untouched.

### 3.3 Callers

- `src/main.py:333` — `evaluate = benchmarks.get(args.data).scorer` and the
  `NotImplementedError` becomes a registry `KeyError` naming the available benchmarks.
- `src/main.py:451/466` and `:518/531` — four copies of the same record-building
  block, differing only in whether `np.round(y, 1)` is applied and whether the key
  is `responses` or `agent_responses`. Collapse to one `build_round_record(...)`
  helper. See section 5.3 — the gsm8k/MCQ key difference is a live bug, and this is
  where it gets fixed.
- `src/team_evaluation.py` — the `'gsm8k'` sentinel becomes the scorer name.
- `scripts/bare_model_baseline.py` — delete its private copy of the suffix table and
  the scoring branch; call the same scorer.

### 3.4 Adding a coding benchmark once this exists

HumanEval/MBPP become: one module in `src/benchmarks/`, plus one `code_tests` scorer
in `scorers.py`. `_build_enhanced_personas` already contains a coding persona set
(`Defensive_Coder`, `Elegant_Minimalist`, `Algorithm_Optimizer`,
`Test_Driven_Developer`, `Creative_Problem_Solver`) under `args.data in ['humaneval',
'mbpp']` at `model_utils.py:1156` — written, in the bank, and currently unreachable
because no loader exists. There is also an unreachable `piqa` set at :1374.

The one genuinely new piece is **sandboxed execution**. Do not run generated code in
the harness process. Minimum viable: a subprocess with a timeout, no network, a
scratch cwd, and a memory cap; better: the same Docker approach section 4 needs
anyway. Decide this before writing the scorer, because it determines whether
`Scorer.correct` can stay synchronous.

---

## 4. Phase 2 — deduplication and tightening

Ordered by payoff. None of these change a number if done correctly; all should be
checked by re-running one small arm and diffing the summary.

1. **`_build_enhanced_personas` is a copy of `chosen_persona_bank()`.** Verified by
   construction: all 50 names the enhanced builder can produce are in the 50-name
   bank, and every definition is byte-identical except one —
   `Elimination_Specialist` under `winogrande`, which the bank carries under the name
   `Elimination_Based_Solver`. That is ~654 lines duplicating ~579 lines, and the two
   have already drifted once. Replace the builder with a per-dataset *name list*:

   ```python
   PAPER_PERSONA_SETS = {
       "gsm8k": ["Conservative_Verifier", "Creative_Explorer", ...],
       "winogrande": [..., "Elimination_Based_Solver"],   # note the rename
       ...
   }
   ```

   ~650 lines deleted. `scripts/paper_persona_baseline.py` and its `personas_override`
   path in `team_evaluation.py` keep working; the file's own docstring already
   explains the `Elimination_Specialist` collision, and this collapses it to a naming
   decision instead of two divergent copies.

   *Check before landing:* the winogrande paper-persona arm is the only one whose
   prompt text changes, and only if it is re-run. It is not in flight.

2. **`_add_nvidia_personas` will raise on 45 of the 50 personas.**
   `model_utils.py:233` reads `persona_data['nvidia_persona']` unconditionally, and
   only the five gsm8k personas define that key. It is called at :1514 behind
   `getattr(args, "persona_prompt", True)` — note the default is `True`, so any entry
   point that does not set `persona_prompt` takes this branch. Make the key optional
   (`if "nvidia_persona" in persona_data`) and give `_build_enhanced_personas` an
   explicit `persona_prompt=False` default. Same trap on `chosen_agents` at
   `model_utils.py:195`, which also defaults `True`.

3. **Three copies of the retry wrapper.** `with_server_retry` is defined identically
   in `src/train_orchestrator.py:149` and `src/holdout_evaluation.py:64`; and
   `holdout_evaluation._select_team` re-implements the retry-on-invalid-names loop
   that `orchestration/orchestrator.team_selection` already has. Move
   `with_server_retry` to a `src/retry.py` (or into `splits.py`'s role as the shared
   module) and give `select_team` its own retry so both callers get it for free.
   `scripts/paper_persona_baseline.py:45` imports the private `_batch_indices` from
   `holdout_evaluation` — that should become a public name in a shared module too.

4. **Persist per-question predictions.** This is the highest-value cleanup in the
   list and it is not obvious from the file sizes. `team_evaluation._run_sample`
   builds `responses` and per-agent predictions, *prints them to stdout*, and returns
   them — and `_accumulate` in `holdout_evaluation.py` folds them into counts and
   drops them. `scripts/bare_model_baseline.py:97` computes a `prediction` field and
   never writes it. So `data-claude/*/holdout_records.jsonl` holds counts only.

   The consequence: **no finished run can be re-scored.** Every parser fix in section
   5 costs a full GPU re-run to measure, and nobody can audit why an agent was marked
   wrong. Write one `predictions.jsonl` per run — question index, gold, per-agent raw
   text (or a hash plus first N chars if size is a worry), extracted value,
   `parsed` flag, per-agent correct, team answer. Do this *before* section 5, because
   it is what makes section 5 cheap.

5. **One record schema across arms.** `team_evaluation` returns `answer_types` as a
   `dict(Counter)`; `bare_model_baseline` writes it as a `list`. `report_run.py` and
   `report_crossval.py` read these files. Pick one shape and have the readers accept
   both for the runs already on disk.

6. **`main.py` is broken against vLLM.** `main.py:429` and `:498` do
   `resp.choices[0].message.content` and `resp.model_dump_json()`, but
   `OpenAICompatChatWrapper.complete` returns a **string**. Only `AzureOpenAIWrapper`
   returns a response object. So the benchmark-harness half of the repo cannot run
   against the local vLLM server — which is the only way models are served here.
   Fix by reusing `team_evaluation._response_text` (promoted to the shared module; it
   already handles all three shapes) and making `responses_json` optional. Check
   whether `scripts/*.sh` sweeps have been run since the vLLM move before assuming
   this is dead code.

7. **`src/model/openai_compat.py:61`.** Delete `max_tokens = 4096`. Then decide the
   default deliberately — a reasoning model needs far more than 512, which is what
   `--max_new_tokens` defaults to, so removing the override without raising the
   default will truncate every answer. Do this together with recording `resp.usage`,
   and treat it as a section 5 change: it will move numbers.

8. **`main.py` end-of-run fragility.** `agent_accs` (`main.py:562`) is assigned
   inside the question loop and read at `:571` after it; an empty test set is a
   `NameError` after a completed run. `args.timestamp` is set only in the
   `__main__` block, so `main(args)` called as a library raises at `:574`.

9. **`requirements.txt`** is 107 fully pinned lines including the Azure SDK tail.
   Split into `requirements.txt` (what the loop needs) and `requirements-azure.txt`,
   or move to a `pyproject.toml` with extras. Low priority, but it is the first thing
   anyone reproducing this hits.

---

## 5. Bugs that change reported numbers — gate these

Each of these is reproducible against the current code. None should be fixed
silently; each needs a flag, a re-run of at least one arm, and a line in
`docs/experiment-log.md` recording which results were produced under which
behaviour. Do item 4.4 first so the re-scoring is free.

### 5.1 `evaluate_mcq` takes the second character of whatever is in braces

`src/evaluator.py:115`. Measured against the current code:

| Response | Extracted |
|---|---|
| `{final answer: (A)}` | `(A)` ✓ |
| `{final answer: A}` | `(A)` ✓ |
| `{Final Answer: (A)}` | `(i)` ✗ |
| `{answer is A}` | `(n)` ✗ |
| `no braces at all, answer (A)` | `""` ✗ |
| `{final answer: (10)}` | `(1)` ✗ |

The strip of `"final answer:"` is case-sensitive, and everything else falls through
to `pred[1]`. A model that capitalises the label — common with reasoning models —
scores zero on that question and is indistinguishable from a model that got it
wrong. `base_evaluate_mcq` (the `--bae` path) is the more robust parser; the
non-`bae` path should gain the same last-`(X)`-match fallback behind a
`--strict_parse/--lenient_parse` flag.

### 5.2 `evaluate_gsm8k` takes the last number anywhere in the response

`src/evaluator.py:48`, ignoring the braces the prompt asked for:

- `"Step 1: 5 apples... {final answer: 8}"` → `8.0` ✓
- `"I computed 8, so the answer is 8 dollars over 3 days"` → `3.0` ✗

`_evaluate_gsm8k` at :69 is the brace-respecting version and is dead code. The fix is
to try braces first and fall back to last-number, again behind a flag.

### 5.3 `K_star_analysis` may be silently dropping every gsm8k history

`main.py:451` writes gsm8k rounds with `responses` = a *list* of `model_dump_json()`
strings and the agent-keyed dict under `agent_responses`; the MCQ branch at :466
writes the agent-keyed dict under `responses` and no JSON list.
`K_star_analysis/analysis.py:177` reads `round_obj.get("responses", {})` and only
proceeds `if isinstance(responses, dict)` — a list yields no text items. At :192 it
also requires the gold answer to be a `str`, and the gsm8k branch writes a float.

Read from the code, not reproduced — there are no history files on this box to test
against. **Verify against a real gsm8k history before acting.** If it holds, every
K\* number computed on a gsm8k run is computed on zero embedded responses. Fix in
the analysis reader (prefer `agent_responses` when present, coerce the gold to
`str`) rather than in `main.py`, so existing histories stay readable — and then fix
`main.py` to emit one schema, per section 3.3.

### 5.4 Peers exchange a bare number during decentralized debate

`main.py:155` injects `extract_number(responses[other_agent])` — the last number in
the peer's text — for every dataset, including MCQ, where the last number is
whatever digit happened to appear. The centralized branch at :182 passes full text.
This is documented in `CLAUDE.md` as deliberate and published results depend on it.
Leave it. If it is ever revisited, it belongs on the `Benchmark` as a
`peer_view(response) -> str` hook, not as a global.

---

## 6. Phase 3 — agentic benchmarks (arXiv:2512.08296)

Do not start this before section 3 lands. Sketched in dependency order.

1. **Trajectory layer.** The unit of work stops being a message and becomes an
   episode: a loop of `(action, observation)` with a tool registry, an iteration cap,
   and a terminal `submit`. The paper's parameters, for reference: single agent 10
   iterations; independent 3 agents, synthesis only; centralized 3 sub-agents + 1
   orchestrator, 5 rounds × 3 iterations; decentralized 3 agents × 3 debate rounds ×
   3 iterations.

   Note this maps onto what the repo already has. `--centralized` is a star topology
   and `--sparse`/default are the decentralized variants; what is missing is the tool
   loop underneath them, not the topologies.

2. **`Scorer` gains a non-text branch.** `correct(prediction, gold)` becomes
   `verify(episode, instance) -> bool`, with the text scorers as the degenerate case
   where the episode is one turn. This is why section 3.1 puts scoring behind a
   protocol rather than a function.

3. **Trace logging.** Turns, inter-agent messages, tool calls, tokens per role.
   Without it none of the paper's secondary metrics (coordination efficiency, error
   amplification, redundancy, message density, information gain) can be computed, and
   they are most of what makes the paper's analysis more than an accuracy table.
   Depends on section 4.7 recording `usage`.

4. **Order of adoption.** Cheapest first:
   - **BrowseComp-Plus** and **Finance-Agent** are closest to the current shape — one
     final answer, graded by an LLM judge against a gold answer or a rubric. A
     `judged` scorer plus a web-search tool gets most of the way. Budget extra
     repeats on BrowseComp-Plus: the paper reports it as the noisiest benchmark
     (σ/μ = 0.32).
   - **WorkBench** and **Plancraft** need the environment hosted and the final state
     asserted on. No sandbox, but real integration work.
   - **SWE-bench Verified** and **Terminal-Bench** need Docker and test execution.
     The paper used 20-instance subsets for exactly this reason, which is a sensible
     first target here too. **Memory contention:** the serving container already
     holds ~50% of the 121 GB. Plan to tear the server down, or to run evaluation
     against a remote endpoint, rather than assume both fit.

5. **What not to inherit.** The paper's implementation is LiteLLM + LangChain. This
   repo has its own wrapper layer that already handles Azure, OpenAI-compatible and
   local HF. Do not pull LangChain in for the tool loop; the `Benchmark`/`Scorer`
   protocols in section 3 plus a small tool dispatcher is less code than adapting to
   someone else's agent abstraction, and it keeps the existing heterogeneous-model
   support (`--vllm_base_urls`, agent *i* → URL *i % n*) working.

---

## 7. Explicitly not doing

- **Parameterising `scripts/*.sh`.** 22 shell scripts over one template, each
  hardcoding its dataset, GPU list and port map. `CLAUDE.md` says to copy the nearest
  one and edit the header; that is the right call for a record of what was run. Leave
  them.
- **Adding a test suite across the board.** Not worth it here. Two exceptions worth
  the effort: a table-driven test for the scorers in section 3.1 (the cases in 5.1
  and 5.2 are the table), and a smoke test that every registered benchmark loads and
  round-trips one instance through its scorer.
- **Touching the debate topology or the peer-opinion asymmetry** (section 5.4).
- **Renaming the agent-name format.** It is parsed in three places and is baked into
  every saved history.

---

## 8. Suggested order

| Order | Item | Blocks a running sweep? |
|---|---|---|
| 1 | §2 dead code, docstrings | no |
| 2 | §4.4 persist per-question predictions | needs the sweep finished |
| 3 | §3 `src/benchmarks/` registry + scorers | needs the sweep finished |
| 4 | §4.1–4.3, 4.5 dedup | needs the sweep finished |
| 5 | §5 parser fixes, behind flags, re-score from §4.4 | needs a re-run |
| 6 | §3.4 coding benchmark + sandbox | — |
| 7 | §4.6–4.7 `main.py`/vLLM, token budget | needs a re-run |
| 8 | §6 trajectory layer | — |

---

## Status — updated 2026-10-05

The branch was rebased onto `main` after the experiment branch merged (#4) and
now carries C1–C5 of the research plan ("Orchestration Meets Agent Scaling") on
top of the registry work below.

### C1–C5, landed

| Commit | Covers |
|---|---|
| `Return what a call cost, and let stages name their model` | C1: `Completion` from `generate()`, `model/registry.py`, `--models_file` |
| `Describe a team as a small graph of stages` | C2: `team_config.py`, `roles.py` |
| `Run every team through one runner and score every stage` | C3, C4: `runner.py`; `run_team_evaluation` wraps it |
| `Record every stage of every question, and re-score from the file` | C5: schema-2 rows, `configs.json`, `scripts/rescore.py`, `scripts/run_config.py` |

Offline: `tests/test_runner.py` (prompts byte-identical to the old vote,
config validation, handoffs, transitions, tie-break, report shape, rows,
rescore agreement) and `tests/test_scorers.py` both pass. An independent review
found a crash in `paper_persona_baseline.py` (stale import), wrong credit for
debate and hub stages, and config validation holes; all three fixed and tested.

### Regression gate (C0)

The 0.8B `random` arm, fold 0, rerun against the same server. Teams are drawn
by a seeded generator, so all four runs answered with identical teams:

| Run | Team correct | Agent answers correct |
|---|---|---|
| old code, 2 Oct (the published result) | 65/140 | 220/560 |
| old code, today | 69/140 | 235/560 |
| new code, run 1 | 69/140 | 236/560 |
| new code, run 2 | 72/140 | 242/560 |

| Pair | Sum over batch x agent of the difference in correct answers (of 560 answers) | Sum of per-batch team differences |
|---|---|---|
| old today vs new run 1 | 89 | 28 |
| new run 1 vs new run 2 | 74 | 13 |
| old 2 Oct vs old today | 99 | 26 |

**Passes.** Old and new code differ by no more than two runs of the same code.
The gap from the published 65 is the server: the 0.8B container was recreated
at 23:30 on 2 October, after the published run, and the old code gives 69 on
today's container too. The somewhat larger team-level churn between old and new
is the tie-break change: 37 of 140 questions had a tied vote.

**The server is not deterministic under load.** The same request repeated in
sequence gives identical text (10/10, with or without a per-request seed), but
under concurrent load 0/10 came back identical. So arms run at
`--eval_workers 5` differ in at least 13–18% of agent answers (the sums above
are a lower bound: flips inside one batch can cancel) and by 3–4 team questions
per fold from serving noise alone - the size of the selection effects being
measured. Comparisons must be paired and repeated, or run with a serving setup
whose outputs do not depend on the batch; which vLLM setting causes it
(speculative decoding is on for the 0.8B) has not been isolated.

### Next

Experiments E2, E10 and E12 can run now with `scripts/run_config.py`. Still
open from the list below: §4.3, §2, §5 (now a re-read, since gate runs carry
predictions), §3.4, §6.

## Status — 2026-10-01 (registry work)

Written while the work lived in a separate clone during the fold-1 sweep.

### Done

| Commit | Covers |
|---|---|
| `Honour the generation budget...` | §1.4, §4.7 |
| `Put benchmarks behind a registry...` | §3 (all), §4.6, §5.1, §5.2, §5.3, part of §2 |
| `Write down what each agent answered...` | §4.4 |
| `Build the paper persona sets from the bank...` | §4.1, §4.2 (key optional; `persona_prompt` fallback now False) |

Verification, all offline — nothing in this branch has contacted the vLLM server:

- Strict parsers reproduce the untouched `evaluator.py` over **3,198 response
  combinations** at team sizes 1–3, and all 7 datasets × 3 prompt styles for the
  instruction suffixes. Zero mismatches.
- The registry's answer-type lookup agrees with the old `float()` sniffing on all
  **699 rows** of `data-claude/tagged_dataset`.
- `tests/test_scorers.py` — 7 tests, runs standalone (`PYTHONPATH=src python
  tests/test_scorers.py`); there is no pytest in the pinned environment.
- Predictions path exercised end to end against the real tagged dataset with the
  network stubbed, on a batch mixing numeric and MCQ questions.

**§5.3 is now confirmed, not suspected.** Run against the untouched
`analysis.py`, a numeric history record yields `texts=0` and an empty ground
truth. Every K\* number computed on a gsm8k run was computed on no embedded text
at all. Fixed in the reader, so histories already on disk become readable.

### Behaviour changes to be aware of

- Everything defaults to the old behaviour. `--parse_mode` defaults to `strict`;
  the token budget resolves to the 4096 that was already in force.
- Local HuggingFace agents now generate up to 4096 tokens rather than 512, since
  they read the same fallback. Nearly everything here goes over HTTP.
- `answer_types` in a record now reads `numeric` rather than `gsm8k`. Nothing
  reads that field — checked against both report scripts and the summariser.
- `main.py` writes one history schema for all benchmarks, and gains
  `final_answers_parsed`. The K\* reader accepts old and new.
- `--bae` parsers now record one prediction per agent instead of dropping
  failures. No sweep script uses `--bae`.

### Next, in order

1. **§4.3** — one `with_server_retry`, currently defined identically in two files.
2. **§2** — remaining dead code: the inert `main.py` flags. The dead second
   `truthfulqa` branch went with §4.1; the paper baseline keeps the first set,
   which is the one it always got.
3. **§5** — re-score a finished arm under `--parse_mode lenient` from the new
   predictions files and record the delta in `docs/experiment-log.md`. No run on
   disk has a predictions file yet, so this needs one arm run on this branch
   first; after that it is a re-read, not a re-run.
4. **§3.4** — coding benchmark. The seam is in place: one module in
   `src/benchmarks/` plus a `code_tests` scorer. The sandbox decision (subprocess
   with timeout and no network, vs. Docker) is still open and should be settled
   before the scorer is written, since it decides whether `Scorer.correct` can
   stay synchronous.
5. **§6** — trajectory layer for the agentic benchmarks.

### Merging back

The branch does not touch `data-claude/`. Merge once `data-claude/crossval/fold1/`
is complete and `scripts/report_crossval.py` has read it. The first run afterwards
should be a repeat of a finished arm under `--parse_mode strict`, to confirm the
numbers land where they did before.
