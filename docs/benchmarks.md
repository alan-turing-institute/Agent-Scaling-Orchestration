# Benchmarks

## Registry

`src/benchmarks/` has one module per dataset. Each module declares:

- `NAME`: the `--data` value and the tagged dataset's `dataset` column.
- `ANSWER_TYPE`: a key into `SCORERS` in `benchmarks/scorers.py`.
- `PERSONA_SET`: the paper's personas for this dataset.
- `load(args, split) -> (questions, labels)`: test splits return a shuffled `head(data_size)`.

To add a benchmark, write the module and add one line to `_MODULES` in `benchmarks/__init__.py`.
The tagger, splits and baselines read the registry. `benchmarks.get(name)` looks one up;
`benchmarks.scorer_for(name)` builds its scorer.

`truthfulqa` and `winogrande` load `truthfulqa/truthful_qa` and `allenai/winogrande` (current
`huggingface_hub` rejects the bare ids). Both map `test` to `validation`.

A module may define `load_instances(args, split) -> list[Instance]` instead of `load`; the registry
derives `load` from it. An `Instance` has a stable `id`, JSON `metadata`, and structural `tags`
(`"key: value"`, e.g. `level: 5`). Modules with only `load` get ids from `question_id(benchmark,
text)`. `tag_dataset.py` keeps `id` and `metadata` as columns and, with `--structural_tags`, adds
the structural tags to `tags`. The 699 pool has neither column; `benchmarks.instance_of(row)` fills
defaults. `Scorer.correct(prediction, gold, instance=None)` gets the `Instance`.

`scripts/fetch_benchmarks.py` downloads every registered set and prints counts. A module may define
`fetch(args)` for data its loader does not pull. gsm8k returns nothing at `--data_size 0`, which the
script flags.

## Harder benchmarks

These have `PERSONA_SET = []`: `paper_persona_names` gives them the default set, and
`paper_persona_baseline.py` refuses them. They live in their own pool, never the 699 one.

- `gpqa_diamond`: 198 four-option questions. Gated: accept the terms on Hugging Face once. Options
  are shuffled per question, seeded by record id.
- `mmlu_pro`: law, engineering, physics, chemistry, 50 each (`--sub_data` to change). Up to ten
  options (A–J). Not `pro_medicine`, which is MMLU's four-option set.
- `aime`: 2022–2025, 120 problems, integer answers, numeric scorer. Needs a large token budget.
- `math500`: MATH-500 levels 4–5 (262; `--sub_data 3,4,5` to change), LaTeX answers, `math` scorer.

The `math` scorer reads the last `\boxed{}` (strict), else a `final answer:` line, else the last
`$...$` (lenient). It compares with math-verify and votes over equivalence classes. math-verify's
timeouts use `signal.alarm`, which fails off the main thread, so on worker threads the scorer turns
them off and rejects inputs over `MATH_MAX_CHARS`. All 262 level 4–5 reference solutions score
correct.

## Baselines for new pools

Both are arms of `small_model_arms.sh`.

- `self_consistency`: `team_config.self_consistency(n, temperature)`. n persona-less stages sampled
  at `Stage.temperature` (default 0.7), then voted. Personas run at temperature 0, so this separates
  sampling variety from persona variety. Stage `temperature`/`top_p` override the persona's and are
  serialised only when set, so existing `config_id`s don't change. Run with
  `run_config.py --topology self_consistency --samples N --temperature T`.
- `canonical_proxy`: `paper_persona_baseline.py --proxy`. A benchmark with no paper set may name,
  per question, a paper benchmark whose set stands in (`proxy_persona_source(metadata)`;
  `ModuleBenchmark.proxy_for`). AIME and MATH-500 borrow gsm8k's set; MMLU-Pro law borrows
  formal_logic's. GPQA, MMLU-Pro science/engineering and the agentic sets have none and are skipped
  and counted (`skipped_questions`). Compare only on the questions it covers.

`report_crossval.py` lists every arm directory it finds, known ones first.

## Agentic benchmarks

A module is agentic when it declares `ANSWER_TYPE = "outcome"`, `ENVIRONMENT`
(`factory(instance) -> Environment`) and `MAX_STEPS`. The registry refuses one without the others.

`benchmarks/environment.py` defines the protocol: `task_prompt()`, `tools()` (OpenAI function
schemas), `call(name, arguments) -> observation` (errors returned as text, never raised), `done`,
`outcome() -> Outcome(success, fingerprint)`, `fork()`, and optional `resume()`. The prediction is
`"success|<fingerprint>"` or `"fail|<fingerprint>"`, so `OutcomeScorer` votes over end states.

**Server flags.** Agentic runs need a server that parses tool calls: `--enable-auto-tool-choice
--tool-call-parser` (`qwen3_xml` for Qwen, `mistral` for Ministral, `gemma4` for Gemma). The
`../vllm/` scripts pass them. Servers launched before 2026-10-19 lack them and return tool calls as
plain text; relaunch first.

**Episode loop.** `episode.run_episode`: generate with tools, run each call, append the observation.
Stops when the environment is done, after two replies in a row with no tool call (one nudge
between), or at the turn cap. No wall-clock caps. `runner.run_question(..., environment=,
max_steps=)` runs each stage as an episode. Where a stage starts:

- reads nothing: fresh;
- reads its own earlier stage (debate): a fork of its last state;
- reads others as critic, reviser or checker: a fork of the last input's state. `resume()` lets it
  overrule a verdict (Plancraft: `impossible`) but keeps a reached goal;
- hub or synthesiser: fresh, with the attempts as advice.

Stages see each other through `episode.report_text`: actions and final message, never the grade.
`team_config.delegated(workers, hub)` (`--topology delegated`) gives the hub a `delegate(agent,
instruction)` tool; workers run only when called, on the hub's environment, and a text reply ends
a worker's turn. Agentic role wording is `roles.AGENT_ROLES`, versioned by
`AGENT_ROLE_TEMPLATES_VERSION`. Steps go to `*_traces.jsonl` beside the predictions file;
prediction rows keep the outcome, calls and tokens. `OpenAICompatChatWrapper.generate(...,
tools=...)` returns `Completion.tool_calls`, with an `error` for non-JSON arguments. The bare-model
baseline runs one persona-less episode.

### The four sets

- `plancraft`: the package's text environment, image renderer stubbed out. Tools: search, move,
  smelt, impossible. Success: target in any slot but [0], or `impossible` on an impossible task.
  Default: first 100 test examples, as in the paper (`--sub_data all` for 580). Install with
  `requirements-agentic.txt` (`plancraft` itself with `--no-deps`). Optimal plans replay 580/580.
- `workbench`: upstream's 27 tools, sandbox CSVs and grader (replays agent calls and ground truth on
  fresh sandboxes, compares tables). Fetched at a pinned commit into
  `data-claude/benchmarks/workbench-upstream` (`scripts/fetch_benchmarks.py workbench`, or
  `WORKBENCH_ROOT`). Each episode owns its `ToolState` and binds it only during a tool call; grading
  runs unbound. A text reply ends the task; the turn cap (20) grades wrong. Default: the paper's 100
  (`random.Random(42)` across domains, alphabetical), current wording; `--sub_data all` for 690,
  `v1` for the 2024 wording. Ground truths replay 690/690; doing nothing passes 124.
- `browsecomp_plus`: the paper's 100 questions over a fixed 100,195-page corpus, searched with BM25
  (`bm25s`). The paper's dense retriever needs flash-attn, which has no ARM build, so these are BM25
  numbers. Tools: search (top 5, ~2,000-char snippets), get_document (clipped at 12,000 chars), done.
  Graded by the judge with BrowseComp-Plus's grader. `fetch` gets questions, qrels and corpus. Build
  the index with `scripts/build_browsecomp_index.py` (8–10 GB peak, not during a sweep), or
  `--subset evidence` (2,568 docs, 0.6 GB) for tests; select with `BROWSECOMP_INDEX`.
- `finance_agent`: Vals AI's 50 public questions. Tools: edgar_search (EDGAR full-text search),
  parse_html_page / read_page / search_page, python (separate interpreter, CPU and memory capped,
  not network-isolated), web_search, submit_final_result. External calls are on by default
  (`FINANCE_AGENT_ONLINE=0` turns them off), cached under `finance-agent/http-cache/`, limited to
  5/s. The SEC contact comes from `SEC_USER_AGENT`, else
  `data-claude/benchmarks/finance-agent/sec_user_agent.txt` (git-ignored; never commit an address).
  `web_search` uses a local SearXNG (`scripts/searxng/run.sh`, 127.0.0.1:8888 or `SEARXNG_URL`),
  offered only when it answers; unlike EDGAR it cannot cap results at 2025-04-07. Graded per rubric
  line by the judge, correct at ≥ 0.5. The contradiction line is asked as "does it contradict?"
  (the paper's grader inverted it).

### Judge

Judged benchmarks (`JUDGED = True`) take `judge=` in their environment factory;
`ModuleBenchmark.environment_for(judge)` supplies it and refuses to run without one. `judge.Judge`
(`--judge_model`/`--judge_api_base_url`; drivers default to the orchestrator's model) caches
verdicts per prompt and counts its tokens separately.

### Length limits

Length limits are not wrong answers.

- A request over the model's context (vLLM 400 "maximum context length",
  `openai_compat.is_context_limit`) is recorded as `limit: "context"`. An episode stops
  (`stopped: "context_limit"`) and is graded on its state; a one-call stage scores as unanswered.
- A reply cut off by the generation budget (`finish_reason: "length"`) is `limit: "max_tokens"`.

Each question row carries the worst stage `limit`. Batch reports, holdout summaries and
`report_crossval.py` count context hits and give accuracy on the questions that fit, to judge
whether `--max-model-len` needs raising. The mock server's `--context_chars N` refuses requests the
same way (`MOCK_ARGS` in `tests/mock_e2e.sh`).

`tests/test_agentic.py` covers the loop, every topology on a toy environment, delegation and traces.
The mock server answers requests that offer tools with deterministic tool calls.
