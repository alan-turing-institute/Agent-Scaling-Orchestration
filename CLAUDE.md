# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

Research code for the paper *Understanding Agent Scaling in LLM-Based Multi-Agent Systems via
Diversity*. `README.md` has the paper story, personas, K\* definition and flags. Benchmarks and
agentic environments: [`docs/benchmarks.md`](docs/benchmarks.md).

Experiment entry points, not a library: no lint, packaging or `__init__.py`. Two pipelines share
`src/model` and `src/benchmarks`:

1. **Benchmark harness**: `src/main.py`. N agents answer, optionally debate R rounds, majority vote.
   Sweeps in `scripts/*.sh`.
2. **Orchestrator loop**: `src/train_orchestrator.py`. An LLM picks a 4-agent team per batch from
   question tags, the team is scored, and the results come back as a markdown memo next iteration.

## Running

Run scripts by path from the repo root. Imports like `from model.model_utils import ...` work only
because `src/` is on `sys.path`; `python -m src.main` fails.

```bash
pip install -r requirements.txt          # fully pinned

python src/main.py --data gsm8k --num_agents 5 --model qwen2.5-7b --solver vote --debate_rounds 0
bash scripts/add.sh                      # sweep; GPU scheduler and vLLM endpoints hardcoded
python src/train_orchestrator.py --api_base_url http://localhost:8001/v1 \
    --model_name Qwen/Qwen3.6-35B-A3B --dataset_path data/tagged_dataset
```

Almost everything calls an OpenAI-compatible endpoint served by `../vllm/`.

## Tests

Offline tests are plain scripts (no pytest in the pinned env):
`PYTHONPATH=src python tests/test_runner.py`, plus `test_scorers.py`, `test_benchmarks.py`,
`test_agentic.py`. `tests/mock_openai_server.py` stands in for a model server;
`tests/mock_e2e.sh` builds a small pool and runs every arm against it. Never point tests at a
server a sweep is using.

## Benchmark harness (`main.py`)

**Generation.** All text goes through `engine(messages, agent, num_agents, persona_configs)` in
`src/model/model_utils.py`. `agent` is one wrapper or a list, indexed `i % len(agents)`: message
order is agent order.

`get_agents()` picks wrappers by their `kind` attribute (no base class):

- `azure_openai`, `openai_compat`: `.complete(messages, ...)`. `OpenAICompatChatWrapper` returns a
  string; `AzureOpenAIWrapper` returns the raw response (retries with doubled `max_tokens` on
  "Insufficient tokens").
- no `kind`: local HF (`LlamaWrapper`/`QwenWrapper`), returns a string.

`main.py` calls `resp.choices[0].message.content`, so it works only with response objects. To
handle all three shapes, use `team_evaluation._response_text()`.

`_make_agent` checks `OPENAI_COMPAT_MODELS` / `AZURE_OPENAI_MODELS` before `use_vllm`. `model_dirs`
maps only 3 local checkpoints; keys like `mistral-7b`, `qwen3-8b` exist only under `--use_vllm`.
With `--vllm_base_urls` (comma-separated), agent *i* uses URL *i % len(urls)*.

**Agent names carry state.** Format: `{data}_{data_size}__{model_key}__{persona_name}__Agent{i+1}`.
`get_new_message()` reads the persona from `split("__")[-2]` and the model from `[-3]`. The name is
also the key in every history record. Changing it breaks debate, saved histories and analysis.

**Personas.** All in `src/personas.py`. `chosen_persona_bank()` defines 50, each `{prompt,
temperature, top_p, style, [nvidia_persona]}`, returned as fresh dicts.

- `_build_enhanced_personas(args)`: the paper's set for `args.data`, from the benchmark module's
  `PERSONA_SET`, else `_UNREGISTERED_PERSONA_SETS` (humaneval, mbpp, piqa, arc_easy), else
  `_DEFAULT_PERSONA_SET`. Returns a no-op `{"None": {...}}` unless `multi_persona`, `baseline_a` or
  `baseline_b` is set. The winogrande `Elimination_Specialist` is stored as
  `Elimination_Based_Solver` and renamed back by `_PAPER_NAMES`.
- `_build_chosen_personas(args)`: picks by name from `args.chosen_personas`. Orchestrator path.
  `build_agent_pool()` renders the bank as the orchestrator's candidate list.

Per-persona `temperature`/`top_p` reach the model as `persona_configs`. Personas cycle
`i % len(personas)` when there are more agents than personas. Role instructions live in `roles.py`,
appended after the persona prompt.

Traps:

- `get_agents` uses `getattr(args, 'chosen_agents', True)`: defaults to True. A new entry point
  without `chosen_agents` takes the chosen-personas branch and fails on missing `chosen_personas`.
  Set both.
- `--persona_prompt` appends `nvidia_persona`; only the five gsm8k personas have one.

**Debate.** Topologies: full mesh (default), `--sparse` (ring), `--centralized` (agent 0 is the hub
and is the only agent scored). The mesh passes peers `extract_number(response)`, a bare number, even
on MCQ; the centralized branch passes full text. Published results depend on this asymmetry; do not
change it silently. `baseline_a` (length-matched padding), `baseline_b` (all agents share persona
#1) and `multi_persona` are mutually exclusive.

**Answer extraction.** `src/evaluator.py`. `get_instruction_suffix` asks for `{final answer: 123}` or
`{final answer: (A)}`; `--cot` adds step-by-step, `--bae` uses looser `base_evaluate_*` parsers.
Evaluators return `(per_agent_answers, majority_answer, majority_is_correct)`, ties by
`random.choice`. `evaluate_gsm8k` takes the last number anywhere (ignores the braces);
`evaluate_mcq` reads the letter by position in the braces. Changing a parser changes every
reported number.

## Orchestrator loop

`train_orchestrator.py` (`--iterations`, default 10).

- `orchestration/orchestrator.py`: `OrchestratorAgent` has its own `OpenAI` client and asks for JSON
  `{selected_agents, reasoning}`. Names are checked against the pool; invalid ones are dropped and
  the selection retried once. A parse failure skips the iteration.
- `team_evaluation.py`: `run_team_evaluation(selected_team, questions, args, config=None)` runs a team
  on each question (parallel, `--eval_workers`). Default config is `team_config.vote(team)`. Answer
  type comes per question from the `dataset` column via the registry; a missing or unknown
  `dataset` raises before any call. Returns accuracies, raw counts keyed by stage id (persona names
  for a vote), `config_id`, call/token totals, and `samples` (written by `predictions.save_report`).
- `summariser.py`: `--summariser counts` (default) folds counts into
  `agent_performance_state.json` and renders markdown from it; `--summariser llm` has the model
  rewrite the markdown. **The markdown is the loop's memory.** Delete it and the state file to
  reset; a stale pair contaminates a new run.
- `holdout_evaluation.py`: `--test_fraction` holds out a split, scored after training with the
  scoreboard frozen, in shuffled batches (`--split_seed`, `--test_batch_size`), one selection per
  batch. `--random_baseline` scores a random team on the same batches.

**Team configs and runner** (the orchestrator path does not use `engine`/`get_agents`):

- `team_config.py`: `TeamConfig` is an ordered tuple of `Stage(id, persona, role, model, inputs,
  max_tokens)`; each stage reads only earlier stages via `Input(source, handoff)`. Aggregate is
  `vote` or `stage` (the `final` stage's answer). Builders: `vote`, `debate`, `centralized`,
  `synthesis`, `pipeline`, `parallel`, `split_budget`, `self_consistency`, `delegated`. `config_id`
  hashes all behaviour, including the role-template version.
- `roles.py`: role templates and handoffs (`answer`, `rationale` = last 600 chars, `full`). A solver
  with no inputs gets the legacy prompt. Bump `ROLE_TEMPLATES_VERSION` on any template change.
- `runner.py`: `run_question` runs a config layer by layer, scores every stage, records transitions
  (`kept_right`, `fixed`, `broke`, `kept_wrong`) and copies, then aggregates. Ties use a
  per-question seeded RNG (`--tie_break global` for the old behaviour). Sends a per-request `seed`
  unless `--request_seed -1`. Connection errors propagate for retry; other failures score as
  unanswered. Shared flags: `add_runner_args`, `add_topology_args`.
- `model/registry.py`: model key -> served name and endpoint. `default` is `--model_name` at the
  base URL (8001 if unset); `--models_file` adds more (`configs/models.example.json`).
  `generate` returns a `Completion` (tokens, finish reason, latency, reasoning).

`--topology` arranges a named team: `vote` (default), `debate --rounds N`, `centralized`/`synthesis`
(last agent leads), `pipeline --roles ...`. Each persona is credited with its last stage's answer.

**Serving noise.** vLLM output depends on what else is in the batch: two runs at `--eval_workers 5`
flip ~20% of agent answers and ~15% of team verdicts. `--max_inflight 1` (per-server semaphore)
makes runs identical at ~2.5–3x the wall-clock. It applies per process: never run two arms against
one server at once.

**Scripts.** `scripts/run_config.py` runs one fixed config on the held-out batches.
`scripts/rescore.py` re-scores predictions under another parser. `scripts/compare_runs.py` pairs
arms per question across folds and repeats (McNemar / sign-flip, Holm) and prints flip rates.

**Outputs** (default `data-claude/orchestrator/`): `agent_performance_by_tag.md`,
`agent_performance_state.json`, `run_records.jsonl`, `team_selection_results.csv`,
`holdout_records.jsonl`, `holdout_summary.json`, `predictions.jsonl` / `holdout_predictions.jsonl`
(one row per question, every stage's response and score), `configs.json` (`config_id` -> stages).

## Question tagging

- `tag_questions.py`: LLM capability tags per question, writes `out/question_tags/*.jsonl`. One
  failing loader aborts the whole run with nothing written; check every dataset loads first.
- `canonicalise_tags.py`: raw -> canonical tag mapping (lexical normalisation, then LLM clustering
  until nothing merges), writes `data-claude/tag_mapping.json`. `--extend <mapping>` adds a new pool
  without changing existing entries.
- `tag_dataset.py`: applies the mapping and `TAG_MAPPING`, drops tags under 5 uses, saves the HF
  dataset.
- `scripts/build_pool.sh`: tag -> extend -> save for a new pool (`POOL`, `DATA`), writing
  `data-claude/tagged_<POOL>`; pass it to drivers as `DATASET=`.

## Sweep scripts

`scripts/` has three families from one template: `add*.sh` (heterogeneous, personas on),
`add*_noperspn.sh` (no `--multi_persona`), `ablation*.sh` (homogeneous, persona vs none). Each
hardcodes dataset, size, rounds, `AGENT_NUMS`, solvers, GPU list and port map, and runs a job pool
with `wait -n`. To add one, copy the nearest script and edit its header; do not parameterise them.

## Outputs (`main.py`)

`<out_dir>/history/<fname>.jsonl` is rewritten after every question, so a run can be inspected or
killed mid-way. A summary row is appended to `<out_dir>/<fname>.csv`. `fname` encodes the full
config and is the only record of how a file was made.

`.gitignore` excludes `*.json`, `*.jsonl`, `*.csv`, `*.npy`, `*.log`, `out/`, `history/`, `cache/`.
Results are never committed; local directories are the only copy.
