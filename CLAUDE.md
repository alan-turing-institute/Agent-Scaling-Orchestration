# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for paper *Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity*.
`README.md` has paper story, persona table, K\* definition, full flag list.

Collection of experiment entry points, not a library. No lint, no packaging, no `__init__.py`.
Offline tests in `tests/` run directly as scripts.

Two pipelines share `src/model` and `src/benchmarks`:

1. **Benchmark harness** — `src/main.py`. N agents answer, optionally debate R rounds, majority-vote.
   Sweeps via `scripts/*.sh`. Histories then scored by `K_star_analysis/`.
2. **Orchestrator loop** — `src/train_orchestrator.py`. LLM orchestrator picks 4-agent team per
   batch from question tags, gets evaluated, reads own past results back as markdown memo next
   iteration. Newer work, see `git log`.

## Running

Invoke by path from repo root. Modules import `from model.model_utils import ...`, works only
because Python puts script dir (`src/`) on `sys.path`. `python -m src.main` fails.

```bash
pip install -r requirements.txt          # fully pinned, heavy Azure SDK tail

python src/main.py --data gsm8k --num_agents 5 --model qwen2.5-7b --solver vote --debate_rounds 0

bash scripts/add.sh                      # sweep: GPU scheduler + vLLM endpoints baked in

python src/train_orchestrator.py --api_base_url http://localhost:8001/v1 \
    --model_name Qwen/Qwen3.6-35B-A3B --dataset_path data/tagged_dataset

python K_star_analysis/analysis.py out/history/ --mode round_agent_avg --out-dir analysis/results/
python K_star_analysis/analysis_improved.py out/history/    # cache-only, no GPU
```

Nearly everything hits an OpenAI-compatible endpoint, not in-process weights. Sibling `../vllm/`
serves those endpoints.

## Architecture

### Generation chokepoint

`engine(messages, agent, num_agents, persona_configs)` in `src/model/model_utils.py` produces all
text. Every caller goes through it. `agent` is one wrapper or a *list* (heterogeneous), indexed
`i % len(agents)` against messages — message order **is** agent order everywhere.

`get_agents()` picks wrappers, identified by `kind` attribute, no base class:

- `kind in {'azure_openai', 'openai_compat'}` → `.complete(messages, ...)`. `OpenAICompatChatWrapper`
  returns **string**. `AzureOpenAIWrapper` returns **raw response object**; retries with doubled
  `max_tokens` on "Insufficient tokens", counts content filtering.
- no `kind` → local HF (`LlamaWrapper`/`QwenWrapper`), returns string.

`main.py` then calls `resp.choices[0].message.content` and `resp.model_dump_json()`, so works only
with response objects. String returns (local HF, `openai_compat`) need shape normalising — reuse
`team_evaluation._response_text()`, handles all three.

`_make_agent` routing: `OPENAI_COMPAT_MODELS` / `AZURE_OPENAI_MODELS` checked before `use_vllm`.
`model_dirs` maps 3 local checkpoints only. Keys sweep scripts use (`mistral-7b`, `qwen3-8b`) exist
**only** under `--use_vllm`; without it, `invalid model key`. With `--vllm_base_urls` (plural,
comma-separated) agent *i* pins to URL *i % len(urls)* — that is how one run spans several
single-model servers.

### Agent names carry state

Built in `main.py`:

```
{data}_{data_size}__{model_key}__{persona_name}__Agent{i+1}
```

`get_new_message()` recovers persona by `agent.split("__")[-2]`, model by `[-3]`. Name also is dict
key in every history record, and `K_star_analysis` parses it back out. Change format, break debate
rounds + saved histories + analysis together.

### Personas

All in `src/personas.py`. Every persona is defined once, in `chosen_persona_bank()`: a flat bank of
50, each `{prompt, temperature, top_p, style, [nvidia_persona]}`. Each call returns fresh dicts, so a
caller may edit them. Everything else selects from that bank:

- `_build_enhanced_personas(args)` — the paper's set for `args.data`. A registered benchmark names
  it in its module's `PERSONA_SET` (`src/benchmarks/`); `_UNREGISTERED_PERSONA_SETS` covers
  `--data` values with no module yet (humaneval, mbpp, piqa, arc_easy), and anything else gets
  `_DEFAULT_PERSONA_SET`. Returns a single no-op `{"None": {...}}` unless one of `multi_persona` /
  `baseline_a` / `baseline_b` is set. The paper used `Elimination_Specialist` for two different
  personas; the bank carries the winogrande one as `Elimination_Based_Solver` and `_PAPER_NAMES`
  renames it back under winogrande, so old histories still match.
- `_build_chosen_personas(args)` — selects by name from `args.chosen_personas` (comma-separated).
  Orchestrator path. `build_agent_pool()` renders the bank as the orchestrator's candidate list
  (`name`, `specialty`, `strengths`, `style`), derived from the bank rather than hand-maintained.

Role instructions (solver, critic, hub, ...) are a separate layer in `roles.py`, appended after
the persona prompt.

Two traps:

- `get_agents` (`model_utils.py`) dispatches on `getattr(args, 'chosen_agents', True)` —
  **defaults True**. New entry point lacking `chosen_agents` silently takes chosen-personas branch,
  then fails on missing `chosen_personas`. Set both explicitly.
- `_add_nvidia_personas` appends the `nvidia_persona` block under `--persona_prompt`. Only the five
  gsm8k personas define one; the rest are left unchanged.

Per-persona `temperature`/`top_p` reach the model as `persona_configs`, passed alongside `messages`
into `engine`, built by `get_persona_config`. Personas cycle `i % len(personas)` when `num_agents`
exceeds persona count — how N=8/12/16 sweeps work.

### Debate loop, known quirks

`main.py` builds round-0 messages per mode (`multi_persona` / `baseline_a` / `baseline_b` / plain),
then loops `debate_rounds` times through `get_new_message`. Topologies: decentralized full mesh
(default), `--sparse` (ring, previous + next agent only), `--centralized` (agent 0 is hub, and
evaluation then scores *only* agent 0).

Decentralized branch injects peer opinions as `extract_number(responses[other_agent])` — bare number,
not peer text — for every dataset. On MCQ that means peers exchange whatever digit appeared last.
Centralized branch passes full text. Asymmetry is pre-existing and published results depend on it;
do not "fix" silently.

`baseline_a` (length-matched digit padding from `_build_length_match_pad`) and `baseline_b` (all
agents share persona #1) are mutually exclusive with each other and with `multi_persona`. `main`
raises on violation.

### Answer extraction

`src/evaluator.py` owns prompt/parse contract. `get_instruction_suffix` appends required format
(`{final answer: 123}` or `{final answer: (A)}`); `--cot` adds step-by-step, `--bae` switches to
looser format plus `base_evaluate_*` parsers. Each evaluator returns
`(per_agent_answers, majority_answer, majority_is_correct)`, ties broken by `random.choice`.

Extraction is loose and mismatched between variants: `evaluate_gsm8k` takes *last number anywhere*,
ignoring the braces it asked for, while unused `_evaluate_gsm8k` parses braces. `evaluate_mcq`
indexes brace contents positionally (`pred[0]` or `pred[1]`) to recover the letter. Change any of
these, change every reported number.

### Orchestrator loop

`train_orchestrator.py` (`--iterations`, default 10) wires:

`orchestration/orchestrator.py` — `OrchestratorAgent` holds own `OpenAI` client, separate from
`model_utils` wrappers. Asks strict JSON `{selected_agents, reasoning}`, tolerates markdown fences.
`sample_tag_questions` picks random tag, samples questions carrying it; `team_selection` turns that
into tag-frequency prompt. Selected names **are** validated against the pool; invalid ones are
dropped and the selection retried once with the rejected names quoted back, so a hallucinated name
no longer reaches `_build_chosen_personas` as a KeyError. A parse failure returns an empty team and
the iteration is skipped and recorded, rather than falling back to agents that never existed.

`team_evaluation.py` — `run_team_evaluation(selected_team, questions, args, config=None)` runs a
team on each question (questions in parallel, `--eval_workers`) and adds up the results. With only
`selected_team` it runs `team_config.vote(selected_team)`; pass a `TeamConfig` for anything else.
Answer type comes from the question's `dataset` column via the benchmark registry, per question
(`benchmarks.answer_type_of_sample`). A question with no `dataset`, or naming an unregistered
benchmark, raises before the batch's first call; nothing guesses from the answer's shape. Entry
points build scorers lazily through `benchmarks.ScorerSet`, so a benchmark with a new answer type
needs a scorer in `SCORERS` and its module, and no edit to any entry point.
Returns accuracies **and** raw counts (`per_agent_correct`, `per_agent_correct_by_tag`,
`per_tag_counts`, `team_correct`) keyed by **stage id** — for a vote the stage ids are the persona
names, so the scoreboard keys are unchanged — plus `config_id`, call and token totals, and
`samples` (per-question detail; callers write it with `predictions.save_report` and drop it).

### Team configs and the runner

The orchestrator path no longer goes through `engine`/`get_agents`. Four modules replace that:

- `team_config.py` — `TeamConfig`: an ordered tuple of `Stage(id, persona, role, model, inputs,
  max_tokens)`, each reading only earlier stages via `Input(source, handoff)`, plus an aggregate
  (`vote` over `voters()`, or `stage` = the `final` stage's answer). Builders: `vote`, `debate`,
  `centralized`, `synthesis`, `pipeline`, `parallel`, and `split_budget` for matched compute.
  `config_id` hashes everything behavioural, including the role-template version.
- `roles.py` — role templates (`solver`, `debater`, `critic`, `reviser`, `planner`, `checker`,
  `hub`, `synthesiser`) and handoff rendering (`answer`, `rationale` = last 600 chars, `full`).
  A solver with no inputs gets the exact legacy prompt (persona, blank line, question + suffix);
  bump `ROLE_TEMPLATES_VERSION` on any template change.
- `runner.py` — `run_question` executes a config layer by layer (stages in a layer run together),
  scores **every** stage, records each stage's transition from the stage it read (`kept_right`,
  `fixed`, `broke`, `kept_wrong`) and whether it copied that answer, and aggregates. Ties break with
  a per-question seeded RNG by default (`--tie_break global` restores the old global-RNG
  behaviour). A per-request `seed` is sent unless `--request_seed -1`. Connection/timeout errors
  propagate to the caller's retry; other stage failures score as unanswered. `add_runner_args`
  holds the shared flags (`--models_file`, `--parse_mode`, `--tie_break`, `--request_seed`,
  `--max_inflight`); `add_topology_args` the team-arrangement flags the training loop takes.
- `model/registry.py` — model key -> served name and endpoint. `default` is `--model_name` at
  `--vllm_base_url`/`--api_base_url`/8001; `--models_file` adds more (see
  `configs/models.example.json`). `OpenAICompatChatWrapper.generate` returns a `Completion` with
  tokens, finish reason, latency and reasoning; `complete` still returns the text.

A team chosen by name (orchestrator or random selector) is arranged by `--topology`
(`team_config.for_team`): `vote` (default), `debate --rounds N`, `centralized`/`synthesis` (the
**last** selected agent leads, the rest work), `pipeline --roles ...` (names take roles in
order). Each persona is credited with its **last** stage's answer (`credited_stage` in the
report), so the scoreboard keeps persona keys whatever the topology.

**Serving noise.** The vLLM servers return different greedy text for the same request depending
on what else is in the batch: two runs of the same arm at `--eval_workers 5` flip ~20% of agent
answers and ~15% of team verdicts. Not caused by speculative decoding (tested), and vLLM's
batch-invariant mode does not support Qwen3.5's GDN layers. `--max_inflight 1` (per-server
semaphore in `model/registry.py`) runs every request alone: two runs of the same 80 calls gave
80/80 identical responses, cold prefix cache included. It costs ~2.5-3x the wall-clock (0.8B:
~5.8 s per call, so a 560-call arm takes ~55 min instead of ~20). It only governs this process: never run two arms against one server at once.

`scripts/run_config.py` runs one fixed config over the held-out split in the same batches as the
orchestrator arms. `scripts/rescore.py` re-scores a predictions file under another parser without
calling a model. `scripts/compare_runs.py` pairs arms question by question across folds and
repeats (McNemar / sign-flip, Holm-corrected) and prints each arm's repeat-to-repeat flip rate. Offline tests: `PYTHONPATH=src python tests/test_runner.py` and
`tests/test_scorers.py` (no pytest in the pinned env).

`summariser.py` — two modes. `save_evaluation_summary` (default, `--summariser counts`) folds those
counts into `agent_performance_state.json` and renders the markdown from it: agent totals, per-tag
accuracy with `correct/seen`, recent selections, untried agents. `save_evaluation_summary_with_llm`
(`--summariser llm`) is the original behaviour, the model rewriting the whole markdown each
iteration. Either way **the markdown is the loop's memory**, read back as `prior_md`; delete it and
the state file to reset, a stale pair contaminates a new experiment.

`holdout_evaluation.py` — with `--test_fraction`, a split is held out before training and scored
after it with the scoreboard frozen: the split is shuffled and chunked (`--split_seed`,
`--test_batch_size`), each batch's tag profile drives one selection, every test question is answered
exactly once. `--random_baseline` scores a randomly drawn team on the same batches for reference.

Answer type is decided **per question** (`answer_type_of`, from the `dataset` column), not per batch: a tag spans gsm8k and
MCQ sets, and judging a batch by its first answer silently mis-scores or drops the rest.

Outputs default under `data-claude/orchestrator/`: `agent_performance_by_tag.md`,
`agent_performance_state.json`, `run_records.jsonl` (one record per iteration),
`team_selection_results.csv` (header written once, not per row), plus `holdout_records.jsonl` and
`holdout_summary.json` when a test split is held out. `predictions.jsonl` (training) and
`holdout_predictions.jsonl` hold one row per question (schema 2: every stage's response, answer,
parsed flag, correctness, transition, tokens), and `configs.json` maps each `config_id` to its
stages.

### Question tagging

`tag_questions.py` asks an LLM for capability tags per question across all 7 datasets (asyncio,
64-way semaphore), writes `out/question_tags/*.jsonl` (`--out_dir`). One failing loader aborts the
whole `asyncio.gather`, so nothing is written — check every dataset loads before a long run.

`canonicalise_tags.py` builds the raw-tag → canonical-tag mapping from the data: lexical
normalisation (case, punctuation, plurals, word order, filler) then LLM clustering in batches,
re-clustering the batch canonicals until a round merges nothing. Writes `data-claude/tag_mapping.json`.

`tag_dataset.py` applies that mapping, then the hand-written `TAG_MAPPING`, drops tags under 5
occurrences, saves the HF dataset. Now argparse behind a `main()` guard (`--tags_file`,
`--tag_mapping`, `--out_dir`, `--threshold`, `--plot_path`) — importing it no longer runs a pipeline.

### Data layer

`src/benchmarks/` is a registry with one module per dataset. Each module declares `NAME` (the
`--data` value and the tagged dataset's `dataset` column), `ANSWER_TYPE` (a key into
`benchmarks/scorers.py`'s `SCORERS`), `PERSONA_SET` (the paper's per-dataset personas) and
`load(args, split) -> (questions, labels)`, which returns a shuffled `head(data_size)` for test
splits. Adding a benchmark means one module plus one line in `_MODULES` in
`benchmarks/__init__.py`; the tagger, splits and baselines read the registry. Look one up with
`benchmarks.get(name)`; `benchmarks.scorer_for(name)` builds its scorer. `truthfulqa` and
`winogrande` load `truthfulqa/truthful_qa` and `allenai/winogrande`, because current
`huggingface_hub` rejects the bare ids they used before. Both remap `test` to `validation`
internally.

### K\* analysis

Consumes `out/history/*.jsonl` from `main.py`, outputs CSV summaries.

`analysis.py` embeds responses with `nvidia/NV-Embed-v2`, computes N\* = exp(H) over normalized
covariance eigenvalues, 3 modes (`round_agent_avg`, `per_question_agent`, `round_cum_text`). Forces
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` **at import**, so embedding model must already be in
HF cache (`--cache-folder`). Embeddings cached append-only as `cache/<stem>.npy` + `.json` beside
each input JSONL, keyed by encoding signature.

`analysis_improved.py` (N\*_conditioned, N\*_weighted, ΔN\*) reads only those caches, no GPU.
`exp2_embedding_robustness.py` re-runs metric under other embedding models.

`analysis.sh` and `analysis_improved.sh` both iterate a hardcoded `DIRS` list of ~36 experiment
directories expected *inside* `K_star_analysis/`, each holding `history/` or `<config>/history/`,
skipping work when output CSV exists.

## Experiment scripts

`scripts/` holds 3 families over one template: `add*.sh` (heterogeneous, personas on),
`add*_noperspn.sh` (same minus `--multi_persona`), `ablation*.sh` (homogeneous, persona vs
no-persona per model). Each hardcodes dataset, data size, rounds, `AGENT_NUMS=(2 4 8 12 16)`, both
solvers, GPU list with jobs-per-GPU cap, own vLLM port map, then hand-rolls a job pool with
`wait -n`. They set `CUDA_VISIBLE_DEVICES` even for vLLM runs — inert, generation is remote. Output
to `./<data>_<config>/` with logs alongside. Copy nearest script and edit its header block; do not
parameterise them.

## Outputs

`main.py` writes `<out_dir>/history/<fname>.jsonl`, rewritten in full after **every** question, so a
run is inspectable or killable mid-flight. Appends summary row to `<out_dir>/<fname>.csv`. `fname`
encodes full config (data, solver, size, models, N, R, plus `_SPARSE`/`_CENTRAL`/`_HETERO_ENHANCED`/
`_BASEA_LENMATCH`/…) and is the only record of how a file was produced.

`.gitignore` excludes `*.json`, `*.jsonl`, `*.csv`, `*.npy`, `*.log`, `out/`, `history/`, `cache/`.
No result ever commits, so local result directories are the only copy.
