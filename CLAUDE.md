# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for paper *Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity*.
`README.md` has paper story, persona table, K\* definition, full flag list.

Collection of experiment entry points, not a library. No tests, no lint, no packaging, no `__init__.py`.

Two pipelines share `src/model` and `src/data`:

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

Two builders, both return `{name: {prompt, temperature, top_p, style, [nvidia_persona]}}`:

- `_build_enhanced_personas(args)` — paper personas, **selected by `args.data`**, one hand-written
  set per dataset, ~600 lines. Returns single no-op `{"None": {...}}` unless one of `multi_persona`
  / `baseline_a` / `baseline_b` set.
- `_build_chosen_personas(args)` — selects by name from `args.chosen_personas` (comma-separated) out
  of `chosen_persona_bank()`, a flat 50-persona bank: the union of the per-dataset sets plus the
  default set. Orchestrator path. `build_agent_pool()` renders that bank as the orchestrator's
  candidate list (`name`, `specialty`, `strengths`, `style`), derived from the bank rather than
  hand-maintained.

Three traps:

- `get_agents` dispatches on `getattr(args, 'chosen_agents', True)` — **defaults True**. New entry
  point lacking `chosen_agents` silently takes chosen-personas branch, then fails on missing
  `chosen_personas`. Set both explicitly.
- `_add_nvidia_personas` gated on `getattr(args, "persona_prompt", True)` — also **defaults True** —
  and reads `persona_data['nvidia_persona']` unconditionally. Only gsm8k set and chosen-persona pool
  define that key, so `--persona_prompt` on any other dataset raises KeyError.
- `_build_enhanced_personas` has two `elif args.data in ['truthfulqa']` branches. Second is dead.

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

`team_evaluation.py` — `run_team_evaluation` rebuilds args for `get_agents` (sets `chosen_agents`,
`chosen_personas`, `num_agents`), infers gsm8k-vs-MCQ from answer shape, one thread per agent,
results placed by index so responses stay aligned with agent names. `vllm_base_url` falls back to
`args.api_base_url` before the 8001 default. MCQ batches borrow `args.data = 'arc'` so
`get_instruction_suffix` asks for `(A)` rather than its numeric fallback, and gsm8k answers are
coerced to float because the tagged dataset stores them as strings. Returns accuracies **and** raw
counts (`per_agent_correct`, `per_agent_correct_by_tag`, `per_tag_counts`, `team_correct`).

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

Answer type is decided **per question** (`_infer_answer_type`), not per batch: a tag spans gsm8k and
MCQ sets, and judging a batch by its first answer silently mis-scores or drops the rest.

Outputs default under `data-claude/orchestrator/`: `agent_performance_by_tag.md`,
`agent_performance_state.json`, `run_records.jsonl` (one record per iteration),
`team_selection_results.csv` (header written once, not per row), plus `holdout_records.jsonl` and
`holdout_summary.json` when a test split is held out.

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

`data/data_utils.load_data` is an if/elif router to one module per dataset, each returning
`(questions, labels)` — shuffled `head(data_size)` for test splits. `truthfulqa` and `winogrande`
load `truthfulqa/truthful_qa` and `allenai/winogrande`; the bare ids they used before are rejected
by current `huggingface_hub`. Both remap `test` to `validation` internally. `base_ds.format_ds` is leftover
from an earlier perturbation study, references args (`reverse_landmark`, `synonym_replacement`, …)
no current entry point defines. Imported, unused.

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
