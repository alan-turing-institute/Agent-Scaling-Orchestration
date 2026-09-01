# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for the paper *Understanding Agent Scaling in LLM-Based Multi-Agent
Systems via Diversity*. Two lines of work live here:

1. **Debate/voting experiments** (`src/main.py`, `scripts/`, `src/analysis/k_star/`) — the
   published pipeline: N persona-guided agents answer a benchmark question, optionally
   debate for R rounds, and the answer is majority-voted. `src/analysis/k_star/` then measures
   semantic diversity (K\*/N\*) of the recorded responses via embedding eigenvalue entropy.
2. **Orchestrator experiments** (`src/train_orchestrator.py`, `src/orchestration/`,
   `src/tag_questions.py`, `src/team_evaluation.py`, `src/summariser.py`) — newer,
   uncommitted-in-parts work where an LLM orchestrator picks a 4-agent team per question
   batch from tag profiles, then learns from a markdown scoreboard of past results.

There are no tests, no linter config, and no build step. `README.md` documents pipeline (1)
in detail and is accurate for it; it does not cover pipeline (2).

## Environment and running

A venv lives in `env/` (Python 3.13, macOS). `requirements.txt` is a full pip freeze.

```bash
source env/bin/activate
```

Nothing is installed as a package: **`src/main.py` and `src/tag_questions.py` are run from
the repo root but import as if `src/` were the root** (`from model.model_utils import ...`).
`main.py` works because Python puts the script's own directory on `sys.path`;
`tag_questions.py` inserts it explicitly. `train_orchestrator.py` (`from orchestration...`,
`from team_evaluation import ...`) must therefore also be run as `python src/train_orchestrator.py`
from the repo root, never as a module.

Single experiment:

```bash
python src/main.py --data gsm8k --num_agents 4 --solver debate --debate_rounds 3 \
  --multi_persona --agent_models "gpt-4.1,o3-mini,DeepSeek-V3.2,claude-sonnet-4-6" \
  --data_size 100 --out_dir ./out --verbose
```

Sweeps: `bash scripts/add*.sh` (heterogeneous personas), `scripts/add*_noperspn.sh`
(no-persona control), `scripts/ablation*.sh` (homogeneous, persona on/off). These scripts
hard-code the dataset, model list, vLLM ports and GPU pool at the top — read and edit those
variables rather than passing arguments. Several (`scripts/add.sh`) reference
`MAX_PARALLEL_JOBS`/`GPU_LIST` that are commented out; they only run as-is if you restore
those definitions or the GPU scheduling block is dead code for that script.

Orchestrator loop (fixed 10 iterations, hard-coded in `__main__`):

```bash
python src/train_orchestrator.py --dataset_path data/tagged_dataset \
  --model_name <served-model> --api_base_url http://localhost:8001/v1 --solver vote
```

Question tagging, which produces the dataset that loop consumes:

```bash
python src/tag_questions.py --data gsm8k arc truthfulqa --split test --data_size 100 \
  --use_vllm --vllm_base_url http://127.0.0.1:8001/v1
```

Diversity analysis over recorded histories:

```bash
python src/analysis/k_star/analysis.py <out_dir>/history --mode round_agent_avg --out-dir <results>
```

`scripts/k_star/analysis.sh` batch-runs this across a hard-coded `DIRS` list of experiment
output directories on multiple GPUs; it expects an `NV-Embed-v2`-class GPU.

## Architecture

### Model dispatch (`src/model/model_utils.py`, ~1450 lines)

`get_agents(args)` is the single factory. It resolves each entry of `--agent_models`
(comma-separated; falls back to `--model`) to a wrapper, in this precedence order:

1. `OPENAI_COMPAT_MODELS` or (`OPENAI_BASE_URL` set and model in `AZURE_OPENAI_MODELS`)
   → `OpenAICompatChatWrapper`
2. `AZURE_OPENAI_MODELS` → `AzureOpenAIWrapper`
3. `--use_vllm` → `OpenAICompatChatWrapper` against `--vllm_base_urls[i % len]`
4. local HuggingFace `LlamaWrapper` / `QwenWrapper` via `model_dirs`

Adding a model means adding it to the right set/dict here — nowhere else.
`--vllm_base_urls` is round-robined across agents by index, which is how heterogeneous runs
map agent *i* to the vLLM server hosting its model; keep the URL order aligned with
`--agent_models`.

`engine(messages, agent, num_agents, persona_configs)` is the only call site for generation.
It broadcasts one message per agent, applies each agent's per-persona sampling config
(`temperature`, `top_p`, `max_new_tokens`), and returns OpenAI-shaped response objects — all
call sites read `resp.choices[0].message.content`.

### Personas

Two mutually exclusive persona builders, both returning `{name: {prompt, temperature, top_p, style, nvidia_persona?}}`:

- `_build_enhanced_personas(args)` — the experiment path. Persona sets are **branched on
  `args.data`** (gsm8k / pro_medicine / formal_logic / truthfulqa / arc / winogrande /
  humaneval+mbpp / piqa). A new dataset needs a new branch or agents silently fall through
  with no persona.
- `_build_chosen_personas(args)` — the orchestrator path. Holds one flat `all_personas` dict
  and returns only the names in `args.chosen_personas`; an unknown name raises `KeyError`.

Traps in `get_agents`: it selects the builder with `getattr(args, 'chosen_agents', True)`,
and `_build_enhanced_personas` appends NVIDIA persona blocks under
`getattr(args, "persona_prompt", True)`. Both default to *True* when the attribute is
missing, while `main.py`'s argparse defaults them to *False*. Any hand-built `args`
namespace (as in `team_evaluation.run_team_evaluation`) must set both explicitly.

Note there are two `elif args.data in ['truthfulqa']` branches in `_build_enhanced_personas`;
the second is unreachable.

### Debate loop (`src/main.py`)

Per question: build round-0 messages (mode chosen by `--multi_persona` / `--baseline_a`
length-matched neutral padding / `--baseline_b` single shared persona / plain), call `engine`,
evaluate, then for each of `--debate_rounds` call `get_new_message` to fold peers' answers
back in and call `engine` again.

Agent identity is encoded in a **string key**, not an object:
`{data}_{data_size}__{model}__{persona}__Agent{i}`. `get_new_message` recovers the persona by
`agent.split("__")[-2]`. Do not change this format without updating every split site.

Topology is chosen by flags: default full mesh, `--sparse` (previous/next neighbour only),
`--centralized` (agent 0 is the hub; only agent 0's answer is scored).

In the decentralized branch, peers' opinions are passed through `extract_number(...)` — peers
see only the extracted final answer, whereas the centralized branch passes full response text.
This asymmetry is deliberate in the current code.

### Evaluation (`src/evaluator.py`)

`get_instruction_suffix(args)` appends the answer-format instruction (`{final answer: 123}`
or `{final answer: (A)}`) that the extractors depend on — the suffix and the parser are a
matched pair, so change them together. `evaluate_gsm8k` / `evaluate_mcq` take
`{agent_name: response}` and return `(per_agent_final_answers, majority_answer, is_correct)`.
`--bae` switches to the looser `base_evaluate_*` parsers.

### Datasets (`src/data/`)

`load_data(args, split)` in `data_utils.py` is a dispatch chain on the `--data` string;
each loader returns `(X, Y)` lists. Local HF dataset caches live in `data/`.

### Orchestrator loop

`tag_questions.py` (async, `--data` accepts several datasets) writes per-question tag JSONL
to `out/question_tags/`; `tag_dataset.py` turns that into `data/tagged_dataset` (HF
`save_to_disk`). Each iteration of `train_orchestrator.py` then:

`team_selection` (samples a tag, gathers its questions, asks `OrchestratorAgent.select_team`
for 4 personas + reasoning as JSON) → `run_team_evaluation` (builds those personas via
`get_agents`, runs them in a thread pool, votes, computes per-agent and per-tag accuracy) →
`save_evaluation_summary_with_llm` (rewrites `out/agent_performance_by_tag.md`).

That markdown file is the loop's only memory: it is read back in as `prior_md` on the next
iteration, so the orchestrator's "learning" is entirely an LLM-maintained scoreboard.
`run_team_evaluation` hard-codes `use_vllm=True` and `vllm_base_url=http://127.0.0.1:8001/v1`,
overriding whatever the caller passed.

## Outputs and conventions

- Per-question histories: `{out_dir}/history/{fname}.jsonl` — one JSON record per question
  holding every round's messages, raw responses, per-agent answers and correctness. This is
  the input to `src/analysis/k_star/`, so preserve its shape.
- Accuracy summary appended to `{out_dir}/{fname}.csv` (README says `.tsv`; the code writes
  `.csv`).
- `fname` encodes the config: `{data}_{solver}_{data_size}__{models}_N={agents}_R={rounds}`
  plus `_SPARSE`/`_CENTRAL`/`_BAE`/`_HETERO_ENHANCED`/`_PERSONA_PROMPT`/`_CHOSEN`/
  `_BASEA_LENMATCH`/`_BASEB_SINGLEP`. Downstream analysis scripts parse these suffixes.
- `.gitignore` excludes `out/`, `out-baseline/`, `results/` and `**/history/`, so experiment
  results are deliberately untracked. The ignore rules are scoped to those directories on
  purpose: an earlier repo-wide `*.json` / `*.csv` pattern silently made
  `data/tagged_dataset/` uncommittable. Do not reintroduce bare extension globs.

## Layout rules

Python goes under `src/` — nothing executable at the repository root. Shell runners go under
`scripts/`. Reference inputs go under `configs/` (currently `personas.json`). Generated output
goes under `results/`; `out/` and `out-baseline/` predate that convention and still hold the
debate-pipeline results, and `--out_dir` still defaults to `out/`.

`src/analysis/` holds post-hoc scripts that are not part of a run. They resolve their own paths
from `__file__` rather than the working directory, so they can be invoked from anywhere.

## Credentials

Read from the environment: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY_ENV` (this holds
the key itself, not the name of another variable, despite the `_ENV` suffix),
`AZURE_OPENAI_API_VERSION`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `VLLM_BASE_URL(S)`,
`VLLM_API_KEY`.
