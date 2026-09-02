# Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity

This repository provides the codebase for studying **how scaling the number of heterogeneous LLM agents with diverse reasoning personas improves collective performance** through debate and voting mechanisms. We introduce the **K\* metric** (effective diversity) based on embedding eigenvalue entropy to quantify semantic diversity among agents, and show that persona-guided multi-agent collaboration yields consistent gains across reasoning benchmarks.

## What is in this repository

Two pipelines share the same agent, persona and evaluation code:

1. **Debate and voting** (`src/main.py`, `scripts/`, `src/analysis/k_star/`) — the
   published pipeline. N persona-guided agents answer a benchmark question,
   optionally debate for R rounds, and a majority vote decides. The K\* analysis
   then measures the semantic diversity of the recorded responses.
2. **Orchestration** (`src/train_orchestrator.py`, `src/orchestration/`) — an LLM
   orchestrator picks a team of agents per batch of questions from their topic
   tags, the team is scored, and the result is written back as a natural-language
   scoreboard the orchestrator reads before its next choice.

Working notes on the architecture, and the traps worth knowing before editing,
are in [CLAUDE.md](CLAUDE.md).

## Setup

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

Every command below is run **from the repository root**. Scripts under `src/`
import as though `src/` were the root (`from model.model_utils import ...`), so
`python src/main.py` works while `python -m src.main` does not.

Credentials are read from the environment: `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_KEY_ENV` (this holds the key itself), `OPENAI_API_KEY`,
`OPENAI_BASE_URL`, `VLLM_BASE_URLS`, `VLLM_API_KEY`.

## The orchestration pipeline

Three stages, run in order. Stages 1 and 2 produce the tagged dataset; stage 3 is
the loop under active development. Full flags for each are in
[Runnable scripts](#runnable-scripts).

**1. Tag the questions.** Labels each question with short topic tags.

```bash
python src/tag_questions.py \
    --data gsm8k arc hellaswag truthfulqa winogrande pro_medicine formal_logic \
    --split test --data_size 100 \
    --use_vllm --vllm_base_url http://127.0.0.1:8001/v1
```

Writes `out/question_tags/<datasets>_<split>_<size>_tags.jsonl`.

**2. Build the tagged dataset.** Unifies tag synonyms, drops tags appearing fewer
than five times, and saves a HuggingFace dataset.

```bash
python src/tag_dataset.py
```

Writes `data/tagged_dataset/` — 699 questions across seven benchmarks, and the
only stage output that is version-controlled.

**3. Run the orchestrator loop.**

```bash
python src/train_orchestrator.py \
    --dataset_path data/tagged_dataset \
    --model_name <served-model> \
    --api_base_url http://localhost:8001/v1 \
    --solver vote
```

Each iteration samples a tag, asks the orchestrator to pick four agents for the
questions carrying it, scores that team, and rewrites
`out/agent_performance_by_tag.md` — the scoreboard that is fed back on the next
iteration as the loop's only memory.

> **Status.** Stage 3 is not yet ready for experiments. Known issues include the
> agent endpoint being hard-coded rather than taken from `--api_base_url`,
> `--solver debate` being accepted but ignored, no seeding, and no
> machine-readable run record. Treat any numbers it prints as provisional.

## Runnable scripts

Every Python entry point is run from the repository root and takes `--help`.
Generation defaults (`--max_new_tokens`, `--temperature`, `--top_p`,
`--thinking_token_budget`) come from `src/defaults.py`, which exists to supply
argparse defaults — pass the flag to change behaviour.

### `src/main.py` — debate and voting runner

Runs N persona-guided agents over a benchmark, optionally debates for R rounds,
and aggregates by majority vote. Writes per-question histories to
`{out_dir}/history/{name}.jsonl` and an accuracy summary to `{out_dir}/{name}.csv`.

```bash
python src/main.py --data gsm8k --num_agents 4 --solver debate --debate_rounds 3 \
    --multi_persona --agent_models "gpt-4.1,o3-mini,DeepSeek-V3.2,claude-sonnet-4-6" \
    --data_size 100 --out_dir ./out --verbose
```

**Data**

| Flag | Default | Purpose |
|---|---|---|
| `--data` | *(required)* | `gsm8k`, `arc`, `hellaswag`, `truthfulqa`, `winogrande`, `pro_medicine`, `formal_logic` |
| `--data_size` | `0` | Number of questions; `0` means all |
| `--split` | `train` | Dataset split |
| `--data_dir` | `./data/` | Where benchmark caches live |
| `--out_dir` | `out/` | Where results are written |
| `--seed` | `42` | RNG seed |
| `--comment` | `""` | Free-text label recorded in the summary row |

**Agents and personas**

| Flag | Default | Purpose |
|---|---|---|
| `--num_agents` | `5` | Team size |
| `--multi_persona` | off | Give each agent a distinct persona (the heterogeneous condition) |
| `--persona_prompt` | off | Append the NVIDIA persona blocks. Only GSM8K personas define them |
| `--baseline_a` | off | Length-matched neutral padding instead of personas |
| `--baseline_b` | off | One shared persona for all agents |
| `--chosen_agents` | off | Use an explicit persona list rather than the per-dataset set |
| `--chosen_personas` | `none` | Comma-separated persona names, used with `--chosen_agents` |

`--multi_persona` is mutually exclusive with both baselines, and the two
baselines are mutually exclusive with each other.

**Models and backends**

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `llama3.1` | Single model for a homogeneous run |
| `--agent_models` | `""` | Comma-separated per-agent models for a heterogeneous run |
| `--use_vllm` | off | Route through vLLM |
| `--vllm_base_urls` | `$VLLM_BASE_URLS` | Comma-separated; agent *i* uses URL *i*, so order must match `--agent_models` |
| `--vllm_base_url` | `$VLLM_BASE_URL` | Single-server fallback |
| `--azure_endpoint` / `--azure_api_key_env` / `--azure_api_version` | env | Azure OpenAI |
| `--openai_api_key` / `--openai_base_url` | env | OpenAI-compatible endpoints |
| `--load_in_4bit` / `--load_in_8bit` | off | bitsandbytes quantisation, local HuggingFace models only |

**Generation**

| Flag | Default | Purpose |
|---|---|---|
| `--max_new_tokens` | `512` | Response length cap |
| `--temperature` | `1.0` | Overridden per agent by persona config |
| `--top_p` | `0.9` | Overridden per agent by persona config |
| `--thinking_token_budget` | `1024` | Reasoning budget; sent as `extra_body`, so endpoints that do not accept it will reject the request |

**Solver and topology**

| Flag | Default | Purpose |
|---|---|---|
| `--solver` | `vote` | `vote` or `debate` |
| `--debate_rounds` | `5` | Refinement rounds after round 0 |
| `--sparse` | off | Each agent sees only its two neighbours |
| `--centralized` | off | Agent 0 is the hub, and only its answer is scored |
| `--bae` | off | Looser answer parsing, no braces required |
| `--cot` | off | Append a chain-of-thought instruction |
| `--verbose` | off | Print per-agent assignments and sampling configs |

### `src/tag_questions.py` — orchestration stage 1

Labels each benchmark question with short topic tags using an LLM. Async, with
one request per question. Writes
`{out_dir}/<datasets>_<split>_<size>_tags.jsonl`.

```bash
python src/tag_questions.py \
    --data gsm8k arc hellaswag truthfulqa winogrande pro_medicine formal_logic \
    --split test --data_size 100 \
    --use_vllm --vllm_base_url http://127.0.0.1:8001/v1
```

| Flag | Default | Purpose |
|---|---|---|
| `--data` | all supported | Space-separated dataset list |
| `--split` | `test` | Dataset split |
| `--data_size` | `0` | Questions per dataset; `0` means all |
| `--out_dir` | `out/question_tags` | Output directory |
| `--output_file` | derived | Override the generated filename |
| `--model` | `Qwen/Qwen3.6-35B-A3B` | Tagging model |
| `--max_new_tokens` | `512` | |
| `--temperature` | `0.0` | Deliberately deterministic for tagging |
| `--top_p` | `0.9` | |
| `--thinking_token_budget` | `1024` | |
| `--use_vllm`, `--vllm_base_url`, `--azure_*`, `--openai_*` | env | Backend selection, as in `main.py` |

### `src/tag_dataset.py` — orchestration stage 2

Unifies tag synonyms through a fixed mapping, drops rare tags, and saves a
HuggingFace dataset. This is the orchestrator's input and the only pipeline
output that is version-controlled.

```bash
python src/tag_dataset.py
```

| Flag | Default | Purpose |
|---|---|---|
| `--tags_file` | the 7-dataset, 100-per-dataset run | Tag JSONL from stage 1 |
| `--out_dir` | `data/tagged_dataset` | Where to `save_to_disk` |
| `--min_tag_count` | `5` | Drop tags appearing fewer times than this |

### `src/train_orchestrator.py` — orchestration stage 3

Each iteration samples a tag, asks the orchestrator to pick a team for the
questions carrying it, scores that team, and rewrites the per-tag scoreboard
that is fed back on the next iteration.

```bash
python src/train_orchestrator.py \
    --dataset_path data/tagged_dataset \
    --model_name <served-model> \
    --api_base_url http://localhost:8001/v1 \
    --iterations 10 --solver vote
```

**Loop and data**

| Flag | Default | Purpose |
|---|---|---|
| `--dataset_path` | `data/tagged_dataset` | Tagged dataset from stage 2 |
| `--iterations` | `10` | Team-selection rounds |
| `--num_samples` | `5` | Questions sampled per round |
| `--solver` | `vote` | **Currently ignored — the team is always majority-voted** |
| `--md_file` | `out/agent_performance_by_tag.md` | The scoreboard |
| `--output_path` | `out/orchestrator_results.json` | **Currently never written** |
| `--debug` | off | Print the orchestrator prompt, its response, and every agent answer |

**Endpoint**

| Flag | Default | Purpose |
|---|---|---|
| `--api_base_url` | `http://localhost:8001/v1` | Orchestrator endpoint. **The selected agents ignore this and always use port 8001** |
| `--api_key` | `none` | |
| `--model_name` | `Qwen/Qwen3.6-35B-A3B` | Used for the orchestrator, the summariser and the agents |

**Generation**

| Flag | Default | Applies to |
|---|---|---|
| `--max_new_tokens` / `--temperature` / `--top_p` | `512` / `1.0` / `0.9` | The agents in the selected team |
| `--orchestrator_max_tokens` / `--orchestrator_temperature` / `--orchestrator_top_p` | `4096` / `0.1` / `0.5` | The team-selection call |
| `--summariser_max_tokens` / `--summariser_temperature` | `8192` / `0.5` | The scoreboard rewrite |
| `--summariser_max_attempts` | `3` | Retries before the run fails and the scoreboard is left unchanged |
| `--thinking_token_budget` | `1024` | All three |

### `src/orchestrator_pool_sweep.py` — agent-pool selection probe

Asks a model to pick a team from a shuffled pool, repeatedly, per task, at three
levels of agent description. Writes
`results/agent-selection/{model}-agents={n}-{detail}-chosen-agents.json` and a
matching `-responses.json`. Needs `AZURE_OPENAI_ENDPOINT` and
`AZURE_OPENAI_API_KEY_ENV`.

```bash
python src/orchestrator_pool_sweep.py --model_name gpt-4.1 --num_agents 4 \
    --agent_detail name --repeats 10
```

| Flag | Default | Purpose |
|---|---|---|
| `--model_name` | `gpt-4.1` | Selecting model |
| `--num_agents` | `4` | Team size to request |
| `--agent_detail` | *(required)* | `name`, `single` or `full` — how much the pool description says about each agent |
| `--repeats` | `10` | Selections sampled per task |
| `--max_completion_tokens` | `1024` | |
| `--temperature` / `--top_p` | `1.0` / `0.9` | |

### `src/agent_selection_sweep.py` — replay selected teams

Reads a chosen-agents file from the sweep above and launches one `src/main.py`
run per selected team, in parallel.

```bash
python src/agent_selection_sweep.py --agent_type chosen --choose_type name \
    --agent_models ministral-3b --max_workers 10
```

| Flag | Default | Purpose |
|---|---|---|
| `--agent_type` | `default` | `chosen` passes the selected personas through to `main.py`; `default` runs the standard per-dataset set |
| `--choose_type` | `name` | Which agent-description variant's selections to replay |
| `--selection_model` | `gpt-4.1` | Whose chosen-agents file to read |
| `--agent_models` | `ministral-3b` | Model the agents run on |
| `--num_agents` / `--data_size` / `--max_new_tokens` | `4` / `100` / `512` | Passed to each run |
| `--solver` / `--debate_rounds` | `vote` / `0` | Passed to each run |
| `--max_workers` | `10` | Concurrent `main.py` processes |

### `src/analysis/k_star/analysis.py` — K\* effective diversity

Embeds recorded agent responses and computes N\* = exp(H) over the normalised
eigenvalue spectrum. Reads the `history/*.jsonl` written by `main.py`.

```bash
python src/analysis/k_star/analysis.py out/history \
    --mode round_agent_avg --out-dir results/k_star
```

| Flag | Default | Purpose |
|---|---|---|
| `jsonl_path` | *(positional)* | A `.jsonl` file or a directory of them |
| `--mode` | `round_agent_avg` | Also `per_question_agent`, `round_cum_text` |
| `--model` | `nvidia/NV-Embed-v2` | SentenceTransformer id |
| `--device` | auto | `cuda` or `cpu` |
| `--batch-size` | auto | 16 on CUDA, 4 on CPU |
| `--max-seq-length` | `32768` | |
| `--out-dir` | beside the input | Where metrics are written |
| `--cache-folder` / `--cache-dir-name` | `cache` | Embedding cache |
| `--agg-mode` / `--agg-sep` | `concat` | How rounds are combined |
| `--no-eos` / `--no-normalize` / `--no-progress` / `--no-intersection` | off | Encoding switches |

Requires `sentence-transformers` and a GPU; not installed by default.

### `src/analysis/k_star/analysis_improved.py` — extended diversity metrics

Adds N\*_conditioned, N\*_weighted and Delta-N\* on top of the same inputs.

| Flag | Default | Purpose |
|---|---|---|
| `jsonl_path` | *(positional)* | A `.jsonl` file or a directory |
| `--out-dir` | beside the input | Output directory |
| `--cache-dir-name` | `cache` | Embedding cache subdirectory |

### `src/analysis/k_star/exp2_embedding_robustness.py` — cross-embedding check

Repeats the diversity measurement under a different embedding model to test
whether the ranking is an artefact of the encoder.

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `gte-qwen2` | Embedding model to validate against |
| `--gpu` | `0` | Single GPU id |
| `--multi_gpu` | `None` | Comma-separated GPU ids |
| `--sample` | `None` | Subsample size |
| `--batch_size` | `64` | |

Requires `scipy` and `sentence-transformers`.

### `src/analysis/overlap_viz.py` — agent-selection overlap plots

Compares the teams a model selected against the canonical set in
`configs/personas.json`, writing CSVs to `results/overlap/csv/` and figures to
`results/overlap/viz/` and `results/overlap/png/`.

```bash
python src/analysis/overlap_viz.py --model_name gpt-4.1 --agent_desc single
```

| Flag | Default | Purpose |
|---|---|---|
| `--model_name` | `gpt-4.1` | Whose chosen-agents file to read |
| `--agent_desc` | `single` | `name`, `single` or `full` |
| `--num_agents` | `4` | Which team size's file to read |

### Shell runners — `scripts/`

These hold their configuration as variables at the top of the file: dataset,
model list, agent counts, vLLM ports and GPU pool. Edit those rather than
passing arguments.

| Script | What it runs |
|---|---|
| `scripts/add*.sh` | Heterogeneous multi-agent sweeps across agent counts 2–16, both solvers |
| `scripts/add*_noperspn.sh` | The same sweeps with personas disabled — the control condition |
| `scripts/ablation*.sh` | Homogeneous runs per model, with and without personas |
| `scripts/k_star/analysis.sh` | Batch K\* analysis across a hard-coded list of experiment directories, round-robin over GPUs |
| `scripts/k_star/analysis_improved.sh` | The same for the extended metrics |

`scripts/add.sh` references `MAX_PARALLEL_JOBS` and `GPU_LIST` whose definitions
are commented out; restore them before running it unqualified.

## Project Structure

Three rules keep the tree predictable:

- **All Python lives under `src/`.** Experiment entry points sit at the top of `src/`;
  `src/analysis/` holds the scripts that read results and draw plots.
- **All shell runners live under `scripts/`.** They hold experiment configuration
  (dataset, model list, GPU pool, vLLM ports) as variables at the top of the file.
- **All generated output lives under `results/`** (plus the legacy `out/` and
  `out-baseline/` directories) and is never committed.

```
.
├── src/                                # All Python
│   ├── main.py                         # Debate / voting loop over N persona agents
│   ├── evaluator.py                    # Answer extraction & scoring (math, MCQ)
│   ├── tag_questions.py                # Stage 1 of the orchestrator pipeline
│   ├── tag_dataset.py                  # Stage 2: tag JSONL -> HF dataset on disk
│   ├── train_orchestrator.py           # Stage 3: the team-selection loop
│   ├── agent_selection_sweep.py        # Sweeps main.py over pre-chosen agent sets
│   ├── orchestrator_pool_sweep.py      # Agent-pool selection probe
│   ├── defaults.py                     # Shared generation defaults
│   ├── team_evaluation.py              # Runs a selected team, scores per agent & per tag
│   ├── summariser.py                   # Rewrites the per-tag performance scoreboard
│   ├── orchestration/
│   │   └── orchestrator.py             # OrchestratorAgent, tag sampling, team selection
│   ├── data/                           # Dataset loaders, routed by data_utils.load_data
│   │   ├── data_utils.py               # Central data router
│   │   ├── gsm8k.py  arc.py  hellaswag.py  truthfulqa.py  winogrande.py
│   │   └── mmlu_pro_medicine.py  mmlu_formal_logic.py
│   ├── model/                          # Backend dispatch and model wrappers
│   │   ├── model_utils.py              # Agent factory, persona definitions, unified engine
│   │   ├── llama.py  qwen.py           # Local HuggingFace wrappers
│   │   ├── openai_compat.py            # OpenAI-compatible client (vLLM, etc.)
│   │   └── azure_openai.py             # Azure OpenAI wrapper
│   └── analysis/                       # Reading data and drawing plots
│       ├── k_star/
│       │   ├── analysis.py             # Core N* (effective diversity) from embeddings
│       │   ├── analysis_improved.py    # N*_conditioned, N*_weighted, Delta-N*
│       │   └── exp2_embedding_robustness.py
│       └── overlap_viz.py              # Agent-selection overlap plots and CSVs
├── scripts/                            # All shell runners
│   ├── add*.sh                         # Heterogeneous multi-agent experiments
│   ├── add*_noperspn.sh                # Same experiments without personas
│   ├── ablation*.sh                    # Ablations (persona impact, agent count)
│   └── k_star/*.sh                     # Batch runners for the K* analysis
├── configs/
│   └── personas.json                   # Canonical persona set per task
├── data/
│   └── tagged_dataset/                 # Tagged questions, the orchestrator's input
├── notebooks/                          # Exploratory analysis
└── results/                            # Generated output (git-ignored)
```

## Key Concepts

### Multi-Agent Debate

Agents iteratively refine their answers by observing peer responses:

1. **Round 0**: Each agent generates an initial answer guided by its persona.
2. **Rounds 1–N**: Agents receive peer responses as context and produce refined answers.
3. **Final answer**: Determined by majority voting or debate consensus.

Three communication topologies are supported:
- **Decentralized** (default): Full mesh — every agent sees all peers.
- **Centralized**: Hub-and-spoke — one coordinator aggregates.
- **Sparse**: Each agent only sees neighboring agents.

### Reasoning Personas

Each agent is assigned a distinct reasoning persona that shapes its problem-solving style. For example, on math tasks:

| Persona | Strategy |
|---|---|
| Conservative Verifier | Step-by-step verification, double-checking |
| Creative Explorer | Pattern recognition, unconventional shortcuts |
| Rigorous Formalist | Precise notation, logical completeness |
| Intuitive Estimator | Rough estimates as sanity checks |
| Systematic Decomposer | Divide-and-conquer sub-problems |

### K\* Metric (Effective Diversity)

K\* measures the **effective number of distinct reasoning strategies** among agents using embedding-space analysis:

- Embed all agent responses with a sentence transformer (e.g., NV-Embed-v2).
- Compute eigenvalues of the embedding covariance matrix.
- **N\* = exp(H)** where H is the Shannon entropy of the normalized eigenvalue distribution.

Extended metrics include:
- **N\*_conditioned**: Diversity within correct vs. incorrect answer groups.
- **N\*_weighted**: Correctness-weighted diversity.
- **Delta-N\***: Marginal diversity contribution of each agent.

### Baselines

- **Baseline A**: Length-matched neutral padding replaces persona prompts (isolates persona effect).
- **Baseline B**: All agents share a single persona (tests heterogeneity benefit).

## Supported Models

| Category | Models |
|---|---|
| Local (HuggingFace) | LLaMA 3.1-8B, Qwen 2.5-7B/32B |
| vLLM Serving | Any model served via OpenAI-compatible API |
| Azure OpenAI | GPT-4o, GPT-4o-mini, o1, o3-mini, o3, o4-mini, GPT-4.1 series |
| OpenAI-Compatible | GPT-5-mini, Gemini-2.5-Flash, etc. |

## Supported Datasets

| Task Type | Datasets |
|---|---|
| Mathematical Reasoning | GSM8K |
| Multiple Choice QA | ARC, HellaSwag, TruthfulQA, WinoGrande |
| Domain-Specific | MMLU-Pro Medicine, MMLU Formal Logic |

## License

This project is for research purposes.


## Citation

If you use Agent-Scaling in your research, please cite:

```bibtex
@article{yang2026understanding,
  title={Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity},
  author={Yang, Yingxuan and Qu, Chengrui and Wen, Muning and Shi, Laixi and Wen, Ying and Zhang, Weinan and Wierman, Adam and Gu, Shangding},
  journal={arXiv preprint arXiv:2602.03794},
  year={2026}
}
```
