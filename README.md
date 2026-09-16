# Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity

This repository provides the codebase for studying **how scaling the number of heterogeneous LLM agents with diverse reasoning personas improves collective performance** through debate and voting mechanisms. We introduce the **K\* metric** (effective diversity) based on embedding eigenvalue entropy to quantify semantic diversity among agents, and show that persona-guided multi-agent collaboration yields consistent gains across reasoning benchmarks.

## Project Structure

```
.
├── src/                          # Core source code
│   ├── main.py                   # Main orchestration: debate/voting loop
│   ├── evaluator.py              # Answer extraction & scoring (math, MCQ)
│   ├── tag_questions.py          # Stage 1: LLM tags each question with capabilities
│   ├── canonicalise_tags.py      # Stage 2: collapses near-duplicate tags into categories
│   ├── tag_dataset.py            # Stage 3: applies the mapping, saves the tagged dataset
│   ├── train_orchestrator.py     # Orchestrator loop: select team, score, summarise, repeat
│   ├── team_evaluation.py        # Runs a selected team, scores per agent and per tag
│   ├── summariser.py             # Rewrites the per-tag performance scoreboard
│   ├── orchestration/            # Orchestrator agent and team selection
│   │   └── orchestrator.py       # OrchestratorAgent, tag sampling, team selection prompt
│   ├── data/                     # Dataset loaders
│   │   ├── data_utils.py         # Central data router
│   │   ├── gsm8k.py              # Grade School Math 8K
│   │   ├── arc.py                # ARC-Challenge / ARC-Easy
│   │   ├── hellaswag.py          # HellaSwag
│   │   ├── truthfulqa.py         # TruthfulQA
│   │   ├── winogrande.py         # WinoGrande
│   │   ├── mmlu_pro_medicine.py  # MMLU-Pro Medicine
│   │   └── mmlu_formal_logic.py  # MMLU Formal Logic
│   └── model/                    # Model wrappers
│       ├── model_utils.py        # Agent factory, persona definitions, unified engine
│       ├── llama.py              # LLaMA (v2/v3) wrapper via HuggingFace
│       ├── qwen.py               # Qwen wrapper via HuggingFace
│       ├── openai_compat.py      # OpenAI-compatible API client (vLLM, etc.)
│       └── azure_openai.py       # Azure OpenAI wrapper
├── scripts/                      # Experiment runner scripts
│   ├── add*.sh                   # Heterogeneous multi-agent experiments
│   ├── add*_noperspn.sh          # Same experiments without personas
│   └── ablation*.sh              # Ablation studies (persona impact, agent count)
├── K_star_analysis/              # K* diversity metric computation
│   ├── analysis.py               # Core N* (effective diversity) from embeddings
│   ├── analysis_improved.py      # Extended metrics: N*_conditioned, N*_weighted, Delta-N*
│   ├── exp2_embedding_robustness.py  # Cross-embedding-model robustness validation
│   ├── analysis.sh               # Runner for analysis.py
│   └── analysis_improved.sh      # Runner for analysis_improved.py
├── data-claude/                  # Orchestrator-track data (tags, mapping, tagged dataset)
└── .gitignore
```

## Installation

### Requirements

- Python >= 3.9
- CUDA-compatible GPU(s)

### Dependencies

```bash
pip install torch transformers accelerate peft
pip install sentence-transformers datasets
pip install openai numpy pandas scipy tqdm
```

`requirements.txt` is a full pinned freeze of an x86 environment. On aarch64
(DGX Spark) the pinned `torch==2.11.0` has no wheel; install `torch` unpinned and
let pip resolve the CUDA build for the platform. The orchestration pipeline below
needs only `openai`, `datasets`, `pandas`, `numpy`, `matplotlib` and `torch`, since
generation happens in a vLLM server rather than in-process.

## Quick Start

### 1. Homogeneous Multi-Agent Voting

```bash
python src/main.py \
    --data gsm8k \
    --num_agents 5 \
    --model qwen2.5-7b \
    --solver vote \
    --debate_rounds 0
```

### 2. Heterogeneous Multi-Agent Debate with Personas

```bash
python src/main.py \
    --data formal_logic \
    --num_agents 4 \
    --agent_models "llama3.1-8b,qwen2.5-7b,mistral-7b,qwen3-8b" \
    --multi_persona \
    --solver debate \
    --debate_rounds 3 \
    --use_vllm \
    --vllm_base_urls "http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1"
```

### 3. Using Azure OpenAI Models

```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY_ENV="your-api-key"

python src/main.py \
    --data gsm8k \
    --num_agents 4 \
    --agent_models "gpt-4o,gpt-4o-mini,o3-mini,gpt-4.1" \
    --multi_persona \
    --solver debate \
    --debate_rounds 3
```

### 4. Run K\* Analysis

```bash
python K_star_analysis/analysis.py \
    --jsonl_dir out/history/ \
    --mode round_agent_avg \
    --output_dir analysis/results/
```

## Orchestration Data Pipeline

The orchestrator selects a team of agents per task, so it needs questions labelled
with the capabilities they demand. Three stages produce that labelled dataset. All
orchestrator-track output goes under `data-claude/`, keeping it separate from the
paper's `data/` and `out/`.

### 0. Serve a model

Every stage talks to an OpenAI-compatible endpoint. The sibling `vllm/` repo serves
one; `HOST_PORT` publishes it on the port this repo defaults to:

```bash
NUM_SPEC_TOKENS=3 MAX_NUM_SEQS=64 HOST_PORT=8001 sh ../vllm/qwen3.6/run_docker_nvfp4.sh
docker logs -f qwen3.6-vllm            # wait for engine init, about 5 minutes
curl -s localhost:8001/v1/models       # confirm the served model id
```

`MAX_NUM_SEQS=64` matches the 64-way semaphore in `tag_questions.py`. The model id
passed as `--model` must be the served HuggingFace handle, not a local model key.

### 1. Tag the questions — `src/tag_questions.py`

Asks the model, for each question, which capabilities or reasoning styles would help
an agent answer it. Runs all datasets concurrently through one semaphore.

```bash
python src/tag_questions.py \
    --data gsm8k arc hellaswag truthfulqa winogrande pro_medicine formal_logic \
    --split test --data_size 100 \
    --data_dir data-claude/benchmarks/ --out_dir data-claude/question_tags \
    --use_vllm --vllm_base_url http://127.0.0.1:8001/v1 \
    --model nvidia/Qwen3.6-35B-A3B-NVFP4
```

Writes `{out_dir}/{datasets}_{split}_{data_size}_tags.jsonl`, one record per question
with its raw tags and the raw model response.

| Argument | Default | Description |
|---|---|---|
| `--data` | all seven | Datasets to tag, space-separated |
| `--split` / `--data_size` | `test` / `0` | Split and per-dataset question cap (`0` means all) |
| `--data_dir` | `./data/` | HuggingFace cache for the benchmark downloads |
| `--out_dir` / `--output_file` | `out/question_tags` / derived | Where the tags JSONL is written |
| `--model` | `Qwen/Qwen3.6-35B-A3B` | Model doing the tagging |
| `--use_vllm` / `--vllm_base_url` | off / `http://127.0.0.1:8001/v1` | Endpoint serving that model |
| `--max_new_tokens` / `--temperature` / `--top_p` | `512` / `0.0` / `0.9` | Generation settings |

### 2. Canonicalise the tag vocabulary — `src/canonicalise_tags.py`

Free-text tagging produces many surface variants of one capability
(`step-by-step reasoning`, `step by step reasoning`, `sequential reasoning`). This
stage derives the mapping from the data rather than relying on a hand-maintained
dictionary, in two passes:

1. **Lexical.** Merges tags differing only in case, punctuation, separators, simple
   plurals or `-ing` endings, filler words, or word order.
2. **Semantic.** The model groups what remains, in alphabetically-sorted batches, then
   re-clusters the canonical labels those batches produced so synonyms split across
   batches still meet. Repeats until a round merges nothing. A tag the model drops or
   renames keeps its own name, so the vocabulary never silently loses one.

```bash
python src/canonicalise_tags.py \
    --tags_file data-claude/question_tags/<name>_tags.jsonl \
    --out_file data-claude/tag_mapping.json \
    --api_base_url http://127.0.0.1:8001/v1 \
    --model nvidia/Qwen3.6-35B-A3B-NVFP4
```

Writes a JSON file holding the `{raw tag: canonical tag}` mapping and the resulting
canonical tag counts.

| Argument | Default | Description |
|---|---|---|
| `--tags_file` | the 700-question default path | JSONL from stage 1 |
| `--out_file` | `data-claude/tag_mapping.json` | Where the mapping is written |
| `--model` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | Model doing the grouping |
| `--api_base_url` / `--api_key` | `http://127.0.0.1:8001/v1` / `EMPTY` | Endpoint serving it |
| `--batch_size` | `120` | Tags per grouping call |
| `--max_rounds` | `4` | Cap on re-clustering rounds |
| `--max_tokens` / `--temperature` / `--top_p` | `8192` / `0.0` / `0.9` | Generation settings |
| `--no_llm` | off | Lexical normalisation only, no model calls |

### 3. Build the tagged dataset — `src/tag_dataset.py`

Applies the mapping from stage 2, then the hand-written `TAG_MAPPING` in the module
(which still catches any synonym pair the canonicaliser left apart), drops tags below
the frequency threshold, and saves a HuggingFace dataset of
`dataset, question, answer, tags`.

```bash
python src/tag_dataset.py \
    --tags_file data-claude/question_tags/<name>_tags.jsonl \
    --tag_mapping data-claude/tag_mapping.json \
    --out_dir data-claude/tagged_dataset
```

| Argument | Default | Description |
|---|---|---|
| `--tags_file` | the 700-question default path | JSONL from stage 1 |
| `--tag_mapping` | `data-claude/tag_mapping.json` | Mapping from stage 2; empty string uses only the built-in `TAG_MAPPING`, and a missing file warns rather than failing |
| `--out_dir` | `data-claude/tagged_dataset` | Where `save_to_disk` writes |
| `--threshold` | `5` | Drop tags occurring fewer than this many times |
| `--plot_path` | `data-claude/tag_frequencies.png` | Tag frequency chart; empty string skips it |

### 4. Run the orchestrator loop — `src/train_orchestrator.py`

```bash
python src/train_orchestrator.py \
    --dataset_path data-claude/tagged_dataset \
    --model_name nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --api_base_url http://127.0.0.1:8001/v1 \
    --iterations 3 --num_samples 5 --team_size 4 --seed 0
```

Each iteration samples a tag, shows the orchestrator the tag profile of the batch
and the scoreboard so far, asks it for a team, evaluates that team on the sampled
questions, and folds the counts back into the scoreboard.

**Candidate agents.** The pool is every persona in `chosen_persona_bank()` — the
union of the paper's per-dataset persona sets plus the default set, 50 in total —
built by `build_agent_pool()` rather than listed by hand, so a persona added to the
bank becomes selectable without a second edit. Names the model invents are rejected
against the pool and the selection is retried once with the rejected names quoted
back; a name that survives that would otherwise reach `_build_chosen_personas` as a
`KeyError`.

**Outputs** (all under `--out_dir`, default `data-claude/orchestrator/`):

| File | Contents |
|---|---|
| `agent_performance_by_tag.md` | The scoreboard, and the loop's entire memory: agent totals, accuracy per tag with `correct/seen` counts, recent team selections, and which pool agents are still untried |
| `agent_performance_state.json` | The counts the scoreboard is rendered from |
| `run_records.jsonl` | One machine-readable record per iteration: tag, team, rejected names, full evaluation report |
| `team_selection_results.csv` | Selection log with the orchestrator's reasoning and reasoning trace |

Delete the scoreboard and the state file to reset the loop; stale ones contaminate a
new experiment.

| Argument | Default | Description |
|---|---|---|
| `--dataset_path` | `data-claude/tagged_dataset` | Tagged dataset from stage 3 |
| `--model_name` | `Qwen/Qwen3.6-35B-A3B` | Model for the orchestrator and, through `team_evaluation`, the agents |
| `--api_base_url` / `--api_key` | `http://localhost:8001/v1` / `none` | Endpoint serving it |
| `--iterations` | `10` | Select-evaluate-summarise cycles |
| `--num_samples` | `5` | Questions sampled per iteration |
| `--team_size` | `4` | Agents the orchestrator must select |
| `--seed` | unset | Seeds tag and question sampling |
| `--summariser` | `counts` | `counts` renders the scoreboard from recorded counts; `llm` has the model rewrite the markdown each iteration (the original behaviour) |
| `--out_dir` | `data-claude/orchestrator` | Directory for all four outputs above |
| `--md_file` / `--state_file` / `--output_path` / `--selection_csv` | derived from `--out_dir` | Override individual paths |
| `--solver` | `vote` | Only `vote` is implemented; `debate` warns and scores by vote |
| `--debug` | off | Print the full orchestrator prompt and raw response |

**Why `counts` is the default summariser.** In the original loop the model rewrote
the markdown from scratch each iteration, so the loop's only memory was whatever
survived a rewrite — a dropped tag or a rounded number changed the record with
nothing to check it against. The `counts` summariser keeps the counts in
`agent_performance_state.json` and renders the markdown from them, so accuracies
always carry the `correct/seen` they are computed from and the record is
reproducible. The model reads the scoreboard; it no longer writes it.

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

## Running Large-Scale Experiments

The `scripts/` directory contains bash scripts for running experiment sweeps:

```bash
# Heterogeneous agents across multiple agent counts
bash scripts/add.sh

# Same experiment without personas (control)
bash scripts/add_noperspn.sh

# Ablation: persona vs. no-persona across models
bash scripts/ablation.sh
```

These scripts manage multi-GPU scheduling and run experiments across varying agent counts (2, 4, 8, 12, 16) with both debate and voting solvers.

## Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--data` | Dataset name | (required) |
| `--num_agents` | Number of agents | 5 |
| `--agent_models` | Comma-separated model list (heterogeneous) | `""` |
| `--multi_persona` | Enable diverse personas per agent | `False` |
| `--baseline_a` | Baseline A: neutral padding | `False` |
| `--baseline_b` | Baseline B: shared persona | `False` |
| `--solver` | Aggregation method: `vote` or `debate` | `vote` |
| `--debate_rounds` | Number of debate rounds | 4 |
| `--sparse` | Sparse communication topology | `False` |
| `--centralized` | Centralized communication topology | `False` |
| `--use_vllm` | Use vLLM backend | `False` |
| `--temperature` | Sampling temperature | 0 |
| `--top_p` | Nucleus sampling threshold | 0.9 |
| `--load_in_4bit` | 4-bit quantization (bitsandbytes) | `False` |
| `--load_in_8bit` | 8-bit quantization (bitsandbytes) | `False` |

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
