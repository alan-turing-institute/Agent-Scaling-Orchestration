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
the loop under active development.

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
python src/analysis/k_star/analysis.py \
    --jsonl_dir out/history/ \
    --mode round_agent_avg \
    --output_dir analysis/results/
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
