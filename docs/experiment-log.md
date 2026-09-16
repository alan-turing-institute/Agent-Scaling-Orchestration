# Experiment log

Running record of orchestrator experiments: what was run, on what data, with which
code, and what came out. Newest entry last. Results tables are filled in when a run
finishes; a run still in flight says so.

The goal of this line of work differs from the paper the repository was built for.
The paper studies how collective performance scales with the number of heterogeneous
persona-guided agents. Here the question is narrower: **can an orchestrator learn to
pick the right agents for a task**, given a pool of candidate personas and a record
of how they have done on questions carrying each tag.

---

## Environment

| | |
|---|---|
| Machine | DGX Spark, one GB10 (SM121), 121 GB unified memory, ~273 GB/s |
| Server | vLLM in Docker, container `qwen3.6-vllm`, host port 8001 |
| Model | `nvidia/Qwen3.6-35B-A3B-NVFP4` (NVFP4 weights, FP8 KV cache) |
| Serve command | `NO_SPEC=1 MAX_NUM_SEQS=64 HOST_PORT=8001 sh vllm/qwen3.6/run_docker_nvfp4.sh` |
| Python | local venv at `env/`, torch 2.14.0+cu130, datasets 5.0.1, openai 3.14.1 |

`MAX_NUM_SEQS=64` is the throughput configuration from `vllm/qwen3.6/README.md`
(631 tok/s aggregate at 64 concurrent), chosen to match the 64-way semaphore in
`tag_questions.py`. The tagging stages below ran with MTP speculative decoding
(`NUM_SPEC_TOKENS=3`); the orchestrator runs do not, for the reason recorded under
experiment 1. `HOST_PORT` was added to the serve script for this work: the
container always listens on 8000, while this repository defaults to 8001 and
`team_evaluation.py` hard-codes it.

All data produced by this line of work lives under `data-claude/`, kept separate
from the paper's `data/` and `out/`. It is gitignored; the repository holds the code
and this log, not the artifacts.

---

## 2026-09-16 — Building the tagged dataset

Three stages, all against the server above.

### Stage 1: tagging

```bash
python src/tag_questions.py \
    --data gsm8k arc hellaswag truthfulqa winogrande pro_medicine formal_logic \
    --split test --data_size 100 \
    --data_dir data-claude/benchmarks/ --out_dir data-claude/question_tags \
    --use_vllm --vllm_base_url http://127.0.0.1:8001/v1 \
    --model nvidia/Qwen3.6-35B-A3B-NVFP4
```

**699 records** — 100 per dataset except truthfulqa at 99, where the loader drops one
question whose choice count falls outside its 2–10 range. 3 questions came back with
no tags. Mean 4.24 raw tags per question, **869 unique raw tags**.

First attempt wrote nothing: the truthfulqa loader asked for the bare dataset id
`truthful_qa`, which current `huggingface_hub` rejects, and because `tag_questions.py`
gathers all datasets in one `asyncio.gather`, that one failure aborted the run after
~500 questions had already been tagged. Fixed to `truthfulqa/truthful_qa` and
`allenai/winogrande`.

### Stage 2: canonicalising the tag vocabulary

869 free-text tags contain many surface variants of one capability, so a
data-derived mapping replaces the hand-maintained `TAG_MAPPING`.

```bash
python src/canonicalise_tags.py \
    --tags_file data-claude/question_tags/<name>_tags.jsonl \
    --out_file data-claude/tag_mapping.json \
    --api_base_url http://127.0.0.1:8001/v1 --model nvidia/Qwen3.6-35B-A3B-NVFP4
```

**869 → 745 canonical tags** over three rounds (855 after the lexical pass, then 777,
then 745; the fourth round merged nothing and stopped).

The first attempt merged almost nothing (869 → 843, all of it lexical). Asked to
return every tag in exactly one group, the model echoed all 120 tags of a batch back,
most as single-member groups, and hit the token cap every time — `finish_reason:
length`, unparseable JSON, every tag kept as-is, reported only as a warning. The
prompt now asks for merge groups only, an unlisted tag stays its own category,
batches dropped to 60 and run 8 at a time, and a response that still overruns is
salvaged for its complete groups.

### Stage 3: the dataset

```bash
python src/tag_dataset.py \
    --tags_file data-claude/question_tags/<name>_tags.jsonl \
    --tag_mapping data-claude/tag_mapping.json \
    --out_dir data-claude/tagged_dataset
```

**699 questions, 96 tags** surviving the frequency-5 threshold (717 unique after both
mappings, 621 dropped as rare). Mean 2.76 canonical tags per question. Per dataset:
gsm8k 100, arc 100, hellaswag 100, truthfulqa 99, winogrande 100, pro_medicine 100,
formal_logic 100.

---

## 2026-09-16 — Code changes these experiments depend on

Branch `orchestrator/agent-selection-pipeline`, commits `30cf7f6`, `8dfe6f2`,
`57e2c5c`, `7c31e51`.

**The candidate pool was 5 of 50.** `AGENT_POOL` was a hand-copied list of the gsm8k
personas, so no medical, logic, science, coreference or truthfulness agent could ever
be selected, whatever the tag. It is now built from `chosen_persona_bank()` — the
union of every per-dataset persona set plus the default set, 50 candidates.

**Selection was unvalidated.** The check against the pool was commented out, so an
invented name reached `_build_chosen_personas` as a `KeyError`, and a parse failure
fell back to two names (`MathReasoner`, `FactChecker`) that exist in no persona set.
Names outside the pool are now dropped, the selection retries once with them quoted
back, and an unparseable response is recorded as `no_valid_team`.

**The scoreboard was an LLM rewrite.** The loop's only memory was whatever survived
the model rewriting the whole markdown each iteration. `--summariser counts` (now the
default) keeps counts in `agent_performance_state.json` and renders the markdown from
them: agent totals, per-tag accuracy as `correct/seen`, recent selections, and which
pool agents remain untried.

Four bugs fixed along the way, each of which corrupts the numbers the scoreboard
records:

| Bug | Effect |
|---|---|
| Threaded responses collected in completion order, zipped against agent names in selection order | Agents credited with each other's answers |
| MCQ batches set `args.data = "mcq"`, unknown to `get_instruction_suffix` | Agents told to answer multiple choice as `{final answer: 123}` |
| gsm8k answers are strings in the tagged dataset; `evaluate_gsm8k` rounds with numpy | Every gsm8k batch would have raised |
| Answer type inferred once per batch from the first answer | In a tag-built batch mixing datasets, questions of the other type were silently dropped — a 14-question test split reported 11 |

Also added: `--iterations` (was hard-coded 10), `--seed`, `--team_size`,
`--summary_every`, `--test_fraction` / `--split_seed` / `--test_batch_size`,
`--eval_workers`, `run_records.jsonl` per iteration, and `team_evaluation` falling
back to `--api_base_url` before the hard-coded endpoint.

**Throughput.** Questions within a batch were evaluated one at a time, leaving the
server idle. They are independent, so they now run concurrently with the per-agent
calls still parallel inside each: one iteration of five questions went from about
four minutes to 2m16s.

---

## Experiment 1 — continual vs batched scoreboard updates

**Question.** The scoreboard is the orchestrator's only memory. Does it matter
whether that memory is refreshed after every task, or in large batches? Continual
updating gives the orchestrator the freshest possible record; batched updating means
it keeps selecting against an older record, seeing several tasks' worth of results at
once. A third run withholds the scoreboard altogether, so the two schedules are
compared against the case where there is nothing to schedule.

**Design.** Two runs identical in every respect except `--summary_every`. The test
split is taken before training, the training loop samples only from the remainder,
and after training the scoreboard is frozen while every held-out question is answered
exactly once. `--split_seed 0` drives the split, the shuffle and the batching in both
runs, so they face the same questions in the same order; `--seed 0` gives them the
same training tags and questions.

Each run writes to its own `--out_dir`, so neither can read the other's scoreboard,
state file or records.

| | |
|---|---|
| Dataset | `data-claude/tagged_dataset`, 699 questions, 96 tags |
| Split | 559 train / 140 held out (`--test_fraction 0.2`, `--split_seed 0`) |
| Training | 30 iterations × 5 questions, `--seed 0` |
| Team size | 4, from a pool of 50 |
| Held-out evaluation | 140 questions in 28 batches of 5, scoreboard frozen |
| Run A `continual` | `--summary_every 1` → `data-claude/orchestrator/continual/` |
| Run B `batched` | `--summary_every 10` → `data-claude/orchestrator/batched/` |
| Run C `no_memory` | `--memory none` → `data-claude/orchestrator/no_memory/` |

```bash
COMMON="--dataset_path data-claude/tagged_dataset \
    --model_name nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --api_base_url http://127.0.0.1:8001/v1 \
    --iterations 30 --num_samples 5 --team_size 4 \
    --seed 0 --split_seed 0 --test_fraction 0.2 --test_batch_size 5 --eval_workers 5"

python src/train_orchestrator.py $COMMON --summary_every 1 \
    --out_dir data-claude/orchestrator/continual
python src/train_orchestrator.py $COMMON --summary_every 10 \
    --out_dir data-claude/orchestrator/batched
python src/train_orchestrator.py $COMMON --memory none --summary_every 1 \
    --out_dir data-claude/orchestrator/no_memory
```

All three arms launched 2026-09-16 and run one after another, in the order
`no_memory`, `continual`, `batched`, against the one server.

**Aborted attempts, and why the arms now run one after another.** The first two
launches both died to the same server fault, not a code fault. At 16:30 UTC, four iterations in, the vLLM engine stopped
making progress: generation throughput fell to zero with fourteen requests still
marked running, and stayed there. The API server kept answering `/v1/models`, so
the container looked healthy from outside, but no completion returned again. Run B
raised `openai.APITimeoutError` and died; run A hung in the same call for another
hour and twenty minutes; run C, launched into the wedged server, never completed an
iteration. Partial outputs are kept under
`data-claude/aborted-2026-09-16-engine-hang/`.

Two changes came out of it, and both are in the code the relaunched runs use.

- **The server no longer speculates.** The engine was running MTP speculative
  decoding (`NUM_SPEC_TOKENS=3`) when it first wedged. It is restarted with
  `NO_SPEC=1`, keeping `MAX_NUM_SEQS=64`. This did not fix it — see below — but
  speculative decoding buys most at low concurrency and these runs keep twenty or
  more requests in flight, so nothing is lost by leaving it off.

The second attempt, with all three arms concurrent and no speculative decoding,
wedged the same way about fifteen minutes in: engine core spinning at 99% CPU and
96% GPU utilisation, no completion returned for ninety minutes, all three runs
blocked in their selection call. So speculative decoding was not the cause. What
both attempts share is sustained concurrent load from three processes — roughly
sixty requests in flight — against a server whose published figures come from
short benchmark bursts (631 tok/s at 64 concurrent, `vllm/qwen3.6/README.md`), not
from hours of long prompts and long generations.

**The arms therefore run sequentially**, driven by `scripts/experiment1_schedules.sh`.
One run keeps about twenty requests in flight. This costs wall-clock — roughly six
hours for three arms rather than two and a half — and buys a server that is inside
its measured envelope. Partial outputs from the second attempt are under
`data-claude/aborted-2026-09-16-engine-hang-2/`. A watchdog probes the server each
minute during a run and restarts the container after three consecutive failures,
so a further wedge costs an iteration rather than the run.
- **A failed call costs an iteration, not a run.** The orchestrator client and the
  agent wrapper now use a 300-second timeout with four retries, and both the
  training loop and the held-out evaluation catch a failure per iteration or per
  batch, record it, and continue. The first attempt lost four completed iterations
  because there was nothing between one timed-out request and the end of the
  process.

**Run C, the no-memory baseline.** `--memory none` writes the scoreboard but never
reads it back, either during training or in the held-out evaluation, so every team
is chosen from the tag profile and the pool of fifty alone. It is the line runs A
and B have to beat for the scoreboard to be worth keeping. It is a different
question from `--random_baseline`, which removes the choosing rather than the
evidence: run C still reasons about which personas suit which tags, it just has no
record of how any of them has actually done.

### Results

_Running. To be filled in: held-out team accuracy for each of the three runs,
per-tag breakdown, which agents each run converged on, how many of the 50 candidates
each tried, and whether selections changed over the course of training. The
comparison that matters first is A and B against C: if neither beats the run with no
memory at all, the update schedule is not the interesting variable._

### Caveats to remember when reading these numbers

- No random-team baseline was run. Run C isolates the value of the *memory*, but not
  the value of the *selecting*: a random team drawn from the same pool might do as
  well as one the orchestrator reasoned its way to. `--random_baseline` scores a
  randomly drawn team on the same held-out batches and should be run before claiming
  the orchestrator is worth anything at all.
- 30 iterations × 5 questions means the scoreboard is built from 150 answered
  questions spread over 96 tags, so most per-tag cells are thin. Differences between
  the two runs may be noise at this scale.
- The arms run sequentially on one server, so their wall-clock timings are
  comparable to each other but were measured across different hours of the day.
- `--solver debate` is accepted but not implemented; everything here is majority vote
  with no debate rounds.
