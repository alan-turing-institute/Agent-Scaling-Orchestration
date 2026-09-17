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
| Run D `random` | `--selection random` → `data-claude/orchestrator/random/` |
| Run E `bare_model` | `scripts/bare_model_baseline.py` → `data-claude/orchestrator/bare_model/` |

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
python src/train_orchestrator.py $COMMON --selection random --memory none --summary_every 1 \
    --out_dir data-claude/orchestrator/random
python scripts/bare_model_baseline.py --out_dir data-claude/orchestrator/bare_model
```

`scripts/experiment1_schedules.sh` runs these in sequence, and takes an `ARMS`
override so a single arm can be added to a finished round:
`ARMS=random ./scripts/experiment1_schedules.sh`.

Runs A to C launched 2026-09-16 and ran one after another, in the order
`no_memory`, `continual`, `batched`, against the one server. Run D was added on
2026-09-17 and queued behind them on the same server and the same split.

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
question from run D, which removes the choosing rather than the evidence: run C
still reasons about which personas suit which tags, it just has no record of how any
of them has actually done.

**Run D, the random-selection baseline.** `--selection random` draws the team
uniformly from the same pool of fifty and never calls the orchestrator model, in
training and in the held-out evaluation alike. Everything else is held constant, so
its held-out accuracy is the floor: if runs A to C do not clear it, the question of
which memory schedule is better does not arise, because the selecting itself is not
paying for the calls it costs. Its RNG is seeded from `--split_seed`, so the arm
reproduces. It was added after the first three had run rather than alongside them,
which means it is scored on the same frozen split but not in the same round.

**Run E, the bare model.** Runs D and C strip away the memory and then the
choosing, but all of them still pay for four persona-prompted agents per question
and a majority vote. Run E answers the same 140 held-out questions with one call
each: no persona prefix, no team, no vote. `scripts/bare_model_baseline.py` shares
the dataset, the split seed and fraction, the batching, the per-question answer-type
inference and the evaluator parsers with the orchestrator runs, so the only things
that differ are the prompt and the number of respondents, and it writes its results
in the shape `scripts/report_run.py` reads. It costs 140 calls against an arm's 560
plus selections.

Read the arms outwards from it. Run E says what the model can do alone; run D says
what four random agents and a vote add to that; run C says what choosing them adds;
runs A and B say what remembering adds to the choosing.

**A held-out split scored 0.00%, and the cause was a swallowed exception.** The first
`random` run finished training at 130/150 and then scored 0/140 on the held-out
split, across all 28 batches, with nothing logged. `run_team_evaluation` calls each
agent in a thread pool and caught every exception from the future, substituting an
empty response; an empty response parses as an empty answer, which scores as
incorrect. The engine had wedged at 14:57 UTC and the watchdog restarted it at
15:02:19, seven seconds before the evaluation began, so every call for the next five
minutes hit a container reloading weights. Nothing was raised for `with_server_retry`
to catch. Connection and timeout errors now propagate; other per-agent failures are
still absorbed but logged with the agent name. The void run is archived at
`data-claude/void-2026-09-17-random-holdout-zeroed/` and run D was restarted.

Runs A to C were checked for the same signature afterwards: no held-out batch and no
training iteration in `no_memory`, `continual` or `batched` has every agent at zero,
so their numbers stand.

### Results

_Running. To be filled in: held-out team accuracy for each of the four runs,
per-tag breakdown, which agents each run converged on, how many of the 50 candidates
each tried, and whether selections changed over the course of training. The
comparison that matters first is A and B against C: if neither beats the run with no
memory at all, the update schedule is not the interesting variable. The ones that
matter before any of those are every arm against D, and every arm against E._

Runs A to C completed 2026-09-17 between 03:08 and 13:41 UTC, 30/30 training
iterations and 28/28 held-out batches each, no errors:

| | `no_memory` | `continual` | `batched` |
|---|---|---|---|
| Training | 142/148 = 95.9% | 135/150 = 90.0% | 136/150 = 90.7% |
| **Held out, 140 questions** | **126/140 = 90.0%** | **131/140 = 93.6%** | **130/140 = 92.9%** |
| matched: best agent in batch | 94.3% | 94.3% | 95.7% |
| matched: mean agent in batch | 88.8% | 91.8% | 92.5% |
| matched: worst agent in batch | 83.6% | 87.1% | 89.3% |
| vote minus best agent | −4.3 | −0.7 | −2.9 |
| Distinct agents in training | 21 of 50 | 18 of 50 | 23 of 50 |
| Most-picked agent | — | `Conservative_Verifier` ×23 | `Conservative_Verifier` ×19 |

Both memory arms beat the no-memory arm, by 3.6 and 2.9 points. The two schedules
differ by one question, which is nothing. The consistent pattern is that memory
lifts the floor — the worst agent on a batch climbs 83.6 → 87.1 → 89.3 — while the
best agent barely moves, and the team vote sits below the best available agent in
every arm. `batched` had the highest mean and worst agent of the three and still
finished behind `continual` on the vote, which points at the aggregation rather than
the selection.

Runs D and E are pending. The watchdog restarted the engine four times during the
A-to-C round, roughly every three to four hours; no iteration was lost, but the
server is not stable at this duty cycle.

### Caveats to remember when reading these numbers

- Run D, the random-selection baseline, was added after runs A to C had finished and
  is scored on the same frozen split, but not in the same round. Anything the server
  drifted on between rounds lands on that comparison.
- 30 iterations × 5 questions means the scoreboard is built from 150 answered
  questions spread over 96 tags, so most per-tag cells are thin. Differences between
  the two runs may be noise at this scale.
- The arms run sequentially on one server, so their wall-clock timings are
  comparable to each other but were measured across different hours of the day.
- `--solver debate` is accepted but not implemented; everything here is majority vote
  with no debate rounds.
