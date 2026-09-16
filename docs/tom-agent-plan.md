# Plan: an agent-modelling partner for the orchestrator

A design note, not yet implemented. It proposes adapting the architecture of
*ToM-SWE: User Mental Modeling for Software Engineering Agents*
(Zhou, Chen, Wang, Neubig, Sap and Wang, arXiv:2510.21903) to the orchestrator
in this repository, and sets out how to test whether it is worth the cost.

---

## 1. What the paper does

ToM-SWE splits a coding agent in two. The SWE agent keeps writing code. A second,
lightweight theory-of-mind agent models the *user* — their goals, preferences and
interaction style — and owns a persistent three-tier memory: raw session
transcripts, a per-session analysis of each one, and a single aggregated user
model. The SWE agent never sees that memory. It calls two tools instead:
`consult_tom` during a session, which returns advisory suggestions with a
confidence score in 0–1, and `update_memory` after a session, which triggers the
analysis and aggregation pass. The paper calls these the wake-time and sleep-time
halves.

Three of its results matter for the design below.

- **Separation does the work, not the information.** A single agent given the same
  memory and asked to do both jobs scored 27.2% on their Stateful SWE benchmark
  against 59.7% for the two-agent split (Claude Sonnet 4, N=100).
- **Retrieval alone is not enough, and can hurt.** A baseline that retrieves raw
  prior sessions into the coding agent's context underperformed the ToM split on
  every base model, and on the weakest model it was worse than no memory at all
  (18.7% → 14.4%). Their reading: a raw history competes for the same attention
  the task needs.
- **The overhead is small.** The consultation added about 16% to the cost of a
  session ($0.17 on $1.08), and a cheap model in the ToM role still helped.

What the paper does not do is model *other agents*. Theory of mind there is
first-order and aimed squarely at the human. The architecture is what transfers;
the target is not.

## 2. Why this maps onto our orchestrator

Our orchestrator selects four personas from a pool of fifty for a batch of
questions described by a tag profile, and its only memory is the scoreboard: a
deterministic table of `correct/seen` per agent, overall and per tag.

Lined up against the paper, our current system is close to its *RAG baseline*,
the one that underperformed. The orchestrator receives a retrieved record and is
asked to do two jobs at once in a single context: work out what each persona is
good at, and decide who should answer this batch. Nothing in the loop ever
reasons about an agent's competence as a standing question; it is re-derived from
a table of numbers at every selection, under the same prompt that has to produce
the team.

The proposal is to add the missing half: a second agent whose only job is to hold
and maintain a model of the *candidate personas* — what each one is good at, how
it fails, and which ones are worth pairing — and to answer the orchestrator's
questions about them. Where ToM-SWE models the user's mind, this models the minds
of the agents being chosen between. That is a second-order use of the same
machinery, and it is the part of the design space the paper explicitly leaves
open: it maintains no beliefs about its partner's competence, and its delegation
is unconditional.

It also fits a problem we already have. Majority voting punishes correlated
errors, so the right team is not the four highest-scoring personas — it is a set
that is individually strong and jointly diverse. Per-agent accuracy cannot express
that. A model of the agents can.

## 3. Design

### 3.1 The agent-model partner

A new `AgentModelAgent`, consulted by the orchestrator and owning its own memory.
The orchestrator keeps making the final selection; the partner is advisory, as in
the paper.

**Wake time (before a selection).** The orchestrator sends the tag profile and the
batch size. The partner replies with a shortlist of candidates, each with a short
justification, the evidence behind it, a confidence in 0–1, and a note on which
candidates complement or duplicate each other. The orchestrator is free to ignore
it and records what it chose either way.

**Sleep time (after an iteration, or after a block of them).** The partner reads
the new raw records, writes a per-episode analysis, and folds that into the
per-persona profiles.

### 3.2 Memory, in three tiers

| Tier | Content | Written by |
|---|---|---|
| 1. Raw | Per question: the prompt, each selected agent's full response, its extracted answer, the gold answer, correctness, the question's tags | `team_evaluation`, mechanically |
| 2. Episode | One record per iteration: which agents were on the team, who got what right, where the wrong answers went wrong, where the team disagreed | The partner, one LLM call per iteration |
| 3. Profile | One record per persona: tag affinities with the counts behind them, characteristic failure modes, observed complementarity with other personas, and a coverage flag when evidence is thin | The partner, aggregating tier 2 |

Tier 1 does not exist today. `run_team_evaluation` builds per-question responses
and per-agent correctness, prints them, and returns only aggregate counts —
`src/team_evaluation.py:201` assembles exactly the record we need and it is
discarded at `src/team_evaluation.py:250`. Persisting it is the first prerequisite
and is useful on its own: it is also what lets us compute *per-question agreement*
between agents, which no current output preserves.

Tier 3 is what the orchestrator's prompt actually consumes, exactly as the paper
passes only distilled suggestions to its SWE agent.

### 3.3 The guard rail: counts stay authoritative

We have already removed one LLM-written memory from this loop. `--summariser llm`
had the model rewrite the whole scoreboard every iteration, so the loop's memory
was whatever survived the rewrite, and nothing was reproducible. The design above
must not reintroduce that failure in a new costume.

The rule is that the deterministic scoreboard remains the ground truth and is never
written by a model. Tier 3 is strictly additive interpretation layered on top, and
every claim in a profile has to carry the counts that support it — the agent, the
tag, and `correct/seen`. A claim whose counts do not exist is dropped by a
validator before the profile is written, and the fraction dropped is logged as a
confabulation rate we can report. The paper's own finding that low-confidence
suggestions (~70%) correlate with rejection points the same way: when the evidence
is thin the useful output is an admission, not a guess.

Cold start needs the same treatment. With fifty personas and a hundred and fifty
training questions, most of the pool has been seen a handful of times or not at
all. Below a threshold of observations a profile should say "untested" rather than
extrapolate, and the partner should say plainly when the right move is to try
someone new.

### 3.4 Complementarity without an LLM

One part of this needs no model at all. Once tier 1 exists, per-question agreement
between agents is arithmetic: for every pair that has answered the same questions,
how often both were right, both wrong, and exactly one right. That is a direct
measure of whether a pair adds anything under majority vote, and it belongs in the
deterministic scoreboard next to the accuracy table, not in the LLM layer.

Doing this first is deliberate. It is cheap, it is reproducible, and it gives the
later experiment a fairer baseline: if the LLM partner only recovers what the
pairwise table already says, we will be able to see that.

## 4. Implementation

| Stage | Work | Depends on |
|---|---|---|
| 0 | Persist tier 1 from `run_team_evaluation`; add pairwise agreement to the counts scoreboard | — |
| 1 | `src/agent_model/` — memory store, episode analysis, profile aggregation, claim validator | 0 |
| 2 | Wake-time consult; `select_team` takes an extra, clearly delimited profile block; profiles frozen during held-out evaluation exactly as the scoreboard is | 1 |
| 3 | Retrieval over tiers 1–2 during a consult, capped at a few actions per call as the paper caps at three | 2, and only once tier 1 is large enough to be worth searching |

New flags on `train_orchestrator.py`, following the shape of the existing ones:
`--agent_model {off,advisory,only}`, `--agent_model_every N` for the sleep-time
cadence, and `--agent_model_name` so the partner can run on a smaller or cheaper
model than the orchestrator. `holdout_evaluation.py` freezes the profiles alongside
the scoreboard, so the held-out numbers measure what was learned rather than what
is still being learned.

## 5. How we would know it worked

Experiment 2, four arms on the split and seeds already used by experiment 1, so the
existing runs serve as arm A without being repeated.

| Arm | Orchestrator sees |
|---|---|
| A | Counts scoreboard (the current system) |
| B | Counts scoreboard plus agent profiles |
| C | Agent profiles only |
| D | Random team, the baseline experiment 1 still owes |

A against D says whether selection is worth anything at all. B against A is the
question the paper answers for its own setting. C against B mirrors the paper's
ablation and asks whether the distilled model can replace the raw record or only
supplement it, which is also the cheaper configuration to run.

Measures: held-out team accuracy first; then pool coverage, meaning how many of the
fifty candidates were ever tried, since a profile that encourages exploration should
move it; selection stability over training; the confabulation rate from the
validator; and wall-clock and token overhead per iteration, to see whether the
paper's 16% figure survives contact with a smaller model.

## 6. Risks

- **It reinvents the LLM summariser.** The strongest argument against this plan is
  that we deleted something adjacent a week ago. Section 3.3 is the answer, and the
  confabulation rate is the number that tells us whether the answer held.
- **Not enough evidence to model.** A hundred and fifty training questions over
  ninety-six tags and fifty personas is thin. Profiles may be confident noise. If
  stage 0 shows most pairwise cells empty, the honest move is to widen training
  before adding a layer that interprets them.
- **Cost.** A consult per selection plus an analysis per iteration on one shared
  server, when an iteration already takes over two minutes. Stage 0 costs nothing
  and should be measured before committing to stages 1 and 2.
- **The analogy is not exact.** The paper's ToM agent models a human who has
  genuine hidden preferences. Ours models agents whose behaviour is, in principle,
  fully observable from their outputs. The claim is only that a distilled,
  maintained model of them beats a flat table re-read under selection pressure —
  and that is precisely what arms B and C are for.
