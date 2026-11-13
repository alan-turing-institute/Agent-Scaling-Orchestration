# Agent Orchestration

An orchestrator chooses which agents work on each task, aiming to be more accurate per unit of
cost than fixed or randomly chosen agents.

## Language

### Work

**Task**:
The unit the orchestrator makes one decision for. Currently a single question; later a multi-step
job with tools whose sub-steps can go to different agents.
_Avoid_: batch, problem, item

**Batch**:
A group of questions sharing a tag, answered by one team. The old unit of selection, now superseded
by per-task selection.
_Avoid_: task (when meaning a batch)

### Agents

**Agent**:
Something that can be asked to do a task, defined by its base model or checkpoint, its thinking mode,
its prompt, and any tools it has. Two agents differ only if they differ in capability or cost, not
just in wording.
_Avoid_: persona (when meaning agent), worker

**Thinking mode**:
Whether an agent reasons before answering (on or off). Part of the agent's identity: the same
checkpoint with thinking on and off is two agents. Thinking may run to the model's context limit
rather than being cut short; what it spends shows up in cost instead.

**Persona**:
A role prompt given to an agent. Most personas on one model differed only by run-to-run noise; a
persona earns a place as a separate agent only when it beats the same model with a neutral prompt.
_Avoid_: agent (when meaning persona)

**Neutral prompt**:
The plain instruction an agent gets when it has no persona.

**Base model**:
A general-purpose checkpoint in the pool, from any open model family. Specialists must be built on
one.

**Specialist**:
An agent built on an existing fine-tuned checkpoint for one domain, on one of the pool's base
models. Taken as published, not trained by us.
_Avoid_: expert, trained agent

**Pool**:
The set of agents the orchestrator may choose from.

### Choosing

**Orchestrator**:
The agent that leads a team. In X1 the team is fixed: the orchestrator reads every worker's answer
and gives the team's final answer, and does not choose the team. In X4 it answers first and decides
whether to escalate; in the delegation experiment it hands sub-steps of a tool task to workers.
Called the hub in X1 until 2026-11-05; the code still says `hub`.
_Avoid_: hub, router, selector, planner

**Direct answer**:
The orchestrator answering a task itself, with no team.

**Delegation**:
The orchestrator handing a task, or a sub-step of a tool task, to other agents.

**Solve first**:
The orchestrator answering the task itself, then deciding whether to escalate. Its own answer is
kept and counts in the cost whether or not it escalates.
_Avoid_: self-check, fallback

**Escalation**:
Asking for help after solving first. The help is a configuration whose first member is the
orchestrator's own answer, in one of three forms:
- **pass on**: another agent answers from scratch and its answer replaces the orchestrator's;
- **second opinion**: other agents answer independently and vote with the orchestrator's answer;
- **review**: another agent sees the orchestrator's answer through the handoff and confirms or
  revises it.
On a tool task, escalation retries the whole task or hands its transcript on.
_Avoid_: fallback, retry (when meaning any of the three)

**Escalation trigger**:
What makes the orchestrator escalate: its own judgement after answering, or a rule fit on training
folds (a cut-off or missing answer, disagreement among its own samples, a stated confidence below a
threshold).

**Configuration**:
What is chosen for a task: which agents, how many, and the topology that combines them.
_Avoid_: setup, team (when meaning the whole choice)

**Team**:
The agents in a configuration.

**Topology**:
How a team's agents work together and how their answers become one: vote, cascade, debate,
centralized, synthesis, or sequential.
_Avoid_: mechanism, aggregation, architecture

**Worker**:
A team member in a centralized topology. It answers without seeing anyone else's work, so its
answer is also its solo answer; the orchestrator reads it.
_Avoid_: solver (when meaning the team member)

**Expert persona**:
A persona written for one source or subject (a GPQA organic chemist, an AIME competition
mathematician), in the same form as the bank's personas.
_Avoid_: specialist (that is a fine-tuned checkpoint)

**Fixed team**:
A team set before the run and used for every task it covers, with no per-task choice. The
alternative to an orchestrator choosing the configuration.
_Avoid_: static team, preset

**Sequential topology**:
A topology where one agent's output is the next agent's input, so a later agent can improve on an
earlier one's answer rather than only outvote it: a fixed pipeline (plan, execute, check), or an
orchestrator delegating sub-steps as it goes. Tested on tool tasks only.
_Avoid_: chain, pipeline (when meaning the topology rather than the code)

**Handoff**:
What a later step in a sequence sees of the earlier steps, with the one fixed instruction that comes
with it. The same for every agent and every step.

**Best single agent**:
The one agent that does best on the training folds, used alone for every task. The fixed baseline;
the agent that does best on the test folds, in hindsight, is reported only as an upper bound.
_Avoid_: best fixed configuration, manual selection

**Bare model**:
One call to one model with no persona and no team. A baseline.

### Judging

**Cost**:
What answering a task spends, including the orchestrator's own calls. Measured three ways: tokens
(thinking included), compute (tokens weighted by each model's active parameters), and wall-clock.

**Development split**:
A slice of questions, stratified by source, set aside for choices made before evaluation, such as
picking a team's members. Never used to report results.

**Oracle**:
For each task, the cheapest single agent that answered it correctly, known only in hindsight. The
ceiling for choosing among agents; a sequential topology can in principle beat it.

**Oracle gate**:
An escalation trigger that fires exactly when the orchestrator's own answer is wrong, known only in
hindsight. The ceiling for deciding when to escalate, as the oracle is for choosing agents.

**Replay**:
Scoring a configuration from answers the agents already gave on their own, without running it
again. Possible only for topologies where agents do not see each other's answers (vote, cascade,
and the pass-on and second-opinion forms of escalation). Interacting topologies (debate,
centralized, synthesis, sequential, review) are run live.

**Efficiency**:
Where a method sits on the accuracy-against-cost curve. One method is more efficient than another
when it reaches at least the same accuracy for less cost. Judged on each cost measure separately;
no measure is the headline.
_Avoid_: speed, throughput
