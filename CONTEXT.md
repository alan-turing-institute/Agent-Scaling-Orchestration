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
The agent that, for each task, either answers it directly or delegates it by choosing a
configuration, weighing what each agent gets right against what it costs.
_Avoid_: router, selector, planner

**Direct answer**:
The orchestrator answering a task itself, with no team.

**Delegation**:
The orchestrator handing a task to a configuration of other agents.

**Decide first**:
The orchestrator choosing between a direct answer and delegation before it has tried the task.
One of two ways it is run; compared with solve first at matched cost.

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

**Hub**:
The agent in a centralized topology that reads every other team member's answer and gives the
team's final answer. It combines answers; it does not choose the team, so a hub is not an
orchestrator.
_Avoid_: orchestrator, aggregator (when meaning the hub)

**Worker**:
A team member in a centralized topology. It answers without seeing anyone else's work, so its
answer is also its solo answer; the hub reads it.
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
earlier one's answer rather than only outvote it. Not a fixed template: for each task the
orchestrator chooses how many steps there are and which pool agent takes each. An agent is the same
in a sequence as in any other topology; there are no separate role prompts, so a reviewer-like step
exists only if such an agent is in the pool.
_Avoid_: chain, pipeline (when meaning the topology rather than the code)

**Cost budget**:
The most a task may spend, stated to the orchestrator. Its own calls, including a direct answer,
count against it. Running at several budgets traces the orchestrator's curve.

**Agent card**:
What the orchestrator is told about each agent: what it is, how accurate it was on the training
folds, and what it cost.

**Continual memory**:
The orchestrator's record of its own past choices and how they turned out. Tested as an ablation
against agent cards alone.

**Handoff**:
What a later step in a sequence sees of the earlier steps, with the one fixed instruction that comes
with it. The same for every agent and every step.

**Best single agent**:
The one agent that does best on the training folds, used alone for every task. The fixed baseline;
it stays the same whatever topology the orchestrator picks.
_Avoid_: best fixed configuration, manual selection

**Random configuration**:
A configuration drawn at random from the pool, at a stated team size. A baseline.

**Bare model**:
One call to one model with no persona and no team. A baseline.

### Judging

**Cost**:
What answering a task spends, including the orchestrator's own calls. Measured three ways: tokens
(thinking included), compute (tokens weighted by each model's active parameters), and wall-clock.

**Development split**:
A slice of questions, stratified by source, set aside for choices made before evaluation:
shortlisting personas, admitting specialists, placing the cost budgets. Never used to report results.

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
