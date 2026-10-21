"""Agentic benchmarks: an environment the agent acts in through tools.

A static benchmark asks a question and reads the answer out of the reply. An
agentic one hands the agent an environment - a crafting table, a mailbox, a
search index - and grades what the agent did to it. Two pieces make that fit
the rest of the harness:

- `Environment`, one instance of a task: the prompt that describes it, the
  tools the agent may call (OpenAI function schemas), what each call returns,
  whether the episode is over, and the outcome. `fork()` copies its state, so
  voters can each try the task from the same start and a pipeline stage can
  carry on from where the last one left off without touching its record.
- `OutcomeScorer`, the scorer for answer type `outcome`. The runner turns an
  episode's outcome into a `Prediction` whose value is
  `"success|<fingerprint>"` or `"fail|<fingerprint>"`. The fingerprint names
  the end state (what was crafted, what was submitted), so a team votes over
  end states the way it votes over answers, and the winning state carries its
  own verdict.

A benchmark module becomes agentic by declaring

    ANSWER_TYPE = "outcome"
    ENVIRONMENT = factory          # factory(instance) -> a fresh Environment
    MAX_STEPS = 30                 # default cap on an agent's turns

and, like any benchmark, `load_instances`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from benchmarks.base import Prediction


@dataclass
class Outcome:
    """How an episode ended."""

    success: bool
    fingerprint: str              # names the end state; equal end states, equal strings
    detail: dict = field(default_factory=dict)

    def prediction(self) -> Prediction:
        verdict = "success" if self.success else "fail"
        return Prediction(value=f"{verdict}|{self.fingerprint}", parsed=True,
                          raw=json.dumps(self.detail, default=str))


@runtime_checkable
class Environment(Protocol):
    """One task instance, acted on through tools."""

    done: bool

    def task_prompt(self) -> str:
        """What the agent is asked to do, with the starting observation."""
        ...

    def tools(self) -> list[dict]:
        """OpenAI function-tool schemas: [{"type": "function", "function": {...}}]."""
        ...

    def call(self, name: str, arguments: dict) -> str:
        """Run one tool call and return the observation. Errors come back as text, never raised."""
        ...

    def outcome(self) -> Outcome:
        ...

    def fork(self) -> "Environment":
        """An independent copy of the current state."""
        ...

    # Optional: `resume()`, called on a fork before another agent carries on
    # from it. It clears endings that are a verdict rather than a state - a
    # declaration that the task is impossible, a submitted answer - so a critic
    # can overrule them, while a state that is itself the goal (an item crafted)
    # stays done. Without it, a finished environment stays finished.
    #
    # Optional: `text_reply_ends = True` for tasks that end with a final message
    # rather than a tool call (WorkBench): a reply without a tool call then ends
    # the episode at once instead of drawing a nudge. And `close(stopped)`,
    # called before `outcome()` with why the episode stopped ("done", "reply",
    # "max_steps", ...), for tasks whose grade depends on it.


def function_tool(name: str, description: str, properties: dict, required: Sequence[str] = ()) -> dict:
    """An OpenAI function-tool schema, so modules do not repeat the boilerplate."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)},
        },
    }


class OutcomeScorer:
    """Agentic tasks: graded on the episode's outcome, not on the reply's text."""

    name = "outcome"

    def __init__(self, mode: str = "strict"):
        self.mode = mode

    def instruction_suffix(self, style: str = "plain") -> str:
        # The tools carry the format; there is no answer line to ask for.
        return ""

    def normalise_gold(self, gold: Any) -> Any:
        return gold

    def extract(self, text: str) -> Prediction:
        # Text alone has no outcome: an episode's prediction comes from its
        # environment (`Outcome.prediction`), never from parsing a reply.
        return Prediction.unparsed(raw=text or "")

    def correct(self, prediction: Prediction, gold: Any, instance=None) -> bool:
        return bool(prediction.parsed and str(prediction.value).startswith("success|"))

    def aggregate(self, predictions: Sequence[Prediction], rng=None) -> Prediction:
        from benchmarks.scorers import _majority
        return _majority(predictions, rng=rng)
