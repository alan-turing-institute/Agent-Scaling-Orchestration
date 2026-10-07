"""One agent working through an agentic task: think, call a tool, read the result, repeat.

`run_episode` is the loop a stage runs when its question comes with an
environment (`benchmarks.environment`). Each turn the model sees the
conversation so far and the environment's tools; every tool call it makes is
run and its observation appended. The episode ends when the environment says
it is done (the item is crafted, the answer submitted), when the agent replies
without calling a tool twice in a row, or at a cap.

Caps are on turns and tokens, never wall-clock: local inference is slow, and a
time limit would cut slow models off mid-task and make the result depend on the
server's load. `delegate` hands one extra tool to a hub: calling it runs a
worker's own episode on the hub's environment and returns the worker's report.

Every step is recorded - the tool, its arguments, a clipped observation, the
tokens - for `traces.jsonl` and the coordination metrics computed from it.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

from openai import APIConnectionError, APITimeoutError

NUDGE = ("You did not call a tool. Act with one of the tools you were given; "
         "if you believe the task is finished or cannot be done, call the tool for that.")
OBSERVATION_CHARS = 4000  # what a trace keeps of each observation; the model sees it whole


def _clip(text, limit=OBSERVATION_CHARS):
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + f"...[{len(text) - limit} more chars]"


def run_episode(client, env, *, system: str, user: str, max_steps: int, max_tokens: int,
                token_budget: Optional[int] = None, temperature=0, top_p=0.9, seed=None,
                delegate: Optional[Dict] = None, nudge: bool = True) -> Dict:
    """Run one agent on `env` and return its record.

    `delegate`, if given, is `{"tool": schema, "run": callable(arguments) -> str}`:
    an extra tool whose calls are answered by `run` instead of the environment.
    With `nudge`, a reply without a tool call gets one reminder before it ends
    the episode; a worker called on by a lead runs without it, since its text
    reply is its report back.
    Connection and timeout errors propagate, so the caller's retry can wait out a
    restarting server; the environment is left as the agent left it.
    """
    tools = list(env.tools())
    if delegate:
        tools.append(delegate["tool"])
    delegate_name = delegate["tool"]["function"]["name"] if delegate else None
    messages: List[Dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    steps, final_text = [], ""
    calls = prompt_tokens = completion_tokens = 0
    finish_reason, nudged, stopped = None, False, "max_steps"
    if env.done:
        # Carrying on from a state that already meets the goal: nothing to do.
        max_steps, stopped = 0, "already_done"

    for turn in range(max_steps):
        completion = client.generate(messages, max_tokens=max_tokens, temperature=temperature,
                                     top_p=top_p, seed=seed, tools=tools)
        calls += 1
        prompt_tokens += completion.prompt_tokens or 0
        completion_tokens += completion.completion_tokens or 0
        finish_reason = completion.finish_reason
        messages.append(completion.assistant_message())
        if completion.content.strip():
            final_text = completion.content

        if not completion.tool_calls:
            steps.append({"turn": turn, "tool": None, "text": _clip(completion.text),
                          "completion_tokens": completion.completion_tokens})
            if nudged or not nudge:
                stopped = "no_tool_call"
                break
            nudged = True
            messages.append({"role": "user", "content": NUDGE})
            continue
        nudged = False

        for call in completion.tool_calls:
            if call["error"]:
                observation = f"ERROR: {call['error']}"
            elif call["name"] == delegate_name:
                observation = delegate["run"](call["arguments"])
            else:
                observation = env.call(call["name"], call["arguments"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
            steps.append({"turn": turn, "tool": call["name"], "arguments": call["arguments"],
                          "observation": _clip(observation),
                          "completion_tokens": completion.completion_tokens})
            if env.done:
                break
        if env.done:
            stopped = "done"
            break
        if token_budget and completion_tokens >= token_budget:
            stopped = "token_budget"
            break

    outcome = env.outcome()
    return {
        "response": final_text,
        "steps": steps,
        "stopped": stopped,
        "outcome": {"success": outcome.success, "fingerprint": outcome.fingerprint, **outcome.detail},
        "prediction": outcome.prediction(),
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "tool_calls": sum(1 for s in steps if s["tool"]),
    }


def report_text(record: Dict) -> str:
    """What a stage that reads this episode sees of it: its actions and its last words.

    Only what the agent itself could observe. Never the grade: on a Plancraft
    task that is impossible, only the grader knows whether `impossible` was the
    right call, and telling the next agent would hand it the answer.
    """
    actions = [s for s in record.get("steps") or [] if s.get("tool")]
    lines = [f"Actions taken: {len(actions)}."]
    for step in actions[-8:]:
        lines.append(f"- {step['tool']}({json.dumps(step.get('arguments'), default=str)[:200]})"
                     f" -> {str(step.get('observation', ''))[:200]}")
    if record.get("response"):
        lines.append(f"Final message: {record['response'][-600:]}")
    return "\n".join(lines)
