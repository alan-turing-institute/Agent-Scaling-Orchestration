"""A frontier model as an agent: each call is one fresh, headless Claude Code session (`claude -p`).

The frontier baseline asks Claude (Sonnet by default) to solve the same questions, in the same
harness, as the open models: `bare_model_baseline.py --claude_cli` swaps this in for
`OpenAICompatChatWrapper`, so prompts, folds, turn caps, tools and scoring are unchanged.

Every call starts a new session with no memory of any other: no tools of Claude Code's own, no
settings, MCP servers or CLAUDE.md, a replaced system prompt, and an empty working directory. A
static question is one call. An agentic task is one call per turn: the conversation so far is
rendered as text, the environment's tools are listed in the system prompt, and the reply is
forced (`--json-schema`) into either one tool call or a message. The harness runs the tools,
as it does for every other model.

What the CLI cannot set: temperature and top_p (the model's defaults apply) and seed.
`max_tokens` is passed as CLAUDE_CODE_MAX_OUTPUT_TOKENS and covers thinking too.

Each call's usage and list-price cost go to `usage_log` (JSON lines), keyed by a hash of the
task's first user message so calls can be summed per question.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import APIConnectionError

from model.openai_compat import Completion

DEFAULT_SYSTEM = "You are a helpful assistant."

TOOL_PROTOCOL = """

You work on the task by choosing actions, one per reply. You cannot run an action yourself and
must not try to call it as a function: instead, your structured output names it. Set "action" to
"tool", "tool" to the action's name and "arguments" to its arguments; the environment runs it and
shows you the result in the next turn, as <tool_result>. When the task asks for a reply, or you
have nothing further to do, set "action" to "message" and put your reply in "message".

Available actions (OpenAI function schemas; every one of them is available):
{tools}"""

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["tool", "message"]},
        "tool": {"type": "string", "description": "The tool's name, when action is tool"},
        "arguments": {"type": "object", "description": "The tool's arguments, when action is tool"},
        "message": {"type": "string", "description": "Your message; may be empty when calling a tool"},
    },
    "required": ["action"],
}

# Failures that mean the model could not be reached (not a wrong answer): the harness waits and retries.
_UNAVAILABLE = ("rate limit", "rate_limit", "usage limit", "overloaded", "529", "503", "502", "timed out",
                "connection", "network", "authentication", "oauth", "credential", "limit reached")


class ClaudeUnavailable(APIConnectionError):
    def __init__(self, message: str):
        super().__init__(message=message, request=httpx.Request("POST", "claude-cli://headless"))


def _render(messages: List[Dict[str, Any]]) -> str:
    """The conversation after the system prompt, as one prompt: the task, then each turn in order."""
    parts = []
    for i, m in enumerate(messages):
        role, content = m["role"], m.get("content") or ""
        if role == "user" and i == 0:
            parts.append(content)
        elif role == "user":
            parts.append(f"<user>\n{content}\n</user>")
        elif role == "assistant":
            calls = [{"tool": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}
                     for c in m.get("tool_calls") or []]
            body = content + ("\n" if content and calls else "") + "\n".join(
                f"[tool call] {json.dumps(c)}" for c in calls)
            parts.append(f"<your_previous_step>\n{body}\n</your_previous_step>")
        elif role == "tool":
            parts.append(f"<tool_result>\n{content}\n</tool_result>")
    if len(messages) > 1:
        parts.append("Take the next step: give your structured output, one action or a message.")
    return "\n\n".join(parts)


class ClaudeCLIChatWrapper:
    kind = "claude_cli"

    def __init__(self, model_name: str = "sonnet", usage_log: Optional[str] = None,
                 timeout: Optional[float] = 3600, effort: Optional[str] = None, max_retries: int = 3,
                 claude_bin: str = "claude", limit_wait_hours: float = 12, limit_poll_seconds: int = 300):
        self.model_name = model_name
        self.usage_log = usage_log
        self.timeout = timeout
        self.effort = effort
        self.max_retries = max_retries
        self.claude_bin = claude_bin
        self.limit_wait_hours = limit_wait_hours
        self.limit_poll_seconds = limit_poll_seconds
        self.workdir = tempfile.mkdtemp(prefix="claude-cli-empty-")
        self._lock = threading.Lock()
        self.last_usage = None

    def _run(self, prompt: str, system: str, max_tokens: Optional[int], schema: Optional[dict]) -> dict:
        cmd = [self.claude_bin, "-p", "--model", self.model_name, "--system-prompt", system,
               "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
               "--exclude-dynamic-system-prompt-sections", "--disable-slash-commands",
               "--output-format", "stream-json", "--verbose"]
        if self.effort:
            cmd += ["--effort", self.effort]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema)]
        env = dict(os.environ)
        if max_tokens:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        last, attempt, waited_since = "", 0, None
        while attempt < self.max_retries:
            attempt += 1
            try:
                proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=self.workdir,
                                      env=env, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                last = f"claude -p timed out after {self.timeout}s"
                continue
            out, attempted = None, []
            for line in proc.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "result":
                    out = event
                elif event.get("type") == "assistant":
                    attempted += [b for b in event["message"].get("content", [])
                                  if b.get("type") == "tool_use" and b.get("name") != "StructuredOutput"]
            if out is not None and not out.get("is_error"):
                out["attempted_tool_calls"] = [{"name": b.get("name"), "input": b.get("input")} for b in attempted]
                return out
            last = ((f"{out.get('result')!r} " + json.dumps(out)[:800]) if out
                    else (proc.stderr or proc.stdout)[-1000:]) or f"exit {proc.returncode}"
            # No model call was made (a usage limit, an outage, an expired login): not an answer, so
            # wait for it to clear rather than score the task. Does not use up the retries.
            if out is not None and not out.get("duration_api_ms") and not out.get("total_cost_usd"):
                waited_since = waited_since or time.monotonic()
                if time.monotonic() - waited_since > self.limit_wait_hours * 3600:
                    raise ClaudeUnavailable(f"no model call for {self.limit_wait_hours}h: {last}")
                print(f"[claude-cli] no model call ({str(out.get('result'))[:200]}); waiting "
                      f"{self.limit_poll_seconds}s", flush=True)
                time.sleep(self.limit_poll_seconds)
                attempt -= 1
                continue
            if "prompt is too long" in last.lower() or "context" in last.lower() and "too long" in last.lower():
                raise RuntimeError(f"prompt is too long: {last}")
            time.sleep(min(60, 10 * attempt))
        if any(marker in last.lower() for marker in _UNAVAILABLE):
            raise ClaudeUnavailable(last)
        raise RuntimeError(f"claude -p failed {self.max_retries} times: {last}")

    def _log(self, messages, out, tools: bool):
        if not self.usage_log:
            return
        first_user = next((m.get("content") or "" for m in messages if m["role"] == "user"), "")
        usage = out.get("usage") or {}
        row = {"ts": time.time(), "task_sha": hashlib.sha1(first_user.encode()).hexdigest()[:16],
               "turn": sum(1 for m in messages if m["role"] == "assistant"), "agentic": tools,
               "cost_usd": out.get("total_cost_usd"), "model_usage": out.get("modelUsage"),
               "input_tokens": usage.get("input_tokens"),
               "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
               "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
               "output_tokens": usage.get("output_tokens"),
               "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
               "stop_reason": out.get("stop_reason"), "duration_ms": out.get("duration_ms"),
               "num_turns": out.get("num_turns"),
               "native_attempts": [c["name"] for c in out.get("attempted_tool_calls") or []]}
        with self._lock, open(self.usage_log, "a") as f:
            f.write(json.dumps(row) + "\n")

    def generate(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, top_p: Optional[float] = None,
                 seed: Optional[int] = None, tools: Optional[list] = None, **kwargs) -> Completion:
        system = DEFAULT_SYSTEM
        if messages and messages[0]["role"] == "system":
            system, messages = messages[0]["content"], messages[1:]
        if tools:
            system += TOOL_PROTOCOL.format(tools=json.dumps(tools, indent=1))
        started = time.monotonic()
        out = self._run(_render(messages), system, max_tokens, REPLY_SCHEMA if tools else None)
        latency = time.monotonic() - started
        self._log(messages, out, bool(tools))
        usage = out.get("usage") or {}
        self.last_usage = usage
        prompt_tokens = sum(usage.get(k) or 0 for k in
                            ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        stop = out.get("stop_reason")
        content, tool_calls = out.get("result") or "", []
        if tools:
            reply = out.get("structured_output") or {}
            content = reply.get("message") or ""
            names = {t["function"]["name"] for t in tools}
            # A tool the model tried to call natively (not available to it) still counts as its step.
            native = [c for c in out.get("attempted_tool_calls") or [] if c["name"] in names]
            call = None
            if reply.get("action") == "tool" and reply.get("tool"):
                call = (reply["tool"], reply.get("arguments"))
            elif native:
                call = (native[0]["name"], native[0]["input"])
            if call:
                arguments = call[1] if isinstance(call[1], dict) else {}
                n = sum(1 for m in messages if m["role"] == "assistant")
                tool_calls = [{"id": f"call_{n}", "name": call[0], "arguments": arguments,
                               "arguments_raw": json.dumps(arguments), "error": None}]
        finish = "length" if stop == "max_tokens" else ("tool_calls" if tool_calls else "stop")
        return Completion(content=content, reasoning="", prompt_tokens=prompt_tokens,
                          completion_tokens=usage.get("output_tokens"), finish_reason=finish,
                          latency_s=round(latency, 3), model=self.model_name, endpoint="claude-cli",
                          seed=None, tool_calls=tool_calls)

    def complete(self, messages, max_tokens=None, temperature=None, top_p=None, **kwargs) -> str:
        return self.generate(messages, max_tokens=max_tokens, **kwargs).text
