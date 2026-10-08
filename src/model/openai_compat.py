from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_MAX_TOKENS = 4096


def thinking_budget_extra_body() -> Dict[str, int]:
    """Request fields that cap a reasoning model's thinking, from the environment.

    A thinking model that is still reasoning when it reaches max_tokens returns
    an empty answer, which scores as wrong. vLLM can cap the thinking instead
    (`thinking_token_budget`) and force the end-of-thinking marker, leaving the
    rest of max_tokens for the answer. THINKING_TOKEN_BUDGET opts in. It is off
    by default because vLLM rejects the field on a server started without
    --reasoning-config, which is every non-thinking model.
    """
    budget = os.environ.get("THINKING_TOKEN_BUDGET")
    return {"thinking_token_budget": int(budget)} if budget else {}


_CONTEXT_MARKERS = ("maximum context length", "max_model_len", "context length exceeded",
                    "context_length_exceeded", "prompt is too long")


def is_context_limit(error) -> bool:
    """Whether a failed call was refused for exceeding the model's context window.

    vLLM answers 400 with "This model's maximum context length is N tokens ..."
    when the messages plus `max_tokens` do not fit. That is a limit of the
    serving setup, not a wrong answer, so callers record it as `limit:
    "context"` and reports count it apart.
    """
    text = str(getattr(error, "message", "") or error).lower()
    return any(marker in text for marker in _CONTEXT_MARKERS)


def limit_of(finish_reason) -> Optional[str]:
    """`"max_tokens"` when a reply was cut off by the generation budget, else None."""
    return "max_tokens" if finish_reason == "length" else None


@dataclass
class Completion:
    """One generation and what it cost.

    `complete()` returns only the text, which is all the debate harness wants.
    Anything that compares configurations needs more: how many tokens a call
    used (so compute can be matched across architectures), why it stopped (a
    `length` stop is a truncation, not an answer), how long it took, and which
    model and server produced it.
    """

    content: str = ""
    reasoning: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    latency_s: Optional[float] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    seed: Optional[int] = None
    # Function calls the model made, when the request offered `tools`:
    # [{"id", "name", "arguments": dict, "arguments_raw": str, "error": str|None}].
    # `error` is set when the arguments were not valid JSON; `arguments` is then {}.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def assistant_message(self) -> Dict[str, Any]:
        """This turn as a chat message, to append to the conversation before the tool results."""
        message: Dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": call["id"], "type": "function",
                 "function": {"name": call["name"], "arguments": call["arguments_raw"]}}
                for call in self.tool_calls
            ]
        return message

    @property
    def text(self) -> str:
        """The text to score: the answer, or the reasoning if the answer is empty.

        A reasoning model that spends its whole budget thinking returns an
        empty `content` with the text under `reasoning_content`. Scoring that
        as an empty answer is indistinguishable from a wrong one.
        """
        return self.content if self.content.strip() else self.reasoning

    @property
    def used_reasoning(self) -> bool:
        """True when `text` came from the reasoning because the answer was empty."""
        return not self.content.strip() and bool(self.reasoning.strip())

    def usage(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "latency_s": self.latency_s,
        }


class OpenAICompatChatWrapper:
    """OpenAI-compatible chat wrapper.

    This is intended for *local* OpenAI-compatible servers such as vLLM's
    OpenAI API server ("vllm serve ..."), so you can run open-source models
    without loading them in this process.

    The rest of the codebase expects an "agent" object.
    For this wrapper we expose:
      - kind = "openai_compat"
      - model_name: str
      - complete(messages, ...) -> str
      - generate(messages, ...) -> Completion

    `complete` is what the debate harness has always called. `generate` returns
    the text with its token usage, stop reason and latency, for the runner. The
    last response's usage is also kept on `last_usage`.
    """

    kind = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        timeout: Optional[float] = 300.0,
        max_retries: int = 4,
        limiter=None,
    ):
        """`limiter`, if given, is a semaphore held for the duration of each call.

        Sharing one across every client of a server caps how many requests are
        in flight there. At 1, every request runs alone, which is what makes
        greedy output reproducible.
        """
        self.base_url = base_url.rstrip("/")
        self.limiter = limiter
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_usage = None

        try:
            # openai>=1.0 provides OpenAI client
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError(
                "OpenAICompatChatWrapper requires the official OpenAI Python SDK. "
                "Install with: pip install 'openai>=1.0.0'"
            ) from e

        # vLLM OpenAI server uses the same endpoints as OpenAI.
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Completion:
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        extra_body = {**thinking_budget_extra_body(), **kwargs.pop("extra_body", {})}
        if seed is not None:
            kwargs["seed"] = seed
        request = dict(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body or None,
            **kwargs,
        )
        if self.limiter is not None:
            with self.limiter:
                started = time.monotonic()
                resp = self._client.chat.completions.create(**request)
                latency = time.monotonic() - started
        else:
            started = time.monotonic()
            resp = self._client.chat.completions.create(**request)
            latency = time.monotonic() - started
        self.last_usage = getattr(resp, "usage", None)

        choice = resp.choices[0]
        message = choice.message
        reasoning = ""
        for attribute in ("reasoning_content", "reasoning"):
            reasoning = getattr(message, attribute, None) or ""
            if reasoning.strip():
                break
        usage = self.last_usage
        tool_calls = []
        for index, call in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(call, "function", None)
            raw = getattr(function, "arguments", None) or "{}"
            try:
                arguments, error = json.loads(raw), None
                if not isinstance(arguments, dict):
                    arguments, error = {}, f"arguments are not a JSON object: {raw[:200]}"
            except json.JSONDecodeError as exc:
                arguments, error = {}, f"arguments are not valid JSON ({exc.msg}): {raw[:200]}"
            tool_calls.append({
                "id": getattr(call, "id", None) or f"call_{index}",
                "name": getattr(function, "name", "") or "",
                "arguments": arguments,
                "arguments_raw": raw,
                "error": error,
            })
        return Completion(
            content=getattr(message, "content", None) or "",
            reasoning=reasoning,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=getattr(choice, "finish_reason", None),
            latency_s=round(latency, 3),
            model=self.model_name,
            endpoint=self.base_url,
            seed=seed,
            tool_calls=tool_calls,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        return self.generate(
            messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, **kwargs
        ).text
