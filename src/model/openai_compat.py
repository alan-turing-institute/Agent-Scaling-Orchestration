from __future__ import annotations

from typing import Dict, List, Optional

# One place that decides how long a generation may be. `complete()` used to
# overwrite whatever the caller asked for with a hardcoded 4096, so
# --max_new_tokens and every persona's max_new_tokens were inert on every
# vLLM run. The constant keeps the old effective value as the *default* while
# letting a caller that passes a budget actually get it.
DEFAULT_MAX_TOKENS = 4096


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

    The last response's token usage is kept on `last_usage` so a caller can
    record what a run cost; the OpenAI SDK object itself is not returned,
    because every caller here wants the text.
    """

    kind = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        timeout: Optional[float] = 300.0,
        max_retries: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
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

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        self.last_usage = getattr(resp, "usage", None)

        message = resp.choices[0].message
        content = getattr(message, "content", None) or ""
        if content.strip():
            return content
        # A reasoning model that spends its whole budget thinking returns an
        # empty `content` with the text under `reasoning_content`. Scoring that
        # as an empty answer is indistinguishable from a wrong one.
        for attribute in ("reasoning_content", "reasoning"):
            fallback = getattr(message, attribute, None) or ""
            if fallback.strip():
                return fallback
        return content
