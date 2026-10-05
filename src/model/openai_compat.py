from __future__ import annotations

import os
from typing import Dict, List, Optional


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
        max_tokens = 4096
        extra_body = {**thinking_budget_extra_body(), **kwargs.pop("extra_body", {})}
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body or None,
            **kwargs,
        )
        return resp.choices[0].message.content or ""
