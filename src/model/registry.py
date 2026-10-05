"""Which model a stage means, and where it is served.

Every team used to run on one model at one endpoint: `run_team_evaluation`
repeated `--model_name` once per agent and pointed them all at one base URL.
Mixed teams (a 35B reviser after a 0.8B drafter, a 3B hub over 0.8B workers)
need each stage to name a model and have that name resolve to a server.

A stage names a model by key. Keys come from two places:

- `default`, always present: `--model_name` at the agent endpoint, which is
  what every run so far has used.
- `--models_file`, a JSON object mapping key -> {"served_name", "base_url",
  and optionally "api_key", "context_length", "supports_tools"}. A file entry
  named `default` replaces the one built from the flags.

The served name is accepted as a key too, so a config can say
`"model": "Qwen/Qwen3.5-0.8B"` as readily as `"model": "qwen0.8b"`.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from model.openai_compat import OpenAICompatChatWrapper

DEFAULT_KEY = "default"
DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    served_name: str
    base_url: str
    api_key: str = "EMPTY"
    context_length: Optional[int] = None
    supports_tools: bool = False

    def describe(self) -> Dict:
        """What a record should say about this model; never the API key."""
        record = asdict(self)
        record.pop("api_key", None)
        return record


class ModelRegistry:
    """Resolves model keys to specs and hands out one client per model."""

    def __init__(self, specs: Dict[str, ModelSpec]):
        if DEFAULT_KEY not in specs:
            raise ValueError(f"a model registry needs a {DEFAULT_KEY!r} entry")
        self._specs = dict(specs)
        self._clients: Dict[str, OpenAICompatChatWrapper] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_args(cls, args) -> "ModelRegistry":
        """`default` from the agent flags, plus anything in `--models_file`.

        The agent endpoint resolves the way `run_team_evaluation` always did:
        `--vllm_base_url`, else `--api_base_url`, else port 8001.
        """
        served = getattr(args, "model_name", None) or getattr(args, "model", None)
        if not served:
            raise ValueError("no agent model: set --model_name")
        base_url = (
            getattr(args, "vllm_base_url", "")
            or getattr(args, "api_base_url", "")
            or DEFAULT_BASE_URL
        )
        specs = {
            DEFAULT_KEY: ModelSpec(
                key=DEFAULT_KEY,
                served_name=served,
                base_url=base_url.rstrip("/"),
                api_key=getattr(args, "vllm_api_key", "EMPTY") or "EMPTY",
            )
        }
        models_file = getattr(args, "models_file", None)
        if models_file:
            specs.update(load_models_file(models_file))
        return cls(specs)

    def resolve(self, key: Optional[str]) -> ModelSpec:
        key = key or DEFAULT_KEY
        if key in self._specs:
            return self._specs[key]
        for spec in self._specs.values():
            if spec.served_name == key:
                return spec
        raise KeyError(f"unknown model {key!r}; registered: {sorted(self._specs)}")

    def client(self, key: Optional[str]) -> OpenAICompatChatWrapper:
        """One client per model, shared across threads (the OpenAI client is thread-safe)."""
        spec = self.resolve(key)
        with self._lock:
            client = self._clients.get(spec.key)
            if client is None:
                client = OpenAICompatChatWrapper(
                    base_url=spec.base_url,
                    model_name=spec.served_name,
                    api_key=spec.api_key,
                )
                self._clients[spec.key] = client
            return client

    def keys(self):
        return sorted(self._specs)


def load_models_file(path) -> Dict[str, ModelSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object mapping model keys to specs")
    specs = {}
    for key, entry in raw.items():
        if key.startswith("_"):
            continue  # room for a "_comment" entry
        missing = {"served_name", "base_url"} - set(entry)
        if missing:
            raise ValueError(f"{path}: model {key!r} is missing {sorted(missing)}")
        specs[key] = ModelSpec(
            key=key,
            served_name=entry["served_name"],
            base_url=entry["base_url"].rstrip("/"),
            api_key=entry.get("api_key", "EMPTY"),
            context_length=entry.get("context_length"),
            supports_tools=bool(entry.get("supports_tools", False)),
        )
    return specs
