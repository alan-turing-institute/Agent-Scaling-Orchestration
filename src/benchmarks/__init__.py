"""The benchmark registry.

Adding a benchmark is one module in this package plus one line in `_MODULES`.
It used to be eight edits spread across `data_utils.py`, `evaluator.py`,
`main.py` (three places), `model_utils.py`, `team_evaluation.py`,
`bare_model_baseline.py` and `tag_questions.py`, with nothing to tell you when
you had missed one - a missed edit did not raise, it scored the new benchmark
with the multiple-choice parser and reported a number.

A module here declares:

    NAME         the value `--data` takes and the tagged dataset records
    ANSWER_TYPE  which scorer reads its answers (benchmarks.scorers.SCORERS)
    PERSONA_SET  the personas the paper assigned it
    load(args, split) -> (questions, labels)

or, instead of `load`, `load_instances(args, split) -> list[Instance]` to attach
ids, metadata and structural tags (the registry derives `load` from it). And
optionally `fetch(args)`, for data its loader does not download itself.
"""

from __future__ import annotations

import importlib
import json
import threading
from dataclasses import dataclass
from typing import Any

from benchmarks.base import Instance, Prediction, ScoreResult, load_instances, question_id
from benchmarks.scorers import LENIENT, STRICT, get_scorer, score_responses

# module path -> imported lazily, so one benchmark's optional dependency cannot
# stop the registry from listing the others.
_MODULES = [
    "benchmarks.gsm8k",
    "benchmarks.arc",
    "benchmarks.hellaswag",
    "benchmarks.truthfulqa",
    "benchmarks.winogrande",
    "benchmarks.mmlu_pro_medicine",
    "benchmarks.mmlu_formal_logic",
    # Harder static sets (H1). No persona set from the paper.
    "benchmarks.gpqa_diamond",
    "benchmarks.mmlu_pro",
    "benchmarks.aime",
    "benchmarks.math500",
]


@dataclass
class ModuleBenchmark:
    """Adapts a benchmark module to the `Benchmark` protocol."""

    name: str
    answer_type: str
    persona_set: list[str]
    _load: Any
    # Optional: downloads the loader does not do itself (scripts/fetch_benchmarks.py).
    fetch: Any = None
    # Optional: the module's own `load_instances`, with ids, metadata and structural tags.
    _instances: Any = None

    def load(self, args, split: str = "test"):
        if self._load is None:
            instances = self._instances(args, split=split)
            return [i.question for i in instances], [i.answer for i in instances]
        return self._load(args, split=split)

    def instances(self, args, split: str = "test") -> list[Instance]:
        if self._instances is not None:
            return self._instances(args, split=split)
        return load_instances(self, args, split=split)

    def scorer(self, mode: str = STRICT, bae: bool = False):
        return get_scorer(self.answer_type, mode=mode, bae=bae)

    def instruction_suffix(self, style: str = "plain") -> str:
        return self.scorer().instruction_suffix(style)


_REGISTRY: dict[str, ModuleBenchmark] = {}


def _load_registry() -> dict[str, ModuleBenchmark]:
    if _REGISTRY:
        return _REGISTRY
    for path in _MODULES:
        module = importlib.import_module(path)
        benchmark = ModuleBenchmark(
            name=module.NAME,
            answer_type=module.ANSWER_TYPE,
            persona_set=list(module.PERSONA_SET),
            _load=getattr(module, "load", None),
            fetch=getattr(module, "fetch", None),
            _instances=getattr(module, "load_instances", None),
        )
        if benchmark._load is None and benchmark._instances is None:
            raise TypeError(f"{path} defines neither load nor load_instances")
        _REGISTRY[benchmark.name] = benchmark
    return _REGISTRY


def get(name: str) -> ModuleBenchmark:
    """Look up a benchmark, or say what there is."""
    registry = _load_registry()
    try:
        return registry[name]
    except KeyError:
        raise KeyError(
            f"unknown benchmark {name!r}; registered: {sorted(registry)}"
        ) from None


def list_names() -> list[str]:
    return sorted(_load_registry())


def answer_type_of(name: str) -> str:
    """Which scorer reads this benchmark's answers.

    This replaces guessing from the gold answer's shape. `float()` on the gold
    answer classified everything non-numeric as multiple choice, so a coding
    benchmark would have been scored by the MCQ letter parser without erroring.
    """
    return get(name).answer_type


def answer_type_of_sample(sample) -> str:
    """Which scorer reads this question: the answer type its benchmark declared.

    Read from the `dataset` field every tagged row carries, per question, since
    a batch can mix benchmarks. A row without one, or naming a benchmark that
    is not registered, is an error. The old fallback guessed from the gold
    answer's shape, and anything non-numeric went to the multiple-choice parser,
    which reads a letter out of any text and reports a number.
    """
    source = sample.get("dataset") if hasattr(sample, "get") else None
    if not source:
        question = str(sample.get("question", "") if hasattr(sample, "get") else sample)
        raise ValueError(
            f"question has no 'dataset' field, so its scorer is unknown: {question[:80]!r}"
        )
    return answer_type_of(source)


class ScorerSet:
    """The scorers one run needs, each built the first time a question asks for it.

    Replaces a dict built up front for `("numeric", "mcq")` in every entry point,
    which meant a benchmark with any other answer type needed edits in each of
    them. Kept for the life of a run, so a scorer that holds state (a judge's
    verdict cache, say) keeps it across questions. Safe to share between the
    threads that score questions concurrently.
    """

    def __init__(self, mode: str = STRICT, bae: bool = False):
        self.mode = mode
        self.bae = bae
        self._scorers = {}
        self._lock = threading.Lock()

    def get(self, answer_type: str):
        with self._lock:
            if answer_type not in self._scorers:
                self._scorers[answer_type] = get_scorer(answer_type, mode=self.mode, bae=self.bae)
            return self._scorers[answer_type]

    def for_sample(self, sample):
        """`(answer_type, scorer)` for one question."""
        answer_type = answer_type_of_sample(sample)
        return answer_type, self.get(answer_type)


def metadata_of(sample) -> dict:
    """A tagged row's metadata: stored as a JSON string, absent in older pools."""
    raw = sample.get("metadata") if hasattr(sample, "get") else None
    if not raw:
        return {}
    return raw if isinstance(raw, dict) else json.loads(raw)


def instance_of(sample) -> Instance:
    """A tagged row as an `Instance`, for scorers that read more than the gold answer.

    Rows from the 699-question pool predate ids and metadata; they get the id
    the registry would have given them and empty metadata.
    """
    question = sample["question"]
    return Instance(
        question=question,
        answer=sample["answer"],
        tags=list(sample.get("tags") or []),
        metadata=metadata_of(sample),
        id=sample.get("id") or question_id(sample.get("dataset") or "", question),
    )


def persona_set(name: str) -> list[str]:
    return list(get(name).persona_set)


def scorer_for(name: str, mode: str = STRICT, bae: bool = False):
    return get(name).scorer(mode=mode, bae=bae)


__all__ = [
    "Instance",
    "LENIENT",
    "ModuleBenchmark",
    "Prediction",
    "STRICT",
    "ScoreResult",
    "ScorerSet",
    "answer_type_of",
    "answer_type_of_sample",
    "get",
    "get_scorer",
    "instance_of",
    "list_names",
    "metadata_of",
    "persona_set",
    "question_id",
    "score_responses",
    "scorer_for",
]
