"""The two contracts a benchmark has to satisfy, and nothing else.

Before this existed, "what shape is an answer" was spread across eight files as
`if args.data in [...]` chains, and the answer *type* was inferred by calling
`float()` on the gold answer - so anything non-numeric was scored as
multiple choice. A coding benchmark's gold answer is a string, so it would have
been handed to the MCQ parser and reported a number rather than an error.

Two objects fix that. A `Benchmark` says where its questions come from and what
shape its answers are. A `Scorer` owns the whole prompt/parse/aggregate
contract for one answer shape, so adding a benchmark whose answers are code, or
a short string, or an environment state, means writing a scorer rather than
editing every caller.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass
class Instance:
    """One question, however it is going to be answered.

    `metadata` is where a benchmark puts what only its own scorer understands:
    a test suite for a coding task, a rubric for a judged one, a repository
    snapshot, an environment seed. It must be JSON-serialisable: the tagged
    dataset stores it as a JSON string.

    `id` is stable across loads and machines, so records from different runs
    can be joined on it. `tags` here are structural, the labels a benchmark
    ships with (a subject, a difficulty level), as `"key: value"` strings;
    the model's capability tags are added when the pool is built.
    """

    question: str
    answer: Any
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    id: str = ""


@dataclass(frozen=True)
class Prediction:
    """What one agent's response was read as.

    `parsed` is the point of this class. Before it, a response the parser could
    not read and a response that was read and wrong were both the empty string
    and both counted as incorrect, so there was no way to tell a weak model from
    a broken extractor - or to measure whether a parser fix helped.
    """

    value: Any = None
    parsed: bool = False
    raw: str = ""

    @property
    def legacy(self) -> Any:
        """The value in the shape the pre-registry code wrote to history files.

        Unparsed answers were the empty string there, and `K_star_analysis` and
        every saved run still expect that.
        """
        return self.value if self.parsed else ""

    @classmethod
    def unparsed(cls, raw: str = "") -> "Prediction":
        return cls(value=None, parsed=False, raw=raw)


@dataclass
class ScoreResult:
    """One question scored across a team."""

    predictions: list[Prediction]
    aggregate: Prediction
    correct: bool
    gold: Any

    def legacy_tuple(self):
        """`(per_agent_answers, majority_answer, majority_is_correct)`.

        The shape every existing caller and every saved history file expects.
        Kept so that moving to scorers does not orphan a single result already
        on disk.
        """
        return (
            [p.legacy for p in self.predictions],
            self.aggregate.legacy,
            self.correct,
        )


@runtime_checkable
class Scorer(Protocol):
    """Owns one answer shape end to end."""

    name: str

    def instruction_suffix(self, style: str = "plain") -> str:
        """What to append to a question so the answer comes back readable.

        `style` is one of `plain`, `cot`, `bae`.
        """
        ...

    def normalise_gold(self, gold: Any) -> Any:
        """Coerce a stored gold answer into the type `correct` compares against.

        The tagged dataset stores every answer as a string, including the
        numeric ones; this is where that is undone, rather than at three
        separate call sites.
        """
        ...

    def extract(self, text: str) -> Prediction:
        ...

    def correct(self, prediction: Prediction, gold: Any, instance: "Instance | None" = None) -> bool:
        """Whether `prediction` answers the question.

        `instance` carries the question and its metadata for scorers that need
        more than the gold answer (a judge reads the question; a rubric lives
        in the metadata). Scorers that compare against `gold` alone ignore it.
        """
        ...

    def aggregate(self, predictions: Sequence[Prediction], rng=None) -> Prediction:
        """Reduce a team's predictions to the team's answer.

        `rng` breaks ties; without one, the global `random` does, as it always has.

        On the scorer rather than in the solver because this is the operation
        that does *not* generalise: majority vote over floats and over `(A)` are
        the same thing, but the useful reduction over a set of candidate
        programs is to run the tests, and over free text is to ask a judge.
        """
        ...


@runtime_checkable
class Benchmark(Protocol):
    """Where a set of questions comes from, and what shape its answers are."""

    name: str
    answer_type: str          # key into benchmarks.scorers.SCORERS
    persona_set: list[str]    # names into personas.chosen_persona_bank()

    def load(self, args, split: str = "test") -> tuple[list[str], list[Any]]:
        """Return `(questions, labels)`.

        The pair, rather than `list[Instance]`, because that is what
        `main.py` consumes. A module may define `load_instances` instead, to
        attach ids, metadata and structural tags; the registry derives `load`
        from it.
        """
        ...


def question_id(benchmark_name: str, question: str) -> str:
    """A stable id for a question that came without one: its benchmark and text hash."""
    return f"{benchmark_name}:{hashlib.sha1((question or '').encode('utf-8')).hexdigest()[:12]}"


def load_instances(benchmark: Benchmark, args, split: str = "test") -> list[Instance]:
    """Adapt a benchmark's `(questions, labels)` into `Instance` objects."""
    questions, labels = benchmark.load(args, split=split)
    return [Instance(question=q, answer=a, id=question_id(benchmark.name, q))
            for q, a in zip(questions, labels)]
