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

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass
class Instance:
    """One question, however it is going to be answered.

    `metadata` is where a benchmark puts what only its own scorer understands:
    a test suite for a coding task, a rubric for a judged one, a repository
    snapshot, an environment seed.
    """

    question: str
    answer: Any
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


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

    def correct(self, prediction: Prediction, gold: Any) -> bool:
        ...

    def aggregate(self, predictions: Sequence[Prediction]) -> Prediction:
        """Reduce a team's predictions to the team's answer.

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
    persona_set: list[str]    # names into model_utils.chosen_persona_bank()

    def load(self, args, split: str = "test") -> tuple[list[str], list[Any]]:
        """Return `(questions, labels)`.

        The pair, rather than `list[Instance]`, because that is what
        `main.py` and `tag_questions.py` already consume. `load_instances`
        below adapts it for anything that wants the richer shape.
        """
        ...


def load_instances(benchmark: Benchmark, args, split: str = "test") -> list[Instance]:
    """Adapt a benchmark's `(questions, labels)` into `Instance` objects."""
    questions, labels = benchmark.load(args, split=split)
    return [Instance(question=q, answer=a) for q, a in zip(questions, labels)]
