"""One scorer per answer shape.

Each scorer owns the whole contract for its shape: what to ask the model for,
how to read the reply back, how to compare it to the gold answer, and how to
reduce a team's replies to one. Adding a benchmark whose answers are code or
free text means adding a scorer here, not editing `main.py`, `evaluator.py`,
`team_evaluation.py` and `bare_model_baseline.py` in step.

Two parsing modes. `strict` reproduces the parsers this repository has always
used, exactly, so every number already in `data-claude/` stays reproducible.
`lenient` fixes what those parsers get wrong. The default is `strict`: a parser
change moves every reported number, so it is opt-in, and a run records which
mode produced it.
"""

from __future__ import annotations

import collections
import random
import re
from typing import Any, Sequence

import numpy as np

from benchmarks.base import Prediction, ScoreResult

STRICT = "strict"
LENIENT = "lenient"

# The last `{...}` group in the response - what every instruction suffix asks for.
_BRACES = re.compile(r"\{(.*?)\}")
_NUMBER = re.compile(r"-?\d+\.?\d*")
_PAREN_LETTER = re.compile(r"\(([A-Za-z])\)")
_LABEL = re.compile(r"^\s*final\s*answer\s*:?\s*", re.IGNORECASE)


def extract_number(text):
    """Last number anywhere in the text, as the original evaluator defined it.

    Kept at module scope and re-exported through `evaluator` because the debate
    loop uses it to build the peer-opinion message, which is a different job
    from scoring.
    """
    if text:
        matches = _NUMBER.findall(text)
        if not matches:
            return ""
        return float(matches[-1])
    return ""


def _majority(predictions: Sequence[Prediction]) -> Prediction:
    """Most common parsed value; ties broken at random, as they always were.

    Uses the global `random`, which every entry point seeds, so a tie resolves
    the same way on a re-run with the same seed.
    """
    parsed = [p for p in predictions if p.parsed]
    if not parsed:
        return Prediction.unparsed()

    counter = collections.Counter(p.value for p in parsed)
    top = max(counter.values())
    winner = random.choice([value for value, count in counter.items() if count == top])
    return Prediction(value=winner, parsed=True, raw=str(winner))


class NumericScorer:
    """Answers that are a single number. GSM8K and anything shaped like it."""

    name = "numeric"

    def __init__(self, mode: str = STRICT):
        self.mode = mode

    def instruction_suffix(self, style: str = "plain") -> str:
        if style == "bae":
            return " Make sure to state your answer at the end of the response."
        if style == "cot":
            return (" Make sure to state your final answer in curly brackets at the very end of your"
                    " response, just like: '{final answer: 123}'. Let's think step by step.")
        return (' Make sure to state your final answer in curly brackets at the very end of your'
                ' response, just like: "{final answer: 123}".')

    def normalise_gold(self, gold: Any) -> Any:
        # The tagged dataset stores every answer as a string, numeric ones
        # included, and numpy raises when asked to round a string.
        return float(gold)

    def extract(self, text: str) -> Prediction:
        if self.mode == LENIENT:
            # Prefer the braces the prompt actually asked for. The strict parser
            # ignores them and takes the last number anywhere, so a response
            # ending "...{final answer: 8} over 3 days" scores as 3.
            for group in reversed(_BRACES.findall(text or "")):
                candidate = extract_number(_LABEL.sub("", group))
                if candidate != "":
                    return Prediction(value=np.round(candidate, 1), parsed=True, raw=group)

        value = extract_number(text)
        if value == "":
            return Prediction.unparsed(raw=text or "")
        return Prediction(value=np.round(value, 1), parsed=True, raw=str(value))

    def correct(self, prediction: Prediction, gold: Any) -> bool:
        if not prediction.parsed:
            return False
        return bool(prediction.value == np.round(gold, 1))

    def aggregate(self, predictions: Sequence[Prediction]) -> Prediction:
        return _majority(predictions)


class MCQScorer:
    """Answers that are one option letter, rendered `(A)`."""

    name = "mcq"

    def __init__(self, mode: str = STRICT):
        self.mode = mode

    def instruction_suffix(self, style: str = "plain") -> str:
        if style == "bae":
            return " Put your final answer in the form (X) at the end of your response."
        if style == "cot":
            return (" Make sure to state your final answer choice in curly brackets at the very end"
                    " of your response, just like: '{final answer: (A)}'. Let's think step by step.")
        return (' Make sure to state your final answer choice in curly brackets at the very end of'
                ' your response, just like: "{final answer: (A)}".')

    def normalise_gold(self, gold: Any) -> Any:
        return gold

    def extract(self, text: str) -> Prediction:
        if self.mode == LENIENT:
            return self._extract_lenient(text)
        return self._extract_strict(text)

    def _extract_strict(self, text: str) -> Prediction:
        """The original parser, character indexing and all.

        It strips a lowercase "final answer:" and then takes `pred[0]` if what
        is left is shorter than three characters and `pred[1]` otherwise. That
        reads `{final answer: (A)}` correctly and `{Final Answer: (A)}` as
        `(i)`, because the strip is case sensitive. Reproduced exactly: every
        result on disk was produced by it.
        """
        try:
            pred = _BRACES.findall(text)[-1]
            pred = pred.replace("final answer:", "").strip()
            if len(pred) == 0:
                return Prediction.unparsed(raw="")
            index = 0 if len(pred) < 3 else 1
            return Prediction(value=f"({pred[index]})", parsed=True, raw=pred)
        except Exception:
            return Prediction.unparsed(raw=text or "")

    def _extract_lenient(self, text: str) -> Prediction:
        """Read the letter rather than a fixed offset into the brace contents."""
        for group in reversed(_BRACES.findall(text or "")):
            body = _LABEL.sub("", group).strip()
            if not body:
                continue
            match = _PAREN_LETTER.search(body)
            if match:
                return Prediction(value=f"({match.group(1).upper()})", parsed=True, raw=group)
            # A bare letter on its own, as in `{A}` or `{final answer: A}`.
            if len(body) == 1 and body.isalpha():
                return Prediction(value=f"({body.upper()})", parsed=True, raw=group)

        # No usable braces: fall back to the last `(X)` in the response, which is
        # what the --bae parser has always done and what a model that ignored the
        # format instruction usually still produces.
        matches = _PAREN_LETTER.findall(text or "")
        if matches:
            return Prediction(value=f"({matches[-1].upper()})", parsed=True, raw=matches[-1])
        return Prediction.unparsed(raw=text or "")

    def correct(self, prediction: Prediction, gold: Any) -> bool:
        if not prediction.parsed:
            return False
        return bool(prediction.value == gold)

    def aggregate(self, predictions: Sequence[Prediction]) -> Prediction:
        return _majority(predictions)


class BaseMCQScorer(MCQScorer):
    """The `--bae` variant: looser prompt, last `(X)` anywhere in the response."""

    name = "mcq_bae"

    def extract(self, text: str) -> Prediction:
        matches = _PAREN_LETTER.findall(text or "")
        if not matches:
            return Prediction.unparsed(raw=text or "")
        return Prediction(value=f"({matches[-1].upper()})", parsed=True, raw=matches[-1])


class BaseNumericScorer(NumericScorer):
    """The `--bae` variant: last whitespace-separated token that parses as a float."""

    name = "numeric_bae"

    def extract(self, text: str) -> Prediction:
        for part in reversed((text or "").split(" ")):
            try:
                return Prediction(value=float(part), parsed=True, raw=part)
            except ValueError:
                continue
        return Prediction.unparsed(raw=text or "")

    def correct(self, prediction: Prediction, gold: Any) -> bool:
        if not prediction.parsed:
            return False
        return bool(prediction.value == np.round(gold, 1))


def score_responses(scorer, responses: dict, gold: Any) -> ScoreResult:
    """Score one question across a team.

    `responses` is `{agent_name: text}`; ordering is the caller's contract, as
    it has always been - message order is agent order everywhere in this
    repository.
    """
    gold = scorer.normalise_gold(gold)
    predictions = [scorer.extract(text) for text in responses.values()]
    aggregate = scorer.aggregate(predictions)
    return ScoreResult(
        predictions=predictions,
        aggregate=aggregate,
        correct=scorer.correct(aggregate, gold),
        gold=gold,
    )


SCORERS = {
    "numeric": NumericScorer,
    "mcq": MCQScorer,
}

# Which scorer the `--bae` flag swaps in for each answer type.
BAE_SCORERS = {
    "numeric": BaseNumericScorer,
    "mcq": BaseMCQScorer,
}


def get_scorer(answer_type: str, mode: str = STRICT, bae: bool = False):
    """Build the scorer for an answer type."""
    table = BAE_SCORERS if bae else SCORERS
    try:
        scorer_class = table[answer_type]
    except KeyError:
        raise KeyError(
            f"no scorer for answer type {answer_type!r}; known: {sorted(table)}"
        ) from None
    return scorer_class(mode=mode)
