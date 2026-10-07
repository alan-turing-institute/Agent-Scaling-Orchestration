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


def _majority(predictions: Sequence[Prediction], rng=None) -> Prediction:
    """Most common parsed value; ties broken at random.

    With no `rng` this uses the global `random`, as every result on disk did.
    That couples tie-breaking to everything else drawing from the global
    generator - which is why two arms with the same `--seed` stop seeing the
    same training tags after a few iterations, and why a threaded run breaks
    ties in whatever order its threads happen to reach them. Pass a generator
    seeded per question to make a tie resolve the same way every time.
    """
    parsed = [p for p in predictions if p.parsed]
    if not parsed:
        return Prediction.unparsed()

    counter = collections.Counter(p.value for p in parsed)
    top = max(counter.values())
    tied = [value for value, count in counter.items() if count == top]
    winner = (rng or random).choice(tied)
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

    def correct(self, prediction: Prediction, gold: Any, instance=None) -> bool:
        if not prediction.parsed:
            return False
        return bool(prediction.value == np.round(gold, 1))

    def aggregate(self, predictions: Sequence[Prediction], rng=None) -> Prediction:
        return _majority(predictions, rng=rng)


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

    def correct(self, prediction: Prediction, gold: Any, instance=None) -> bool:
        if not prediction.parsed:
            return False
        return bool(prediction.value == gold)

    def aggregate(self, predictions: Sequence[Prediction], rng=None) -> Prediction:
        return _majority(predictions, rng=rng)


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

    def correct(self, prediction: Prediction, gold: Any, instance=None) -> bool:
        if not prediction.parsed:
            return False
        return bool(prediction.value == np.round(gold, 1))


# The longest answer `MathScorer` will hand to sympy. math-verify's own timeouts
# use signal.alarm, which only works on the main thread, and questions are
# scored on worker threads; capping the length is what stops a pathological
# expression from stalling a worker instead.
MATH_MAX_CHARS = 400


def last_boxed(text: str):
    """The contents of the last `\\boxed{...}` (or `\\fbox{...}`), braces balanced."""
    text = text or ""
    for marker in ("\\boxed", "\\fbox"):
        start = text.rfind(marker)
        while start != -1:
            open_at = text.find("{", start)
            if open_at != -1 and text[start + len(marker):open_at].strip() == "":
                depth = 0
                for i in range(open_at, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            return text[open_at + 1:i].strip()
            start = text.rfind(marker, 0, start)
    return None


class MathScorer:
    """Free-form maths answers (MATH-style LaTeX), compared by symbolic equivalence.

    Asks for the answer in `\\boxed{}`, because answers like `\\frac{7}{4}` contain
    braces that the `{final answer: ...}` format cannot hold. Two answers match
    when math-verify finds them equivalent: `\\frac{7}{4}` and `1.75`, or
    `(3, \\pi/2)` and `\\left( 3, \\frac{\\pi}{2} \\right)`.

    `strict` reads only the last `\\boxed{}`. `lenient` falls back to a
    `final answer:` line and then the last `$...$`. This shape has no earlier
    results to reproduce, so the two modes differ only in that fallback.

    A team's answer is a vote over equivalence classes, not over strings, so
    three agents writing `0.5`, `\\frac{1}{2}` and `1/2` agree.

    Needs `math-verify` (pinned in requirements.txt), imported only when a
    maths answer is scored.
    """

    name = "math"

    def __init__(self, mode: str = STRICT):
        self.mode = mode

    def instruction_suffix(self, style: str = "plain") -> str:
        if style == "bae":
            return " Make sure to state your answer at the end of the response."
        if style == "cot":
            return (" Make sure to state your final answer in a LaTeX box at the very end of your"
                    " response, just like: '\\boxed{\\frac{1}{2}}'. Let's think step by step.")
        return (" Make sure to state your final answer in a LaTeX box at the very end of your"
                " response, just like: \"\\boxed{\\frac{1}{2}}\".")

    def normalise_gold(self, gold: Any) -> Any:
        return str(gold).strip()

    def _parse(self, latex: str):
        import logging
        import threading
        from math_verify import parse
        # It warns on every call made without its signal-based timeout, which is
        # every call on a worker thread; the length cap above stands in for it.
        logging.getLogger("math_verify").setLevel(logging.ERROR)
        timeout = 5 if threading.current_thread() is threading.main_thread() else None
        return parse(f"${latex}$", parsing_timeout=timeout)

    def _equivalent(self, gold_latex: str, answer_latex: str) -> bool:
        if len(gold_latex) > MATH_MAX_CHARS or len(answer_latex) > MATH_MAX_CHARS:
            return gold_latex == answer_latex
        import threading
        from math_verify import verify
        gold, answer = self._parse(gold_latex), self._parse(answer_latex)
        if not gold or not answer:
            return gold_latex.replace(" ", "") == answer_latex.replace(" ", "")
        timeout = 5 if threading.current_thread() is threading.main_thread() else None
        return bool(verify(gold, answer, timeout_seconds=timeout))

    def extract(self, text: str) -> Prediction:
        answer = last_boxed(text)
        if answer is None and self.mode == LENIENT:
            line = re.findall(r"final answer\s*:?\s*(.+)", text or "", re.IGNORECASE)
            dollars = re.findall(r"\$([^$]+)\$", text or "")
            answer = (line[-1].strip().strip("$. ") if line else None) or (dollars[-1].strip() if dollars else None)
        if not answer:
            return Prediction.unparsed(raw=text or "")
        return Prediction(value=answer, parsed=True, raw=answer)

    def correct(self, prediction: Prediction, gold: Any, instance=None) -> bool:
        if not prediction.parsed:
            return False
        return self._equivalent(str(gold), str(prediction.value))

    def aggregate(self, predictions: Sequence[Prediction], rng=None) -> Prediction:
        """Majority over equivalence classes; ties broken as `_majority` breaks them."""
        classes = []  # [representative value, count]
        for prediction in predictions:
            if not prediction.parsed:
                continue
            for cls in classes:
                if self._equivalent(cls[0], str(prediction.value)):
                    cls[1] += 1
                    break
            else:
                classes.append([str(prediction.value), 1])
        if not classes:
            return Prediction.unparsed()
        top = max(count for _, count in classes)
        tied = [value for value, count in classes if count == top]
        winner = (rng or random).choice(tied)
        return Prediction(value=winner, parsed=True, raw=winner)


def score_responses(scorer, responses: dict, gold: Any, rng=None, instance=None) -> ScoreResult:
    """Score one question across a team.

    `responses` is `{agent_name: text}`; ordering is the caller's contract, as
    it has always been - message order is agent order everywhere in this
    repository.
    """
    gold = scorer.normalise_gold(gold)
    predictions = [scorer.extract(text) for text in responses.values()]
    aggregate = scorer.aggregate(predictions, rng=rng)
    return ScoreResult(
        predictions=predictions,
        aggregate=aggregate,
        correct=scorer.correct(aggregate, gold, instance=instance),
        gold=gold,
    )


SCORERS = {
    "numeric": NumericScorer,
    "mcq": MCQScorer,
    "math": MathScorer,
}

# Which scorer the `--bae` flag swaps in for each answer type.
BAE_SCORERS = {
    "numeric": BaseNumericScorer,
    "mcq": BaseMCQScorer,
    "math": MathScorer,
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
