"""Backwards-compatible face of `benchmarks.scorers`.

The parsers themselves moved to `benchmarks/scorers.py`, where they are one
object per answer shape rather than four near-duplicate functions. This module
stays because `main.py`, `team_evaluation.py` and `scripts/bare_model_baseline.py`
import these names, and because every result in `data-claude/` was produced
through them: the shapes returned here are exactly what the saved histories and
`K_star_analysis` expect.

New code should ask the registry for a scorer:

    scorer = benchmarks.scorer_for("gsm8k")
    result = benchmarks.score_responses(scorer, {name: text}, gold)

which additionally reports whether each response could be parsed at all -
something the tuple below cannot express, because it writes an unreadable
response and a wrong one identically as "".
"""

from types import SimpleNamespace

import benchmarks
from benchmarks.scorers import extract_number, get_scorer, score_responses

__all__ = [
    "get_instruction_suffix",
    "extract_number",
    "evaluate_gsm8k",
    "evaluate_mcq",
    "base_evaluate_gsm8k",
    "base_evaluate_mcq",
]


def get_instruction_suffix(args):
    """Required answer format for `args.data`, honouring `--cot` and `--bae`."""
    data = getattr(args, "data", None)
    bae = getattr(args, "bae", False)
    style = "bae" if bae else ("cot" if getattr(args, "cot", False) else "plain")

    try:
        return benchmarks.get(data).scorer(bae=bae).instruction_suffix(style)
    except KeyError:
        # The old behaviour for an unregistered dataset was to ask for a number.
        # Kept so an ad-hoc `--data something` still runs, but it is a guess:
        # register the benchmark instead of relying on it.
        return get_scorer("numeric", bae=bae).instruction_suffix(style)


def _legacy(answer_type, responses, answer, bae=False):
    scorer = get_scorer(answer_type, bae=bae)
    return score_responses(scorer, responses, answer).legacy_tuple()


def evaluate_gsm8k(responses, answer):
    return _legacy("numeric", responses, answer)


def evaluate_mcq(responses, answer):
    return _legacy("mcq", responses, answer)


def base_evaluate_gsm8k(responses, answer):
    return _legacy("numeric", responses, answer, bae=True)


def base_evaluate_mcq(responses, answer):
    return _legacy("mcq", responses, answer, bae=True)
