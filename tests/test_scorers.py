"""What each parser does with a response, written down.

Run it directly - there is no pytest in the pinned environment:

    PYTHONPATH=src python tests/test_scorers.py

Two things are pinned here. That `strict` reproduces the parsers every result in
`data-claude/` was produced by, so a refactor cannot quietly move a number. And
what `lenient` fixes, so the cost of switching is a table someone can read
rather than a claim.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmarks
from benchmarks.scorers import LENIENT, get_scorer, score_responses

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


# --------------------------------------------------------------------------
# Multiple choice, strict: the parser strips a lowercase "final answer:" and
# then indexes a fixed offset into what is left.
# --------------------------------------------------------------------------
MCQ_STRICT = [
    ("{final answer: (A)}",                  "(A)"),
    ("{final answer: A}",                    "(A)"),
    ("{(B)}",                                "(B)"),
    ("{B}",                                  "(B)"),
    ("{final answer:(C)}",                   "(C)"),
    ("I think {A} then reconsider {B}",      "(B)"),   # last group wins
    ("", ""),
    ("no braces at all, answer (A)",         ""),      # cannot see it
    # The known failures. These are wrong, and reproduced on purpose: every
    # number on disk was produced by them. `lenient` below is the fix.
    ("{Final Answer: (A)}",                  "(i)"),   # strip is case sensitive
    ("{FINAL ANSWER: (D)}",                  "(I)"),
    ("{answer is A}",                        "(n)"),   # second character
    ("{final answer: (10)}",                 "(1)"),
]

MCQ_LENIENT = [
    ("{final answer: (A)}",                  "(A)"),
    ("{final answer: A}",                    "(A)"),
    ("{Final Answer: (A)}",                  "(A)"),   # fixed
    ("{FINAL ANSWER: (D)}",                  "(D)"),   # fixed
    ("no braces at all, answer (A)",         "(A)"),   # fixed
    ("reasoning (B) then {garbage}",         "(B)"),   # fixed
    # Unreadable is now reported as unreadable rather than as an invented letter.
    ("{answer is A}",                        ""),
    ("{final answer: (10)}",                 ""),
    ("", ""),
]

# --------------------------------------------------------------------------
# Numeric. Strict takes the last number anywhere and ignores the braces it
# asked for, so trailing units or dates win.
# --------------------------------------------------------------------------
NUMERIC_STRICT = [
    ("{final answer: 8}",                                    8.0),
    ("The answer is 42.",                                   42.0),
    ("{final answer: 8} over 3 days",                        3.0),   # wrong, on purpose
    ("{final answer: 8} costing $12 each",                  12.0),   # wrong, on purpose
    ("no digits here",                                        ""),
    ("", ""),
]

NUMERIC_LENIENT = [
    ("{final answer: 8}",                                    8.0),
    ("{final answer: 8} over 3 days",                        8.0),   # fixed
    ("{final answer: 8} costing $12 each",                   8.0),   # fixed
    ("The answer is 42.",                                   42.0),   # no braces: unchanged
    ("no digits here",                                        ""),
]


def run_table(name, scorer, table):
    for text, want in table:
        check(f"{name} {text!r}", scorer.extract(text).legacy, want)


def test_parsers():
    run_table("mcq/strict",      get_scorer("mcq"),                    MCQ_STRICT)
    run_table("mcq/lenient",     get_scorer("mcq", mode=LENIENT),      MCQ_LENIENT)
    run_table("numeric/strict",  get_scorer("numeric"),                NUMERIC_STRICT)
    run_table("numeric/lenient", get_scorer("numeric", mode=LENIENT),  NUMERIC_LENIENT)


def test_parsed_flag_separates_unreadable_from_wrong():
    """The distinction the old `""` return could not make."""
    scorer = get_scorer("mcq")
    check("unreadable is not parsed", scorer.extract("mumbling").parsed, False)
    check("read is parsed",  scorer.extract("{final answer: (A)}").parsed, True)
    # Read, and wrong: parsed is still True.
    prediction = scorer.extract("{final answer: (B)}")
    check("wrong answer still parsed", (prediction.parsed, scorer.correct(prediction, "(A)")),
          (True, False))


def test_every_agent_gets_a_prediction():
    """One prediction per response, including the ones that could not be read.

    The `--bae` parsers used to skip failures rather than record them, so
    `final_answers` came back shorter than the team and per-agent accuracy was
    attributed to the wrong agent from the first failure onwards.
    """
    responses = {"a": "answer (A)", "b": "no letter here", "c": "answer (B)"}
    for bae in (False, True):
        result = score_responses(get_scorer("mcq", bae=bae), dict(responses), "(A)")
        check(f"mcq bae={bae} length", len(result.predictions), 3)
    result = score_responses(get_scorer("numeric", bae=True),
                             {"a": "5", "b": "nothing", "c": "7"}, 5.0)
    check("numeric bae length", len(result.predictions), 3)


def test_majority_and_ties():
    scorer = get_scorer("mcq")
    result = score_responses(
        scorer, {"a": "{final answer: (A)}", "b": "{final answer: (A)}",
                 "c": "{final answer: (B)}"}, "(A)")
    check("majority", (result.aggregate.legacy, result.correct), ("(A)", True))

    # All unreadable: no team answer, and not counted as correct.
    result = score_responses(scorer, {"a": "", "b": ""}, "(A)")
    check("all unreadable", (result.aggregate.legacy, result.correct), ("", False))


def test_registry():
    names = benchmarks.list_names()
    check("registry is populated", len(names) >= 7, True)
    check("gsm8k is numeric", benchmarks.answer_type_of("gsm8k"), "numeric")
    check("arc is mcq", benchmarks.answer_type_of("arc"), "mcq")
    check("every benchmark has a persona set",
          all(len(benchmarks.persona_set(n)) == 5 for n in names), True)
    check("every benchmark has a scorer",
          all(benchmarks.get(n).scorer() is not None for n in names), True)
    try:
        benchmarks.get("humaneval")
    except KeyError:
        pass
    else:
        FAILURES.append("registry: an unregistered benchmark should raise, not fall back")


def test_gold_normalisation():
    """The tagged dataset stores every answer as a string, numeric ones included."""
    check("numeric gold coerced", get_scorer("numeric").normalise_gold("8"), 8.0)
    check("mcq gold untouched", get_scorer("mcq").normalise_gold("(A)"), "(A)")


def test_legacy_shape_is_unchanged():
    """`(answers, majority, is_correct)` - what every saved history expects."""
    result = score_responses(get_scorer("mcq"),
                             {"a": "{final answer: (A)}", "b": "mumbling"}, "(A)")
    check("legacy tuple", result.legacy_tuple(), (["(A)", ""], "(A)", True))


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for failure in FAILURES:
            print("  -", failure)
        raise SystemExit(1)
    print(f"ok - {len(tests)} tests passed")
