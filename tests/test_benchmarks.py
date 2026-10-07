"""Offline checks for the benchmarks added after the paper, and for growing the tag vocabulary.

Run it directly - there is no pytest in the pinned environment:

    PYTHONPATH=src python tests/test_benchmarks.py

Nothing here downloads data or calls a model. Loading the real sets is
`scripts/fetch_benchmarks.py`; running them end to end is `tests/mock_e2e.sh`.
"""

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmarks  # noqa: E402
import canonicalise_tags  # noqa: E402
import personas  # noqa: E402
from benchmarks import gpqa_diamond  # noqa: E402

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def test_new_benchmarks_are_registered():
    check("gpqa_diamond is mcq", benchmarks.answer_type_of("gpqa_diamond"), "mcq")
    check("mmlu_pro is mcq", benchmarks.answer_type_of("mmlu_pro"), "mcq")
    check("aime is numeric", benchmarks.answer_type_of("aime"), "numeric")


def test_persona_fallback():
    """No paper set: main.py gets the default set, as an unknown --data always did."""
    check("gpqa falls back to the default set", personas.paper_persona_names("gpqa_diamond"),
          list(personas._DEFAULT_PERSONA_SET))
    check("gsm8k keeps the paper's set", personas.paper_persona_names("gsm8k"),
          benchmarks.persona_set("gsm8k"))


def test_gpqa_option_shuffle():
    """Same order on every call, correct answer tracked, letters spread out."""
    def row(record_id):
        return {"Record ID": record_id, "Correct Answer": "right",
                "Incorrect Answer 1": "w1", "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"}
    options, letter = gpqa_diamond._options(row("recA"))
    check("shuffle is deterministic", gpqa_diamond._options(row("recA")), (options, letter))
    check("correct letter points at the right answer", options["ABCD".index(letter)], "right")
    check("all four options kept", sorted(options), ["right", "w1", "w2", "w3"])
    spread = Counter(gpqa_diamond._options(row(f"rec{i}"))[1] for i in range(400))
    check("correct letter lands on every position", sorted(spread), list("ABCD"))
    check("and roughly evenly", all(70 < n < 130 for n in spread.values()), True)


class FakeClient:
    """Answers the assignment prompt with whatever `assign` says, clustering with nothing."""

    def __init__(self, assign):
        self.assign = assign
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "EXISTING tags" in prompt:
            import json
            new = [line[2:] for line in prompt.split("New tags:", 1)[1].splitlines() if line.startswith("- ")]
            content = json.dumps({"assignments": {t: self.assign.get(t) for t in new}})
        else:
            content = '{"groups": []}'
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def test_extend_mapping():
    args = canonicalise_tags.parse_args([])
    base = {"Step by step reasoning": "step-by-step reasoning", "step-by-step reasoning": "step-by-step reasoning",
            "arithmetic": "arithmetic", "math": "arithmetic"}
    base_counts = {"step-by-step reasoning": 10, "arithmetic": 8}
    counts = Counter({
        "math": 3,                          # already mapped: must keep its entry
        "Arithmetic": 2,                    # lexical match to an existing canonical
        "multi-step reasoning": 2,          # the model says: same as step-by-step reasoning
        "quantum mechanics": 2,             # the model says: nothing fits
        "invented": 1,                      # the model names a tag that does not exist
    })
    client = FakeClient({"multi-step reasoning": "step-by-step reasoning",
                         "quantum mechanics": None, "invented": "made-up tag"})
    mapping = canonicalise_tags.extend_mapping(counts, base, base_counts, args, client)
    check("existing entries unchanged", {k: mapping[k] for k in base}, base)
    check("lexical match onto the existing vocabulary", mapping["Arithmetic"], "arithmetic")
    check("model match onto the existing vocabulary", mapping["multi-step reasoning"], "step-by-step reasoning")
    check("no match stays new vocabulary", mapping["quantum mechanics"], "quantum mechanics")
    check("an assignment to a tag that does not exist is ignored", mapping["invented"], "invented")

    offline = canonicalise_tags.extend_mapping(counts, base, base_counts, args, None)
    check("without a model, unmatched tags stay themselves", offline["multi-step reasoning"], "multi-step reasoning")


def test_parse_assignments():
    existing = {"arithmetic", "algebra"}
    got = canonicalise_tags.parse_assignments(
        '```json\n{"assignments": {"sums": "Arithmetic", "x": "algebra", "y": null, "z": "geometry"}}\n```',
        {"sums", "y", "z"}, existing)
    check("only asked-about tags onto real tags", got, {"sums": "arithmetic"})
    check("garbage parses to nothing", canonicalise_tags.parse_assignments("no json here", {"a"}, existing), {})


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
