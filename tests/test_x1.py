"""Offline checks for X1: worker prompts, team filters, the stage-1 plan, deferred judging, the hub's handoff.

    PYTHONPATH=src python tests/test_x1.py
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from bare_model_baseline import SUMMARY_INSTRUCTION, matches_sources, question_prompt, sampling_kwargs
from benchmarks.base import Prediction, local_root
from expert_personas import EXPERT_PERSONAS
from judge import DEFERRED, DeferredJudge, regrade
from x1_hub import summary_handoff, team_workers
from x1_plan import command, load_config, runs, workers

SUFFIX = ' Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (A)}".'


def args(**kw):
    base = dict(persona=None, summary=False, thinking="server", thinking_budget=None, top_k=None, min_p=None,
                repetition_penalty=None, presence_penalty=None)
    return SimpleNamespace(**{**base, **kw})


def test_question_prompt():
    # Without X1's options the prompt is X0's: the question and its format line.
    assert question_prompt("Q?", SUFFIX, args()) == "Q?" + SUFFIX
    p = question_prompt("Q?", SUFFIX, args(persona="Causal_Chain_Biologist", summary=True))
    assert p.startswith(EXPERT_PERSONAS["Causal_Chain_Biologist"] + "\n\nQ?\n\n" + SUMMARY_INSTRUCTION)
    assert p.endswith(SUFFIX)


def test_sampling_kwargs():
    assert sampling_kwargs(args()) == {}
    kw = sampling_kwargs(args(thinking="on", thinking_budget=26624, top_k=20, repetition_penalty=1.05))
    assert kw["extra_body"] == {"top_k": 20, "repetition_penalty": 1.05,
                                "chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 26624}


def test_matches_sources():
    gpqa = {"dataset": "gpqa_diamond", "metadata": json.dumps({"domain": "Physics"})}
    law = {"dataset": "mmlu_pro", "metadata": json.dumps({"category": "law"})}
    plan = {"dataset": "plancraft", "metadata": "{}"}
    assert matches_sources(gpqa, "gpqa_diamond:Physics,mmlu_pro:physics")
    assert not matches_sources(gpqa, "gpqa_diamond:Chemistry")
    assert matches_sources(law, "mmlu_pro:law") and not matches_sources(law, "mmlu_pro:physics")
    assert matches_sources(plan, "plancraft") and matches_sources(plan, None)


def test_plan():
    config = load_config()
    names = {p for t in config["teams"] for p in t["personas"]}
    assert names <= set(EXPERT_PERSONAS) and len(names) == 21
    ws = workers(config)
    big = [w for w in ws if w["worker"] == "bigbang" and w["pool"] == "hard"]
    assert {w["mode"] for w in big} == {"on", "off"}
    assert big[0]["sources"].split(",") == ["gpqa_diamond:Physics", "mmlu_pro:physics", "gpqa_diamond:Chemistry",
                                            "mmlu_pro:chemistry", "gpqa_diamond:Biology", "mmlu_pro:law"]
    every = runs(config)
    assert len(every) == len(ws) * 5 * 3 and len({r["out_dir"] for r in every}) == len(every)
    by = {(r["worker"], r["mode"], r["pool"]): r for r in runs(config, folds=[0], samples=[0])}
    on = command(config, by[("bigbang", "on", "hard")], "http://x/v1")
    assert on[on.index("--thinking_budget") + 1] == "26624" and "--summary" in on and "--defer_judge" not in on
    off = command(config, by[("bigbang", "off", "agentic")], "http://x/v1")
    assert "--thinking_budget" not in off and "--defer_judge" in off and "--summary" not in off
    mini = command(config, by[("ministral-r", "always", "hard")], "http://x/v1")
    assert mini[mini.index("--thinking") + 1] == "server" and "--thinking_budget" not in mini
    assert "--system_file" in mini
    persona = command(config, by[("Crafting_Planner", "off", "agentic")], "http://x/v1")
    assert persona[persona.index("--persona") + 1] == "Crafting_Planner" and "--defer_judge" not in persona
    t1 = next(t for t in config["teams"] if t["id"] == "T1")
    assert team_workers(config, t1, "headline")[2:] == [("rl-math", "rl-math-off"), ("agentmercury", "agentmercury-on")]
    t6 = next(t for t in config["teams"] if t["id"] == "T6")
    assert team_workers(config, t6, "on")[3] == ("ministral-r", "ministral-r-always")


class FakeJudge:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def answer_matches(self, *a):
        return self.verdicts.pop(0), "correct: yes"

    def criterion(self, *a):
        return self.verdicts.pop(0), "correct: yes"


def test_deferred_judge():
    judge = DeferredJudge()
    assert judge.answer_matches("q", "Exact Answer: x", "x") == (False, DEFERRED)
    ok, detail = regrade("browsecomp_plus", json.loads(json.dumps(judge.asked)), FakeJudge([True]))
    assert ok and "judge" in detail
    judge = DeferredJudge()
    for op in ("correctness", "correctness", "contradiction"):
        judge.criterion("q", "answer", "c", op)
    ok, detail = regrade("finance_agent", judge.asked, FakeJudge([True, False, False]))
    assert not ok and detail["rubric_score"] == round(1 / 3, 3)
    ok, _ = regrade("finance_agent", judge.asked, FakeJudge([True, True, False]))
    assert ok
    assert regrade("browsecomp_plus", [], FakeJudge([])) == (False, {})


def test_summary_handoff():
    text = "Long working...\nSummary: Step one. Step two.\n{final answer: (B)}"
    handed, fell_back = summary_handoff(text, Prediction("B", True))
    assert not fell_back and handed.startswith("Summary: Step one.") and handed.endswith("Final answer: B")
    handed, fell_back = summary_handoff("x" * 900, Prediction(None, False))
    assert fell_back and handed.startswith("...") and handed.endswith("Final answer: (no readable answer)")
    _, fell_back = summary_handoff("work\nSummary:   \n", Prediction("3", True))
    assert fell_back  # an empty summary is not a summary


def test_local_root(tmp=Path("/tmp")):
    assert local_root("/no/such/machine/path", tmp) == tmp
    assert local_root(str(tmp), "/elsewhere") == tmp


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
