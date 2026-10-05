"""The runner, the team configs and the prediction rows, checked without a server.

Run it directly, as with the scorer tests:

    PYTHONPATH=src python tests/test_runner.py

The first check is the one the rest of the repository depends on: a vote of
solvers sends exactly the requests `run_team_evaluation` always sent, so moving
the orchestrator arms onto the runner cannot move a number by itself.
"""

import concurrent.futures
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from openai import APIConnectionError  # noqa: E402

import benchmarks  # noqa: E402
import team_config  # noqa: E402
import team_evaluation  # noqa: E402
from model.model_utils import chosen_persona_bank  # noqa: E402
from model.openai_compat import Completion  # noqa: E402
from predictions import rows_from_report, save_report  # noqa: E402
from roles import RATIONALE_CHARS, render_handoff, render_prompt  # noqa: E402
from runner import run_question  # noqa: E402

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def raises(label, fn, exc=Exception):
    try:
        fn()
    except exc:
        return
    FAILURES.append(f"{label}: did not raise {exc.__name__}")


class FakeClient:
    """Answers by looking up which persona's prompt opens the user message."""

    def __init__(self, answers, calls):
        self.answers = answers      # persona prompt prefix -> response text, or a callable(content)
        self.calls = calls

    def generate(self, messages, max_tokens=None, temperature=None, top_p=None, seed=None, **kwargs):
        self.calls.append({"messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "top_p": top_p, "seed": seed})
        content = messages[-1]["content"]
        for prefix, answer in self.answers.items():
            if content.startswith(prefix):
                text = answer(content) if callable(answer) else answer
                return Completion(content=text, prompt_tokens=10, completion_tokens=5,
                                  finish_reason="stop", latency_s=0.01, model="fake")
        return Completion(content="", prompt_tokens=10, completion_tokens=0, finish_reason="stop")


class FakeRegistry:
    def __init__(self, client):
        self._client = client

    def resolve(self, key):
        return key

    def client(self, key):
        return self._client


BANK = chosen_persona_bank()
A, B, C, D = "Conservative_Verifier", "Creative_Explorer", "Rigorous_Formalist", "Intuitive_Estimator"
PERSONAS = {name: BANK[name] for name in (A, B, C, D)}
NUMERIC = benchmarks.get_scorer("numeric")
QUESTION = "Tom has 3 apples and buys 5 more. How many apples does he have?"


def prefix(name):
    return BANK[name]["prompt"][:60]


# --------------------------------------------------------------------------
# A vote sends the requests the old run_team_evaluation sent, byte for byte.
# --------------------------------------------------------------------------
calls = []
client = FakeClient({prefix(A): "{final answer: 8}", prefix(B): "{final answer: 8}",
                     prefix(C): "{final answer: 7}", prefix(D): "{final answer: 8}"}, calls)
result = run_question(team_config.vote([A, B, C, D]), QUESTION, NUMERIC.normalise_gold("8"),
                      scorer=NUMERIC, personas=PERSONAS, registry=FakeRegistry(client),
                      max_tokens=4096, request_seed=None)
legacy_suffix = (' Make sure to state your final answer in curly brackets at the very end of your'
                 ' response, just like: "{final answer: 123}".')
for name, call in zip((A, B, C, D), calls):
    check(f"vote prompt {name}", call["messages"], [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{BANK[name]['prompt']}\n\n{QUESTION + legacy_suffix}"},
    ])
    check(f"vote temperature {name}", call["temperature"], BANK[name]["temperature"])
    check(f"vote top_p {name}", call["top_p"], BANK[name]["top_p"])
    check(f"vote max_tokens {name}", call["max_tokens"], 4096)
    check(f"vote no seed {name}", call["seed"], None)
check("vote team answer", result["team_answer"], "8.0")
check("vote team correct", result["team_correct"], True)
check("vote stage ids are persona names", [s["id"] for s in result["stages"]], [A, B, C, D])
check("vote per-stage correctness", [s["correct"] for s in result["stages"]], [True, True, False, True])
check("vote calls", result["calls"], 4)
check("vote depth", result["depth"], 1)
check("vote tokens", (result["prompt_tokens"], result["completion_tokens"]), (40, 20))

# --------------------------------------------------------------------------
# Configs: validation, identity, layers.
# --------------------------------------------------------------------------
S = team_config.Stage
I = team_config.Input
raises("duplicate ids", lambda: team_config.TeamConfig(stages=(S("x", A), S("x", B))), ValueError)
raises("forward reference", lambda: team_config.TeamConfig(stages=(S("x", A, inputs=(I("y"),)), S("y", B))), ValueError)
raises("unknown role", lambda: team_config.TeamConfig(stages=(S("x", A, role="oracle"),)), ValueError)
raises("unknown handoff", lambda: team_config.TeamConfig(stages=(S("x", A), S("y", B, inputs=(I("x", "gist"),)))), ValueError)
raises("stage aggregate needs final", lambda: team_config.TeamConfig(stages=(S("x", A),), aggregate="stage"), ValueError)

pipe = team_config.pipeline([("solver", A), ("critic", B), ("reviser", C)])
check("config round trip keeps id", team_config.TeamConfig.from_dict(json.loads(json.dumps(pipe.to_dict()))).config_id,
      pipe.config_id)
check("name does not change id", team_config.pipeline([("solver", A), ("critic", B), ("reviser", C)], name="other").config_id,
      pipe.config_id)
check("handoff changes id", team_config.pipeline([("solver", A), ("critic", B), ("reviser", C)], handoff="answer").config_id
      != pipe.config_id, True)
check("budget changes id", team_config.split_budget(pipe, 3000).config_id != pipe.config_id, True)
check("budget splits evenly", team_config.split_budget(pipe, 3000).max_tokens, 1000)
check("pipeline layers", pipe.layers(), [[pipe.stage_ids[0]], [pipe.stage_ids[1]], [pipe.stage_ids[2]]])
check("reviser reads solver and critic", [i.source for i in pipe.stages[2].inputs], pipe.stage_ids[:2])

deb = team_config.debate([A, B, C], rounds=2)
check("debate layers", [len(layer) for layer in deb.layers()], [3, 3, 3])
check("debate votes over last round", deb.voters(), [f"{A}@r2", f"{B}@r2", f"{C}@r2"])
cen = team_config.centralized([A, B, C], hub=D)
check("centralized answer stage", cen.answer_stages(), [f"hub:{D}"])
par = team_config.parallel(team_config.pipeline([("solver", A), ("checker", B)]),
                           team_config.pipeline([("solver", C), ("checker", D)]))
check("parallel votes over each branch's last stage", par.voters(),
      ["b1/2:checker:" + B, "b2/2:checker:" + D])
check("repeated persona gets a unique id", team_config.vote([A, A]).stage_ids, [A, f"{A}#2"])

# --------------------------------------------------------------------------
# Handoffs and prompts.
# --------------------------------------------------------------------------
pred = NUMERIC.extract("so {final answer: 8}")
check("answer handoff", render_handoff("long text {final answer: 8}", pred, "answer"), "Answer: 8.0")
check("answer handoff unparsed", render_handoff("no number", NUMERIC.extract("no number"), "answer"),
      "Answer: (no readable answer)")
long_text = "x" * (RATIONALE_CHARS + 50) + "{final answer: 8}"
check("rationale keeps the end", render_handoff(long_text, pred, "rationale").endswith("{final answer: 8}"), True)
check("rationale is clipped", len(render_handoff(long_text, pred, "rationale")), RATIONALE_CHARS + 3)
check("full handoff", render_handoff("all of it", pred, "full"), "all of it")
critic_prompt = render_prompt("PERSONA", "Q?", " SUFFIX", role="critic", inputs=[("Agent 1 (solver)", "work")])
check("critic prompt layout", critic_prompt.startswith("PERSONA\n\nQuestion:\nQ?\n\nWork from your team:"), True)
check("critic prompt ends with suffix", critic_prompt.endswith(" SUFFIX"), True)
check("persona name never shown", A not in render_prompt(BANK[A]["prompt"], "Q?", " S", role="critic",
                                                         inputs=[("Agent 1 (solver)", "w")]), True)

# --------------------------------------------------------------------------
# A pipeline credits the stage that fixed the answer, and spots copying.
# --------------------------------------------------------------------------
calls = []
client = FakeClient({prefix(A): "I get {final answer: 7}",
                     prefix(B): "The solver slipped. {final answer: 8}",
                     prefix(C): "Agreed with the review. {final answer: 8}"}, calls)
result = run_question(pipe, QUESTION, NUMERIC.normalise_gold("8"), scorer=NUMERIC, personas=PERSONAS,
                      registry=FakeRegistry(client), max_tokens=4096)
stages = {s["role"]: s for s in result["stages"]}
check("solver has no transition", stages["solver"]["transition"], None)
check("critic fixed it", stages["critic"]["transition"], "fixed")
check("critic did not copy", stages["critic"]["copied"], False)
check("reviser kept it", stages["reviser"]["transition"], "kept_right")
check("reviser copied the critic", stages["reviser"]["copied"], True)
check("pipeline answer is the last stage's", (result["team_answer"], result["team_correct"]), ("8.0", True))
check("pipeline depth", result["depth"], 3)
check("critic saw the solver's work", "I get {final answer: 7}" in calls[1]["messages"][-1]["content"], True)
check("reviser saw both", all(t in calls[2]["messages"][-1]["content"]
                              for t in ("I get {final answer: 7}", "The solver slipped.")), True)
check("seeds differ by stage", len({c["seed"] for c in calls}), 3)

# --------------------------------------------------------------------------
# Debaters are credited against their own earlier answer; a hub against the
# vote of the workers it read.
# --------------------------------------------------------------------------
stubborn = FakeClient({prefix(A): "{final answer: 8}", prefix(B): "{final answer: 7}"}, [])
result = run_question(team_config.debate([A, B], rounds=1), QUESTION, 8.0, scorer=NUMERIC, personas=PERSONAS,
                      registry=FakeRegistry(stubborn), max_tokens=4096)
by_id = {s["id"]: s for s in result["stages"]}
check("debater references own answer", by_id[f"{A}@r1"]["reference"], A)
check("debater kept its right answer", (by_id[f"{A}@r1"]["transition"], by_id[f"{A}@r1"]["copied"]), ("kept_right", True))
check("other debater kept its wrong one", (by_id[f"{B}@r1"]["transition"], by_id[f"{B}@r1"]["copied"]), ("kept_wrong", True))
check("debater records every input", sorted(by_id[f"{A}@r1"]["inputs_correct"]), sorted([A, B]))

hub_client = FakeClient({prefix(A): "{final answer: 8}", prefix(B): "{final answer: 8}",
                         prefix(C): "{final answer: 7}", prefix(D): "{final answer: 8}"}, [])
result = run_question(team_config.centralized([A, B, C], hub=D), QUESTION, 8.0, scorer=NUMERIC, personas=PERSONAS,
                      registry=FakeRegistry(hub_client), max_tokens=4096)
hub = result["stages"][-1]
check("hub references the workers' vote", hub["reference"], "vote")
check("hub agreeing with the vote kept it", (hub["transition"], hub["copied"]), ("kept_right", True))

raises("unknown config key", lambda: team_config.TeamConfig.from_dict(
    {"stages": [{"id": "x", "persona": A}], "aggregation": "stage"}), ValueError)
raises("unknown stage key", lambda: team_config.TeamConfig.from_dict(
    {"stages": [{"id": "x", "persona": A, "roll": "critic"}]}), ValueError)
raises("bare-string input", lambda: team_config.TeamConfig.from_dict(
    {"stages": [{"id": "x", "persona": A}, {"id": "y", "persona": B, "inputs": ["x"]}]}), ValueError)
raises("hub needs a persona", lambda: team_config.centralized([A, B], hub=None), ValueError)
raises("synthesiser needs a persona", lambda: team_config.synthesis([A, B], synthesiser=""), ValueError)

# --------------------------------------------------------------------------
# Seeded tie-break: a 2-2 vote resolves the same way every time.
# --------------------------------------------------------------------------
tie = FakeClient({prefix(A): "{final answer: 8}", prefix(B): "{final answer: 8}",
                  prefix(C): "{final answer: 7}", prefix(D): "{final answer: 7}"}, [])
answers = {run_question(team_config.vote([A, B, C, D]), QUESTION, 8.0, scorer=NUMERIC, personas=PERSONAS,
                        registry=FakeRegistry(tie), max_tokens=4096)["team_answer"] for _ in range(20)}
check("seeded tie is stable", len(answers), 1)

# --------------------------------------------------------------------------
# A server that is down reaches the caller; any other failure scores as unanswered.
# --------------------------------------------------------------------------
class DownClient(FakeClient):
    def generate(self, *a, **k):
        raise APIConnectionError(request=None)


class BrokenClient(FakeClient):
    def generate(self, *a, **k):
        raise RuntimeError("bad request")


raises("connection error propagates",
       lambda: run_question(team_config.vote([A]), QUESTION, 8.0, scorer=NUMERIC, personas=PERSONAS,
                            registry=FakeRegistry(DownClient({}, [])), max_tokens=4096), APIConnectionError)
broken = run_question(team_config.vote([A]), QUESTION, 8.0, scorer=NUMERIC, personas=PERSONAS,
                      registry=FakeRegistry(BrokenClient({}, [])), max_tokens=4096)
check("other failure is recorded", broken["stages"][0]["error"] is not None, True)
check("other failure scores as unanswered", (broken["stages"][0]["parsed"], broken["team_correct"]), (False, False))

# --------------------------------------------------------------------------
# run_team_evaluation keeps its report shape; rows carry every stage; rescore agrees.
# --------------------------------------------------------------------------
class Args:
    model_name = "fake"
    api_base_url = "http://unused/v1"
    eval_workers = 3
    parse_mode = "strict"
    tie_break = "seeded"
    request_seed = 0
    max_new_tokens = 4096


samples = [
    {"question": QUESTION, "answer": "8", "dataset": "gsm8k", "tags": ["arithmetic"]},
    {"question": "Which is a mammal? (A) trout (B) whale", "answer": "(B)", "dataset": "arc", "tags": ["biology"]},
]
answers = {prefix(A): lambda c: "{final answer: (B)}" if "(A)" in c else "{final answer: 8}",
           prefix(B): lambda c: "{final answer: (A)}" if "(A)" in c else "{final answer: 8}",
           prefix(C): lambda c: "{final answer: (B)}" if "(A)" in c else "{final answer: 9}"}
original = team_evaluation.registry_for
team_evaluation.registry_for = lambda args: FakeRegistry(FakeClient(answers, []))
try:
    report = team_evaluation.run_team_evaluation([A, B, C], samples, Args())
finally:
    team_evaluation.registry_for = original

check("report n_samples", report["n_samples"], 2)
check("report team_correct", report["team_correct"], 2)
check("report per_agent_correct keyed by persona", report["per_agent_correct"], {A: 2, B: 1, C: 1})
check("report per-tag counts", report["per_tag_counts"], {"arithmetic": 1, "biology": 1})
check("report per-tag agent counts", report["per_agent_correct_by_tag"]["biology"], {A: 1, B: 0, C: 1})
check("report answer types", report["answer_types"], {"numeric": 1, "mcq": 1})
check("report calls", report["calls"], 6)
check("report config id", report["config_id"], team_config.vote([A, B, C]).config_id)

team_evaluation.registry_for = lambda args: FakeRegistry(BrokenClient({}, []))
try:
    raises("a batch where every call failed raises",
           lambda: team_evaluation.run_team_evaluation([A, B], samples, Args()), RuntimeError)
finally:
    team_evaluation.registry_for = original

rows = rows_from_report(report, batch=1, arm="test", team=[A, B, C], question_indices=[10, 11])
check("rows: one per question", len(rows), 2)
check("rows: schema", rows[0]["schema"], 2)
check("rows: every stage", [s["id"] for s in rows[0]["stages"]], [A, B, C])
check("rows: question index", [r["question_index"] for r in rows], [10, 11])
check("rows: aggregation", rows[0]["aggregation"]["rule"], "vote")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "holdout_predictions.jsonl"
    stripped = save_report(path, dict(report), batch=1, arm="test", team=[A, B, C], question_indices=[10, 11])
    check("save_report strips samples and config", ("samples" in stripped, "config" in stripped), (False, False))
    check("save_report writes the config once", list(json.loads((Path(tmp) / "configs.json").read_text())),
          [report["config_id"]])
    import rescore
    import paper_persona_baseline
    check("paper baseline imports save_report", hasattr(paper_persona_baseline, "save_report"), True)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        check(f"rescore strict agrees, q{row['question_index']}",
              rescore.rescore_row(row, "strict")["team_correct"], row["team_correct"])


# --------------------------------------------------------------------------
# --topology arranges a selected team; each persona is credited with its last stage.
# --------------------------------------------------------------------------
check("for_team vote", team_config.for_team([A, B, C, D]).config_id, team_config.vote([A, B, C, D]).config_id)
check("for_team centralized: last leads", team_config.for_team([A, B, C, D], "centralized").final, f"hub:{D}")
check("for_team synthesis: last leads", team_config.for_team([A, B, C], "synthesis").final, f"synth:{C}")
check("for_team pipeline default roles", [s.role for s in team_config.for_team([A, B, C, D], "pipeline").stages],
      ["solver", "critic", "critic", "reviser"])
raises("pipeline roles must match the team", lambda: team_config.for_team([A, B], "pipeline", roles=["solver"]), ValueError)


class TopologyArgs(Args):
    topology = "debate"
    rounds = 1
    handoff = "full"
    roles = None


# A changes its mind in round 1 (wrong, then right); B stays wrong.
flip_answers = {prefix(A): lambda c: "{final answer: 8}" if "Work from your team" in c else "{final answer: 6}",
                prefix(B): "{final answer: 7}"}
team_evaluation.registry_for = lambda args: FakeRegistry(FakeClient(flip_answers, []))
try:
    report = team_evaluation.run_team_evaluation([A, B], samples[:1], TopologyArgs())
finally:
    team_evaluation.registry_for = original
check("debate report keyed by persona", report["agents"], [A, B])
check("debater credited with its final round", report["per_agent_correct"], {A: 1, B: 0})
check("credited stage", report["credited_stage"], {A: f"{A}@r1", B: f"{B}@r1"})
check("debate votes over the final round", report["samples"][0]["answer_stages"], [f"{A}@r1", f"{B}@r1"])
check("debate ran two rounds", report["calls"], 4)


# --------------------------------------------------------------------------
# --max_inflight: one semaphore per server, shared by every client of it.
# --------------------------------------------------------------------------
import threading  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from model.registry import ModelRegistry, ModelSpec  # noqa: E402

registry = ModelRegistry({
    "default": ModelSpec("default", "m0", "http://a/v1"),
    "other": ModelSpec("other", "m1", "http://a/v1"),
    "far": ModelSpec("far", "m2", "http://b/v1"),
}, max_inflight=1)
check("same server shares a limiter", registry.client("default").limiter is registry.client("other").limiter, True)
check("other server has its own", registry.client("default").limiter is registry.client("far").limiter, False)

state = {"now": 0, "peak": 0}
lock = threading.Lock()


def fake_create(**kwargs):
    with lock:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
    time.sleep(0.02)
    with lock:
        state["now"] -= 1
    message = SimpleNamespace(content="{final answer: 8}", reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


for key in ("default", "other"):
    registry.client(key)._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
with concurrent.futures.ThreadPoolExecutor(8) as pool:
    list(pool.map(lambda k: registry.client(k).generate([{"role": "user", "content": "q"}]),
                  ["default", "other"] * 8))
check("at most one request in flight", state["peak"], 1)


if FAILURES:
    print(f"{len(FAILURES)} FAILED")
    for failure in FAILURES:
        print("  " + failure)
    sys.exit(1)
print("all runner checks passed")
