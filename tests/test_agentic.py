"""Offline checks for agentic tasks: tool calls, episodes, environments through every topology.

Run it directly - there is no pytest in the pinned environment:

    PYTHONPATH=src python tests/test_agentic.py

A toy environment (count up to a target, then submit) and scripted models stand
in for real tasks and servers. The Plancraft checks run only when the
`plancraft` package is installed (requirements-agentic.txt).
"""

import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmarks  # noqa: E402
import team_config  # noqa: E402
from benchmarks.environment import Outcome, OutcomeScorer, function_tool  # noqa: E402
from episode import report_text, run_episode  # noqa: E402
from model.openai_compat import Completion, OpenAICompatChatWrapper  # noqa: E402
from predictions import rows_from_report, save_report, traces_path  # noqa: E402
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


class Counter:
    """Count from 0 to `target` with `inc`, then `submit`. Submitting is a verdict; the count is state."""

    def __init__(self, target=3):
        self.target, self.n, self.submitted, self.done = target, 0, False, False

    def task_prompt(self):
        return f"Reach {self.target}, then submit. Now at {self.n}."

    def tools(self):
        return [function_tool("inc", "Add to the count.", {"by": {"type": "integer"}}, ["by"]),
                function_tool("submit", "Finish.", {}, [])]

    def call(self, name, arguments):
        if self.done:
            return "over"
        if name == "inc":
            self.n += int(arguments.get("by", 1))
            return f"now {self.n}"
        if name == "submit":
            self.submitted, self.done = True, True
            return "submitted"
        return f"ERROR: unknown tool {name}"

    def outcome(self):
        return Outcome(success=self.submitted and self.n == self.target, fingerprint=f"n={self.n}")

    def fork(self):
        return copy.deepcopy(self)

    def resume(self):
        self.submitted, self.done = False, False


def call(name, **arguments):
    return {"id": f"c_{name}", "name": name, "arguments": arguments,
            "arguments_raw": json.dumps(arguments), "error": None}


class Scripted:
    """A model whose next turn is the next item of the script for whoever is asking.

    `scripts` maps a marker that appears in the user message (a persona's text,
    a role's instruction) to a list of turns; each turn is a list of tool calls
    or a string reply.
    """

    def __init__(self, scripts):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.requests = []

    def generate(self, messages, tools=None, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), "tools": tools})
        user = messages[1]["content"]
        key = next(k for k in self.scripts if k in user)
        turn = self.scripts[key].pop(0) if self.scripts[key] else "nothing left to do"
        if isinstance(turn, str):
            return Completion(content=turn, completion_tokens=5, prompt_tokens=10, finish_reason="stop")
        return Completion(content="", tool_calls=turn, completion_tokens=5, prompt_tokens=10,
                          finish_reason="tool_calls")


class Registry:
    def __init__(self, client):
        self._client = client

    def client(self, key):
        return self._client


PERSONAS = {name: {"prompt": f"<{name}>", "temperature": 0, "top_p": 0.9} for name in "ABCD"}
SCORER = OutcomeScorer()


def run(config, client, target=3, max_steps=10):
    return run_question(config, "count", "unused", scorer=SCORER, personas=PERSONAS,
                        registry=Registry(client), max_tokens=100, instance=SimpleNamespace(metadata={}),
                        environment=lambda instance: Counter(target), max_steps=max_steps)


def test_completion_reads_tool_calls():
    wrapper = OpenAICompatChatWrapper(base_url="http://unused/v1", model_name="fake")
    message = SimpleNamespace(content="", tool_calls=[
        SimpleNamespace(id="a", function=SimpleNamespace(name="inc", arguments='{"by": 2}')),
        SimpleNamespace(id="b", function=SimpleNamespace(name="inc", arguments='{"by": ')),
    ])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4))
    wrapper._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: response)))
    completion = wrapper.generate([{"role": "user", "content": "x"}], tools=[{}])
    check("parsed arguments", completion.tool_calls[0]["arguments"], {"by": 2})
    check("bad JSON is an error, not a crash", (completion.tool_calls[1]["arguments"],
                                                completion.tool_calls[1]["error"] is not None), ({}, True))
    check("assistant message carries the calls", completion.assistant_message()["tool_calls"][0]["function"]["name"], "inc")


def test_episode_loop():
    client = Scripted({"go": [[call("inc", by=2)], [call("inc", by=1)], [call("submit")]]})
    record = run_episode(client, Counter(3), system="s", user="go", max_steps=10, max_tokens=50)
    check("succeeds", record["prediction"].value, "success|n=3")
    check("three turns, three tool calls", (record["calls"], record["tool_calls"]), (3, 3))
    check("stopped because done", record["stopped"], "done")
    check("tool results go back to the model", client.requests[1]["messages"][-1]["role"], "tool")

    talker = Scripted({"go": ["thinking", "still thinking"]})
    record = run_episode(talker, Counter(3), system="s", user="go", max_steps=10, max_tokens=50)
    check("two replies without a tool call end it", (record["stopped"], record["calls"]), ("no_tool_call", 2))
    check("and it fails", record["prediction"].value, "fail|n=0")

    looper = Scripted({"go": [[call("inc", by=1)]] * 20})
    record = run_episode(looper, Counter(30), system="s", user="go", max_steps=4, max_tokens=50)
    check("turn cap", (record["stopped"], record["calls"]), ("max_steps", 4))

    finished = Counter(0)
    finished.call("submit", {})
    record = run_episode(Scripted({"go": []}), finished, system="s", user="go", max_steps=5, max_tokens=50)
    check("already done: no call made", (record["stopped"], record["calls"]), ("already_done", 0))


def test_reports_never_carry_the_grade():
    record = run_episode(Scripted({"go": [[call("inc", by=3)], [call("submit")]]}), Counter(3),
                         system="s", user="go", max_steps=5, max_tokens=50)
    text = report_text(record).lower()
    check("report names the actions", "inc" in text and "submit" in text, True)
    check("report does not say whether it succeeded", "success" in text or "succeeded" in text, False)


def test_vote_over_outcomes():
    client = Scripted({
        "<A>": [[call("inc", by=3)], [call("submit")]],
        "<B>": [[call("inc", by=3)], [call("submit")]],
        "<C>": [[call("inc", by=1)], [call("submit")]],
    })
    result = run(team_config.vote(["A", "B", "C"]), client)
    check("majority end state wins", result["team_answer"], "success|n=3")
    check("team correct", result["team_correct"], True)
    check("each voter graded on its own fork", [s["correct"] for s in result["stages"]], [True, True, False])
    check("calls counted across episodes", result["calls"], 6)


def test_pipeline_builds_on_the_last_state():
    client = Scripted({
        "<A>": [[call("inc", by=2)], [call("submit")]],          # stops one short
        "<B>": [[call("inc", by=1)], [call("submit")]],          # carries on from 2
    })
    config = team_config.pipeline([("solver", "A"), ("critic", "B")])
    result = run(config, client)
    solver, critic = result["stages"]
    check("solver's own record unchanged", solver["prediction"], "fail|n=2")
    check("critic overruled the submission and finished", critic["prediction"], "success|n=3")
    check("critic recorded as a fix", critic["transition"], "fixed")
    check("pipeline's answer is the last stage's", result["team_correct"], True)
    check("the critic saw the solver's actions",
          "inc" in client.requests[2]["messages"][1]["content"], True)


def test_debate_round_continues_own_attempt():
    client = Scripted({
        "<A>": [[call("inc", by=2)], [call("submit")], [call("inc", by=1)], [call("submit")]],
        "<B>": [[call("inc", by=3)], [call("submit")], [call("submit")]],
    })
    result = run(team_config.debate(["A", "B"], rounds=1), client)
    final = {s["id"]: s["prediction"] for s in result["stages"] if s["role"] == "debater"}
    check("each debater continues its own state", sorted(final.values()), ["success|n=3", "success|n=3"])


def test_hub_starts_fresh():
    client = Scripted({
        "<A>": [[call("inc", by=5)], [call("submit")]],
        "<B>": [[call("inc", by=3)], [call("submit")]],
    })
    result = run(team_config.centralized(["A"], hub="B"), client)
    check("hub's count starts from zero, not the worker's 5", result["stages"][-1]["prediction"], "success|n=3")


def test_delegation():
    # Markers are checked in order; the worker's prompt quotes the instruction, so
    # its marker comes first and the hub's persona marker never appears in it.
    client = Scripted({
        "ADD-TWO": [[call("inc", by=2)], "done"],
        "<HUB>": [[call("delegate", agent="A", instruction="ADD-TWO")],
                  [call("delegate", agent="nobody", instruction="x")],
                  [call("inc", by=1)], [call("submit")]],
    })
    personas = dict(PERSONAS, B={"prompt": "<HUB>", "temperature": 0, "top_p": 0.9})
    config = team_config.delegated(["A"], hub="B")
    result = run_question(config, "count", "unused", scorer=SCORER, personas=personas,
                          registry=Registry(client), max_tokens=100, instance=SimpleNamespace(metadata={}),
                          environment=lambda instance: Counter(3), max_steps=10)
    worker, hub = result["stages"]
    check("worker acted on the hub's environment", hub["prediction"], "success|n=3")
    check("worker called once", worker["times_called"], 1)
    check("worker's calls counted", worker["calls"], 2)
    check("an unknown teammate is an error the hub reads, not a crash",
          any("no teammate" in (s.get("observation") or "") for s in hub["steps"]), True)
    check("the team's answer is the hub's", result["answer_stages"], [hub["id"]])
    check("delegation layers: hub only", config.layers(), [[hub["id"]]])
    raises("delegates on a static question",
           lambda: run_question(config, "q", "1", scorer=SCORER, personas=personas, registry=Registry(client),
                                max_tokens=10), ValueError)


def test_new_stage_fields_keep_old_config_ids():
    stage = team_config.vote(["A", "B"]).stages[0].to_dict()
    check("unset max_steps and delegates are not serialised", "max_steps" in stage or "delegates" in stage, False)
    raw = team_config.delegated(["A"], hub="B").to_dict()
    check("delegated round-trips", team_config.TeamConfig.from_dict(raw).to_dict(), raw)


def test_traces_split_from_predictions():
    client = Scripted({"<A>": [[call("inc", by=3)], [call("submit")]]})
    result = run(team_config.vote(["A"]), client)
    sample = dict(result, question="count", gold="unused", dataset="toy", tags=[])
    report = {"samples": [sample], "config_id": "x", "config": {}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "holdout_predictions.jsonl"
        save_report(path, report, batch=1, arm="test")
        rows = [json.loads(line) for line in path.open()]
        traces = [json.loads(line) for line in traces_path(path).open()]
    check("traces file beside predictions", traces_path(path).name, "holdout_traces.jsonl")
    check("prediction rows carry no steps", "steps" in rows[0]["stages"][0], False)
    check("trace rows carry them", [s["tool"] for s in traces[0]["steps"]], ["inc", "submit"])


def test_plancraft():
    try:
        import plancraft  # noqa: F401
    except ImportError:
        print("  (plancraft not installed; skipping its checks)")
        return
    bench = benchmarks.get("plancraft")
    instances = bench.instances(SimpleNamespace(data_dir="", data_size=0, sub_data=""), "test")
    check("the paper's first 100", len(instances), 100)
    possible = next(i for i in instances if not i.metadata["impossible"])
    impossible = next(i for i in instances if i.metadata["impossible"])

    env = bench.environment(impossible)
    env.call("impossible", {"reason": "x"})
    check("impossible on an impossible task succeeds", env.outcome().success, True)
    env = bench.environment(possible)
    env.call("impossible", {"reason": "x"})
    check("impossible on a craftable task fails", env.outcome().success, False)
    twin = env.fork()
    twin.resume()
    check("resume lets the next agent overrule it", (twin.done, env.done), (False, True))

    env = bench.environment(possible)
    before = env.observation()
    twin = env.fork()
    first = [line for line in before.splitlines() if "[I" in line][0]
    slot = first.split("[")[1].split("]")[0]
    env.call("move", {"slot_from": f"[{slot}]", "slot_to": "[A1]", "quantity": 1})
    check("a fork does not move when the original does", twin.observation(), before)
    check("a bad slot is an error message", env.call("move", {"slot_from": "I1", "slot_to": "[0]", "quantity": 1})
          .startswith("ERROR"), True)
    check("search answers with a recipe", "recipe" in env.call("search", {"recipe_name": possible.metadata["target"]}).lower(), True)


def test_workbench():
    import ast
    import re
    from benchmarks import workbench
    if not (workbench.upstream_root() / "src" / "tools" / "state.py").exists():
        print("  (WorkBench not fetched; skipping its checks: scripts/fetch_benchmarks.py workbench)")
        return
    bench = benchmarks.get("workbench")
    instances = bench.instances(SimpleNamespace(data_dir=str(workbench.upstream_root().parent), data_size=0,
                                                sub_data=""), "test")
    check("the paper's 100", len(instances), 100)

    def replay(instance, stopped="reply"):
        env = bench.environment(instance)
        for action in instance.metadata["ground_truth"]:
            tree = ast.parse(action, mode="eval").body
            parts, node = [], tree.func
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            parts = [node.id] + parts[::-1]
            env.call(re.sub(r"[^a-zA-Z0-9_-]", "_", ".".join(parts[:-1])),
                     {k.arg: ast.literal_eval(k.value) for k in tree.keywords})
        env.close(stopped)
        return env

    writing = next(i for i in instances if i.metadata["ground_truth"])
    check("ground truth, replayed as tool calls, grades correct", replay(writing).outcome().success, True)
    check("the same calls at the turn cap grade wrong, as upstream", replay(writing, "max_steps").outcome().success, False)
    env = bench.environment(writing)
    env.close("reply")
    check("doing nothing on a task that needs a change fails", env.outcome().success, False)
    env = replay(writing)
    check("a fork keeps the actions so far", env.fork().actions, env.actions)
    check("unknown tools are an error message", env.call("email_nuke", {}).startswith("ERROR"), True)


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
