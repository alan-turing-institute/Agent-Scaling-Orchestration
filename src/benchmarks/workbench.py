"""WorkBench: workplace tasks done through 27 tools over email, calendar, CRM, analytics and projects.

Styles et al., 2024 (github.com/olly-styles/WorkBench, MIT), one of the six
benchmarks of Kim et al. (arXiv:2512.08296). The agent works in a sandboxed
company (500 emails, 300 calendar events, a CRM, web analytics, a project
board) and is graded on the sandbox's state afterwards: upstream replays the
agent's calls and the ground-truth calls on fresh copies and compares the
resulting tables. Reads are harmless; a wrong write, or a write to the wrong
record, fails the task.

This uses upstream's own tools, sandbox data and grader (`is_correct`),
fetched at a pinned commit by `fetch` into `<data_dir>/workbench-upstream`, so
the grading is theirs exactly. The paper harness's adapter is not used: it has
four stub tools whose searches return nothing and grades argument names, not
values or state.

Upstream keeps the sandbox in a per-thread `ToolState`. Each episode here owns
its own `ToolState` and binds it into the thread only for the length of a tool
call, so concurrent episodes, forks and a hub's delegated workers each see
their own state, and grading always runs on fresh copies.

Default set: the 100 tasks the paper's harness sampled (17 from each of
analytics, calendar, CRM and email, 16 from multi-domain and project
management, one `random.Random(42)` shuffled through the domains in
alphabetical order), on upstream's current task wording and ground truth.
`--sub_data all` takes all 690; add `,v1` for the 2024 wording the paper ran
on (e.g. `--sub_data v1` or `--sub_data all,v1`).

A task ends when the agent replies without a tool call: upstream's agents end
with a final message, and there is no "submit". An episode cut off at the turn
cap is graded wrong, as upstream grades an agent that hits its iteration limit.
"""
NAME = 'workbench'
ANSWER_TYPE = 'outcome'
# The paper assigned no persona set to this benchmark; see gpqa_diamond.py.
PERSONA_SET = []
# Upstream's agent runs at most 20 iterations.
MAX_STEPS = 20

UPSTREAM_REPO = "https://github.com/olly-styles/WorkBench"
UPSTREAM_COMMIT = "49c7dfd00c03d384ec59ea57374f50b766aa5613"
DOMAINS = ("analytics", "calendar", "customer_relationship_manager", "email", "multi_domain",
           "project_management")
PAPER_SAMPLE, PAPER_SEED = 100, 42
CURRENT_TIME_NOTE = ("Today's date is Thursday, 2023-11-30 and the current time is 00:00:00. Remember the "
                     "current date and time when completing tasks. Meetings must not start before 9am or "
                     "end after 6pm.")

import ast
import contextlib
import hashlib
import os
import random
import re
import subprocess
import sys
import threading
from pathlib import Path

from benchmarks.base import Instance
from benchmarks.environment import Outcome

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data-claude" / "benchmarks" / "workbench-upstream"
_IMPORT_LOCK = threading.Lock()
_UPSTREAM = None


def upstream_root(data_dir=None):
    """Where upstream is checked out: `WORKBENCH_ROOT`, else `<data_dir>/workbench-upstream`."""
    if os.environ.get("WORKBENCH_ROOT"):
        return Path(os.environ["WORKBENCH_ROOT"])
    return Path(data_dir) / "workbench-upstream" if data_dir else _DEFAULT_ROOT


def fetch(args):
    """Clone upstream and check out the pinned commit (scripts/fetch_benchmarks.py)."""
    root = upstream_root(args.data_dir)
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", UPSTREAM_REPO, str(root)], check=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != UPSTREAM_COMMIT:
        subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "origin"], check=False)
        subprocess.run(["git", "-C", str(root), "checkout", "--quiet", UPSTREAM_COMMIT], check=True)


def _upstream(root=None):
    """Upstream's tools, state and grader, imported once from its checkout.

    Upstream's code is a package named `src` with CSV paths relative to its
    repository root; its root goes first on `sys.path` and the paths are made
    absolute, so nothing depends on the working directory.
    """
    global _UPSTREAM
    with _IMPORT_LOCK:
        if _UPSTREAM is not None:
            return _UPSTREAM
        root = Path(root or upstream_root()).resolve()
        if not (root / "src" / "tools" / "state.py").exists():
            raise FileNotFoundError(f"WorkBench is not checked out at {root}; run "
                                    f"scripts/fetch_benchmarks.py workbench (or set WORKBENCH_ROOT)")
        sys.path.insert(0, str(root))
        import src.tools.state as state
        state._CSV_PATHS = {name: str(root / path) for name, path in state._CSV_PATHS.items()}
        from src.evals import actions, evaluation
        from src.tools import toolkits
        from src.tools.tool import tool_to_openai_schema
        tools = {re.sub(r"[^a-zA-Z0-9_-]", "_", t.name): t for t in toolkits.all_tools}
        schemas = []
        for sanitized, t in tools.items():
            schema = tool_to_openai_schema(t)
            schema["function"]["name"] = sanitized
            schemas.append(schema)
        _UPSTREAM = {
            "state": state, "actions": actions, "evaluation": evaluation, "root": root,
            "tools": tools, "schemas": schemas,
            "writes": {t.name for t in toolkits.tools_with_side_effects},
        }
        return _UPSTREAM


@contextlib.contextmanager
def _bound(state_module, tool_state):
    """Make `tool_state` the calling thread's sandbox for the duration, then put back what was there."""
    previous = getattr(state_module._local, "tool_state", None)
    state_module._local.tool_state = tool_state
    try:
        yield
    finally:
        state_module._local.tool_state = previous


class WorkBenchTask:
    """One WorkBench task as an `Environment` (see benchmarks.environment)."""

    # Upstream's agents end with a final message; a reply without a tool call ends the task.
    text_reply_ends = True

    def __init__(self, metadata, _state=None, _actions=None):
        self.up = _upstream()
        self.metadata = metadata
        self.task = metadata["task"]
        self.ground_truth = list(metadata["ground_truth"])
        self.tool_state = _state if _state is not None else self.up["state"]._pristine_state().copy()
        self.actions = list(_actions or [])   # every call, in upstream's string form, reads included
        self.done = False
        self.error = ""

    def task_prompt(self):
        return f"{CURRENT_TIME_NOTE}\n\n{self.task}"

    def tools(self):
        return self.up["schemas"]

    def call(self, name, arguments):
        tool = self.up["tools"].get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}"
        # Upstream passes every argument as a string, and grades the call in that form.
        arguments = {k: str(v) for k, v in (arguments or {}).items() if v is not None}
        self.actions.append(self.up["actions"].convert_intermediate_step_to_function_call(tool.name, arguments))
        try:
            with _bound(self.up["state"], self.tool_state):
                return str(tool.func(**arguments))
        except Exception as error:  # a bad argument name or value, as upstream reports it
            return f"ERROR: {type(error).__name__}: {error}"

    def close(self, stopped):
        """Upstream grades an agent that hits its iteration limit as wrong."""
        self.error = "hit the turn cap" if stopped == "max_steps" else ""

    def outcome(self):
        # Grading replays both call lists on fresh sandboxes, through the thread's
        # state; unbind first so it can never reset an episode's own state.
        with _bound(self.up["state"], None):
            success = bool(self.up["evaluation"].is_correct(self.actions, self.ground_truth, self.error))
        writes = sorted(a.lower() for a in self.actions if a.split(".func(")[0] in self.up["writes"])
        fingerprint = hashlib.sha1("\n".join(writes).encode("utf-8")).hexdigest()[:10]
        return Outcome(success=success, fingerprint=f"writes:{len(writes)}:{fingerprint}",
                       detail={"actions": len(self.actions), "writes": len(writes), "error": self.error})

    def fork(self):
        return WorkBenchTask(self.metadata, _state=self.tool_state.copy(), _actions=self.actions)


def ENVIRONMENT(instance):
    return WorkBenchTask(instance.metadata)


def _tasks(root, version):
    import pandas as pd
    folder = root / "data" / "processed" / "tasks_and_outcomes"
    if version == "v1":
        folder = folder / "v1"
    records = []
    for domain in DOMAINS:
        frame = pd.read_csv(folder / f"{domain}_tasks_and_outcomes.csv")
        task_column = "task" if "task" in frame.columns else "query"
        outcome_column = "outcome" if "outcome" in frame.columns else "answer"
        for row, (task, outcome, domains) in enumerate(zip(frame[task_column], frame[outcome_column],
                                                            frame["domains"])):
            records.append({"task_id": f"{domain}-{row:04d}", "domain": domain, "task": str(task),
                            "ground_truth": list(ast.literal_eval(outcome)) if isinstance(outcome, str) else [],
                            "domains": list(ast.literal_eval(domains)) if isinstance(domains, str) else [domain]})
    return records


def paper_sample(records, sample_size=PAPER_SAMPLE, seed=PAPER_SEED):
    """The paper harness's stratified sample (`stratified_subset` in its _convert_workbench.py)."""
    by_domain = {}
    for record in records:
        by_domain.setdefault(record["task_id"].split("-", 1)[0], []).append(record)
    domains = sorted(by_domain)
    base, extra = divmod(sample_size, len(domains))
    rng = random.Random(seed)
    picked = []
    for i, domain in enumerate(domains):
        pool = list(by_domain[domain])
        rng.shuffle(pool)
        picked.extend(pool[:base + (1 if i < extra else 0)])
    return picked


def load_instances(args, split='test'):
    root = upstream_root(args.data_dir)
    options = {o.strip() for o in (getattr(args, 'sub_data', '') or '').split(',') if o.strip()}
    version = "v1" if "v1" in options else "current"
    records = _tasks(root, version)
    if "all" not in options:
        records = paper_sample(records)
    if args.data_size and args.data_size > 0:
        records = records[:args.data_size]

    instances = []
    for record in records:
        writes = len(record["ground_truth"])
        instances.append(Instance(
            question=record["task"],
            answer="; ".join(record["ground_truth"]) or "(no change)",
            id=f"{NAME}:{record['task_id']}",
            tags=[f"benchmark: {NAME}", f"domain: {record['domain'].replace('_', ' ')}",
                  f"expected writes: {'none' if writes == 0 else '1-2' if writes <= 2 else '3+'}", "tools: 27"],
            metadata={"task_id": record["task_id"], "task": record["task"], "domain": record["domain"],
                      "domains": record["domains"], "ground_truth": record["ground_truth"],
                      "version": version, "upstream_commit": UPSTREAM_COMMIT},
        ))
    return instances
