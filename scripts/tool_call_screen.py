"""Tool-calling screen: how well a model drives the agentic environments on its own.

Reads the traces a bare-model run wrote (scripts/bare_model_baseline.py, or the
bare_model arm of scripts/small_model_arms.sh) and reports, per benchmark:

  - turns that made a tool call, against plain-text replies;
  - plain-text replies that look like a tool call the server failed to parse
    (`<tool_call>`, `<function=`, `[TOOL_CALLS]`): a sign the serve script's
    --tool-call-parser is wrong or missing;
  - tool calls with arguments that were not JSON, calls to a tool that does not
    exist, and calls the environment answered with an error;
  - how episodes stopped (`done` means the agent reached a submit or finished the
    task), and how many succeeded.

The plan's bar (plan.md §6.4, H3): a model that rarely produces valid calls or
rarely reaches a submit on Plancraft starts agentic work at the next tier up.

    python scripts/tool_call_screen.py data-claude/crossval-small-agentic/qwen3.5-0.8b/fold*/bare_model
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

UNPARSED = re.compile(r"<tool_call>|<function=|\[TOOL_CALLS\]|\"name\"\s*:\s*\"\w+\"\s*,\s*\"arguments\"")
STOPS = ("done", "reply", "no_tool_call", "max_steps", "context_limit", "token_budget", "already_done")


def classify(step):
    """One tool-call step -> ok | malformed | unknown_tool | env_error."""
    observation = step.get("observation") or ""
    if not observation.startswith("ERROR"):
        return "ok"
    if "arguments are not" in observation[:60]:
        return "malformed"
    if "unknown tool" in observation[:40]:
        return "unknown_tool"
    return "env_error"


def load_run(run_dir):
    """Prediction rows keyed by question_key, and the trace rows."""
    run_dir = Path(run_dir)
    rows, traces = {}, []
    for path in sorted(run_dir.glob("*predictions*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            rows[row.get("question_key")] = row
    for path in sorted(run_dir.glob("*traces*.jsonl")):
        traces += [json.loads(line) for line in path.open()]
    return rows, traces


def summarise(run_dirs):
    stats = defaultdict(lambda: Counter())
    stops = defaultdict(Counter)
    for run_dir in run_dirs:
        rows, traces = load_run(run_dir)
        if not traces:
            print(f"[warn] no traces in {run_dir}", file=sys.stderr)
        for trace in traces:
            row = rows.get(trace.get("question_key"), {})
            dataset = row.get("dataset") or "unknown"
            s = stats[dataset]
            s["episodes"] += 1
            s["success"] += bool(row.get("correct"))
            stops[dataset][trace.get("stopped") or "unknown"] += 1
            turns = set()
            for step in trace["steps"]:
                turns.add(step.get("turn"))
                if step.get("tool") is None:
                    text = step.get("text") or ""
                    if text.startswith("[context limit"):
                        continue
                    s["text_turns"] += 1
                    s["unparsed_calls"] += bool(UNPARSED.search(text))
                else:
                    s["tool_calls"] += 1
                    s[classify(step)] += 1
            s["turns"] += len(turns)
            s["call_turns"] += len({st.get("turn") for st in trace["steps"] if st.get("tool") is not None})
    return stats, stops


def pct(part, whole):
    return f"{100 * part / whole:5.1f}%" if whole else "    -"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", help="bare-model output directories (one per fold)")
    parser.add_argument("--json", help="also write the numbers here")
    args = parser.parse_args(argv)

    stats, stops = summarise(args.run_dirs)
    print(f"{'benchmark':16s} {'tasks':>5s} {'success':>8s} {'turns/task':>10s} {'call turns':>10s} "
          f"{'unparsed':>8s} {'calls':>6s} {'valid':>6s} {'malformed':>9s} {'unknown':>7s} {'env err':>7s} {'reached done':>12s}")
    out = {}
    for dataset in sorted(stats):
        s, n = stats[dataset], stats[dataset]["episodes"]
        print(f"{dataset:16s} {n:5d} {pct(s['success'], n):>8s} {s['turns'] / n:10.1f} "
              f"{pct(s['call_turns'], s['turns']):>10s} {s['unparsed_calls']:8d} {s['tool_calls']:6d} "
              f"{pct(s['ok'], s['tool_calls']):>6s} {pct(s['malformed'], s['tool_calls']):>9s} "
              f"{pct(s['unknown_tool'], s['tool_calls']):>7s} {pct(s['env_error'], s['tool_calls']):>7s} "
              f"{pct(stops[dataset]['done'], n):>12s}")
        out[dataset] = {**s, "stopped": dict(stops[dataset])}
    print("\nhow episodes stopped:")
    for dataset in sorted(stops):
        parts = [f"{k} {v}" for k, v in sorted(stops[dataset].items(), key=lambda kv: (kv[0] not in STOPS, kv[0]))]
        print(f"  {dataset:16s} " + ", ".join(parts))
    print("\n'reached done' counts episodes the environment marked finished (a submit, an `impossible`"
          " verdict, BrowseComp's `done`); WorkBench ends on a text reply instead ('reply').")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
