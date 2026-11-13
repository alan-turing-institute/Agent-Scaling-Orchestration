"""Frontier baseline: Claude (via `bare_model_baseline.py --claude_cli`) on X1's questions, per benchmark.

Reads one fold/sample of the Claude runs under `--root` (`stage1/<set>/<agent>/fold<f>/s<k>[-rerun]/`;
a rerun's rows replace the original's failed ones, and regraded predictions are used where they
exist), sums `claude_usage.jsonl` (every call, including ones whose episode was later rerun), and
puts the 35B alone (`--l2` roots) and the orchestrator (`--stage2` roots, `--modes`) beside it on
the same questions.

    python3 scripts/frontier_summary.py --root data-claude/frontier-sonnet --fold 0 --sample 0 \
        --l2 data-claude/x1-l2 data-claude/x1-l2-t8 --stage2 data-claude/x1-teams/stage2 SPARK/stage2 \
        --out data-claude/frontier-sonnet/summary_fold0_s0.json
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import types
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

POOLS = ("data-claude/tagged_hard", "data-claude/tagged_agentic")
# Workers built on a ~35B model (the rest are 3–4B), for splitting the team's tokens.
LARGE_WORKERS = ("BigBang", "occamy", "Aquila", "Fx-Work", "35B")


def task_datasets(fold, split_seed=0, n_folds=5):
    """task_sha, as claude_cli logs it (the task's first user message), -> dataset, for one fold.

    Rebuilds each held-out task's first message as bare_model_baseline.py does: the question with
    X1's summary request and its format line, or the environment's task prompt.
    """
    from datasets import load_from_disk
    import benchmarks
    import bare_model_baseline as b
    from judge import DeferredJudge
    shas = {}
    for path in POOLS:
        args = types.SimpleNamespace(split_seed=split_seed, n_folds=n_folds, fold=fold, test_fraction=None,
                                     summary="hard" in path, persona=None, system_file=None)
        _, test = b.make_split(load_from_disk(path), args)
        scorers = benchmarks.ScorerSet(mode="strict")
        for s in test:
            bench = benchmarks.get(s["dataset"])
            if bench.environment is None:
                prompt = b.question_prompt(s["question"], scorers.for_sample(s)[1].instruction_suffix(), args)
            else:
                prompt = bench.environment_for(DeferredJudge())(benchmarks.instance_of(s)).task_prompt()
            shas[hashlib.sha1(prompt.encode()).hexdigest()[:16]] = s["dataset"]
    return shas


def rows_of(run_dir):
    path = os.path.join(run_dir, "holdout_predictions.jsonl")
    regraded = path.replace(".jsonl", ".regraded.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(regraded if os.path.exists(regraded) else path)]


def failed(row):
    return bool(row["stages"][0].get("error"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--agent", default="sonnet")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--l2", nargs="*", default=[])
    p.add_argument("--stage2", nargs="*", default=[])
    p.add_argument("--modes", default="headline")
    p.add_argument("--out")
    p.add_argument("--extra_usage", nargs="*", default=[],
                   help="More claude_usage.jsonl files counted only in the windows table (e.g. smoke tests)")
    p.add_argument("--window_start", default=None,
                   help="UTC start of the first 5-hour usage window, e.g. 2026-11-12T10:20; with --windows_out")
    p.add_argument("--limit_logs", nargs="*", default=[], help="run.log files whose 'no model call' lines mark limits")
    p.add_argument("--windows_out")
    args = p.parse_args()

    claude, usage = {}, []
    for run in sorted(glob.glob(f"{args.root}/stage1/*/{args.agent}/fold{args.fold}/s{args.sample}")):
        rerun = {r["question_key"]: r for r in rows_of(run + "-rerun")}
        for r in rows_of(run):
            if failed(r) and r["question_key"] in rerun:
                r = rerun[r["question_key"]]
            claude[r["question_key"]] = r
        for d in (run, run + "-rerun"):
            log = os.path.join(d, "claude_usage.jsonl")
            if os.path.exists(log):
                usage += [json.loads(line) for line in open(log)]

    alone = {}
    for root in args.l2:
        for run in glob.glob(f"{root}/stage1/*/qwen35b-off/fold{args.fold}/s{args.sample}"):
            alone.update({r["question_key"]: r for r in rows_of(run)})
    orch = {}
    for root in args.stage2:
        for run in glob.glob(f"{root}/{args.modes}/T*/fold{args.fold}/s{args.sample}"):
            for r in rows_of(run):
                orch.setdefault(r["question_key"], r)

    sha_dataset = task_datasets(args.fold)
    spend = defaultdict(lambda: defaultdict(float))
    for u in usage:
        s = spend[sha_dataset.get(u["task_sha"], "unmatched")]
        s["calls"] += 1
        s["cost_usd"] += u["cost_usd"] or 0
        s["input_tokens"] += sum(u.get(k) or 0 for k in
                                 ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        s["cache_read_input_tokens"] += u.get("cache_read_input_tokens") or 0
        s["output_tokens"] += u["output_tokens"] or 0
        s["thinking_tokens"] += u["thinking_tokens"] or 0

    table = defaultdict(lambda: defaultdict(float))
    for key, r in claude.items():
        t = table[r["dataset"]]
        t["n"] += 1
        t["claude"] += bool(r["team_correct"])
        t["failed"] += failed(r)
        t["claude_completion_tokens"] += r.get("completion_tokens") or 0
        t["claude_prompt_tokens"] += r.get("prompt_tokens") or 0
        if key in alone:
            t["alone_n"] += 1
            t["alone"] += bool(alone[key]["team_correct"])
            t["alone_completion_tokens"] += alone[key].get("completion_tokens") or 0
            t["alone_prompt_tokens"] += alone[key].get("prompt_tokens") or 0
        if key in orch:
            t["orch_n"] += 1
            t["orch"] += bool(orch[key]["team_correct"])
            t["orch_claude"] += bool(r["team_correct"])
            t["orch_completion_tokens"] += orch[key].get("completion_tokens") or 0
            t["orch_prompt_tokens"] += orch[key].get("prompt_tokens") or 0
            for st in orch[key]["stages"]:
                out_tokens = st.get("completion_tokens") or 0
                if st.get("role") == "hub":
                    t["hub_completion_tokens"] += out_tokens
                elif any(m.lower() in (st.get("model") or "").lower() for m in LARGE_WORKERS):
                    t["large_worker_completion_tokens"] += out_tokens
                else:
                    t["small_worker_completion_tokens"] += out_tokens

    total = defaultdict(float)
    for u in usage:
        total["calls"] += 1
        total["cost_usd"] += u["cost_usd"] or 0
        total["input_tokens"] += u.get("input_tokens") or 0
        total["cache_creation_input_tokens"] += u.get("cache_creation_input_tokens") or 0
        total["cache_read_input_tokens"] += u.get("cache_read_input_tokens") or 0
        total["output_tokens"] += u["output_tokens"] or 0
        total["thinking_tokens"] += u["thinking_tokens"] or 0

    print(f"fold {args.fold} sample {args.sample}; accuracy on the same questions; tokens per question")
    print(f"{'benchmark':16} {'n':>4} {'Claude':>7} {'35B':>7} {'orch':>7} {'fail':>4} {'C out tok':>9} "
          f"{'C in tok':>9} {'35B out':>8} {'orch out':>8}")
    out = {}
    for ds in sorted(table):
        t = table[ds]
        n = t["n"]
        row = {"n": int(n), "claude": t["claude"] / n, "failed": int(t["failed"]),
               "claude_correct": int(t["claude"]), "alone_correct": int(t["alone"]), "alone_n": int(t["alone_n"]),
               "orch_correct": int(t["orch"]), "orch_n": int(t["orch_n"]),
               "alone_35b": t["alone"] / t["alone_n"] if t["alone_n"] else None,
               "orchestrator": t["orch"] / t["orch_n"] if t["orch_n"] else None,
               "claude_on_orch_questions": t["orch_claude"] / t["orch_n"] if t["orch_n"] else None,
               "claude_completion_tokens": t["claude_completion_tokens"] / n,
               "claude_prompt_tokens": t["claude_prompt_tokens"] / n,
               "alone_completion_tokens": t["alone_completion_tokens"] / t["alone_n"] if t["alone_n"] else None,
               "orch_completion_tokens": t["orch_completion_tokens"] / t["orch_n"] if t["orch_n"] else None,
               "totals": {k: v for k, v in t.items() if k.endswith("_tokens") or k.endswith("_n") or k == "n"}}
        out[ds] = row
        fmt = lambda v: f"{v:7.1%}" if v is not None else f"{'-':>7}"
        fmt_t = lambda v: f"{v:8.0f}" if v is not None else f"{'-':>8}"
        print(f"{ds:16} {n:4.0f} {fmt(row['claude'])} {fmt(row['alone_35b'])} {fmt(row['orchestrator'])} "
              f"{row['failed']:4d} {row['claude_completion_tokens']:9.0f} {row['claude_prompt_tokens']:9.0f} "
              f"{fmt_t(row['alone_completion_tokens'])} {fmt_t(row['orch_completion_tokens'])}")
    print("\nspend (every call, including episodes lost to a usage limit and rerun):")
    print(f"{'benchmark':16} {'calls':>6} {'input tok':>11} {'cache read':>11} {'output tok':>10} {'thinking':>9} {'cost $':>8} {'$/task':>7}")
    for ds in sorted(spend):
        s = spend[ds]
        n = table[ds]["n"] if ds in table else 0
        print(f"{ds:16} {s['calls']:6.0f} {s['input_tokens']:11.0f} {s['cache_read_input_tokens']:11.0f} "
              f"{s['output_tokens']:10.0f} {s['thinking_tokens']:9.0f} {s['cost_usd']:8.2f} "
              f"{(s['cost_usd'] / n if n else 0):7.3f}")
    print("total usage:", {k: round(v, 2) for k, v in total.items()})
    if args.windows_out and args.window_start:
        import datetime as dt
        start = dt.datetime.fromisoformat(args.window_start).replace(tzinfo=dt.timezone.utc).timestamp()
        calls = usage + [json.loads(line) for path in args.extra_usage for line in open(path)]
        limited = set()
        for path in args.limit_logs:
            for line in open(path):
                if "no model call" in line or "hit your" in line:
                    limited.add(path)
        windows = defaultdict(lambda: {"calls": 0, "cost_usd": 0.0, "last": 0})
        for u in calls:
            w = windows[int((u["ts"] - start) // (5 * 3600))]
            w["calls"] += 1
            w["cost_usd"] += u["cost_usd"] or 0
            w["last"] = max(w["last"], u["ts"])
        rows = []
        for i in sorted(windows):
            w = windows[i]
            a = dt.datetime.fromtimestamp(start + i * 5 * 3600, dt.timezone.utc)
            b = a + dt.timedelta(hours=5)
            rows.append({"window": f"{a:%d %b %H:%M}–{b:%H:%M}", "calls": w["calls"],
                         "cost_usd": round(w["cost_usd"], 2), "last_call": w["last"],
                         "ended": "hit the 5-hour limit" if i < max(windows) else "run finished"})
        with open(args.windows_out, "w") as f:
            json.dump(rows, f, indent=1)
        for r in rows:
            print(r["window"], r["calls"], r["cost_usd"], r["ended"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"fold": args.fold, "sample": args.sample, "benchmarks": out,
                       "spend": {k: dict(v) for k, v in spend.items()}, "usage_total": total}, f, indent=1)


if __name__ == "__main__":
    main()
