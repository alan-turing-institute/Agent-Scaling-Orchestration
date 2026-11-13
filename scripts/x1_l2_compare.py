"""X1 L2: the orchestrator with its team against the 35B answering alone, question by question.

Joins stage 2 (`<stage2>/<modes>/<team>/fold*/s*/holdout_predictions.jsonl`, one or more roots;
on a run present in several, the first root wins) with the L2 bare-35B runs
(one or more `<l2>/stage1/{hard,agentic}/qwen35b-off/fold*/s*/`, regraded predictions where they exist) on
(fold, sample, question). Per team: accuracy of the orchestrator, the 35B alone, the plain vote and
the best worker (the team member with the most right on these questions, picked after the fact);
the paired orchestrator − 35B difference with a 95% interval over distinct questions; and
completion tokens per question.

    python3 scripts/x1_l2_compare.py --stage2 ROOT1/stage2 ROOT2/stage2 --l2 data-claude/x1-l2 data-claude/x1-l2-t8 \
        --modes headline --out l2_compare.json
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict


def load_l2(roots):
    rows = {}
    paths = [p for root in roots for p in glob.glob(f"{root}/stage1/*/qwen35b-off/fold*/s*/holdout_predictions.jsonl")]
    for path in paths:
        regraded = path.replace(".jsonl", ".regraded.jsonl")
        for line in open(regraded if os.path.exists(regraded) else path):
            r = json.loads(line)
            fold, sample = path.split("/")[-3][4:], path.split("/")[-2][1:]
            rows[(int(fold), int(sample), r["question_key"])] = r
    return rows


def load_stage2(roots, modes):
    seen, rows = set(), []
    for root in roots:
        for path in sorted(glob.glob(f"{root}/{modes}/T*/fold*/s*/holdout_predictions.jsonl")):
            run = tuple(path.split("/")[-4:-1])
            if run in seen:
                continue
            seen.add(run)
            rows += [json.loads(line) for line in open(path)]
    return rows


def interval(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2", nargs="+", required=True)
    p.add_argument("--l2", nargs="+", required=True)
    p.add_argument("--modes", default="headline")
    p.add_argument("--out")
    args = p.parse_args()

    alone = load_l2(args.l2)
    teams = defaultdict(list)
    missing = defaultdict(int)
    for r in load_stage2(args.stage2, args.modes):
        key = (int(r["fold"]), int(r["sample"]), r["question_key"])
        if key not in alone:
            missing[r["team_id"]] += 1
            continue
        teams[r["team_id"]].append((r, alone[key]))

    out = {}
    print(f"modes {args.modes}; accuracy over (question, sample) pairs; Δ = orchestrator − 35B alone, "
          "95% interval over distinct questions; tokens = completion tokens per question")
    print(f"{'team':5} {'pairs':>5} {'qs':>4} {'orch':>6} {'35B':>6} {'vote':>6} {'best wkr':>22} "
          f"{'Δ (95% CI)':>22} {'orch tok':>9} {'hub tok':>8} {'35B tok':>8}")
    for team in sorted(teams, key=lambda t: int(t[1:])):
        pairs = teams[team]
        n = len(pairs)
        orch = sum(bool(s["team_correct"]) for s, _ in pairs)
        bare = sum(bool(a["team_correct"]) for _, a in pairs)
        vote = sum(bool(s.get("vote_correct")) for s, _ in pairs)
        workers = defaultdict(int)
        for s, _ in pairs:
            for st in s["stages"]:
                if st.get("role") != "hub":
                    workers[st["id"]] += bool(st.get("correct"))
        best, best_n = max(workers.items(), key=lambda kv: kv[1])
        per_q = defaultdict(list)
        for s, a in pairs:
            per_q[s["question_key"]].append(bool(s["team_correct"]) - bool(a["team_correct"]))
        mean, lo, hi = interval([sum(v) / len(v) for v in per_q.values()])
        tok_orch = sum(s.get("completion_tokens") or 0 for s, _ in pairs) / n
        tok_hub = sum(st.get("completion_tokens") or 0 for s, _ in pairs
                      for st in s["stages"] if st.get("role") == "hub") / n
        tok_bare = sum(a.get("completion_tokens") or 0 for _, a in pairs) / n
        by_ds = defaultdict(lambda: [0, 0, 0])
        for s, a in pairs:
            d = by_ds[s["dataset"]]
            d[0] += 1
            d[1] += bool(s["team_correct"])
            d[2] += bool(a["team_correct"])
        out[team] = {"pairs": n, "questions": len(per_q), "orchestrator": orch, "alone_35b": bare,
                     "vote": vote, "best_worker": best, "best_worker_correct": best_n,
                     "diff": mean, "diff_lo": lo, "diff_hi": hi, "tokens_orchestrator": tok_orch,
                     "tokens_hub": tok_hub, "tokens_35b": tok_bare, "missing": missing.get(team, 0),
                     "by_dataset": {k: {"n": v[0], "orchestrator": v[1], "alone_35b": v[2]}
                                    for k, v in by_ds.items()}}
        print(f"{team:5} {n:5} {len(per_q):4} {orch / n:6.1%} {bare / n:6.1%} {vote / n:6.1%} "
              f"{best[:14]:>14} {best_n / n:6.1%} {mean:+7.1%} [{lo:+6.1%}, {hi:+6.1%}] "
              f"{tok_orch:9.0f} {tok_hub:8.0f} {tok_bare:8.0f}")
    if missing:
        print("stage-2 rows with no L2 match:", dict(missing))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"modes": args.modes, "teams": out}, f, indent=1)


if __name__ == "__main__":
    main()
