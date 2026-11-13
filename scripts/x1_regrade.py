"""Grade stage-1 episodes whose judge was deferred (`--defer_judge`), as x1_hub.py does.

Stage 1 serves one worker at a time, so BrowseComp-Plus (and Finance-Agent) episodes record what
they would have asked the 35B judge and count as wrong. This asks the judge now, with the same
`judge.regrade` the orchestrator runs use, and writes `holdout_predictions.regraded.jsonl` next to
each run's predictions (the originals are left alone), plus `regraded.json` under ROOT with the
per-run, per-dataset counts. Already-regraded runs are skipped unless --force.

    ./env/bin/python scripts/x1_regrade.py --root data-claude/x1-l2 --api http://127.0.0.1:8001/v1
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from judge import Judge, regrade  # noqa: E402
from model.openai_compat import OpenAICompatChatWrapper  # noqa: E402

JUDGE_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"


def regrade_row(row, judge):
    stage = row["stages"][0]
    if stage.get("judge_deferred") is None:
        return row
    ok, detail = regrade(row["dataset"], stage["judge_deferred"], judge)
    fingerprint = (stage.get("outcome") or {}).get("fingerprint", "no_answer")
    stage["prediction"] = f"{'success' if ok else 'fail'}|{fingerprint}"
    stage["correct"] = ok
    stage["regraded"] = detail
    row["team_answer"], row["team_correct"] = stage["prediction"], ok
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--api", default="http://127.0.0.1:8001/v1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    judge = Judge(OpenAICompatChatWrapper(base_url=args.api, model_name=JUDGE_MODEL))

    summary = {}
    for path in sorted(glob.glob(f"{args.root}/stage1/*/*/fold*/s*/holdout_predictions.jsonl")):
        out = path.replace(".jsonl", ".regraded.jsonl")
        rows = [json.loads(line) for line in open(path)]
        if not any(r["stages"][0].get("judge_deferred") is not None for r in rows):
            continue
        if os.path.exists(out) and not args.force:
            rows = [json.loads(line) for line in open(out)]
        else:
            with ThreadPoolExecutor(args.workers) as pool:
                rows = list(pool.map(lambda r: regrade_row(r, judge), rows))
            with open(out + ".tmp", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(out + ".tmp", out)
        counts = defaultdict(lambda: [0, 0])
        for r in rows:
            counts[r["dataset"]][0] += bool(r["team_correct"])
            counts[r["dataset"]][1] += 1
        run = os.path.relpath(os.path.dirname(path), args.root)
        summary[run] = {k: {"correct": c, "n": n} for k, (c, n) in counts.items()}
        total_c = sum(c for c, _ in counts.values())
        total_n = sum(n for _, n in counts.values())
        print(f"{run}: {total_c}/{total_n} = {total_c / total_n:.1%}  "
              + "  ".join(f"{k} {c}/{n}" for k, (c, n) in sorted(counts.items())), flush=True)
    with open(os.path.join(args.root, "regraded.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
