"""X1 stage 1 so far: each worker's accuracy on each team's questions, from the finished runs.

    python scripts/x1_stage1_summary.py [--root data-claude/x1-teams] > stage1.json

One row per team, worker and mode, summed over the folds and samples finished: questions,
correct, answers cut off at the token cap, completion tokens; `by_run` holds the same counts per
finished fold and sample ("f0s0"), so workers can be compared on the runs they share. BrowseComp-Plus and
Finance-Agent are judged in stage 2, so for them `correct` stays 0 and `answered` counts the
episodes that submitted an answer.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from bare_model_baseline import matches_sources
from x1_plan import JUDGED, POOLS, ROOT, load_config, workers

COUNTS = ("questions", "correct", "answered", "cut_off", "completion_tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()
    from datasets import load_from_disk
    config = load_config()
    pools = {pool: {r["id"]: r for r in load_from_disk(str(REPO / path))} for pool, path in POOLS.items()}
    rows = []
    for team in config["teams"]:
        names = set(team["personas"]) | set(team["specialists"])
        for w in workers(config):
            if w["worker"] not in names or w["pool"] != team["pool"]:
                continue
            row = {"team": team["id"], "worker": w["worker"], "mode": w["mode"], "persona": bool(w["persona"]),
                   "runs": 0, **{k: 0 for k in COUNTS}, "by_run": {}}
            base = REPO / args.root / "stage1" / w["pool"] / f"{w['worker']}-{w['mode']}"
            for summary in sorted(base.glob("fold*/s*/holdout_summary.json")):
                row["runs"] += 1
                run = {k: 0 for k in COUNTS}
                for line in (summary.parent / "holdout_predictions.jsonl").open():
                    p = json.loads(line)
                    if not matches_sources(pools[w["pool"]][p["question_id"]], team["sources"]):
                        continue
                    s = p["stages"][0]
                    run["questions"] += 1
                    run["correct"] += bool(s.get("correct"))
                    run["answered"] += not str(s.get("prediction") or "").endswith("no_answer")
                    run["cut_off"] += s.get("finish_reason") == "length"
                    run["completion_tokens"] += s.get("completion_tokens") or 0
                row["by_run"][f"f{summary.parent.parent.name[4:]}s{summary.parent.name[1:]}"] = run
                for k in COUNTS:
                    row[k] += run[k]
            row["judged"] = any(src.split(":")[0] in JUDGED for src in team["sources"].split(","))
            rows.append(row)
    json.dump(rows, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
