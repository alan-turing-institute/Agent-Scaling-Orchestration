"""X1 stage 2 results split by source dataset, one record per finished run and dataset.

Each team draws its questions from one or more datasets (T1 Maths: MATH-500 and AIME; T2 Physics:
GPQA Diamond and MMLU-Pro; ...). This reads every finished stage 2 run (one with a
holdout_summary.json) and writes, per run and dataset, the same counts that x1_hub.summarise writes
per run, plus "dataset":

    python scripts/x1_stage2_by_dataset.py [--root data-claude/x1-teams] > stage2_datasets.json

Read by x1_stage1_tab.py --stage2-datasets.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRANSITIONS = ("kept_right", "fixed", "broke", "kept_wrong")


def records(root):
    out = []
    for summary in sorted(Path(root).glob("stage2/*/*/fold*/s*/holdout_summary.json")):
        run = json.loads(summary.read_text())
        groups = {}
        for line in (summary.parent / "holdout_predictions.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                groups.setdefault(row["dataset"], []).append(row)
        for dataset, rows in sorted(groups.items()):
            workers = [s["id"] for s in rows[0]["stages"][:-1]]
            out.append({
                "team": run["team"], "modes": run["modes"], "fold": run["fold"], "sample": run["sample"],
                "dataset": dataset, "questions": len(rows),
                "hub_correct": sum(r["team_correct"] for r in rows),
                "vote_correct": sum(r["vote_correct"] for r in rows),
                "workers_correct": {w: sum(bool(r["stages"][i]["correct"]) for r in rows) for i, w in enumerate(workers)},
                "hub_transitions": {t: sum(r["stages"][-1]["transition"] == t for r in rows) for t in TRANSITIONS},
            })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(REPO / "data-claude" / "x1-teams"))
    print(json.dumps(records(parser.parse_args().root)))


if __name__ == "__main__":
    main()
