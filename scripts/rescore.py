"""Re-score a finished run from its predictions file, without calling a model.

Every stage's response is stored in full in `predictions.jsonl` /
`holdout_predictions.jsonl`, so changing the parser is a re-read, not a re-run.
This re-extracts each stage's answer under `--parse_mode`, recomputes each
stage's correctness and the team's answer (vote over the row's answer stages,
or the final stage), and reports what moved.

    python scripts/rescore.py data-claude/.../holdout_predictions.jsonl --parse_mode lenient

Reads both row shapes: schema 2 (`stages`, written by the runner) and the
earlier per-agent shape (`agents`, a vote over every agent). Vote ties are
broken with the per-question seeded generator, so a run made with
`--tie_break global` can differ on tied questions even under the same parser. Rows whose
responses were clipped with `--response_chars` cannot be re-scored faithfully;
they are counted and reported.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmarks  # noqa: E402
from runner import _stable_int, question_key  # noqa: E402
from team_evaluation import _infer_answer_type  # noqa: E402


def stages_of(row):
    """(stage_id, response) pairs, and the ids whose answers decide the team's."""
    if row.get("stages") is not None:
        stages = [(s["id"], s.get("response") or "") for s in row["stages"]]
        aggregation = row.get("aggregation") or {}
        over = aggregation.get("over") or [sid for sid, _ in stages]
        return stages, aggregation.get("rule") or "vote", over
    agents = row.get("agents") or {}
    stages = [(name, (a or {}).get("response") or "") for name, a in agents.items()]
    return stages, "vote", [sid for sid, _ in stages]


def rescore_row(row, parse_mode, tie_break_seed=0):
    answer_type = row.get("answer_type")
    if answer_type == "gsm8k":
        answer_type = "numeric"  # what pre-registry rows called it
    elif answer_type is None:
        answer_type = _infer_answer_type(row.get("gold"))
    scorer = benchmarks.get_scorer(answer_type, mode=parse_mode)
    gold = scorer.normalise_gold(row["gold"])
    stages, rule, over = stages_of(row)
    predictions = {sid: scorer.extract(text) for sid, text in stages}
    per_stage = {sid: bool(scorer.correct(p, gold)) for sid, p in predictions.items()}
    if rule == "stage":
        team = predictions[over[0]]
    else:
        key = row.get("question_key") or question_key(row.get("question"))
        rng = random.Random(_stable_int(tie_break_seed, key))
        team = scorer.aggregate([predictions[sid] for sid in over], rng=rng)
    return {
        "team_correct": bool(scorer.correct(team, gold)),
        "per_stage": per_stage,
        "parsed": {sid: p.parsed for sid, p in predictions.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", nargs="+", help="predictions JSONL file(s)")
    parser.add_argument("--parse_mode", choices=["strict", "lenient"], default="lenient")
    parser.add_argument("--show", type=int, default=5, help="Print this many questions whose team verdict changed")
    args = parser.parse_args()

    totals = Counter()
    stage_parsed = Counter()
    changed = []
    for path in args.predictions:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stages, _, _ = stages_of(row)
            if any("...[" in text and "more chars]" in text for _, text in stages):
                totals["clipped"] += 1
                continue
            new = rescore_row(row, args.parse_mode)
            old = bool(row.get("team_correct"))
            totals["questions"] += 1
            totals["old_correct"] += int(old)
            totals["new_correct"] += int(new["team_correct"])
            for sid, ok in new["parsed"].items():
                stage_parsed["stages"] += 1
                stage_parsed["parsed"] += int(ok)
            if old != new["team_correct"]:
                changed.append((path, row.get("question_index"), old, new["team_correct"], row.get("question", "")[:80]))

    n = totals["questions"]
    print(f"questions re-scored: {n}  (skipped as clipped: {totals['clipped']})")
    if n:
        print(f"team correct as recorded: {totals['old_correct']}/{n} = {totals['old_correct'] / n:.1%}")
        print(f"team correct, {args.parse_mode}: {totals['new_correct']}/{n} = {totals['new_correct'] / n:.1%}")
        print(f"stage answers parsed, {args.parse_mode}: {stage_parsed['parsed']}/{stage_parsed['stages']}")
        gained = sum(1 for c in changed if c[3])
        print(f"verdicts changed: {len(changed)} ({gained} now right, {len(changed) - gained} now wrong)")
        for path, index, old, new, text in changed[:args.show]:
            print(f"  q{index}: {'right' if old else 'wrong'} -> {'right' if new else 'wrong'}  {text!r}")


if __name__ == "__main__":
    main()
