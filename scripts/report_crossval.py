"""Aggregate a cross-validation sweep: every arm, pooled over the folds it finished.

Folds are disjoint, so pooling them is a single estimate over every question the
sweep has reached rather than an average of averages. Arms are only comparable on
the folds they have both finished, so the table reports which folds each arm covers
and the pooled figure is computed over those; an arm still running shows fewer.

The spread across folds is the number the first round was missing. A difference
between arms means something only if it is larger than the difference the same arm
shows between one fold and the next.
"""

import argparse
import json
import math
from pathlib import Path

ARM_ORDER = ["bare_model", "paper_personas", "paper_personas_4", "random",
             "no_memory", "batched", "continual"]


def wilson(correct, total, z=1.96):
    """Binomial interval that stays sane at small n and near the ceiling."""
    if not total:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return 100 * (centre - half), 100 * (centre + half)


def read_fold(path):
    """Return (correct, questions) for one finished arm, or None if it has not finished."""
    summary_path = path / "holdout_summary.json"
    if not summary_path.exists():
        return None
    records = path / "holdout_records.jsonl"
    if not records.exists():
        return None
    correct = questions = 0
    zeroed = 0
    with records.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            report = record.get("report")
            if not report:
                continue
            correct += report["team_correct"]
            questions += report["n_samples"]
            if all(v == 0 for v in report["per_agent_correct"].values()):
                zeroed += 1
    return correct, questions, zeroed


def main():
    parser = argparse.ArgumentParser(description="Summarise a cross-validation sweep")
    parser.add_argument("root", nargs="?", default="data-claude/crossval")
    args = parser.parse_args()

    root = Path(args.root)
    fold_dirs = sorted(p for p in root.glob("fold*") if p.is_dir())
    if not fold_dirs:
        print(f"no folds under {root}")
        return

    results = {}
    for arm in ARM_ORDER:
        per_fold = {}
        for fold_dir in fold_dirs:
            got = read_fold(fold_dir / arm)
            if got:
                per_fold[fold_dir.name.replace("fold", "")] = got
        if per_fold:
            results[arm] = per_fold

    if not results:
        print(f"no finished arms under {root}")
        return

    print(f"# {root}\n")
    header = f"{'arm':18s} {'pooled':>14s}  {'95% CI':>14s}  {'per fold':>28s}  folds"
    print(header)
    print("-" * len(header))
    for arm in ARM_ORDER:
        if arm not in results:
            continue
        per_fold = results[arm]
        correct = sum(v[0] for v in per_fold.values())
        questions = sum(v[1] for v in per_fold.values())
        low, high = wilson(correct, questions)
        each = " ".join(f"{100 * c / n:.1f}" for c, n, _ in
                        (per_fold[k] for k in sorted(per_fold)))
        folds = ",".join(sorted(per_fold))
        print(f"{arm:18s} {correct:4d}/{questions:<4d} {100 * correct / questions:5.1f}%  "
              f"[{low:4.1f},{high:5.1f}]  {each:>28s}  {folds}")

    # The comparison the sweep exists to make: is any gap between arms bigger than
    # the gap the same arm shows between folds?
    spreads = []
    for arm, per_fold in results.items():
        if len(per_fold) < 2:
            continue
        rates = [100 * c / n for c, n, _ in per_fold.values()]
        spreads.append((max(rates) - min(rates), arm))
    if spreads:
        print("\nFold-to-fold spread within one arm (the noise floor):")
        for spread, arm in sorted(spreads, reverse=True):
            print(f"  {arm:18s} {spread:4.1f} points")
        print(f"\n  Largest within-arm spread: {max(spreads)[0]:.1f} points. "
              "Treat any gap between arms smaller than this as unresolved.")

    zeroed = [(arm, fold) for arm, per_fold in results.items()
              for fold, (_, _, z) in per_fold.items() if z]
    if zeroed:
        print("\nBatches with every agent at zero (check for a wedged server, not a hard batch):")
        for arm, fold in zeroed:
            print(f"  {arm} fold {fold}")


if __name__ == "__main__":
    main()
