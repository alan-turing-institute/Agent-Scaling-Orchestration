"""Compare arms question by question, from their predictions files.

`report_crossval.py` pairs arms by held-out batch, from the records files. That
throws away which questions moved, and it cannot use repeated runs. This joins
arms on the question itself, across every fold and every repeat given, and
tests each pair of arms on the per-question differences.

    python scripts/compare_runs.py \\
        --arm "vote=data-claude/e10/vote/fold*/holdout_predictions.jsonl" \\
        --arm "pipeline=data-claude/e10/pipeline/fold*/holdout_predictions.jsonl" \\
        --arm "pipeline=data-claude/e10/pipeline-repeat/fold*/holdout_predictions.jsonl"

Naming an arm more than once adds repeats of it. A question answered by several
repeats of an arm counts as the fraction of them that got it right, and the
arm's own repeat-to-repeat flip rate is printed: that is the noise floor any
difference between arms has to clear.

Tests: with one run per arm, an exact McNemar test on the discordant questions;
with repeats, a two-sided sign-flip permutation test on per-question
differences. p-values are Holm-corrected across every pair of arms compared.
"""

import argparse
import glob
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runner import question_key  # noqa: E402


def load_run(pattern):
    """{question_key: team_correct} for one run, which may span several fold files."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern!r}")
    run = {}
    for path in files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("question_key") or question_key(row.get("question"))
            run[key] = bool(row.get("team_correct"))
    return run


def mcnemar_exact(b, c):
    """Two-sided exact test on b (first arm right only) vs c (second arm right only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def sign_flip(diffs, n=20000, seed=0):
    rng = random.Random(seed)
    observed = abs(sum(diffs))
    if observed == 0:
        return 1.0
    hits = sum(1 for _ in range(n) if abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12)
    return hits / n


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True,
                        help="name=glob of predictions files making one run of that arm; repeat a name to add repeats")
    parser.add_argument("--baseline", default=None, help="Only compare every arm against this one")
    args = parser.parse_args()

    runs = defaultdict(list)
    for spec in args.arm:
        name, _, pattern = spec.partition("=")
        if not pattern:
            raise SystemExit(f"--arm wants name=glob, got {spec!r}")
        runs[name].append(load_run(pattern))

    # Only questions every run of every arm answered, so arms are compared on the same set.
    common = set.intersection(*(set(run) for arm_runs in runs.values() for run in arm_runs))
    if not common:
        raise SystemExit("the arms share no questions")
    keys = sorted(common)
    score = {name: {k: sum(run[k] for run in arm_runs) / len(arm_runs) for k in keys}
             for name, arm_runs in runs.items()}

    print(f"questions compared: {len(keys)}\n")
    print(f"{'arm':24s} {'runs':>4s} {'accuracy':>9s}  repeat-to-repeat flips")
    for name, arm_runs in runs.items():
        accuracy = sum(score[name].values()) / len(keys)
        flips = ""
        if len(arm_runs) > 1:
            pairs = list(itertools.combinations(arm_runs, 2))
            rate = sum(sum(a[k] != b[k] for k in keys) for a, b in pairs) / (len(pairs) * len(keys))
            flips = f"{rate:.1%} of questions"
        print(f"{name:24s} {len(arm_runs):4d} {accuracy:9.1%}  {flips}")

    names = list(runs)
    pairs = [(args.baseline, n) for n in names if n != args.baseline] if args.baseline else \
        list(itertools.combinations(names, 2))
    rows = []
    for a, b in pairs:
        diffs = [score[b][k] - score[a][k] for k in keys]
        if len(runs[a]) == 1 and len(runs[b]) == 1:
            only_b = sum(d > 0 for d in diffs)
            only_a = sum(d < 0 for d in diffs)
            p, test = mcnemar_exact(only_a, only_b), f"McNemar, {only_b} vs {only_a} discordant"
        else:
            p, test = sign_flip(diffs), "sign-flip over questions"
        rows.append((a, b, 100 * sum(diffs) / len(keys), p, test))

    adjusted = holm([r[3] for r in rows])
    print(f"\n{'comparison':40s} {'diff':>8s} {'p':>7s} {'p (Holm)':>9s}  test")
    for (a, b, diff, p, test), p_holm in zip(rows, adjusted):
        print(f"{b + ' - ' + a:40s} {diff:+7.1f}pt {p:7.3f} {p_holm:9.3f}  {test}")


if __name__ == "__main__":
    main()
