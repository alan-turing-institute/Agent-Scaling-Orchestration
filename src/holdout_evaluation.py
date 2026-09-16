"""Score a learned scoreboard on held-out questions.

The training loop samples tags and updates its scoreboard as it goes, so its own
accuracy numbers are measured on the questions it learned from. This module runs
the finished scoreboard over a test split the loop never saw: every test question
is evaluated exactly once, in batches, with the scoreboard frozen.

The split, the shuffle and the batching are all driven by `--split_seed`, so two
training runs that differ only in when they updated their scoreboard are compared
on exactly the same questions in the same order.
"""

import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from team_evaluation import run_team_evaluation


def _batch_indices(n_rows, batch_size, seed):
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, n_rows, batch_size)]


def _tag_frequencies(batch):
    counts = Counter()
    for tags in batch["tags"]:
        counts.update(tags or [])
    return counts


def _accumulate(totals, team, report):
    """Fold one batch's counts into the running totals."""
    n_samples = report.get("n_samples", 0) or 0
    totals["questions"] += n_samples
    totals["team_correct"] += report.get("team_correct", 0) or 0
    totals["batches"] += 1

    for name in team:
        agent = totals["agents"].setdefault(name, {"selected": 0, "questions": 0, "correct": 0})
        agent["selected"] += 1
        agent["questions"] += n_samples
        agent["correct"] += report.get("per_agent_correct", {}).get(name, 0)

    for tag, count in (report.get("per_tag_counts") or {}).items():
        tag_state = totals["tags"].setdefault(tag, {"questions": 0, "agents": {}})
        tag_state["questions"] += count
        for name in team:
            agent = tag_state["agents"].setdefault(name, {"questions": 0, "correct": 0})
            agent["questions"] += count
            agent["correct"] += (report.get("per_agent_correct_by_tag") or {}).get(tag, {}).get(name, 0)


def _empty_totals():
    return {"questions": 0, "team_correct": 0, "batches": 0, "agents": {}, "tags": {}}


def _select_team(orchestrator, tag_frequencies, batch_size, scoreboard_md, team_size, max_attempts=2):
    """Ask the orchestrator for a team, retrying once on out-of-pool names."""
    tag_profile = list(tag_frequencies.keys())
    invalid_feedback = None
    team_result = {"agents": [], "invalid_agents": [], "reasoning": ""}

    for attempt in range(1, max_attempts + 1):
        team_result = orchestrator.select_team(
            tag_profile,
            tag_frequencies,
            batch_size=batch_size,
            prior_md=scoreboard_md,
            team_size=team_size,
            invalid_feedback=invalid_feedback,
        )
        if len(team_result.get("agents", [])) >= team_size:
            break
        if attempt == max_attempts:
            break
        invalid = team_result.get("invalid_agents") or []
        invalid_feedback = ", ".join(invalid) if invalid else None

    return team_result


def evaluate_holdout(orchestrator, test_dataset, args, pool_names, scoreboard_md=None,
                     random_baseline=False):
    """Run every test question once and report how the selected teams did.

    `scoreboard_md` is read once and passed to every selection, so the scoreboard
    is frozen for the whole evaluation - nothing learned here feeds back.
    """
    batch_size = args.test_batch_size or args.num_samples
    batches = _batch_indices(len(test_dataset), batch_size, args.split_seed)
    print(f"\nHeld-out evaluation: {len(test_dataset)} questions in {len(batches)} batches of up to {batch_size}")

    totals = _empty_totals()
    baseline_totals = _empty_totals() if random_baseline else None
    baseline_rng = random.Random(args.split_seed)

    records_path = Path(args.out_dir) / "holdout_records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)

    for batch_index, indices in enumerate(batches, start=1):
        batch = test_dataset.select(indices)
        tag_frequencies = _tag_frequencies(batch)

        print("\n" + "-" * 60)
        print(f"HELD-OUT BATCH {batch_index}/{len(batches)} ({len(indices)} questions)")

        try:
            team_result = _select_team(
                orchestrator,
                tag_frequencies,
                len(indices),
                scoreboard_md,
                args.team_size,
            )
        except Exception as error:
            # One failed call should cost this batch, not the whole evaluation.
            print(f"[warn] selection failed: {error!r}; skipping this batch")
            team_result = {"agents": [], "invalid_agents": [], "reasoning": f"selection failed: {error!r}"}
        team = team_result.get("agents", [])
        print(f"Selected team: {team}")

        record = {
            "batch": batch_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_indices": indices,
            "tag_frequencies": dict(tag_frequencies),
            "selected_team": team,
            "invalid_agents": team_result.get("invalid_agents", []),
            "reasoning": team_result.get("reasoning", ""),
        }

        if team:
            try:
                report = run_team_evaluation(team, batch, args)
            except Exception as error:
                print(f"[warn] evaluation failed: {error!r}; skipping this batch")
                report = None
                record["status"] = "evaluation_error"
                record["error"] = repr(error)
            if report is not None:
                _accumulate(totals, team, report)
                record["report"] = report
                print(f"Team accuracy: {report['team_accuracy']:.2%}")
        else:
            record["status"] = "no_valid_team"
            print("[warn] no valid team selected for this batch")

        if random_baseline:
            baseline_team = baseline_rng.sample(pool_names, args.team_size)
            baseline_report = run_team_evaluation(baseline_team, batch, args)
            _accumulate(baseline_totals, baseline_team, baseline_report)
            record["baseline_team"] = baseline_team
            record["baseline_report"] = baseline_report
            print(f"Random baseline accuracy: {baseline_report['team_accuracy']:.2%}")

        with records_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    summary = {
        "run": str(args.out_dir),
        "summary_every": args.summary_every,
        "iterations_trained": args.iterations,
        "test_questions": totals["questions"],
        "batches": totals["batches"],
        "team_accuracy": (totals["team_correct"] / totals["questions"]) if totals["questions"] else 0.0,
        "agents": totals["agents"],
        "tags": totals["tags"],
    }
    if random_baseline:
        summary["baseline_team_accuracy"] = (
            (baseline_totals["team_correct"] / baseline_totals["questions"])
            if baseline_totals["questions"]
            else 0.0
        )
        summary["baseline_agents"] = baseline_totals["agents"]

    summary_path = Path(args.out_dir) / "holdout_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 60)
    print("HELD-OUT RESULT")
    print("=" * 60)
    print(f"Questions: {summary['test_questions']} in {summary['batches']} batches")
    print(f"Team accuracy: {summary['team_accuracy']:.2%}")
    if random_baseline:
        print(f"Random-team baseline: {summary['baseline_team_accuracy']:.2%}")
    print(f"Written to {summary_path} and {records_path}")

    return summary
