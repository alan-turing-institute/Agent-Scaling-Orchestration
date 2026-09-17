"""Summarise one orchestrator run: team accuracy, per-agent accuracy, coverage.

Reads a run directory written by `train_orchestrator.py` and prints the numbers
the experiment log wants. Held-out figures come from `holdout_summary.json`;
training figures come from the scoreboard state, so the two can be compared.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def agent_table(agents, min_questions=1):
    rows = []
    for name, counts in agents.items():
        questions = counts.get("questions", 0)
        if questions < min_questions:
            continue
        correct = counts.get("correct", 0)
        rows.append((correct / questions, correct, questions, counts.get("selected", 0), name))
    return sorted(rows, reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Summarise one orchestrator run")
    parser.add_argument("run_dir")
    parser.add_argument("--top", type=int, default=12, help="Agents to list at each end")
    parser.add_argument("--min_questions", type=int, default=5, help="Ignore agents seen fewer times than this")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = load(run_dir / "holdout_summary.json")
    state = load(run_dir / "agent_performance_state.json")

    print(f"# {run_dir}")

    if summary:
        print(f"\nHeld out: {summary['test_questions']} questions in {summary['batches']} batches")
        print(f"Team accuracy: {summary['team_accuracy']:.1%}")
        if "baseline_team_accuracy" in summary:
            print(f"Random-team baseline: {summary['baseline_team_accuracy']:.1%}")

        rows = agent_table(summary.get("agents", {}), args.min_questions)
        print(f"\nAgents used on the held-out split: {len(summary.get('agents', {}))}")
        print(f"{'accuracy':>9}  {'correct/seen':>13}  {'picks':>5}  agent")
        for accuracy, correct, questions, selected, name in rows[: args.top]:
            print(f"{accuracy:>8.1%}  {correct:>6}/{questions:<6}  {selected:>5}  {name}")
        if len(rows) > 2 * args.top:
            print("   ...")
        for accuracy, correct, questions, selected, name in rows[-args.top:][max(0, 2 * args.top - len(rows)):]:
            print(f"{accuracy:>8.1%}  {correct:>6}/{questions:<6}  {selected:>5}  {name}")

        if rows:
            best, worst = rows[0], rows[-1]
            spread = best[0] - worst[0]
            print(f"\nSpread across agents seen at least {args.min_questions} times: "
                  f"{spread:.1%} ({worst[4]} {worst[0]:.1%} to {best[4]} {best[0]:.1%})")
            print(f"Team accuracy minus best single agent: {summary['team_accuracy'] - best[0]:+.1%}")
            mean = sum(r[1] for r in rows) / sum(r[2] for r in rows)
            print(f"Team accuracy minus mean selected agent: {summary['team_accuracy'] - mean:+.1%}")

    records_path = run_dir / "holdout_records.jsonl"
    if records_path.exists():
        # Agents are only comparable within a batch, because each batch is a
        # different set of questions. Compare the team against the agents that
        # answered the same five questions, then average over batches.
        team, best, mean, worst, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        for line in records_path.open(encoding="utf-8"):
            record = json.loads(line)
            report = record.get("report")
            if not report or not report.get("n_samples"):
                continue
            size = report["n_samples"]
            per_agent = [c / size for c in report.get("per_agent_correct", {}).values()]
            if not per_agent:
                continue
            n_batches += 1
            team += report.get("team_correct", 0) / size
            best += max(per_agent)
            worst += min(per_agent)
            mean += sum(per_agent) / len(per_agent)
        if n_batches:
            print(f"\nMatched within batch, averaged over {n_batches} batches:")
            print(f"  team vote            {team / n_batches:.1%}")
            print(f"  best agent in batch  {best / n_batches:.1%}")
            print(f"  mean agent in batch  {mean / n_batches:.1%}")
            print(f"  worst agent in batch {worst / n_batches:.1%}")
            print(f"  vote over mean agent {(team - mean) / n_batches:+.1%}")
            print(f"  vote over best agent {(team - best) / n_batches:+.1%}")

    if state:
        agents = state.get("agents", {})
        tried = [name for name, counts in agents.items() if counts.get("selected", 0)]
        picks = Counter({name: counts.get("selected", 0) for name, counts in agents.items()})
        print(f"\nTraining: {len(tried)} distinct agents selected")
        print("Most-picked during training: " + ", ".join(f"{n} x{c}" for n, c in picks.most_common(8) if c))


if __name__ == "__main__":
    main()
