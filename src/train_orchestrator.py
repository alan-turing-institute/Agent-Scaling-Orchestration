"""Orchestrator loop: select a team per tag, score it, record what happened.

Each iteration samples a tag from the tagged dataset, asks the orchestrator to
pick a team from the persona bank, evaluates that team on the sampled questions,
and folds the counts into a scoreboard the next iteration reads back.
"""

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_from_disk

from model.model_utils import build_agent_pool
from orchestration.orchestrator import OrchestratorAgent, team_selection
from team_evaluation import run_team_evaluation
from summariser import save_evaluation_summary, save_evaluation_summary_with_llm

# Every persona in the bank is a candidate, so the orchestrator can staff a team
# across tasks rather than within one. Built from the bank itself, so a persona
# added there is selectable without editing this file.
AGENT_POOL = build_agent_pool()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the orchestrator team-selection workflow")
    parser.add_argument("--api_base_url", default="http://localhost:8001/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", default="none", help="API key for the OpenAI-compatible endpoint")
    parser.add_argument("--model_name", default="Qwen/Qwen3.6-35B-A3B", help="Model name to request from the API")
    parser.add_argument("--dataset_path", default="data-claude/tagged_dataset", help="Path to the Hugging Face dataset on disk")
    parser.add_argument("--iterations", type=int, default=10, help="Number of select-evaluate-summarise cycles")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of questions to sample for team selection")
    parser.add_argument("--team_size", type=int, default=4, help="Number of agents the orchestrator must select")
    parser.add_argument("--seed", type=int, default=None, help="Seed for tag and question sampling")
    parser.add_argument("--solver", choices=["vote", "debate"], default="vote", help="How to aggregate the selected team answers (only vote is implemented)")
    parser.add_argument("--out_dir", default="data-claude/orchestrator", help="Directory for the run's outputs")
    parser.add_argument("--output_path", default=None, help="Run record JSONL (default: {out_dir}/run_records.jsonl)")
    parser.add_argument("--md_file", default=None, help="Scoreboard markdown (default: {out_dir}/agent_performance_by_tag.md)")
    parser.add_argument("--state_file", default=None, help="Scoreboard counts JSON (default: {out_dir}/agent_performance_state.json)")
    parser.add_argument("--selection_csv", default=None, help="Team selection log (default: {out_dir}/team_selection_results.csv)")
    parser.add_argument(
        "--summariser",
        choices=["counts", "llm"],
        default="counts",
        help="counts: scoreboard rendered from recorded counts. llm: the model rewrites the markdown each iteration",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    args.output_path = args.output_path or str(out_dir / "run_records.jsonl")
    args.md_file = args.md_file or str(out_dir / "agent_performance_by_tag.md")
    args.state_file = args.state_file or str(out_dir / "agent_performance_state.json")
    args.selection_csv = args.selection_csv or str(out_dir / "team_selection_results.csv")
    return args


def resolve_dataset_path(dataset_path):
    path = Path(dataset_path)
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root = Path(__file__).resolve().parent.parent
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate

    return path


SELECTION_HEADERS = [
    "iteration",
    "chosen_tag",
    "batch_size",
    "tag_profile",
    "selected_team",
    "invalid_agents",
    "reasoning",
    "reasoning_trace",
]


def write_team_selection(filename, result):
    """Append one selection to the CSV, writing the header only for a new file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SELECTION_HEADERS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "iteration": result.get("iteration"),
            "chosen_tag": result.get("chosen_tag"),
            "batch_size": result.get("batch_size"),
            "tag_profile": result.get("tag_profile"),
            "selected_team": result.get("selected_team"),
            "invalid_agents": result.get("invalid_agents"),
            "reasoning": result.get("reasoning"),
            "reasoning_trace": result.get("reasoning trace"),
        })


def write_run_record(path, record):
    """Append one machine-readable record per iteration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.solver == "debate":
        print("[warn] --solver debate is not implemented in team_evaluation; scoring by vote")

    print("📂 Loading dataset...")
    dataset_path = resolve_dataset_path(args.dataset_path)
    dataset = load_from_disk(str(dataset_path))
    print(f"✓ Dataset loaded: {len(dataset)} questions")
    print(f"✓ Agent pool: {len(AGENT_POOL)} candidate personas")

    orchestrator = OrchestratorAgent(
        args.model_name,
        AGENT_POOL,
        api_key=args.api_key,
        base_url=args.api_base_url,
        debug=args.debug,
    )

    pool_names = [agent["name"] for agent in AGENT_POOL]
    evaluations = []

    for iteration in range(1, args.iterations + 1):
        print("\n" + "#" * 60)
        print(f"ITERATION {iteration}/{args.iterations}")
        print("#" * 60)

        # The scoreboard is the loop's memory: read it back before every choice.
        md_path = Path(args.md_file)
        prior_md = md_path.read_text(encoding="utf-8") if md_path.exists() else None

        result = team_selection(
            orchestrator,
            dataset,
            num_samples=args.num_samples,
            prior_md=prior_md,
            team_size=args.team_size,
        )
        if not result:
            continue

        result["iteration"] = iteration

        print("\n" + "=" * 60)
        print("ORCHESTRATION SUMMARY")
        print("=" * 60)
        print(f"Chosen tag: {result['chosen_tag']}")
        print(f"Batch size: {result['batch_size']}")
        print(f"Tag profile: {result['tag_profile']}")
        print(f"Selected team: {result['selected_team']}")
        if result.get("invalid_agents"):
            print(f"Rejected names (not in pool): {result['invalid_agents']}")
        print(f"Reasoning: {result['reasoning']}")
        print("=" * 60)

        write_team_selection(args.selection_csv, result)

        selected_team = result["selected_team"]
        if not selected_team:
            print("[warn] no valid agents selected; skipping evaluation for this iteration")
            write_run_record(args.output_path, {
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chosen_tag": result.get("chosen_tag"),
                "tag_profile": result.get("tag_profile"),
                "selected_team": [],
                "invalid_agents": result.get("invalid_agents"),
                "status": "no_valid_team",
            })
            continue

        if len(selected_team) != args.team_size:
            print(f"[warn] team has {len(selected_team)} agents, expected {args.team_size}; evaluating anyway")

        sampled_questions = result["sampled_questions"]
        report = run_team_evaluation(selected_team, sampled_questions, args)

        print("\n" + "=" * 60)
        print("TEAM EVALUATION")
        print("=" * 60)
        print(f"Team accuracy ({args.solver}): {report['team_accuracy']:.2%}")
        for agent_name, accuracy in report["per_agent_accuracy"].items():
            print(f"{agent_name}: {accuracy:.2%}")

        by_tag = report.get("per_agent_accuracy_by_tag", {})
        if by_tag:
            print("\nPer-agent accuracies by tag:")
            for tag, accs in by_tag.items():
                parts = ", ".join(f"{name}: {acc:.2%}" for name, acc in accs.items())
                print(f" - {tag}: {parts}")
        print("=" * 60)

        evaluation = {
            "iteration": iteration,
            "chosen_tag": result.get("chosen_tag"),
            "batch_size": result.get("batch_size"),
            "tag_profile": result.get("tag_profile"),
            "selected_team": selected_team,
            "report": report,
        }
        evaluations.append(evaluation)

        write_run_record(args.output_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model_name,
            "num_samples": args.num_samples,
            "team_size": args.team_size,
            "invalid_agents": result.get("invalid_agents"),
            "reasoning": result.get("reasoning"),
            "status": "evaluated",
            **evaluation,
        })

        # Rewrite the scoreboard so the next iteration selects with this result in view.
        if args.summariser == "llm":
            save_evaluation_summary_with_llm(orchestrator, evaluations, out_md_path=args.md_file)
        else:
            save_evaluation_summary(
                evaluation,
                out_md_path=args.md_file,
                state_path=args.state_file,
                pool_names=pool_names,
            )

    print(f"\nRun records: {args.output_path}")
    print(f"Scoreboard:  {args.md_file}")


if __name__ == "__main__":
    main()
