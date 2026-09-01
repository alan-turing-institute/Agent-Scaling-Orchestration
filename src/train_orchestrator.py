import argparse
from pathlib import Path
import csv
import json
import os
import tempfile

from datasets import load_from_disk

from defaults import THINKING_TOKEN_BUDGET

from orchestration.orchestrator import OrchestratorAgent, team_selection
from team_evaluation import run_team_evaluation
from summariser import save_evaluation_summary_with_llm

AGENT_POOL = [
    {
        "name": "Conservative_Verifier",
        "specialty": "Careful, methodical reasoning with step-by-step validation and error checking",
        "strengths": ["verification", "accuracy", "step-by-step reasoning", "error detection"],
    },
    {
        "name": "Creative_Explorer",
        "specialty": "Innovative problem solving with pattern-seeking and alternative reasoning paths",
        "strengths": ["pattern recognition", "creative reasoning", "alternative strategies", "insight"],
    },
    {
        "name": "Rigorous_Formalist",
        "specialty": "Precise mathematical formalism, clear definitions, and logically complete derivations",
        "strengths": ["formal reasoning", "logical rigor", "precise notation", "assumption checking"],
    },
    {
        "name": "Intuitive_Estimator",
        "specialty": "Intuitive estimation and reasonableness checks for numerical solutions",
        "strengths": ["estimation", "sanity checking", "intuition", "plausibility assessment"],
    },
    {
        "name": "Systematic_Decomposer",
        "specialty": "Breaking complex problems into manageable subproblems and structured solving steps",
        "strengths": ["decomposition", "planning", "modular reasoning", "solution structure"],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the orchestrator team-selection workflow")
    parser.add_argument("--api_base_url", default="http://localhost:8001/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", default="none", help="API key for the OpenAI-compatible endpoint")
    parser.add_argument("--model_name", default="Qwen/Qwen3.6-35B-A3B", help="Model name to request from the API")
    parser.add_argument("--dataset_path", default="data/tagged_dataset", help="Path to the Hugging Face dataset on disk")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of questions to sample for team selection")
    parser.add_argument("--solver", choices=["vote", "debate"], default="vote", help="How to aggregate the selected team answers")
    parser.add_argument("--output_path", default="out/orchestrator_results.json", help="Where to save the evaluation report")
    parser.add_argument("--thinking_token_budget", type=int, default=THINKING_TOKEN_BUDGET,
                        help="Reasoning budget for API models that accept it")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")
    parser.add_argument("--md_file", default="out/agent_performance_by_tag.md", help="Path to the markdown summary file")
    return parser.parse_args()


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


def write_team_selection(filename, result):
    headers = [
        "iteration",
        "chosen_tag",
        "batch_size",
        "tag_profile",
        "selected_team",
        "reasoning",
        "reasoning_trace"
    ]

    with open(filename, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerow({
            "iteration": result.get("iteration"),
            "chosen_tag": result.get("chosen_tag"),
            "batch_size": result.get("batch_size"),
            "tag_profile": result.get("tag_profile"),
            "selected_team": result.get("selected_team"),
            "reasoning": result.get("reasoning"),
            "reasoning_trace": result.get("reasoning trace")
        })


if __name__ == "__main__":
    args = parse_args()

    print("📂 Loading dataset...")
    dataset_path = resolve_dataset_path(args.dataset_path)
    dataset = load_from_disk(str(dataset_path))
    print(f"✓ Dataset loaded: {len(dataset)} questions")

    orchestrator = OrchestratorAgent(
        args.model_name,
        AGENT_POOL,
        api_key=args.api_key,
        base_url=args.api_base_url,
        thinking_token_budget=args.thinking_token_budget,
        debug=args.debug,
    )

    evaluations = []

    for i in range(10):
        # Read existing performance markdown (if any) so the orchestrator can use it
        md_path = Path(args.md_file)
        prior_md = md_path.read_text(encoding="utf-8") if md_path.exists() else None

        result = team_selection(orchestrator, dataset, num_samples=args.num_samples, prior_md=prior_md)

        if result:
            print("\n" + "=" * 60)
            print("ORCHESTRATION SUMMARY")
            print("=" * 60)
            print(f"Chosen tag: {result['chosen_tag']}")
            print(f"Batch size: {result['batch_size']}")
            print(f"Tag profile: {result['tag_profile']}")
            print(f"Selected team: {result['selected_team']}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Full reasoning trace: {result['reasoning trace']}")
            print("=" * 60)

            write_team_selection("out/team_selection_results.csv", result)

            selected_team = result["selected_team"]
            sampled_questions = result["sampled_questions"]
            report = run_team_evaluation(selected_team, sampled_questions, args)

            print("\n" + "=" * 60)
            print("TEAM EVALUATION")
            print("=" * 60)
            print(f"Team accuracy ({args.solver}): {report['team_accuracy']:.2%}")
            for agent_name, accuracy in report["per_agent_accuracy"].items():
                print(f"{agent_name}: {accuracy:.2%}")

            # Print per-tag per-agent accuracies
            by_tag = report.get("per_agent_accuracy_by_tag", {})
            if by_tag:
                print("\nPer-agent accuracies by tag:")
                for tag, accs in by_tag.items():
                    parts = ", ".join(f"{name}: {acc:.2%}" for name, acc in accs.items())
                    print(f" - {tag}: {parts}")
            print("=" * 60)

            # Collect evaluation metadata to inform the LLM summary
            evaluations.append({
                "chosen_tag": result.get("chosen_tag"),
                "batch_size": result.get("batch_size"),
                "tag_profile": result.get("tag_frequencies"),
                "selected_team": selected_team,
                "report": report,
            })

            # After each epoch/evaluation, update the markdown summary so the next
            # epoch can read and use it when selecting teams.
            save_evaluation_summary_with_llm(orchestrator, evaluations, out_md_path=args.md_file)