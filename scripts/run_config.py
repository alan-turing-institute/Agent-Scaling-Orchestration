"""Run one fixed team configuration over the held-out split.

The orchestrator arms choose a team per batch; this runs the same team, in the
same topology, on every batch. It is how a vote, a debate, a hub, a pipeline or
a synthesis is scored on the split the orchestrator arms use, with the same
batching, so the results pair batch for batch.

Describe the team with a JSON file (`--config`, the `TeamConfig.to_dict()`
shape) or with flags:

    --topology vote        --personas A,B,C,D
    --topology debate      --personas A,B,C --rounds 1 [--handoff full|rationale|answer]
    --topology centralized --personas A,B,C --hub H
    --topology synthesis   --personas A,B,C --synthesiser S
    --topology pipeline    --steps solver:A,critic:B,reviser:C
    --topology parallel    --steps "solver:A,checker:B|solver:C,checker:D"

A step may name a model key after a second colon (`reviser:C:qwen35b`); keys
resolve through `--models_file`. `--budget N` splits N tokens per question
evenly over the stages, for comparisons at matched compute.

Writes `holdout_records.jsonl`, `holdout_summary.json`,
`holdout_predictions.jsonl` and `configs.json` to `--out_dir`, in the shapes
`scripts/report_run.py` and `scripts/report_crossval.py` read.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_from_disk  # noqa: E402

import team_config  # noqa: E402
from holdout_evaluation import _accumulate, _batch_indices, _empty_totals, with_server_retry  # noqa: E402
from model.model_utils import DEFAULT_MAX_NEW_TOKENS  # noqa: E402
from predictions import save_report  # noqa: E402
from runner import add_runner_args  # noqa: E402
from splits import add_split_args, make_split, split_label  # noqa: E402
from team_evaluation import run_team_evaluation  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset_path", default="data-claude/tagged_dataset")
    parser.add_argument("--model_name", required=True, help="Served name of the 'default' agent model")
    parser.add_argument("--api_base_url", default="http://127.0.0.1:8001/v1")
    add_split_args(parser)
    add_runner_args(parser)
    parser.add_argument("--test_batch_size", type=int, default=5)
    parser.add_argument("--eval_workers", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Token cap per stage unless the config or --budget sets one")

    team = parser.add_argument_group("the team")
    team.add_argument("--config", default=None, help="TeamConfig JSON file; overrides the flags below")
    team.add_argument("--topology", choices=["vote", "debate", "centralized", "synthesis", "pipeline", "parallel"])
    team.add_argument("--personas", default="", help="Comma-separated persona names")
    team.add_argument("--rounds", type=int, default=1, help="Debate exchange rounds after the first answer")
    team.add_argument("--hub", default=None)
    team.add_argument("--synthesiser", default=None)
    team.add_argument("--steps", default="", help="Comma-separated role:persona[:model] steps")
    team.add_argument("--branches", type=int, default=2,
                      help="parallel only: copies of --steps when it names one branch. Separate branches with "
                           "'|' to give each its own personas; identical branches at temperature 0 answer identically")
    team.add_argument("--handoff", choices=["full", "rationale", "answer"], default="full")
    team.add_argument("--model", default="default", help="Model key for stages that do not name one")
    team.add_argument("--budget", type=int, default=0, help="Tokens per question, split evenly over the stages")
    team.add_argument("--name", default="", help="Label stored with the config")

    parser.add_argument("--limit", type=int, default=0, help="Answer only the first N held-out questions; 0 means all")
    parser.add_argument("--response_chars", type=int, default=0,
                        help="Clip stored responses to this many characters; 0 keeps them whole")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _steps(spec):
    steps = []
    for item in [s for s in spec.split(",") if s.strip()]:
        parts = item.strip().split(":")
        if len(parts) < 2:
            raise SystemExit(f"--steps entry {item!r} should be role:persona[:model]")
        steps.append(tuple(parts[:3]))
    return steps


def build_config(args):
    if args.config:
        config = team_config.TeamConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    else:
        personas = [p.strip() for p in args.personas.split(",") if p.strip()]
        name = args.name or args.topology
        if args.topology == "vote":
            config = team_config.vote(personas, model=args.model, name=name)
        elif args.topology == "debate":
            config = team_config.debate(personas, rounds=args.rounds, model=args.model,
                                        handoff=args.handoff, name=name)
        elif args.topology == "centralized":
            config = team_config.centralized(personas, hub=args.hub, model=args.model,
                                             handoff=args.handoff, name=name)
        elif args.topology == "synthesis":
            config = team_config.synthesis(personas, synthesiser=args.synthesiser, model=args.model,
                                           handoff=args.handoff, name=name)
        elif args.topology == "pipeline":
            config = team_config.pipeline(_steps(args.steps), model=args.model, handoff=args.handoff, name=name)
        elif args.topology == "parallel":
            specs = [b for b in args.steps.split("|") if b.strip()]
            if len(specs) == 1:
                specs = specs * args.branches
            branches = [team_config.pipeline(_steps(b), model=args.model, handoff=args.handoff) for b in specs]
            config = team_config.parallel(*branches, name=name)
        else:
            raise SystemExit("give --config or --topology")
    if args.budget:
        config = team_config.split_budget(config, args.budget)
    return config


def main():
    args = parse_args()
    config = build_config(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.dataset_path)
    _, test_dataset = make_split(dataset, args)
    if args.limit:
        test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))
    print(f"✓ Held out: {len(test_dataset)} questions [{split_label(args)}]")
    print(f"✓ Config {config.config_id} ({config.name or 'unnamed'}): "
          f"{len(config.stages)} stages, {len(config.layers())} layers, aggregate {config.aggregate}")

    records_path = out_dir / "holdout_records.jsonl"
    predictions_path = out_dir / "holdout_predictions.jsonl"
    for path in (records_path, predictions_path, out_dir / "configs.json"):
        path.unlink(missing_ok=True)

    team = config.stage_ids
    totals = _empty_totals()
    tokens = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    batches = _batch_indices(len(test_dataset), args.test_batch_size, args.split_seed)
    for batch_index, indices in enumerate(batches, start=1):
        batch = test_dataset.select(indices)
        record = {
            "batch": batch_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_indices": indices,
            "selected_team": team,
            "invalid_agents": [],
            "reasoning": f"Fixed config {config.config_id}; no selection",
        }
        try:
            report = with_server_retry(
                lambda: run_team_evaluation(None, batch, args, config=config), "evaluation")
        except Exception as error:
            print(f"[warn] evaluation failed: {error!r}; skipping this batch")
            record.update({"status": "evaluation_error", "error": repr(error)})
            report = None
        if report is not None:
            _accumulate(totals, team, report)
            for key in tokens:
                tokens[key] += report.get(key, 0) or 0
            record["report"] = save_report(predictions_path, report, batch=batch_index,
                                           arm=config.name or "fixed", team=team,
                                           question_indices=indices, response_chars=args.response_chars)
            print(f"batch {batch_index}/{len(batches)}: {report['team_correct']}/{report['n_samples']} "
                  f"| running {totals['team_correct']}/{totals['questions']}")
        with records_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    summary = {
        "run": str(out_dir),
        "selection": f"fixed config {config.config_id}",
        "config_id": config.config_id,
        "config": config.to_dict(),
        "agent_model": args.model_name,
        "parse_mode": args.parse_mode,
        "tie_break": args.tie_break,
        "test_questions": totals["questions"],
        "batches": totals["batches"],
        "team_accuracy": totals["team_correct"] / totals["questions"] if totals["questions"] else 0.0,
        **tokens,
        "agents": totals["agents"],
        "tags": totals["tags"],
    }
    summary_path = out_dir / "holdout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nTeam accuracy: {summary['team_accuracy']:.2%} on {summary['test_questions']} questions; "
          f"{tokens['calls']} calls, {tokens['completion_tokens']} completion tokens")
    print(f"Written to {summary_path}, {records_path} and {predictions_path}")


if __name__ == "__main__":
    main()
