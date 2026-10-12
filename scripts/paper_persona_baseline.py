"""Score the held-out split with the paper's hand-assigned per-dataset personas.

The orchestration experiment asks a model to assemble a team per batch of questions.
The paper it grew out of did not: it assigned one fixed set of five personas to each
dataset by hand, written for that dataset's failure modes. That assignment is the
obvious thing to beat, and nothing so far has measured it.

There is no training here. Each held-out question is routed by the `dataset` column
it came from, answered by that dataset's five personas, and scored by majority vote
on the same evaluator the other arms use. Because the assignment is fixed and known
in advance, this baseline is also the cheapest of them all.

Two differences from the orchestrator arms are deliberate and worth holding in mind
when reading the number:

- The paper's sets have five personas; the arms select four. `--team_size 4` runs a
  truncated version, so the two effects can be separated.
- The per-dataset sets are drawn from the same 50-persona bank the orchestrator
  selects from, with one naming wrinkle: `Elimination_Specialist` is written twice in
  the paper - a science-MCQ solver for `arc`, a pronoun-resolution solver for
  `winogrande`. The bank kept the `arc` one under that name and carries the
  `winogrande` one as `Elimination_Based_Solver`. Under `winogrande` this script gets
  it back under the paper's name, and passes the definitions explicitly so each
  dataset gets the prompt written for it.
- Only the five gsm8k personas define an extra NVIDIA-format block.
  `--nvidia_persona` appends it to those; the orchestrator arms never append it.

`--proxy` is the domain-matched version for pools the paper never saw. A
benchmark with no paper set may name, per question, a paper benchmark whose set
stands in for it (`proxy_persona_source` in its module): AIME and MATH-500 borrow
GSM8K's maths set, MMLU-Pro's law questions the formal-logic set. Questions with
no stand-in are left out and counted, never forced onto a poor match, so compare
this arm with the others only on the questions it covers (the summary lists
them by dataset). These pairings are this repository's choice, not the paper's.

Output is written in the shape `scripts/report_run.py` reads.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_from_disk
import benchmarks

from model.model_utils import DEFAULT_MAX_NEW_TOKENS
from runner import add_runner_args
from splits import add_split_args, make_split, split_label

from holdout_evaluation import _batch_indices, with_server_retry
from predictions import save_report
from personas import _build_enhanced_personas
from team_evaluation import run_team_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset_path", default="data-claude/tagged_dataset")
    parser.add_argument("--model_name", default="nvidia/Qwen3.6-35B-A3B-NVFP4")
    parser.add_argument("--api_base_url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api_key", default="EMPTY")
    add_split_args(parser)
    add_runner_args(parser)
    parser.add_argument("--test_batch_size", type=int, default=5)
    parser.add_argument("--eval_workers", type=int, default=5)
    parser.add_argument("--team_size", type=int, default=0,
                        help="0 uses each dataset's whole assigned set (five, as the paper wrote it); "
                             "4 truncates to the first four, matching the team size the orchestrator picks")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Token cap per agent answer")
    parser.add_argument("--proxy", action="store_true",
                        help="Benchmarks the paper assigned no set to borrow a domain-matched paper set "
                             "(proxy_persona_source); questions with none are skipped and counted")
    parser.add_argument("--nvidia_persona", action="store_true",
                        help="Append the NVIDIA-format block where a persona defines one, as the orchestrator arms do")
    parser.add_argument("--limit", type=int, default=0, help="Answer only the first N held-out questions; 0 means all")
    parser.add_argument("--response_chars", type=int, default=0,
                        help="Clip stored responses to this many characters; 0 keeps them whole")
    parser.add_argument("--out_dir", default="data-claude/orchestrator/paper_personas")
    return parser.parse_args()


def persona_sets(datasets, team_size, nvidia, proxy=False):
    """Build the paper's assigned set for each dataset (or stand-in source) named."""
    sets = {}
    unassigned = sorted(n for n in datasets if n in benchmarks.list_names() and not benchmarks.persona_set(n))
    if unassigned and not proxy:
        # The canonical arm means "the paper's persona set for this dataset". These
        # benchmarks postdate the paper, so there is nothing canonical to run.
        raise SystemExit(f"no paper persona set for {unassigned}; the canonical arm is undefined "
                         f"for them (see PERSONA_SET in src/benchmarks/)")
    for name in sorted(datasets):
        built = _build_enhanced_personas(SimpleNamespace(
            data=name, multi_persona=True, baseline_a=False, baseline_b=False,
            persona_prompt=nvidia,
        ))
        names = list(built)
        if team_size:
            names = names[:team_size]
        sets[name] = (names, {n: built[n] for n in names})
        print(f"  {name:14s} {len(names)} personas: {', '.join(names)}")
    return sets


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.dataset_path)
    _, test_dataset = make_split(dataset, args)
    if args.limit:
        test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))
    print(f"✓ Held out: {len(test_dataset)} questions [{split_label(args)}]")

    # Each question's team comes from its own dataset's paper set, or under --proxy
    # from the set its benchmark borrows; questions with neither are skipped.
    source_of, skipped = [], defaultdict(int)
    for position, name in enumerate(test_dataset["dataset"]):
        if benchmarks.persona_set(name) or not args.proxy:
            source_of.append(name)
            continue
        source = benchmarks.get(name).proxy_for(benchmarks.metadata_of(test_dataset[position]))
        source_of.append(source)
        if source is None:
            skipped[name] += 1
    print("✓ Assigned persona sets:")
    sets = persona_sets({s for s in source_of if s}, args.team_size, args.nvidia_persona, proxy=args.proxy)
    if skipped:
        print(f"✓ Skipped, no stand-in set: {dict(skipped)}")

    # Questions are grouped by their source dataset (and the set it borrows)
    # because that is what picks the team; within a group the shared batching
    # keeps batch sizes comparable.
    by_dataset = defaultdict(list)
    for position, (name, source) in enumerate(zip(test_dataset["dataset"], source_of)):
        if source is not None:
            by_dataset[name if source == name else f"{name}<-{source}"].append(position)

    records_path = out_dir / "holdout_records.jsonl"
    records_path.unlink(missing_ok=True)
    predictions_path = out_dir / "holdout_predictions.jsonl"
    predictions_path.unlink(missing_ok=True)

    total = team_correct = 0
    agent_totals = defaultdict(lambda: {"correct": 0, "questions": 0, "selected": 0})
    tag_totals = defaultdict(lambda: {"correct": 0, "questions": 0})
    per_dataset = {}
    batch_number = 0

    for name in sorted(by_dataset):
        positions = by_dataset[name]
        team, definitions = sets[name.split("<-")[-1]]
        batches = _batch_indices(len(positions), args.test_batch_size, args.split_seed)
        correct_here = seen_here = 0

        for local_indices in batches:
            batch_number += 1
            indices = [positions[i] for i in local_indices]
            batch = test_dataset.select(indices)

            try:
                report = with_server_retry(
                    lambda: run_team_evaluation(team, batch, args, personas_override=definitions),
                    "evaluation",
                )
            except Exception as error:
                print(f"[warn] {name} batch {batch_number} failed: {error!r}; skipping")
                with records_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"batch": batch_number, "source_dataset": name,
                                         "question_indices": indices, "selected_team": team,
                                         "status": "evaluation_error", "error": repr(error)}) + "\n")
                continue

            save_report(predictions_path, report, batch=batch_number,
                        arm=f"{'proxy' if '<-' in name else 'paper'}_personas:{name}", team=team,
                        question_indices=indices, response_chars=args.response_chars)

            n = report["n_samples"]
            total += n
            team_correct += report["team_correct"]
            correct_here += report["team_correct"]
            seen_here += n
            for agent, correct in report["per_agent_correct"].items():
                agent_totals[agent]["correct"] += correct
                agent_totals[agent]["questions"] += n
                agent_totals[agent]["selected"] += 1
            for tag, count in report.get("per_tag_counts", {}).items():
                tag_totals[tag]["questions"] += count

            print(f"{name:14s} batch {batch_number}: {report['team_correct']}/{n} "
                  f"| running {team_correct}/{total} = {team_correct / total:.1%}")

            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "batch": batch_number,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_dataset": name,
                    "question_indices": indices,
                    "selected_team": team,
                    "invalid_agents": [],
                    "reasoning": (f"Fixed assignment: the paper's persona set for {name}" if "<-" not in name else
                                  f"Fixed stand-in: {name.split('<-')[0]} borrows the paper's set for "
                                  f"{name.split('<-')[1]} (this repository's choice)"),
                    "report": report,
                }, default=str) + "\n")

        per_dataset[name] = {"correct": correct_here, "questions": seen_here,
                             "accuracy": correct_here / seen_here if seen_here else 0.0,
                             "team": team}

    summary = {
        "run": str(out_dir),
        "selection": (f"fixed per-dataset assignment (team_size {args.team_size or 'all'})"
                      + ("; domain-matched stand-ins for unassigned benchmarks" if args.proxy else "")),
        # Questions with no set at all (under --proxy): compare arms only on the rest.
        "skipped_questions": sum(skipped.values()),
        "skipped_by_dataset": dict(skipped),
        "model": args.model_name,
        "test_questions": total,
        "batches": batch_number,
        "team_accuracy": team_correct / total if total else 0.0,
        "agents": dict(agent_totals),
        "tags": dict(tag_totals),
        "per_dataset": per_dataset,
    }
    (out_dir / "holdout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"PAPER PERSONAS: {team_correct}/{total} = {team_correct / total:.1%}" if total else "no questions")
    for name in sorted(per_dataset):
        d = per_dataset[name]
        print(f"  {name:14s} {d['correct']:>3}/{d['questions']:<4} = {d['accuracy']:.1%}")
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()
