"""Answer the held-out split with the bare model: no orchestrator, no personas, no vote.

This is the floor under every arm of the orchestration experiment. The other arms
vary what the orchestrator knows (`--memory`) or whether it chooses at all
(`--selection random`), but all of them still pay for four persona-prompted agents
per question and a majority vote. This script pays for one call and nothing else, so
its accuracy is what the whole apparatus has to beat to have earned its cost.

Everything that decides the score is shared with the orchestrator runs: the same
dataset, the same `train_test_split` seed and fraction, the same batching, the same
per-question answer-type inference, and the same `evaluator` parsers. What differs
is only the prompt - the question and its format instruction, with no persona
prefix - and that one model answers instead of four.

Output is written in the shape `holdout_records.jsonl` and `holdout_summary.json`
take, so `scripts/report_run.py` reads this run like any other arm.
"""

import argparse
import concurrent.futures
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_from_disk

from splits import add_split_args, make_split, split_label

import benchmarks
from benchmarks import score_responses
from holdout_evaluation import _batch_indices
from predictions import SCHEMA_VERSION, write_predictions
from runner import question_key
from model.openai_compat import OpenAICompatChatWrapper
from team_evaluation import answer_type_of

# Named so the report tables read sensibly: it occupies the slot a persona would.
AGENT_NAME = "bare_model"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset_path", default="data-claude/tagged_dataset")
    parser.add_argument("--model_name", default="nvidia/Qwen3.6-35B-A3B-NVFP4")
    parser.add_argument("--api_base_url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api_key", default="EMPTY")
    add_split_args(parser)
    parser.add_argument("--test_batch_size", type=int, default=5,
                        help="Only groups the records; one model answering alone is unaffected by batching")
    parser.add_argument("--workers", type=int, default=5,
                        help="Questions answered concurrently, matching --eval_workers on the other arms")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--parse_mode", choices=["strict", "lenient"], default="strict",
                        help="strict reproduces the parsers every result so far used; lenient fixes them")
    parser.add_argument("--limit", type=int, default=0, help="Answer only the first N held-out questions; 0 means all")
    parser.add_argument("--response_chars", type=int, default=0,
                        help="Clip stored responses to this many characters; 0 keeps them whole")
    parser.add_argument("--out_dir", default="data-claude/orchestrator/bare_model")
    return parser.parse_args()


def answer_one(agent, sample, scorers, args):
    """Ask the model one question and score it exactly as a team agent is scored."""
    question = sample["question"]
    answer = sample["answer"]
    answer_type = answer_type_of(sample)
    scorer = scorers[answer_type]
    # The tagged dataset stores every answer as a string, numeric ones included.
    answer = scorer.normalise_gold(answer)

    prompt = question + scorer.instruction_suffix()
    completion, error_text = None, None
    try:
        completion = agent.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        text = completion.text
    except Exception as error:
        print(f"[warn] call failed: {error!r}; scoring as incorrect")
        text, error_text = "", repr(error)

    result = score_responses(scorer, {AGENT_NAME: text}, answer)
    prediction = result.predictions[0]

    # With one respondent the majority answer is that respondent's, so the team
    # verdict and the agent verdict are the same fact recorded in both places.
    return {
        "tags": sample["tags"] or [],
        "answer_type": answer_type,
        "correct": bool(result.correct),
        "parsed": prediction.parsed,
        "prediction": str(prediction.legacy),
        "gold": str(result.gold),
        "question": question,
        "response": text,
        "dataset": sample.get("dataset"),
        "completion": completion,
        "error": error_text,
    }


def prediction_row(r, batch_index, question_index, args):
    completion = r.get("completion")
    usage = completion.usage() if completion else {}
    response = r.get("response") or ""
    reasoning = completion.reasoning if completion else ""
    if args.response_chars:
        response = response[:args.response_chars]
        reasoning = reasoning[:args.response_chars]
    stage = {
        "id": AGENT_NAME, "persona": None, "role": "solver", "model": args.model_name,
        "inputs": [], "max_tokens": args.max_tokens, "seed": None,
        "response": response,
        "reasoning": reasoning,
        "used_reasoning": completion.used_reasoning if completion else False,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": usage.get("finish_reason"),
        "latency_s": usage.get("latency_s"),
        "served_model": args.model_name,
        "error": r.get("error"),
        "prediction": r["prediction"], "parsed": r["parsed"], "correct": r["correct"],
        "transition": None, "copied": None,
    }
    return {
        "schema": SCHEMA_VERSION,
        "arm": "bare_model",
        "batch": batch_index,
        "question_index": question_index,
        "question_key": question_key(r.get("question")),
        "dataset": r.get("dataset"),
        "answer_type": r["answer_type"],
        "tags": r["tags"],
        "question": r.get("question"),
        "gold": r.get("gold"),
        "team": [AGENT_NAME],
        "config_id": None,
        "aggregation": {"rule": "stage", "over": [AGENT_NAME], "tie_break": None},
        "team_answer": r["prediction"],
        "team_correct": r["correct"],
        "calls": 1,
        "depth": 1,
        "prompt_tokens": stage["prompt_tokens"] or 0,
        "completion_tokens": stage["completion_tokens"] or 0,
        "versions": {"parse_mode": args.parse_mode, "role_templates": None},
        "stages": [stage],
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.dataset_path)
    _, test_dataset = make_split(dataset, args)
    if args.limit:
        test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))
    print(f"✓ Held out: {len(test_dataset)} questions [{split_label(args)}]")

    agent = OpenAICompatChatWrapper(
        base_url=args.api_base_url,
        model_name=args.model_name,
        api_key=args.api_key,
    )

    # Each scorer states the answer format it can read, so a split that mixes
    # numeric and multiple-choice questions asks each one for the right thing.
    scorers = {
        answer_type: benchmarks.get_scorer(answer_type, mode=args.parse_mode)
        for answer_type in ("numeric", "mcq")
    }

    batches = _batch_indices(len(test_dataset), args.test_batch_size, args.split_seed)
    records_path = out_dir / "holdout_records.jsonl"
    records_path.unlink(missing_ok=True)
    predictions_path = out_dir / "holdout_predictions.jsonl"
    predictions_path.unlink(missing_ok=True)

    total = correct = 0
    tag_counts, tag_correct = Counter(), Counter()

    for batch_index, indices in enumerate(batches, start=1):
        batch = test_dataset.select(indices)
        samples = [batch[i] for i in range(len(batch))]

        # Placed by index rather than completion order, so a result cannot be
        # attributed to the wrong question.
        results = [None] * len(samples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(answer_one, agent, s, scorers, args): i
                       for i, s in enumerate(samples)}
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()

        batch_correct = sum(r["correct"] for r in results)
        total += len(results)
        correct += batch_correct
        for r in results:
            for tag in r["tags"]:
                tag_counts[tag] += 1
                tag_correct[tag] += int(r["correct"])

        print(f"batch {batch_index}/{len(batches)}: {batch_correct}/{len(results)} "
              f"| running {correct}/{total} = {correct / total:.1%}")

        record = {
            "batch": batch_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_indices": indices,
            "selected_team": [AGENT_NAME],
            "invalid_agents": [],
            "reasoning": "No selection: the bare model answers every question itself",
            "report": {
                "n_samples": len(results),
                "team_correct": batch_correct,
                "team_accuracy": batch_correct / len(results) if results else 0.0,
                "per_agent_correct": {AGENT_NAME: batch_correct},
                "per_agent_accuracy": {AGENT_NAME: batch_correct / len(results) if results else 0.0},
                "answer_types": [r["answer_type"] for r in results],
            },
        }
        with records_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        # Same per-question detail the team arms write, in the same shape: one
        # stage, no persona, so the arms can be compared question by question.
        write_predictions(predictions_path, [prediction_row(r, batch_index, indices[position], args)
                                             for position, r in enumerate(results)])

    summary = {
        "run": str(out_dir),
        "selection": "none (bare model)",
        "model": args.model_name,
        "test_questions": total,
        "batches": len(batches),
        "team_accuracy": correct / total if total else 0.0,
        # Keyed the way report_run.py reads an arm: "questions" answered,
        # "selected" batches it was picked for - here, every batch.
        "agents": {AGENT_NAME: {"correct": correct, "questions": total, "selected": len(batches)}},
        "tags": {t: {"correct": tag_correct[t], "questions": tag_counts[t]} for t in sorted(tag_counts)},
    }
    summary_path = out_dir / "holdout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"BARE MODEL: {correct}/{total} = {correct / total:.1%}" if total else "BARE MODEL: no questions")
    print(f"Written to {summary_path}, {records_path} and {predictions_path}")


if __name__ == "__main__":
    main()
