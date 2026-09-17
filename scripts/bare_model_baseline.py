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
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from datasets import load_from_disk

from evaluator import evaluate_gsm8k, evaluate_mcq, get_instruction_suffix
from holdout_evaluation import _batch_indices
from model.openai_compat import OpenAICompatChatWrapper
from team_evaluation import _infer_answer_type

# Named so the report tables read sensibly: it occupies the slot a persona would.
AGENT_NAME = "bare_model"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset_path", default="data-claude/tagged_dataset")
    parser.add_argument("--model_name", default="nvidia/Qwen3.6-35B-A3B-NVFP4")
    parser.add_argument("--api_base_url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--test_fraction", type=float, default=0.2,
                        help="Must match the orchestrator runs being compared against")
    parser.add_argument("--split_seed", type=int, default=0,
                        help="Must match the orchestrator runs; this is what makes the split identical")
    parser.add_argument("--test_batch_size", type=int, default=5,
                        help="Only groups the records; one model answering alone is unaffected by batching")
    parser.add_argument("--workers", type=int, default=5,
                        help="Questions answered concurrently, matching --eval_workers on the other arms")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="Answer only the first N held-out questions; 0 means all")
    parser.add_argument("--out_dir", default="data-claude/orchestrator/bare_model")
    return parser.parse_args()


def answer_one(agent, sample, suffixes, args):
    """Ask the model one question and score it exactly as a team agent is scored."""
    question = sample["question"]
    answer = sample["answer"]
    answer_type = _infer_answer_type(answer)
    # The tagged dataset stores answers as strings and the gsm8k evaluator rounds
    # with numpy, which raises on a string.
    if answer_type == "gsm8k":
        answer = float(answer)

    prompt = question + suffixes[answer_type]
    try:
        text = agent.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except Exception as error:
        print(f"[warn] call failed: {error!r}; scoring as incorrect")
        text = ""

    responses = {AGENT_NAME: text}
    if answer_type == "gsm8k":
        final_answers, _, is_correct = evaluate_gsm8k(responses, answer)
    else:
        final_answers, _, is_correct = evaluate_mcq(responses, answer)

    # With one respondent the majority answer is that respondent's, so the team
    # verdict and the agent verdict are the same fact recorded in both places.
    return {
        "tags": sample["tags"] or [],
        "answer_type": answer_type,
        "correct": bool(is_correct),
        "prediction": str(final_answers[0]) if final_answers else "",
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.dataset_path)
    split = dataset.train_test_split(test_size=args.test_fraction, seed=args.split_seed)
    test_dataset = split["test"]
    if args.limit:
        test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))
    print(f"✓ Held out: {len(test_dataset)} questions (split_seed {args.split_seed})")

    agent = OpenAICompatChatWrapper(
        base_url=args.api_base_url,
        model_name=args.model_name,
        api_key=args.api_key,
    )

    # get_instruction_suffix keys off dataset names and its fallback asks for a
    # numeric answer, so both suffixes are built up front and chosen per question.
    suffixes = {
        "gsm8k": get_instruction_suffix(SimpleNamespace(data="gsm8k", bae=False, cot=False)),
        "mcq": get_instruction_suffix(SimpleNamespace(data="arc", bae=False, cot=False)),
    }

    batches = _batch_indices(len(test_dataset), args.test_batch_size, args.split_seed)
    records_path = out_dir / "holdout_records.jsonl"
    records_path.unlink(missing_ok=True)

    total = correct = 0
    tag_counts, tag_correct = Counter(), Counter()

    for batch_index, indices in enumerate(batches, start=1):
        batch = test_dataset.select(indices)
        samples = [batch[i] for i in range(len(batch))]

        # Placed by index rather than completion order, so a result cannot be
        # attributed to the wrong question.
        results = [None] * len(samples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(answer_one, agent, s, suffixes, args): i
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
    print(f"Written to {summary_path} and {records_path}")


if __name__ == "__main__":
    main()
