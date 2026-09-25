"""Write down what each agent actually answered.

A finished run used to be a set of counts. `run_team_evaluation` built per-agent
predictions and raw responses, printed them to stdout, and the accumulators
folded them into integers; `holdout_records.jsonl` kept the integers. So nothing
could be audited after the fact - not why an agent was marked wrong, not whether
a response was unreadable rather than incorrect - and re-scoring a run under a
fixed parser cost another run on the GPU instead of a re-read of a file.

One row per question per batch, appended as it happens, so a killed run keeps
what it had.
"""

import json
from pathlib import Path


def _clip(text, limit):
    if not limit or limit <= 0 or text is None:
        return text
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[{len(text) - limit} more chars]"


def write_predictions(path, rows):
    """Append prediction rows. Directories are created as needed."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def rows_from_report(report, *, batch=None, arm=None, team=None,
                     question_indices=None, response_chars=0):
    """Turn one evaluation's `samples` into prediction rows.

    `question_indices` positions each row in the split it came from, so a row
    can be traced back to the exact question. Pass `response_chars` to clip the
    stored response text; 0 keeps it whole, which is what makes a re-score
    possible.
    """
    samples = (report or {}).get("samples") or []
    rows = []
    for position, sample in enumerate(samples):
        index = None
        if question_indices is not None and position < len(question_indices):
            index = question_indices[position]
        rows.append({
            "arm": arm,
            "batch": batch,
            "question_index": index,
            "dataset": sample.get("dataset"),
            "answer_type": sample.get("answer_type"),
            "tags": sample.get("tags"),
            "question": sample.get("question"),
            "gold": sample.get("gold"),
            "team": list(team) if team else None,
            "team_answer": sample.get("team_answer"),
            "team_correct": sample.get("team_correct"),
            "agents": {
                name: {
                    "prediction": sample.get("predictions", {}).get(name),
                    "correct": sample.get("correct_by_agent", {}).get(name),
                    "parsed": sample.get("parsed_by_agent", {}).get(name),
                    "response": _clip(sample.get("responses", {}).get(name), response_chars),
                }
                for name in (sample.get("correct_by_agent") or {})
            },
        })
    return rows


def strip_samples(report):
    """Take `samples` off a report so it can go into a records file unchanged."""
    if not report:
        return report
    report.pop("samples", None)
    return report
