"""Write down what each agent actually answered.

A finished run used to be a set of counts. `run_team_evaluation` built per-agent
predictions and raw responses, printed them to stdout, and the accumulators
folded them into integers; `holdout_records.jsonl` kept the integers. So nothing
could be audited after the fact - not why an agent was marked wrong, not whether
a response was unreadable rather than incorrect - and re-scoring a run under a
fixed parser cost another run on the GPU instead of a re-read of a file.

One row per question per batch, appended as it happens, so a killed run keeps
what it had. Rows written before the runner existed have no `schema` field and
an `agents` mapping instead of `stages`; `scripts/rescore.py` reads both.
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


SCHEMA_VERSION = 2


def rows_from_report(report, *, batch=None, arm=None, team=None,
                     question_indices=None, response_chars=0):
    """Turn one evaluation's `samples` into prediction rows, one per question.

    Each row records every stage of the team: its persona, role, model and
    inputs, what it answered and whether that parsed and was right, the
    transition from the stage it read (`kept_right`, `fixed`, `broke`,
    `kept_wrong`), whether it copied that stage's answer, and what the call
    cost. With the row's `aggregation` that is enough to re-score a run under a
    different parser without calling a model (`scripts/rescore.py`).

    `question_indices` positions each row in the split it came from. Pass
    `response_chars` to clip stored text; 0 keeps it whole, which is what makes
    a re-score possible.
    """
    report = report or {}
    samples = report.get("samples") or []
    rows = []
    for position, sample in enumerate(samples):
        index = None
        if question_indices is not None and position < len(question_indices):
            index = question_indices[position]
        stages = []
        for stage in sample.get("stages") or []:
            stage = dict(stage)
            stage["response"] = _clip(stage.get("response"), response_chars)
            stage["reasoning"] = _clip(stage.get("reasoning"), response_chars)
            stages.append(stage)
        rows.append({
            "schema": SCHEMA_VERSION,
            "arm": arm,
            "batch": batch,
            "question_index": index,
            "question_key": sample.get("question_key"),
            "question_id": sample.get("question_id"),
            "dataset": sample.get("dataset"),
            "answer_type": sample.get("answer_type"),
            "tags": sample.get("tags"),
            "question": sample.get("question"),
            "gold": sample.get("gold"),
            "team": list(team) if team else None,
            "config_id": report.get("config_id"),
            "aggregation": {
                "rule": sample.get("aggregate"),
                "over": sample.get("answer_stages"),
                "tie_break": report.get("tie_break"),
            },
            "team_answer": sample.get("team_answer"),
            "team_correct": sample.get("team_correct"),
            "calls": sample.get("calls"),
            "depth": sample.get("depth"),
            "prompt_tokens": sample.get("prompt_tokens"),
            "completion_tokens": sample.get("completion_tokens"),
            "versions": {
                "parse_mode": report.get("parse_mode"),
                "role_templates": sample.get("role_templates_version"),
            },
            "stages": stages,
        })
    return rows


def write_config(path, report):
    """Record the full config a run used, once, beside its predictions.

    Rows carry only `config_id`; this file maps each id to the stages it means.
    Rewritten in place, keyed by id, so a run that uses several configs keeps
    all of them.
    """
    if not report or not report.get("config_id"):
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if report["config_id"] not in configs:
        configs[report["config_id"]] = report.get("config")
        path.write_text(json.dumps(configs, indent=2), encoding="utf-8")


def strip_samples(report):
    """Take `samples` and `config` off a report so it can go into a records file.

    Both are written elsewhere: the samples as prediction rows, the config once
    to `configs.json`. The records file keeps its counts and `config_id`.
    """
    if not report:
        return report
    report.pop("samples", None)
    report.pop("config", None)
    return report


def save_report(predictions_path, report, **row_kwargs):
    """Write a report's prediction rows and config, then return it stripped for the records file."""
    write_predictions(predictions_path, rows_from_report(report, **row_kwargs))
    write_config(Path(predictions_path).with_name("configs.json"), report)
    return strip_samples(report)
