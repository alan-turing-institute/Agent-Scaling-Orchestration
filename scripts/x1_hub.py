"""X1 stage 2: the hub reads each team's saved worker answers and gives the team's answer.

Stage 1 (scripts/x1_plan.py) ran every worker alone and saved its answers. Here only the hub
is served (Qwen3.6-35B, thinking off), and it also judges: BrowseComp-Plus and Finance-Agent
episodes from stage 1 recorded what they would have asked the judge (`--defer_judge`), and
are graded now with the same questions (`judge.regrade`).

Per question the hub sees, labelled "Agent 1 (solver)" and so on, never by name:

- hard questions: each worker's "Summary:" block and its final answer; where a worker left the
  block out, the last 600 characters instead (counted as a fallback, per worker);
- agentic tasks: each worker's actions and final message (`episode.report_text`), then it
  works the task itself in a fresh environment.

Each row also records the majority vote over the workers, so hub vs vote vs each worker is
read from one file. `--modes` picks which thinking mode's answers a specialist contributes:
`headline` (the config's choice per specialist), `on` or `off` (where a specialist has it).

    python scripts/x1_hub.py --fold 0 --sample 0                 # every team
    python scripts/x1_hub.py --fold 0 --sample 0 --teams T1,T7 --modes off

Writes <root>/stage2/<modes>/<team>/fold<f>/s<k>/holdout_predictions.jsonl (+ traces) and
holdout_summary.json. Questions already in the predictions file are skipped, so a rerun resumes.
"""

import argparse
import concurrent.futures
import hashlib
import json
import random
import shlex
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from datasets import load_from_disk
from openai import APIConnectionError, APITimeoutError

import benchmarks
from benchmarks.base import Prediction
from episode import report_text, run_episode
from holdout_evaluation import with_server_retry
from judge import Judge, regrade
from model.openai_compat import OpenAICompatChatWrapper, is_context_limit, limit_of
from predictions import SCHEMA_VERSION, split_traces, traces_path, write_predictions
from roles import AGENT_ROLE_TEMPLATES_VERSION, ROLE_TEMPLATES_VERSION, render_agent_prompt, render_prompt
from runner import SYSTEM_PROMPT, question_key
from x1_plan import CONFIG, POOLS, ROOT, load_config, run_dir

RATIONALE_CHARS = 600   # the fallback when a worker wrote no summary (roles.RATIONALE_CHARS)
SUMMARY_CHARS = 2000    # five sentences and an answer line, with room to spare
SERVER_RETRIES = 20
HUB_ID = "hub"


def parse_sampling(flags):
    """'--temperature 0.7 --top_p 0.8 ...' -> (temperature, top_p, request kwargs), as stage 1 sent them."""
    parser = argparse.ArgumentParser()
    for name in ("temperature", "top_p", "min_p", "presence_penalty", "repetition_penalty"):
        parser.add_argument(f"--{name}", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    s = parser.parse_args(shlex.split(flags))
    extra = {k: getattr(s, k) for k in ("top_k", "min_p", "repetition_penalty") if getattr(s, k) is not None}
    extra["chat_template_kwargs"] = {"enable_thinking": False}  # the hub thinks off (Q7)
    kwargs = {"extra_body": extra}
    if s.presence_penalty is not None:
        kwargs["presence_penalty"] = s.presence_penalty
    return s.temperature, s.top_p, kwargs


def summary_handoff(response, prediction):
    """What the hub reads of one worker's answer, and whether it fell back to the response's end."""
    text = response or ""
    start = text.rfind("Summary:")
    if start >= 0 and text[start + len("Summary:"):].strip():
        block, fallback = text[start:].strip()[:SUMMARY_CHARS], False
    else:
        block = text if len(text) <= RATIONALE_CHARS else "..." + text[-RATIONALE_CHARS:]
        fallback = True
    answer = prediction.value if prediction is not None and prediction.parsed else "(no readable answer)"
    return f"{block}\nFinal answer: {answer}", fallback


def team_workers(config, team, modes):
    """[(label, stage-1 worker dir name)] for one team: personas, then specialists in the chosen mode."""
    out = [(p, f"{p}-{config['persona_model']['mode']}") for p in team["personas"]]
    for key in team["specialists"]:
        spec = config["specialists"][key]
        mode = spec["headline"] if modes == "headline" else (modes if modes in spec["modes"] else spec["headline"])
        out.append((key, f"{key}-{mode}"))
    return out


def load_worker(root, pool, worker_dir, fold, sample):
    """A worker's stage-1 rows by question id, with each agentic stage's steps joined back from traces."""
    directory = run_dir(root, pool, worker_dir.rsplit("-", 1)[0], worker_dir.rsplit("-", 1)[1], fold, sample)
    if not (directory / "holdout_summary.json").exists():
        raise SystemExit(f"stage 1 is not finished for {directory}")
    rows = {}
    for line in (directory / "holdout_predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            rows[row["question_id"]] = row
    traces = traces_path(directory / "holdout_predictions.jsonl")
    if traces.exists():
        steps = {}
        for line in traces.read_text(encoding="utf-8").splitlines():
            if line:
                t = json.loads(line)
                steps[t["question_id"]] = t["steps"]
        for qid, row in rows.items():
            if qid in steps:
                row["stages"][0]["steps"] = steps[qid]
    return rows


def _stable_int(*parts):
    return int(hashlib.sha1("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


class Hub:
    def __init__(self, args, config):
        hub = config["hub"]
        self.config = config
        self.args = args
        self.model = hub["handle"]
        self.client = OpenAICompatChatWrapper(base_url=args.api, model_name=self.model,
                                              timeout=config["settings"]["request_timeout"])
        self.judge = Judge(OpenAICompatChatWrapper(base_url=args.api, model_name=self.model))
        self.temperature, self.top_p, self.sampling = parse_sampling(config["sampling"][hub["sampling"]])
        self.scorers = benchmarks.ScorerSet(mode="strict")
        self.judge_lock = threading.Lock()

    def worker_stage(self, row, label, sample_row, scorer, instance, gold):
        """A worker's stage-1 record, regraded where the judge was deferred, with its prediction."""
        stage = {k: v for k, v in row["stages"][0].items() if k != "reasoning"}
        stage["id"], stage["worker"] = label, row["stages"][0]["id"]
        if stage.get("judge_deferred") is not None:
            ok, detail = regrade(sample_row["dataset"], stage["judge_deferred"], self.judge)
            fingerprint = (stage.get("outcome") or {}).get("fingerprint", "no_answer")
            stage["prediction"] = f"{'success' if ok else 'fail'}|{fingerprint}"
            stage["correct"] = ok
            stage["regraded"] = detail
        if (stage.get("outcome") is not None) or stage.get("judge_deferred") is not None:
            prediction = Prediction(value=stage["prediction"], parsed=bool(stage.get("parsed", True)))
        else:
            prediction = scorer.extract(stage.get("response") or "")
            stage["correct"] = bool(scorer.correct(prediction, gold, instance=instance))
        return stage, prediction

    def answer(self, team, workers, sample_row):
        answer_type, scorer = self.scorers.for_sample(sample_row)
        instance = benchmarks.instance_of(sample_row)
        gold = scorer.normalise_gold(sample_row["answer"])
        qid = instance.id
        stages, predictions = [], []
        for label, rows in workers:
            stage, prediction = self.worker_stage(rows[qid], label, sample_row, scorer, instance, gold)
            stages.append(stage)
            predictions.append(prediction)

        benchmark = benchmarks.get(sample_row["dataset"])
        qkey = question_key(sample_row["question"])
        hub = {"id": HUB_ID, "persona": None, "role": "hub", "model": self.model,
               "inputs": [s["id"] for s in stages], "response": "", "reasoning": "", "error": None,
               "prompt_tokens": None, "completion_tokens": None, "finish_reason": None, "latency_s": None,
               "limit": None}
        fallbacks = {}
        if benchmark.environment is not None:
            env = benchmark.environment_for(self.judge)(instance)
            inputs = [(f"Agent {i} (solver)", report_text(s)) for i, s in enumerate(stages, start=1)]
            user = render_agent_prompt("", env.task_prompt(), role="hub", inputs=inputs)
            for s in stages:
                s.pop("steps", None)  # already in stage 1's traces; the hub's own go to this run's
            hub["agent_role_templates_version"] = AGENT_ROLE_TEMPLATES_VERSION
            hub["max_tokens"] = self.config["settings"]["agentic"]["max_tokens"]
            try:
                episode = run_episode(self.client, env, system=SYSTEM_PROMPT, user=user,
                                      max_steps=benchmark.max_steps or 30, max_tokens=hub["max_tokens"],
                                      temperature=self.temperature, top_p=self.top_p, sampling=self.sampling)
                hub_prediction = episode.pop("prediction")
                hub.update(episode)
            except (APIConnectionError, APITimeoutError):
                raise
            except Exception as error:
                print(f"[warn] hub episode failed on {qid}: {error!r}")
                hub["error"], hub_prediction = repr(error), scorer.extract("")
                if is_context_limit(error):
                    hub["limit"] = "context"
        else:
            inputs = []
            for i, (s, p) in enumerate(zip(stages, predictions), start=1):
                text, fell_back = summary_handoff(s.get("response"), p)
                fallbacks[s["id"]] = fell_back
                inputs.append((f"Agent {i} (solver)", text))
            content = render_prompt("", sample_row["question"], scorer.instruction_suffix(), role="hub", inputs=inputs)
            hub["max_tokens"] = self.config["settings"]["hard"]["max_tokens"]
            try:
                completion = self.client.generate(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
                    max_tokens=hub["max_tokens"], temperature=self.temperature, top_p=self.top_p, **self.sampling)
                hub.update({"response": completion.text, "reasoning": completion.reasoning,
                            "prompt_tokens": completion.prompt_tokens,
                            "completion_tokens": completion.completion_tokens,
                            "finish_reason": completion.finish_reason, "latency_s": completion.latency_s,
                            "limit": limit_of(completion.finish_reason)})
            except (APIConnectionError, APITimeoutError):
                raise
            except Exception as error:
                print(f"[warn] hub call failed on {qid}: {error!r}")
                hub["error"] = repr(error)
                if is_context_limit(error):
                    hub["limit"] = "context"
            hub_prediction = scorer.extract(hub["response"])
        hub["prediction"], hub["parsed"] = str(hub_prediction.legacy), hub_prediction.parsed
        hub["correct"] = bool(scorer.correct(hub_prediction, gold, instance=instance))

        vote = scorer.aggregate(predictions, rng=random.Random(_stable_int(self.args.fold, self.args.sample, qkey)))
        vote_correct = bool(scorer.correct(vote, gold, instance=instance))
        hub["reference"] = "vote"
        hub["transition"] = ("kept_right" if hub["correct"] else "broke") if vote_correct else (
            "fixed" if hub["correct"] else "kept_wrong")
        hub["copied"] = bool(hub_prediction.parsed and vote.parsed and hub_prediction.value == vote.value)
        hub["inputs_correct"] = {s["id"]: s["correct"] for s in stages}
        all_stages = stages + [hub]
        limits = {s.get("limit") for s in all_stages} - {None}
        return {
            "schema": SCHEMA_VERSION, "arm": f"x1-{team['id']}-{self.args.modes}", "team_id": team["id"],
            "fold": self.args.fold, "sample": self.args.sample, "modes": self.args.modes,
            "question_key": qkey, "question_id": qid, "dataset": sample_row["dataset"],
            "answer_type": answer_type, "tags": sample_row.get("tags") or [], "question": sample_row["question"],
            "gold": str(sample_row["answer"]), "team": [s["id"] for s in stages] + [HUB_ID],
            "aggregation": {"rule": "stage", "over": [HUB_ID], "tie_break": None},
            "limit": "context" if "context" in limits else ("max_tokens" if limits else None),
            "team_answer": hub["prediction"], "team_correct": hub["correct"],
            "vote_answer": str(vote.legacy), "vote_correct": vote_correct,
            "summary_fallbacks": fallbacks,
            "calls": sum(s.get("calls") or 1 for s in all_stages),
            "prompt_tokens": sum(s.get("prompt_tokens") or 0 for s in all_stages),
            "completion_tokens": sum(s.get("completion_tokens") or 0 for s in all_stages),
            "versions": {"parse_mode": "strict", "role_templates": ROLE_TEMPLATES_VERSION},
            "stages": all_stages,
        }


def summarise(rows, team, args):
    n = len(rows)
    workers = [s["id"] for s in rows[0]["stages"][:-1]] if rows else []
    per_worker = {w: sum(r["stages"][i]["correct"] for r in rows) for i, w in enumerate(workers)}
    fallbacks = {w: sum(bool(r["summary_fallbacks"].get(w)) for r in rows) for w in workers}
    return {
        "team": team["id"], "name": team["name"], "fold": args.fold, "sample": args.sample, "modes": args.modes,
        "questions": n,
        "hub_correct": sum(r["team_correct"] for r in rows),
        "vote_correct": sum(r["vote_correct"] for r in rows),
        "workers_correct": per_worker,
        "summary_fallbacks": fallbacks,
        "hub_transitions": {t: sum(r["stages"][-1]["transition"] == t for r in rows)
                            for t in ("kept_right", "fixed", "broke", "kept_wrong")},
        "hub_limits": sum(r["stages"][-1].get("limit") is not None for r in rows),
        "hub_completion_tokens": sum(r["stages"][-1].get("completion_tokens") or 0 for r in rows),
        "written": datetime.now(timezone.utc).isoformat(),
    }


def run_team(hub, config, team, pools, args):
    workers = [(label, load_worker(args.root, team["pool"], wdir, args.fold, args.sample))
               for label, wdir in team_workers(config, team, args.modes)]
    # The personas ran on exactly this team's questions; the specialists on theirs and other teams'.
    qids = list(workers[0][1])
    for label, rows in workers:
        missing = [q for q in qids if q not in rows]
        if missing:
            raise SystemExit(f"{team['id']}: worker {label} has no answer for {len(missing)} questions, e.g. {missing[0]}")
    out_dir = Path(args.root) / "stage2" / args.modes / team["id"] / f"fold{args.fold}" / f"s{args.sample}"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "holdout_predictions.jsonl"
    done = {}
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                done[row["question_id"]] = row
    todo = [q for q in qids if q not in done]
    print(f"{team['id']} {team['name']}: {len(qids)} questions, {len(done)} already answered")
    pool_workers = config["settings"][team["pool"]]["workers"]
    lock = threading.Lock()

    def one(qid):
        row = with_server_retry(lambda: hub.answer(team, workers, pools[team["pool"]][qid]),
                                f"hub on {qid}", attempts=SERVER_RETRIES)
        with lock:
            # One row at a time, so an interrupted run keeps every finished question.
            write_predictions(traces_path(predictions_path), split_traces([row]))
            write_predictions(predictions_path, [row])
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_workers) as executor:
        for row in executor.map(one, todo):
            done[row["question_id"]] = row
    rows = [done[q] for q in qids]
    summary = summarise(rows, team, args)
    (out_dir / "holdout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{team['id']}: hub {summary['hub_correct']}/{len(rows)}, vote {summary['vote_correct']}/{len(rows)}, "
          f"workers {summary['workers_correct']}, fallbacks {summary['summary_fallbacks']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--teams", default=None, help="Comma-separated team ids; default all")
    parser.add_argument("--modes", choices=["headline", "on", "off"], default="headline",
                        help="Which thinking mode's answers each specialist contributes")
    parser.add_argument("--api", default="http://127.0.0.1:8001/v1", help="The hub's server, which also judges")
    args = parser.parse_args()
    config = load_config(args.config)
    benchmarks._load_registry()
    wanted = set(args.teams.split(",")) if args.teams else None
    teams = [t for t in config["teams"] if not wanted or t["id"] in wanted]
    pools = {}
    for pool in {t["pool"] for t in teams}:
        pools[pool] = {benchmarks.instance_of(row).id: row for row in load_from_disk(str(REPO / POOLS[pool]))}
    hub = Hub(args, config)
    for team in teams:
        run_team(hub, config, team, pools, args)
    print(f"judge: {hub.judge.calls} calls")


if __name__ == "__main__":
    main()
