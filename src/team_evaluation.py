"""Evaluate a team on a batch of questions.

`run_team_evaluation(selected_team, sampled_questions, args)` runs a team on each
question and returns per-agent and team accuracy plus the raw counts the
scoreboard folds in. With only `selected_team` it runs a vote of those personas,
which is what every orchestrator arm so far has done; pass `config` to run any
`TeamConfig` instead (a debate, a hub, a pipeline). The work itself is done by
`runner.run_question`; this module decides each question's answer type, runs the
questions concurrently and adds up the results.
"""
import concurrent.futures
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional

import benchmarks
import team_config
from model.model_utils import DEFAULT_MAX_NEW_TOKENS
from personas import chosen_persona_bank
from model.registry import ModelRegistry
from runner import run_question


@lru_cache(maxsize=8)
def _registry(model_name, base_url, models_file, api_key, max_inflight):
    """One registry, and so one client per model, for the life of the process.

    `get_agents` built fresh clients for every batch; nothing about a client is
    batch-specific, and reusing it keeps connections open between batches.
    """
    class _Args:
        pass
    a = _Args()
    a.model_name, a.vllm_base_url, a.models_file, a.vllm_api_key = model_name, base_url, models_file, api_key
    a.max_inflight = max_inflight
    return ModelRegistry.from_args(a)


def registry_for(args) -> ModelRegistry:
    """The agents' model registry, resolved the way `run_team_evaluation` always did.

    The agents run on `--vllm_base_url`, else `--api_base_url` (the endpoint the
    orchestrator was pointed at unless it has its own), else port 8001.
    """
    base_url = (getattr(args, "vllm_base_url", "") or getattr(args, "api_base_url", "")
                or "http://127.0.0.1:8001/v1")
    model_name = getattr(args, "model_name", None) or getattr(args, "model", None)
    return _registry(model_name, base_url, getattr(args, "models_file", None),
                     getattr(args, "vllm_api_key", "EMPTY") or "EMPTY",
                     getattr(args, "max_inflight", 0) or 0)


def run_team_evaluation(selected_team: Optional[List[str]], sampled_questions, args,
                        personas_override=None, config: Optional[team_config.TeamConfig] = None) -> Dict:
    """Run a team on the sampled questions and return accuracies and counts.

    Args:
        selected_team: persona names to run as a vote. Ignored when `config` is given.
        sampled_questions: a HuggingFace Dataset or list of dicts with at least
            `question` and `answer`; `dataset` and `tags` are used when present.
        args: runtime options: `model_name` and the agent endpoint, plus the
            runner flags (`parse_mode`, `tie_break`, `request_seed`, `models_file`),
            `max_new_tokens` and `eval_workers`.
        personas_override: persona definitions to use instead of the shared bank.
            The paper's per-dataset sets come from that bank, but
            `Elimination_Specialist` names a science-MCQ solver under `arc` and a
            pronoun-resolution solver under `winogrande` (the bank's
            `Elimination_Based_Solver`), so a caller reproducing those sets under
            the paper's names has to supply the definitions it means.
        config: the team to run. Default: `team_config.vote(selected_team)`.

    Per-agent counts are keyed by persona: each persona is credited with the
    answer of the last stage it played (`credited_stage` in the report). For a
    vote that is its only answer, so every saved record keeps its meaning.
    """
    if config is None:
        if not selected_team:
            raise ValueError("run_team_evaluation needs selected_team or config")
        roles = getattr(args, "roles", None)
        config = team_config.for_team(
            list(selected_team), topology=getattr(args, "topology", "vote") or "vote",
            rounds=getattr(args, "rounds", 1), handoff=getattr(args, "handoff", "full") or "full",
            roles=[r.strip() for r in roles.split(",")] if roles else None,
        )

    # Credit: each agent (a persona, or a stage with none) is credited with the
    # answer of the last stage it played - a debater's final round, the hub's
    # verdict, a worker's own answer. Report keys are those agents, so for any
    # team picked by name the scoreboard and the holdout accumulators keep the
    # persona keys they have always used. Per-stage detail is in `samples`.
    credit = {}
    for stage in config.stages:
        credit[stage.persona or stage.id] = stage.id
    agents = list(credit)

    if personas_override is not None:
        personas = personas_override
    else:
        bank = chosen_persona_bank()
        unknown = [p for p in config.personas() if p not in bank]
        if unknown:
            raise KeyError(f"personas not in the bank: {unknown}")
        personas = {p: bank[p] for p in config.personas()}
    missing = [p for p in config.personas() if p not in personas]
    if missing:
        raise KeyError(f"no definition for personas {missing}")

    registry = registry_for(args)
    for model in config.models():
        registry.resolve(model)  # fail before the first call, not halfway through a batch

    # Each question is scored as its own benchmark declared, so a batch that mixes
    # numeric and multiple-choice questions asks each one for the right thing.
    # Look every question up before the first call: a question nobody can score
    # should stop the batch, not fail halfway through it.
    parse_mode = getattr(args, "parse_mode", benchmarks.STRICT)
    scorers = benchmarks.ScorerSet(mode=parse_mode)
    samples = [s if isinstance(s, dict) else dict(s) for s in sampled_questions]
    looked_up = [scorers.for_sample(sample) for sample in samples]
    tie_break = getattr(args, "tie_break", "seeded")
    request_seed = getattr(args, "request_seed", 0)
    max_tokens = getattr(args, "max_new_tokens", None) or DEFAULT_MAX_NEW_TOKENS

    def _run_sample(item):
        sample, (answer_type, scorer) = item
        # The tagged dataset stores every answer as a string, numeric ones
        # included; the scorer knows what its own comparison needs.
        gold = scorer.normalise_gold(sample["answer"])
        result = run_question(
            config, sample["question"], gold, scorer=scorer, personas=personas,
            registry=registry, max_tokens=max_tokens, tie_break=tie_break,
            request_seed=None if request_seed is None or request_seed < 0 else request_seed,
            instance=benchmarks.instance_of(sample),
        )
        by_id = {r["id"]: r for r in result["stages"]}
        result.update({
            "tags": sample.get("tags") or [],
            "dataset": sample.get("dataset"),
            "question_id": benchmarks.instance_of(sample).id,
            "answer_type": answer_type,
            "question": sample["question"],
            "gold": str(gold),
            # The per-agent views the counters below and older readers use.
            # A response nobody could parse is kept apart from a wrong one.
            "correct_by_agent": {a: by_id[credit[a]]["correct"] for a in agents},
            "parsed_by_agent": {a: by_id[credit[a]]["parsed"] for a in agents},
            "predictions": {a: by_id[credit[a]]["prediction"] for a in agents},
            "responses": {a: by_id[credit[a]]["response"] for a in agents},
        })
        return result

    # Questions are independent, so run them together rather than one at a time:
    # a batch of 5 questions with 4 agents is 20 requests the server can overlap.
    workers = max(1, min(int(getattr(args, "eval_workers", 5) or 1), len(samples) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_run_sample, zip(samples, looked_up)))

    # A batch in which every call failed is a broken setup - a wrong served
    # name in --models_file, a prompt over the context limit, a bad budget -
    # not a team that got everything wrong. Raise, so the caller records an
    # evaluation error instead of a zero.
    stage_records = [stage for r in results for stage in r["stages"]]
    if stage_records and all(stage["error"] for stage in stage_records):
        raise RuntimeError(f"every call in this batch failed; first error: {stage_records[0]['error']}")

    per_agent_correct = {sid: 0 for sid in agents}
    per_agent_correct_by_tag: Dict[str, Dict[str, int]] = {}
    per_tag_counts: Dict[str, int] = {}
    team_correct = 0
    for result in results:
        print("\n" + "=" * 60)
        print("AGENT RESPONSES")
        print("=" * 60)
        print(f"{result['responses']}\n")
        print("=" * 60)
        for sid, correct in result["correct_by_agent"].items():
            per_agent_correct[sid] += int(correct)
        for tag in result["tags"]:
            per_tag_counts[tag] = per_tag_counts.get(tag, 0) + 1
            row = per_agent_correct_by_tag.setdefault(tag, {sid: 0 for sid in agents})
            for sid, correct in result["correct_by_agent"].items():
                row[sid] += int(correct)
        team_correct += int(result["team_correct"])

    total = len(results)
    return {
        "team_accuracy": team_correct / total if total else 0.0,
        "per_agent_accuracy": {sid: (per_agent_correct[sid] / total if total else 0.0) for sid in agents},
        "per_agent_accuracy_by_tag": {
            tag: {sid: counts[sid] / (per_tag_counts[tag] or 1) for sid in agents}
            for tag, counts in per_agent_correct_by_tag.items()
        },
        "n_samples": total,
        # Raw counts, so results from several iterations can be summed rather than
        # averaged over batches of different sizes.
        "team_correct": team_correct,
        "per_agent_correct": per_agent_correct,
        "per_agent_correct_by_tag": per_agent_correct_by_tag,
        "per_tag_counts": per_tag_counts,
        "answer_types": dict(Counter(r["answer_type"] for r in results)),
        # What was run and what it cost, so configurations can be compared at
        # matched compute.
        "agents": agents,
        "credited_stage": credit,
        "config_id": config.config_id,
        "config": config.to_dict(),
        "calls": sum(r["calls"] for r in results),
        "prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "completion_tokens": sum(r["completion_tokens"] for r in results),
        "parse_mode": parse_mode,
        "tie_break": tie_break,
        # Per-question detail, for the caller to write to predictions.jsonl and
        # then drop with `predictions.strip_samples`, so `holdout_records.jsonl`
        # keeps its schema.
        "samples": results,
    }
