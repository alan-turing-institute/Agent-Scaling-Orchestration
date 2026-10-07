"""Run any team configuration on one question, and record every stage.

`run_question` executes a `TeamConfig` layer by layer: stages that depend on
nothing run together, then the stages that read them, and so on. Every stage's
response is scored with the question's scorer, so the record says which stage
got the answer right, which one fixed or broke what it was handed, and what each
call cost. The team's answer is a vote over the answer stages, or the answer of
the final stage.

A vote of solvers sends the same requests `run_team_evaluation` always sent:
the same system message, the same user message, the persona's temperature and
top_p, and the same token cap. Two things are new and can move a number:

- Ties are broken by a generator seeded per question (`tie_break="seeded"`,
  the default) rather than the global one. `tie_break="global"` restores the
  old behaviour.
- A per-request `seed` is sent unless `request_seed` is None. At temperature 0,
  which every persona in the bank uses, it changes nothing.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import random
from typing import Any, Dict, Optional

from benchmarks.environment import function_tool
from episode import report_text, run_episode

from openai import APIConnectionError, APITimeoutError

from personas import get_persona_config
from roles import (AGENT_ROLE_TEMPLATES_VERSION, RATIONALE_CHARS, ROLE_TEMPLATES_VERSION,
                   render_agent_prompt, render_handoff, render_prompt)
from team_config import TeamConfig

SYSTEM_PROMPT = "You are a helpful assistant."
TIE_BREAKS = ("seeded", "global")


def add_runner_args(parser):
    """The flags every entry point that runs teams shares."""
    parser.add_argument("--models_file", default=None,
                        help="JSON mapping model keys to {served_name, base_url}, for configs whose stages "
                             "use more than one model. 'default' is always --model_name at the agent endpoint")
    parser.add_argument("--parse_mode", choices=["strict", "lenient"], default="strict",
                        help="strict reproduces the parsers every result so far used; lenient fixes them")
    parser.add_argument("--tie_break", choices=list(TIE_BREAKS), default="seeded",
                        help="seeded: a vote tie resolves the same way for the same question every time. "
                             "global: the old behaviour, drawn from the global random state")
    parser.add_argument("--request_seed", type=int, default=0,
                        help="Base for the per-request seed sent with every call; -1 sends none")
    parser.add_argument("--max_inflight", type=int, default=0,
                        help="Cap on agent requests in flight per server; 0 = no cap. 1 makes greedy "
                             "output reproducible: the servers' output depends on what else is in the batch")
    return parser


def add_topology_args(parser):
    """How a team chosen by name works together. For entry points that select teams."""
    parser.add_argument("--topology", choices=["vote", "debate", "centralized", "synthesis", "pipeline", "delegated"],
                        default="vote",
                        help="How a selected team works together. centralized/synthesis: the last selected "
                             "agent leads, the rest are workers. pipeline: agents take --roles in order. "
                             "delegated (agentic tasks only): the last agent leads and calls on the rest "
                             "through a delegate tool")
    parser.add_argument("--rounds", type=int, default=1, help="debate: exchange rounds after the first answer")
    parser.add_argument("--roles", default=None,
                        help="pipeline: comma-separated roles, one per agent (default solver,critic...,reviser)")
    parser.add_argument("--handoff", choices=["full", "rationale", "answer"], default="full",
                        help="What a stage sees of the stages it reads")
    return parser


def _stable_int(*parts) -> int:
    blob = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha1(blob).hexdigest()[:8], 16)


def question_key(question: str) -> str:
    """A short, stable name for a question, for seeding and for joining records."""
    return hashlib.sha1((question or "").encode("utf-8")).hexdigest()[:12]


def _persona_text(name, personas):
    if not name:
        return ""
    data = personas.get(name)
    if isinstance(data, dict):
        return data.get("prompt", "")
    return data or ""


def _generation_params(name, personas):
    if name and name in personas and isinstance(personas[name], dict):
        config = get_persona_config(name, personas)
        return config["temperature"], config["top_p"]
    return 0, 0.9


def _transition(previous_correct: Optional[bool], correct: bool) -> Optional[str]:
    if previous_correct is None:
        return None
    if previous_correct:
        return "kept_right" if correct else "broke"
    return "fixed" if correct else "kept_wrong"


def run_question(config: TeamConfig, question: str, gold: Any, *, scorer, personas: Dict,
                 registry, max_tokens: int, tie_break: str = "seeded", tie_break_seed: int = 0,
                 request_seed: Optional[int] = 0, instance=None, environment=None,
                 max_steps: Optional[int] = None) -> Dict:
    """Run `config` on one question and return a record of every stage.

    `gold` must already be normalised by `scorer.normalise_gold`. Connection and
    timeout errors propagate, so the caller's retry can wait out a restarting
    server; any other failure in a stage is recorded and scores as unanswered,
    as a failed agent call always has. `instance` is the question as a
    `benchmarks.Instance`, handed to `scorer.correct` for scorers that read
    more than the gold answer.

    `environment`, for an agentic task, is a factory `instance -> Environment`
    (`benchmarks.environment`). Each stage then runs an episode instead of one
    call (`episode.run_episode`), capped at the stage's `max_steps` or
    `max_steps`, and its prediction is the environment's outcome. Where a stage
    starts:

    - reading nothing: a fresh environment;
    - reading an earlier stage of its own persona (a debater's next round): a
      fork of the state its own last attempt left;
    - reading others, as a critic, reviser or checker: a fork of the state the
      last stage it reads left, so a pipeline builds on its predecessor;
    - a hub or synthesiser: fresh, with the attempts above as advice;
    - a stage called on through `delegate`: the delegating stage's own
      environment, shared, so the lead sees what the worker did.

    Stages read each other as action logs and final messages, never as grades.
    """
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"unknown tie_break {tie_break!r}; known: {TIE_BREAKS}")

    suffix = scorer.instruction_suffix()
    qkey = question_key(question)
    results: Dict[str, Dict] = {}
    predictions: Dict[str, Any] = {}
    if environment is None and config.on_call():
        raise ValueError("a config with delegates needs an agentic task: delegate is a tool, "
                         "and a static question has no environment to act on")
    envs: Dict[str, Any] = {}          # stage id -> the environment it left
    delegated_runs: Dict[str, list] = {}

    def base_record(stage, budget, seed):
        return {
            "id": stage.id,
            "persona": stage.persona,
            "role": stage.role,
            "model": stage.model,
            "inputs": [i.to_dict() for i in stage.inputs],
            "max_tokens": budget,
            "seed": seed,
            "response": "",
            "reasoning": "",
            "used_reasoning": False,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
            "latency_s": None,
            "served_model": None,
            "error": None,
        }

    def start_env(stage):
        own = [i.source for i in stage.inputs
               if stage.persona and config.stage(i.source).persona == stage.persona]
        if own:
            base = envs[own[-1]]
        elif stage.inputs and stage.role not in ("hub", "synthesiser"):
            base = envs[stage.inputs[-1].source]
        else:
            return environment(instance)
        env = base.fork()
        if hasattr(env, "resume"):
            env.resume()  # a verdict the last agent gave (impossible, submitted) can be overruled
        return env

    def agent_handoff(record, handoff):
        if handoff == "answer":
            return f"Final message: {record.get('response') or '(none)'}"
        text = report_text(record)
        if handoff == "rationale" and len(text) > RATIONALE_CHARS:
            return "..." + text[-RATIONALE_CHARS:]
        return text

    def run_agent_stage(stage, env=None, instruction=None, call=0):
        """One stage's episode. `env` and `instruction` are set when a hub calls on it."""
        if env is None:
            env = start_env(stage)
        inputs = []
        for item in stage.inputs:
            position = config.stage_ids.index(item.source) + 1
            inputs.append((f"Agent {position} ({config.stage(item.source).role})",
                           agent_handoff(results[item.source], item.handoff)))
        if instruction is not None:
            inputs.append(("Instruction from your team lead", instruction))
        content = render_agent_prompt(_persona_text(stage.persona, personas), env.task_prompt(),
                                      role=stage.role, inputs=inputs, delegating=bool(stage.delegates))
        temperature, top_p = _generation_params(stage.persona, personas)
        budget = stage.max_tokens or config.max_tokens or max_tokens
        seed = None if request_seed is None or request_seed < 0 else (
            _stable_int(request_seed, qkey, stage.id, call) % (2 ** 31))

        delegate = None
        if stage.delegates:
            names = list(stage.delegates)
            calls_made = {name: 0 for name in names}

            def run_delegate(arguments):
                name = arguments.get("agent")
                if name not in names:
                    return f"ERROR: there is no teammate called {name!r}; choose one of {names}"
                calls_made[name] += 1
                sub = run_agent_stage(config.stage(name), env=env, call=calls_made[name],
                                      instruction=str(arguments.get("instruction") or ""))
                delegated_runs.setdefault(name, []).append(sub)
                return report_text(sub)

            delegate = {
                "tool": function_tool(
                    "delegate",
                    "Give one teammate an instruction. They act on the same environment you see, "
                    "then report what they did.",
                    {"agent": {"type": "string", "enum": names, "description": "Which teammate"},
                     "instruction": {"type": "string", "description": "What they should do"}},
                    ["agent", "instruction"]),
                "run": run_delegate,
            }

        record = base_record(stage, budget, seed)
        record["agent_role_templates_version"] = AGENT_ROLE_TEMPLATES_VERSION
        try:
            episode = run_episode(registry.client(stage.model), env, system=SYSTEM_PROMPT, user=content,
                                  max_steps=stage.max_steps or max_steps or 30, max_tokens=budget,
                                  temperature=temperature, top_p=top_p, seed=seed, delegate=delegate,
                                  nudge=instruction is None)
            prediction = episode.pop("prediction")
            record.update(episode)
            record["_prediction"] = prediction
        except (APIConnectionError, APITimeoutError):
            raise
        except Exception as error:
            print(f"[warn] stage {stage.id} failed on one question: {error!r}")
            record["error"] = repr(error)
        if instruction is None:
            envs[stage.id] = env
        return record

    def merge_delegated(stage):
        """A worker's record: every time the hub called on it, in order."""
        runs = delegated_runs.get(stage.id) or []
        record = base_record(stage, stage.max_tokens or config.max_tokens or max_tokens, None)
        record.update({"times_called": len(runs), "steps": [], "calls": 0, "tool_calls": 0,
                       "prompt_tokens": 0, "completion_tokens": 0,
                       "agent_role_templates_version": AGENT_ROLE_TEMPLATES_VERSION})
        for run in runs:
            record["steps"].extend(run.get("steps") or [])
            for key in ("calls", "tool_calls", "prompt_tokens", "completion_tokens"):
                record[key] += run.get(key) or 0
            record["response"] = run.get("response") or record["response"]
            record["outcome"] = run.get("outcome")
            record["error"] = run.get("error") or record["error"]
            if "_prediction" in run:
                record["_prediction"] = run["_prediction"]
        return record

    def run_stage(index, stage):
        if environment is not None:
            return index, run_agent_stage(stage)
        inputs = []
        for item in stage.inputs:
            source = config.stage(item.source)
            position = config.stage_ids.index(item.source) + 1
            inputs.append((
                f"Agent {position} ({source.role})",
                render_handoff(results[item.source]["response"], predictions[item.source], item.handoff),
            ))
        content = render_prompt(_persona_text(stage.persona, personas), question, suffix,
                                role=stage.role, inputs=inputs)
        temperature, top_p = _generation_params(stage.persona, personas)
        budget = stage.max_tokens or config.max_tokens or max_tokens
        seed = None if request_seed is None or request_seed < 0 else (
            _stable_int(request_seed, qkey, stage.id) % (2 ** 31))

        record = {
            "id": stage.id,
            "persona": stage.persona,
            "role": stage.role,
            "model": stage.model,
            "inputs": [i.to_dict() for i in stage.inputs],
            "max_tokens": budget,
            "seed": seed,
            "response": "",
            "reasoning": "",
            "used_reasoning": False,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
            "latency_s": None,
            "served_model": None,
            "error": None,
        }
        try:
            completion = registry.client(stage.model).generate(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": content}],
                max_tokens=budget, temperature=temperature, top_p=top_p, seed=seed,
            )
            record.update({
                "response": completion.text,
                "reasoning": completion.reasoning,
                "used_reasoning": completion.used_reasoning,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "finish_reason": completion.finish_reason,
                "latency_s": completion.latency_s,
                "served_model": completion.model,
            })
        except (APIConnectionError, APITimeoutError):
            # A server that is down is not a wrong answer; let the caller retry.
            raise
        except Exception as error:
            print(f"[warn] stage {stage.id} failed on one question: {error!r}")
            record["error"] = repr(error)
        return index, record

    def settle(record):
        # An episode's prediction is its environment's outcome; a reply's is parsed from its text.
        prediction = record.pop("_prediction", None) or scorer.extract(record["response"])
        record["prediction"] = str(prediction.legacy)
        record["parsed"] = prediction.parsed
        record["correct"] = bool(scorer.correct(prediction, gold, instance=instance))
        results[record["id"]] = record
        predictions[record["id"]] = prediction

    for layer in config.layers():
        stages = [(config.stage_ids.index(sid), config.stage(sid)) for sid in layer]
        if len(stages) == 1:
            done = [run_stage(*stages[0])]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(stages))) as pool:
                done = list(pool.map(lambda pair: run_stage(*pair), stages))
        for _, record in done:
            settle(record)

    # Stages called on through `delegate` ran inside the stage that called them.
    for stage_id in config.on_call():
        settle(merge_delegated(config.stage(stage_id)))

    # Credit: compare each stage with the answer it was most directly building on.
    # `transition` says whether it fixed or broke that answer; `copied` says it
    # handed the same answer back, which is how anchoring shows up. The reference:
    #   - its own earlier answer, if it read one (a debater in a later round);
    #   - else, for a hub or synthesiser reading several stages, their vote,
    #     since that is what the team would have answered without it;
    #   - else the last stage it read (the previous step of a pipeline).
    # `inputs_correct` keeps every input's verdict, for any other definition.
    for stage in config.stages:
        record = results[stage.id]
        record["inputs_correct"] = {i.source: results[i.source]["correct"] for i in stage.inputs}
        if not stage.inputs:
            record.update({"reference": None, "transition": None, "copied": None})
            continue
        own = [i.source for i in stage.inputs
               if stage.persona and config.stage(i.source).persona == stage.persona]
        if own:
            reference = own[-1]
            theirs, their_correct = predictions[reference], results[reference]["correct"]
        elif stage.role in ("hub", "synthesiser") and len(stage.inputs) > 1:
            reference = "vote"
            rng = random.Random(_stable_int(tie_break_seed, qkey, stage.id))
            theirs = scorer.aggregate([predictions[i.source] for i in stage.inputs], rng=rng)
            their_correct = bool(scorer.correct(theirs, gold, instance=instance))
        else:
            reference = stage.inputs[-1].source
            theirs, their_correct = predictions[reference], results[reference]["correct"]
        mine = predictions[stage.id]
        record["reference"] = reference
        record["transition"] = _transition(their_correct, record["correct"])
        record["copied"] = bool(mine.parsed and theirs.parsed and mine.value == theirs.value)

    answer_ids = config.answer_stages()
    if config.aggregate == "stage":
        team_prediction = predictions[config.final]
    else:
        rng = random.Random(_stable_int(tie_break_seed, qkey)) if tie_break == "seeded" else None
        team_prediction = scorer.aggregate([predictions[sid] for sid in answer_ids], rng=rng)

    stage_records = [results[sid] for sid in config.stage_ids]
    return {
        "question_key": qkey,
        "stages": stage_records,
        "answer_stages": answer_ids,
        "aggregate": config.aggregate,
        "team_answer": str(team_prediction.legacy),
        "team_correct": bool(scorer.correct(team_prediction, gold, instance=instance)),
        # An episode makes many calls; a one-call stage records none and counts as one.
        "calls": sum(r.get("calls", 1) for r in stage_records),
        "depth": len(config.layers()),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in stage_records),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in stage_records),
        "role_templates_version": ROLE_TEMPLATES_VERSION,
    }
