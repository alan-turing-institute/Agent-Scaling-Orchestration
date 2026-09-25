"""Evaluate a selected team of agent personas on a batch of questions.

Provides `run_team_evaluation(selected_team, sampled_questions, args)` which:
- Instantiates agents using existing `model_utils.get_agents`
- Runs each agent on every question using `model_utils.engine`
- Uses the repository `evaluator` voting logic to compute the team's final answer
- Computes per-agent accuracies and returns a report dict
"""
from collections import Counter
from copy import deepcopy
from typing import List, Dict


from openai import APIConnectionError, APITimeoutError

from model.model_utils import get_agents, engine, get_persona_config
from responses import response_text as _response_text
import concurrent.futures
import benchmarks
from benchmarks import score_responses




def _infer_answer_type(answer) -> str:
    """Guess an answer type from the answer's shape. Last resort only.

    Numeric-looking -> numeric scoring, anything else -> MCQ scoring. That
    second branch is why this is a last resort: a coding benchmark's answer is
    a string, so it lands on the MCQ letter parser, which reads a character out
    of it and reports a number rather than failing. Prefer `answer_type_of`,
    which asks the registry what the benchmark declared.
    """
    if answer is None:
        return "mcq"
    try:
        float(answer)
        return "numeric"
    except (TypeError, ValueError):
        return "mcq"


def answer_type_of(sample) -> str:
    """Which scorer reads this question's answers.

    Decided per question, not per batch: a tag like "step-by-step reasoning"
    pulls questions from gsm8k and from the multiple-choice sets at once, and
    judging a batch by its first answer silently mis-scores the rest.

    Taken from the `dataset` column the tagged dataset already carries, so a
    benchmark declares its own answer shape once and every scoring path agrees.
    Falls back to sniffing the answer only when that column is absent or names
    something unregistered.
    """
    source = sample.get("dataset") if isinstance(sample, dict) else None
    if source:
        try:
            return benchmarks.answer_type_of(source)
        except KeyError:
            print(f"[warn] {source!r} is not a registered benchmark; "
                  f"guessing its answer type from the answer's shape")

    answer = sample.get("answer") if isinstance(sample, dict) else None
    return _infer_answer_type(answer)


def run_team_evaluation(selected_team: List[str], sampled_questions, args,
                        personas_override=None) -> Dict:
    """Run the selected team on the sampled questions and return accuracies.

    Args:
        selected_team: list of persona names (strings)
        sampled_questions: a HuggingFace Dataset or list-like with dicts containing at least `question` and `answer`
        args: namespace with runtime options (model_name, api keys, etc.)
        personas_override: use these persona definitions instead of the ones
            `get_agents` builds from the shared bank. The paper's per-dataset sets
            are not a subset of that bank: `Elimination_Specialist` names a
            science-MCQ solver under `arc` and a pronoun-resolution solver under
            `winogrande`, and the bank kept one of them. A caller reproducing those
            sets has to supply the definitions it means.

    Returns:
        dict with keys: `team_accuracy` (float), `per_agent_accuracy` (dict mapping persona->accuracy)
    """
    # Prepare args so get_agents builds the chosen personas
    args = deepcopy(args)
    # Normalize model fields expected by model_utils
    if hasattr(args, 'model_name') and not hasattr(args, 'model'):
        setattr(args, 'model', args.model_name)
    # Ensure agent_models is supplied (repeat model_name to match number of agents)
    n_agents = len(selected_team)
    args.chosen_agents = True
    args.chosen_personas = ",".join(selected_team)
    args.num_agents = n_agents
    args.agent_models = ",".join([getattr(args, 'model', getattr(args, 'model_name', ''))] * n_agents)
    args.use_vllm = True
    # The selected team runs on the same endpoint the orchestrator was pointed at,
    # unless the caller set an agent-specific one.
    args.vllm_base_url = (
        getattr(args, 'vllm_base_url', '')
        or getattr(args, 'api_base_url', '')
        or "http://127.0.0.1:8001/v1"
    )

    agents, personas = get_agents(args)
    if personas_override is not None:
        # The wrappers from get_agents are model clients and carry no persona
        # state, so only the definitions need replacing.
        personas = personas_override

    # Each scorer states the format it can read, so a batch that mixes numeric and
    # multiple-choice questions asks each one for the right thing. This used to
    # build a fake args namespace with `data = 'arc'` to trick the dataset-keyed
    # suffix function into producing the MCQ wording.
    parse_mode = getattr(args, 'parse_mode', benchmarks.STRICT)
    SCORERS = {
        answer_type: benchmarks.get_scorer(answer_type, mode=parse_mode)
        for answer_type in ('numeric', 'mcq')
    }
    SUFFIXES = {name: scorer.instruction_suffix() for name, scorer in SCORERS.items()}

    # Build per-agent counters
    per_agent_correct = {name: 0 for name in selected_team}
    total = 0
    team_correct = 0

    # Build tag set across samples
    tag_set = set()
    for s in sampled_questions:
        tags = s.get('tags') if isinstance(s, dict) else s['tags']
        if tags:
            for t in tags:
                tag_set.add(t)

    per_agent_correct_by_tag = {t: {name: 0 for name in selected_team} for t in tag_set}
    per_tag_counts = {t: 0 for t in tag_set}

    # Persona configs and message prefix per agent (cycle if needed)
    persona_list = list(personas.items())

    def _run_sample(sample):
        """Run the team on one question and return what it got right."""
        question = sample.get('question') if isinstance(sample, dict) else sample['question']
        answer = sample.get('answer') if isinstance(sample, dict) else sample['answer']
        answer_type = answer_type_of(sample if isinstance(sample, dict) else dict(sample))
        scorer = SCORERS[answer_type]
        # The tagged dataset stores every answer as a string, numeric ones
        # included; the scorer knows what its own comparison needs.
        answer = scorer.normalise_gold(answer)
        SUFFIX = SUFFIXES[answer_type]
        sample_tags = sample.get('tags') if isinstance(sample, dict) else sample['tags']
        if sample_tags is None:
            sample_tags = []

        # Build messages and persona configs in the same order as selected_team
        messages = []
        persona_configs = []
        agent_names = []

        for pname in selected_team:
            p_data = personas.get(pname)
            if isinstance(p_data, dict):
                content = f"{p_data.get('prompt','')}\n\n{question + SUFFIX}"
                persona_configs.append(get_persona_config(pname, personas))
            else:
                content = f"{p_data}\n\n{question + SUFFIX}" if p_data else f"{question + SUFFIX}"
                persona_configs.append(None)

            messages.append({"role": "user", "content": content})
            agent_names.append(pname)

        # If `agents` is a list we can call each agent separately in parallel.
        if isinstance(agents, (list, tuple)) and len(agents) >= n_agents:
            def _call_one(i, msg, cfg):
                # engine returns a list for the given call; extract single element
                res = engine([msg], agents[i % len(agents)], 1, persona_configs=[cfg])
                if isinstance(res, (list, tuple)) and len(res) > 0:
                    return _response_text(res[0])
                return _response_text(res)

            # Results are placed by index, not by completion order: agent_names is
            # zipped against this list, so a response landing in the wrong slot
            # would credit one agent's answer to another.
            response_texts = [""] * n_agents
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, n_agents)) as ex:
                futures = {
                    ex.submit(_call_one, i, msg, cfg): i
                    for i, (msg, cfg) in enumerate(zip(messages, persona_configs))
                }
                for fut in concurrent.futures.as_completed(futures):
                    idx = futures[fut]
                    try:
                        response_texts[idx] = fut.result()
                    except (APIConnectionError, APITimeoutError):
                        # A server that is down is not a wrong answer. Swallowing
                        # this scored a whole held-out split at 0% once, silently:
                        # every agent returned "", every answer parsed as empty,
                        # and nothing above ever saw an exception to retry on.
                        # Let it reach the caller's retry instead.
                        raise
                    except Exception as error:
                        print(f"[warn] agent {agent_names[idx]} failed on one question: {error!r}")
                        response_texts[idx] = ""
        else:
            # Fallback: call engine in batch mode (synchronous)
            responses = engine(messages, agents, n_agents, persona_configs=persona_configs)
            response_texts = [_response_text(r) for r in responses]

        # Preserve original agent order when zipping names -> responses
        agent_responses = dict(zip(agent_names, response_texts))

        result = score_responses(scorer, agent_responses, answer)

        correct_by_agent = {
            name: scorer.correct(prediction, result.gold)
            for name, prediction in zip(agent_names, result.predictions)
        }
        # Kept apart from `correct_by_agent`: a response nobody could parse is
        # not the same event as a wrong answer, and only one of the two is the
        # model's fault.
        parsed_by_agent = {
            name: prediction.parsed
            for name, prediction in zip(agent_names, result.predictions)
        }

        return {
            "tags": sample_tags,
            "answer_type": answer_type,
            "correct_by_agent": correct_by_agent,
            "parsed_by_agent": parsed_by_agent,
            "predictions": {
                name: str(prediction.legacy)
                for name, prediction in zip(agent_names, result.predictions)
            },
            "gold": str(result.gold),
            "question": question,
            "team_answer": str(result.aggregate.legacy),
            "team_correct": bool(result.correct),
            "responses": agent_responses,
        }

    # Questions are independent, so run them together rather than one at a time:
    # a batch of 5 questions with 4 agents is 20 requests the server can overlap.
    workers = max(1, min(int(getattr(args, 'eval_workers', 5) or 1), len(sampled_questions)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_run_sample, list(sampled_questions)))

    answer_types = Counter(r["answer_type"] for r in results if r)

    for result in results:
        if result is None:
            continue
        total += 1

        print("\n" + "=" * 60)
        print("AGENT RESPONSES")
        print("=" * 60)
        print(f"{result['responses']}\n")
        print("=" * 60)

        for name, correct in result["correct_by_agent"].items():
            if correct:
                per_agent_correct[name] += 1

        for t in result["tags"]:
            per_tag_counts[t] = per_tag_counts.get(t, 0) + 1
            for name, correct in result["correct_by_agent"].items():
                if correct:
                    per_agent_correct_by_tag.setdefault(t, {})
                    per_agent_correct_by_tag[t][name] = per_agent_correct_by_tag[t].get(name, 0) + 1

        team_correct += 1 if result["team_correct"] else 0

    # Compute accuracies
    per_agent_accuracy = {name: (per_agent_correct[name] / total if total > 0 else 0.0) for name in selected_team}
    # Compute per-agent accuracy per tag
    per_agent_accuracy_by_tag = {}
    for t in tag_set:
        denom = per_tag_counts.get(t, 0) or 1
        per_agent_accuracy_by_tag[t] = {name: (per_agent_correct_by_tag.get(t, {}).get(name, 0) / denom) for name in selected_team}
    team_accuracy = (team_correct / total) if total > 0 else 0.0

    return {
        "team_accuracy": team_accuracy,
        "per_agent_accuracy": per_agent_accuracy,
        "per_agent_accuracy_by_tag": per_agent_accuracy_by_tag,
        "n_samples": total,
        # Raw counts, so results from several iterations can be summed rather than
        # averaged over batches of different sizes.
        "team_correct": team_correct,
        "per_agent_correct": dict(per_agent_correct),
        "per_agent_correct_by_tag": {t: dict(v) for t, v in per_agent_correct_by_tag.items()},
        "per_tag_counts": dict(per_tag_counts),
        "answer_types": dict(answer_types),
    }
