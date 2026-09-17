"""Evaluate a selected team of agent personas on a batch of questions.

Provides `run_team_evaluation(selected_team, sampled_questions, args)` which:
- Instantiates agents using existing `model_utils.get_agents`
- Runs each agent on every question using `model_utils.engine`
- Uses the repository `evaluator` voting logic to compute the team's final answer
- Computes per-agent accuracies and returns a report dict
"""
from collections import Counter
from copy import deepcopy
import re
from typing import List, Dict

import numpy as np

from openai import APIConnectionError, APITimeoutError

from model.model_utils import get_agents, engine, get_persona_config
import concurrent.futures
from evaluator import get_instruction_suffix, evaluate_gsm8k, evaluate_mcq, base_evaluate_gsm8k, base_evaluate_mcq


def _response_text(resp):
    """Extract text content from various response shapes returned by `engine`."""
    # OpenAI-like response objects
    try:
        return resp.choices[0].message.content
    except Exception:
        pass
    try:
        return resp.choices[0].text
    except Exception:
        pass
    # Already a plain string
    if isinstance(resp, str):
        return resp
    # Fallback to str()
    return str(resp)


def _infer_answer_type(answer) -> str:
    """Numeric answer -> gsm8k scoring, anything else -> MCQ scoring.

    Decided per question, not per batch: a tag like "step-by-step reasoning"
    pulls questions from gsm8k and from the multiple-choice sets at once, and
    judging the batch by its first answer silently mis-scores the rest.
    """
    if answer is None:
        return "mcq"
    try:
        float(answer)
        return "gsm8k"
    except (TypeError, ValueError):
        return "mcq"


def run_team_evaluation(selected_team: List[str], sampled_questions, args) -> Dict:
    """Run the selected team on the sampled questions and return accuracies.

    Args:
        selected_team: list of persona names (strings)
        sampled_questions: a HuggingFace Dataset or list-like with dicts containing at least `question` and `answer`
        args: namespace with runtime options (model_name, api keys, etc.)

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

    # get_instruction_suffix keys off dataset names, and its fallback branch asks
    # for a numeric answer, so MCQ questions borrow an MCQ dataset name or the
    # agents are told the wrong answer format. Both suffixes are built up front
    # and chosen per question.
    numeric_args = deepcopy(args)
    numeric_args.data = 'gsm8k'
    mcq_args = deepcopy(args)
    mcq_args.data = 'arc'
    SUFFIXES = {
        'gsm8k': get_instruction_suffix(numeric_args),
        'mcq': get_instruction_suffix(mcq_args),
    }

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
        answer_type = _infer_answer_type(answer)
        # The tagged dataset stores answers as strings; the gsm8k evaluator rounds
        # them with numpy, which raises on a string.
        if answer_type == 'gsm8k':
            answer = float(answer)
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

        # Use repository evaluator voting logic to get team decision and per-agent final answers
        if answer_type == 'gsm8k':
            final_answers, debate_answer, is_corr = evaluate_gsm8k(agent_responses, answer)
        else:
            final_answers, debate_answer, is_corr = evaluate_mcq(agent_responses, answer)

        correct_by_agent = {}
        for name, pred in zip(agent_names, final_answers):
            try:
                if answer_type == 'gsm8k':
                    correct = (pred != "" and pred == np.round(answer, 1))
                else:
                    correct = (pred != "" and pred == answer)
            except Exception:
                # conservative: treat as incorrect on errors
                correct = False
            correct_by_agent[name] = bool(correct)

        return {
            "tags": sample_tags,
            "answer_type": answer_type,
            "correct_by_agent": correct_by_agent,
            "team_correct": bool(is_corr),
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
