"""Evaluate a selected team of agent personas on a batch of questions.

Provides `run_team_evaluation(selected_team, sampled_questions, args)` which:
- Instantiates agents using existing `model_utils.get_agents`
- Runs each agent on every question using `model_utils.engine`
- Uses the repository `evaluator` voting logic to compute the team's final answer
- Computes per-agent accuracies and returns a report dict
"""
from copy import deepcopy
from typing import List, Dict

import numpy as np

from model.model_utils import get_agents, engine, get_persona_config
import concurrent.futures
from evaluator import evaluate_question, get_instruction_suffix


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
    args.vllm_base_url = "http://127.0.0.1:8001/v1"

    agents, personas = get_agents(args)

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

    for sample in sampled_questions:
        print(f"Processing sample: {sample}\n")
        total += 1
        question = sample.get('question') if isinstance(sample, dict) else sample['question']
        answer = sample.get('answer') if isinstance(sample, dict) else sample['answer']
        data_type = sample.get('dataset') if isinstance(sample, dict) else sample['dataset']
        args.data = data_type
        suffix = get_instruction_suffix(args)
        sample_tags = sample.get('tags') if isinstance(sample, dict) else sample['tags']
        if sample_tags is None:
            sample_tags = []

        # Build messages and persona configs in the same order as selected_team
        messages = []
        persona_configs = []
        agent_names = []

        for i, pname in enumerate(selected_team):
            p_data = personas.get(pname)
            if isinstance(p_data, dict):
                content = f"{p_data.get('prompt','')}\n\n{question + suffix}"
                persona_configs.append(get_persona_config(pname, personas))
            else:
                content = f"{p_data}\n\n{question + suffix}" if p_data else f"{question + suffix}"
                persona_configs.append(None)

            messages.append({"role": "user", "content": content})
            agent_names.append(pname)

        # Run agents concurrently (send all agent requests at once)
        response_texts = []

        # If `agents` is a list we can call each agent separately in parallel.
        if isinstance(agents, (list, tuple)) and len(agents) >= n_agents:
            def _call_one(i, msg, cfg):
                # engine returns a list for the given call; extract single element
                res = engine([msg], agents[i % len(agents)], 1, persona_configs=[cfg])
                if isinstance(res, (list, tuple)) and len(res) > 0:
                    return _response_text(res[0])
                return _response_text(res)

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, n_agents)) as ex:
                futures = [ex.submit(_call_one, i, msg, cfg) for i, (msg, cfg) in enumerate(zip(messages, persona_configs))]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        response_texts.append(fut.result())
                    except Exception:
                        response_texts.append("")
        else:
            # Fallback: call engine in batch mode (synchronous)
            responses = engine(messages, agents, n_agents, persona_configs=persona_configs)
            response_texts = [_response_text(r) for r in responses]

        # Preserve original agent order when zipping names -> responses
        agent_responses = dict(zip(agent_names, response_texts))

        # print("\n" + "=" * 60)
        # print("AGENT RESPONSES")
        # print("=" * 60)
        # print(f"{agent_responses}\n")
        # print("=" * 60)

        # Use repository evaluator voting logic to get team decision and per-agent final answers
        final_answers, debate_answer, is_corr = evaluate_question(
            agent_responses, answer, data_type
        )

        # final_answers is a list in the same order as agent_responses insertion
        for name, pred in zip(agent_names, final_answers):
            try:
                if data_type == 'gsm8k':
                    correct = (pred != "" and pred == np.round(answer, 1))
                else:
                    correct = (pred != "" and pred == answer)

                if correct:
                    per_agent_correct[name] += 1
            except Exception:
                # conservative: treat as incorrect on errors
                pass

        # Update per-tag counters for this sample
        for t in sample_tags:
            per_tag_counts[t] = per_tag_counts.get(t, 0) + 1
            for name, pred in zip(agent_names, final_answers):
                try:
                    if data_type == 'gsm8k':
                        correct = (pred != "" and pred == np.round(answer, 1))
                    else:
                        correct = (pred != "" and pred == answer)
                    if correct:
                        per_agent_correct_by_tag.setdefault(t, {})
                        per_agent_correct_by_tag[t][name] = per_agent_correct_by_tag[t].get(name, 0) + 1
                except Exception:
                    pass

        team_correct += 1 if is_corr else 0

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
    }
