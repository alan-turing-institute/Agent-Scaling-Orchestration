import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defaults import MAX_NEW_TOKENS, THINKING_TOKEN_BUDGET, TOP_P
from data.data_utils import load_data
from model.model_utils import get_agents


SUPPORTED_DATASETS = [
    'gsm8k',
    'arc',
    'hellaswag',
    'truthfulqa',
    'winogrande',
    'pro_medicine',
    'formal_logic',
]


def parse_args():
    parser = argparse.ArgumentParser(description="Tag dataset questions with an LLM")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--data', nargs='*', default=None, help='List of datasets to tag, e.g. --data gsm8k truthfulqa')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--data_size', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='out/question_tags')
    parser.add_argument('--output_file', type=str, default=None)

    parser.add_argument('--model', type=str, default='Qwen/Qwen3.6-35B-A3B')
    parser.add_argument('--agent_models', type=str, default='')
    parser.add_argument('--max_new_tokens', type=int, default=MAX_NEW_TOKENS)
    parser.add_argument('--thinking_token_budget', type=int, default=THINKING_TOKEN_BUDGET,
                        help='Reasoning budget for API models that accept it')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=TOP_P)

    parser.add_argument('--use_vllm', action='store_true')
    parser.add_argument('--vllm_base_url', type=str, default=os.getenv('VLLM_BASE_URL', 'http://127.0.0.1:8001/v1'))
    parser.add_argument('--vllm_api_key', type=str, default=os.getenv('VLLM_API_KEY', 'EMPTY'))
    parser.add_argument('--azure_endpoint', type=str, default=os.getenv('AZURE_OPENAI_ENDPOINT', ''))
    parser.add_argument('--azure_api_key_env', type=str, default=os.getenv('AZURE_OPENAI_API_KEY_ENV', 'API_KEY'))
    parser.add_argument('--openai_api_key', type=str, default=os.getenv('OPENAI_API_KEY', ''))
    parser.add_argument('--openai_base_url', type=str, default=os.getenv('OPENAI_BASE_URL', ''))

    return parser.parse_args()


def resolve_datasets(raw_value):
    if not raw_value:
        return SUPPORTED_DATASETS

    datasets = [item.strip() for item in raw_value if item and str(item).strip()]
    if not datasets:
        return SUPPORTED_DATASETS

    invalid = [ds for ds in datasets if ds not in SUPPORTED_DATASETS]
    if invalid:
        raise ValueError(f"Unsupported dataset(s): {invalid}")
    return datasets


def build_prompt(question: str) -> str:
    return f"""You are assigning tags to a question for the purpose of selecting a multi-agent team to solve it.
Each tag should describe a capability or reasoning style, or domain expertise that would help choose the right agents.
Return valid JSON with a single key named "tags" whose value is a list of short topic labels (1-3 words each).
Focus on what kinds of agents would be useful, not just the surface topic.
Examples:
- math word problem -> ["math", "step-by-step reasoning", "algebra"]
- reading comprehension -> ["language understanding", "context tracking"]
- scientific reasoning -> ["science", "formal reasoning"]

Question:
{question}
"""


def normalize_response(response):
    if hasattr(response, 'choices') and response.choices:
        return response.choices[0].message.content
    return response


def parse_tags(raw_text):
    if not raw_text:
        return []

    text = raw_text.strip()
    if not text:
        return []

    fenced = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if isinstance(parsed.get('tags'), list):
                return [str(tag).strip() for tag in parsed['tags'] if str(tag).strip()]
            if isinstance(parsed.get('tag'), str) and parsed['tag'].strip():
                return [parsed['tag'].strip()]
            if isinstance(parsed.get('labels'), list):
                return [str(tag).strip() for tag in parsed['labels'] if str(tag).strip()]
        elif isinstance(parsed, list):
            return [str(tag).strip() for tag in parsed if str(tag).strip()]
    except Exception:
        pass

    parts = [part.strip() for part in re.split(r"[,\n]+", text) if part.strip()]
    if len(parts) > 1:
        return parts

    return [text]


async def tag_one_question(agent, dataset_name, idx, question, answer, args, semaphore):
    async with semaphore:
        prompt = build_prompt(question)
        message = [{"role": "user", "content": prompt}]
        try:
            if hasattr(agent, 'complete'):
                raw_text = await asyncio.to_thread(
                    agent.complete,
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                raw_text = normalize_response(raw_text)
            else:
                raise AttributeError("agent does not expose a complete() method")
        except Exception as exc:
            raw_text = ""
            print(f"[error] {dataset_name} sample {idx}: {exc}")

        tags = parse_tags(raw_text)
        print(f"[{dataset_name} sample {idx}] Question: {question}\nTags: {tags}\nRaw response: {raw_text}\n")
        return {
            'dataset': dataset_name,
            'question': question,
            'answer': answer,
            'tags': tags,
            'raw_response': raw_text,
        }


def main():
    args = parse_args()
    datasets = resolve_datasets(args.data)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    agent_objs, _ = get_agents(args)
    if isinstance(agent_objs, (list, tuple)):
        agent = agent_objs[0]
    else:
        agent = agent_objs

    agent.max_new_tokens = getattr(args, 'max_new_tokens', 128)
    agent.temperature = getattr(args, 'temperature', 0.0)
    agent.top_p = getattr(args, 'top_p', 0.9)

    os.makedirs(args.out_dir, exist_ok=True)
    dataset_name_slug = '_'.join(datasets)
    output_file = args.output_file or f"{dataset_name_slug}_{args.split}_{args.data_size}_tags.jsonl"
    output_path = os.path.join(args.out_dir, output_file)

    all_records = []
    semaphore = asyncio.Semaphore(64)

    async def run_dataset(dataset_name):
        print(f"Processing dataset: {dataset_name}")
        args.data = dataset_name
        test_X, test_Y = load_data(args, split=args.split)
        if args.data_size and args.data_size > 0:
            test_X = test_X[:args.data_size]
            test_Y = test_Y[:args.data_size]

        tasks = [
            tag_one_question(agent, dataset_name, idx, question, answer, args, semaphore)
            for idx, (question, answer) in enumerate(zip(test_X, test_Y))
        ]
        return await asyncio.gather(*tasks)

    async def run_all():
        nonlocal all_records
        dataset_tasks = [run_dataset(dataset_name) for dataset_name in datasets]
        dataset_results_list = await asyncio.gather(*dataset_tasks)
        for dataset_results in dataset_results_list:
            all_records.extend(dataset_results)

    asyncio.run(run_all())

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Saved {len(all_records)} tagged records to {output_path}")


if __name__ == '__main__':
    main()
