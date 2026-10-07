"""MMLU-Pro, its hardest subjects: up to ten options per question.

MMLU-Pro (Wang et al., 2024; `TIGER-Lab/MMLU-Pro`, MIT) rewrote MMLU with more
reasoning-heavy questions and ten options instead of four, which pulls chance
down to 10%. This is not the `pro_medicine` set, which is MMLU's four-option
professional medicine.

By default the set is law, engineering, physics and chemistry, the subjects
models score lowest on, with an equal number from each (`data_size` split
across them; 50 each when unset). `--sub_data` takes a comma-separated list to
choose other categories. Some questions have fewer than ten options (4, 8 or
9), so the letters run only as far as each question needs.
"""
NAME = 'mmlu_pro'
ANSWER_TYPE = 'mcq'
# The paper assigned no persona set to this benchmark; see gpqa_diamond.py.
PERSONA_SET = []

import pandas as pd
from datasets import load_dataset

from benchmarks.base import Instance

LETTERS = "ABCDEFGHIJ"
DEFAULT_CATEGORIES = ("law", "engineering", "physics", "chemistry")
DEFAULT_PER_CATEGORY = 50


def categories(args):
    chosen = [c.strip() for c in (getattr(args, 'sub_data', '') or '').split(',') if c.strip()]
    return tuple(chosen) or DEFAULT_CATEGORIES


def load_instances(args, split='test'):
    split = 'validation' if split == 'train' else 'test'
    dataset = pd.DataFrame(load_dataset('TIGER-Lab/MMLU-Pro', cache_dir=args.data_dir)[split])
    chosen = categories(args)
    unknown = sorted(set(chosen) - set(dataset['category']))
    if unknown:
        raise ValueError(f"unknown MMLU-Pro categories {unknown}; known: {sorted(set(dataset['category']))}")
    per_category = (args.data_size // len(chosen)) if args.data_size and args.data_size > 0 else DEFAULT_PER_CATEGORY

    # Each category shuffled on its own seed and cut, so adding a category does
    # not change which questions the others contribute.
    parts = [
        dataset[dataset['category'] == name].sample(frac=1, random_state=0).head(per_category)
        for name in chosen
    ]
    picked = pd.concat(parts).sample(frac=1, random_state=0).reset_index(drop=True)

    instances = []
    for _, row in picked.iterrows():
        body = "\n".join(f"({LETTERS[i]}) {str(text).strip()}" for i, text in enumerate(row['options']))
        instances.append(Instance(
            question=f"{str(row['question']).strip()}\n{body}\n\n",
            answer=f"({row['answer']})",
            id=f"{NAME}:{row['question_id']}",
            tags=[f"benchmark: {NAME}", f"subject: {row['category']}"],
            metadata={"question_id": int(row['question_id']), "category": row['category'],
                      "n_options": len(row['options']), "src": row['src']},
        ))
    return instances
