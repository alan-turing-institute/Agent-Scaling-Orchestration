"""AIME 2022-2025: competition maths with integer answers from 0 to 999.

The 90 problems from 2022-2024 (`AI-MO/aimo-validation-aime`, Apache-2.0) plus
the 30 from 2025 (`math-ai/aime25`). Every answer is an integer, so the numeric
scorer reads them as it reads GSM8K. Solutions run long, so give agents a
generous `--max_new_tokens`. With 120 problems the intervals are wide; treat it
as a stress test rather than the headline set.
"""
NAME = 'aime'
ANSWER_TYPE = 'numeric'
# The paper assigned no persona set to this benchmark; see gpqa_diamond.py.
PERSONA_SET = []

import re

import pandas as pd
from datasets import load_dataset

from benchmarks.base import Instance


def _source(url):
    """`2022_AIME_I` and problem 1 from an AoPS wiki url."""
    match = re.search(r"/(\d{4})_AIME_(I+)_Problems/Problem_(\d+)", url or "")
    return (int(match.group(1)), match.group(2), int(match.group(3))) if match else (None, None, None)


def load_instances(args, split='test'):
    # Both sources ship one split; every split here is the same 120 problems.
    earlier = pd.DataFrame(load_dataset('AI-MO/aimo-validation-aime', cache_dir=args.data_dir)['train'])
    latest = pd.DataFrame(load_dataset('math-ai/aime25', cache_dir=args.data_dir)['test'])
    earlier['year'], earlier['paper'], earlier['number'] = zip(*earlier['url'].map(_source))
    earlier['source_id'] = earlier['id'].astype(str)
    latest['year'], latest['paper'], latest['number'] = 2025, None, None
    latest['source_id'] = '2025-' + latest['id'].astype(str)
    columns = ['problem', 'answer', 'year', 'paper', 'number', 'source_id']
    problems = pd.concat([earlier[columns], latest[columns]])
    problems = problems.sample(frac=1, random_state=0).reset_index(drop=True)
    if args.data_size and args.data_size > 0:
        problems = problems.head(args.data_size)

    instances = []
    for _, row in problems.iterrows():
        year = None if pd.isna(row['year']) else int(row['year'])
        instances.append(Instance(
            question=str(row['problem']).strip(),
            answer=int(str(row['answer']).strip()),
            id=f"{NAME}:{row['source_id']}",
            tags=[f"benchmark: {NAME}"] + ([f"year: {year}"] if year else []),
            metadata={"year": year, "paper": row['paper'], "number": None if pd.isna(row['number']) else int(row['number'])},
        ))
    return instances
