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

import pandas as pd
from datasets import load_dataset


def load_data(args, split='test'):
    # Both sources ship one split; every split here is the same 120 problems.
    earlier = pd.DataFrame(load_dataset('AI-MO/aimo-validation-aime', cache_dir=args.data_dir)['train'])
    latest = pd.DataFrame(load_dataset('math-ai/aime25', cache_dir=args.data_dir)['test'])
    problems = pd.concat([earlier[['problem', 'answer']], latest[['problem', 'answer']]])
    problems = problems.sample(frac=1, random_state=0).reset_index(drop=True)
    if args.data_size and args.data_size > 0:
        problems = problems.head(args.data_size)

    questions = [str(p).strip() for p in problems['problem']]
    labels = [int(str(a).strip()) for a in problems['answer']]
    return questions, labels


load = load_data
