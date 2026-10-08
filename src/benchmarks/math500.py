"""MATH-500, its two hardest levels: competition maths with free-form LaTeX answers.

The 500-problem MATH subset used by "Let's Verify Step by Step"
(`HuggingFaceH4/MATH-500`, MIT). Answers are LaTeX (`\\frac{7}{4}`,
`(3, \\frac{\\pi}{2})`, `\\text{even}`), so they are read from `\\boxed{}` and
compared by symbolic equivalence (the `math` scorer), not by string.

By default only levels 4 and 5 (262 problems): levels 1-3 are within reach of
the small agents and would dilute the pool. `--sub_data` takes a comma-separated
list of levels, e.g. `3,4,5`.
"""
NAME = 'math500'
ANSWER_TYPE = 'math'
# The paper assigned no persona set to this benchmark; see gpqa_diamond.py.
PERSONA_SET = []

import pandas as pd
from datasets import load_dataset

from benchmarks.base import Instance

DEFAULT_LEVELS = (4, 5)


def levels(args):
    chosen = [int(x) for x in (getattr(args, 'sub_data', '') or '').split(',') if x.strip()]
    return tuple(chosen) or DEFAULT_LEVELS


def load_instances(args, split='test'):
    # One split of 500; every split here is that set, filtered by level.
    problems = pd.DataFrame(load_dataset('HuggingFaceH4/MATH-500', cache_dir=args.data_dir)['test'])
    problems = problems[problems['level'].isin(levels(args))]
    problems = problems.sample(frac=1, random_state=0).reset_index(drop=True)
    if args.data_size and args.data_size > 0:
        problems = problems.head(args.data_size)

    return [
        Instance(
            question=str(row['problem']).strip(),
            answer=str(row['answer']).strip(),
            id=f"{NAME}:{row['unique_id']}",
            tags=[f"benchmark: {NAME}", f"subject: {str(row['subject']).lower()}", f"level: {int(row['level'])}"],
            metadata={"unique_id": row['unique_id'], "subject": row['subject'], "level": int(row['level'])},
        )
        for _, row in problems.iterrows()
    ]
