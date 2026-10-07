"""GPQA-Diamond: graduate-level science questions written to resist web search.

198 four-option questions in physics, chemistry and biology (Rein et al.,
2023). Gated on Hugging Face (`Idavidrein/gpqa`): accept the terms once on the
dataset page; the token comes from the local HF login or `HF_TOKEN`, never
from a file in this repository. The dataset's terms ask that questions are not
republished, so nothing here prints or stores them outside `data_dir`.

The dataset stores the correct answer and three distractors in separate
columns, so the options are shuffled here, seeded by each question's record id:
the order is the same on every load and every machine, and the correct letter
lands evenly on A-D rather than always first.
"""
NAME = 'gpqa_diamond'
ANSWER_TYPE = 'mcq'
# The paper assigned no persona set to this benchmark; `personas.paper_persona_names`
# falls back to the default set, and the canonical arm refuses it.
PERSONA_SET = []

import random

import pandas as pd
from datasets import load_dataset

from benchmarks.base import Instance

LETTERS = "ABCD"


def _options(row):
    """The four options in their fixed shuffled order, and the correct letter."""
    correct = str(row["Correct Answer"]).strip()
    options = [correct] + [str(row[f"Incorrect Answer {i}"]).strip() for i in (1, 2, 3)]
    random.Random(row["Record ID"]).shuffle(options)
    return options, LETTERS[options.index(correct)]


def load_instances(args, split='test'):
    # GPQA ships one split, `train`; every split here is that one set of 198.
    dataset = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', cache_dir=args.data_dir)['train']
    dataset = pd.DataFrame(dataset).sample(frac=1, random_state=0).reset_index(drop=True)
    if args.data_size and args.data_size > 0:
        dataset = dataset.head(args.data_size)

    instances = []
    for _, row in dataset.iterrows():
        options, letter = _options(row)
        body = "\n".join(f"({LETTERS[i]}) {text}" for i, text in enumerate(options))
        domain, subdomain = str(row['High-level domain']).strip(), str(row['Subdomain']).strip()
        instances.append(Instance(
            question=f"{str(row['Question']).strip()}\n{body}\n\n",
            answer=f"({letter})",
            id=f"{NAME}:{row['Record ID']}",
            tags=[f"benchmark: {NAME}", f"domain: {domain.lower()}", f"subdomain: {subdomain.lower()}"],
            metadata={"record_id": row['Record ID'], "domain": domain, "subdomain": subdomain},
        ))
    return instances
