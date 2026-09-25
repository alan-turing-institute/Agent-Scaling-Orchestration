"""Commonsense sentence completion.

Declares what the registry needs to know about this benchmark; `load`
is the loader that used to live in `src/data/`, unchanged.
"""
NAME = 'hellaswag'
ANSWER_TYPE = 'mcq'
# The set this dataset was assigned in the paper. Names index into
# model_utils.chosen_persona_bank().
PERSONA_SET = [
    'Careful_Analyst',
    'Broad_Thinker',
    'Practical_Reasoner',
    'Detail_Oriented',
    'Intuitive_Judge',
]

from datasets import load_dataset
import pandas as pd

def load_data(args, split='validation'):
    split = 'validation' if split == 'test' else split
    dataset = load_dataset('Rowan/hellaswag', cache_dir=args.data_dir)[split]
    dataset = pd.DataFrame(dataset)
    if split == 'train':
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)
    else :
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(300)

    questions, labels = [], []
    choices = "ABCD"
    template = 'Can you choose the option that best follows:\n"{}"?\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n'
    for ctx, options, answer in zip(dataset['ctx'], dataset['endings'], dataset['label']):
        if len(options) != 4 :
            continue
        question = template.format(ctx, options[0], options[1], options[2], options[3])
        label = f"({choices[int(answer)]})"
        questions.append(question)
        labels.append(label)
    
    return questions, labels

load = load_data
