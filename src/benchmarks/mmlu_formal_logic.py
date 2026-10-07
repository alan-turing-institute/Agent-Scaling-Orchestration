"""MMLU formal logic.

Declares what the registry needs to know about this benchmark; `load`
is the loader that used to live in `src/data/`, unchanged.
"""
NAME = 'formal_logic'
ANSWER_TYPE = 'mcq'
# The set this dataset was assigned in the paper. Names index into
# personas.chosen_persona_bank().
PERSONA_SET = [
    'Logical_Formalist',
    'Commonsense_Reasoner',
    'Critical_Analyzer',
    'Eliminative_Reasoner',
    'Analogical_Thinker',
]

from datasets import load_dataset
import pandas as pd


def load_data(args, split='validation'):
    split = 'validation' if split == 'train' else 'test'
    dataset = load_dataset('cais/mmlu', 'formal_logic', cache_dir=args.data_dir)[split]
    dataset = pd.DataFrame(dataset)
    
    questions, labels = [], []
    choices = "ABCD"
    template = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n'
    for query, options, answer in zip(dataset['question'], dataset['choices'], dataset['answer']):
        if len(options) != 4 :
            continue
        question = template.format(query, options[0], options[1], options[2], options[3])
        label = f"({choices[int(answer)]})"
        questions.append(question)
        labels.append(label)

    return questions, labels


load = load_data
