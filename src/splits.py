"""One definition of the train/test split, shared by every entry point.

Two schemes live here. `--test_fraction` with `--split_seed` draws one random
holdout, which is what the first round of experiments used; test sets drawn with
different seeds overlap, so repeating it does not give independent estimates.
`--n_folds` with `--fold` partitions the dataset instead, so across a full sweep
every question is tested exactly once and the folds are disjoint.

Every arm must use the same function, or the arms stop being comparable: the point
of the split is that a question is either trained on or tested on, identically, for
all of them.
"""

import random
from collections import defaultdict


def add_split_args(parser):
    """Attach the split flags. Shared so no entry point can drift from the others."""
    parser.add_argument("--test_fraction", type=float, default=0.2,
                        help="Fraction held out when --n_folds is 0. Must match across runs being compared")
    parser.add_argument("--split_seed", type=int, default=0,
                        help="Seeds the split, the fold partition and the held-out batching. "
                             "Keep it equal across runs being compared")
    parser.add_argument("--n_folds", type=int, default=0,
                        help="0 uses --test_fraction. Otherwise partition into this many disjoint folds "
                             "for cross validation, and hold out the one named by --fold")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold to hold out, counting from 0. Ignored unless --n_folds is set")
    return parser


def make_split(dataset, args):
    """Return (train, test) for this run.

    Folds are cut from one shuffle of the whole dataset, so `--split_seed` fixes the
    partition and `--fold` picks which part of it is held out. Two runs with the same
    seed and fold see exactly the same questions on each side.
    """
    if not getattr(args, "n_folds", 0):
        split = dataset.train_test_split(test_size=args.test_fraction, seed=args.split_seed)
        return split["train"], split["test"]

    n_folds, fold = args.n_folds, args.fold
    if not 0 <= fold < n_folds:
        raise ValueError(f"--fold must be in [0, {n_folds}), got {fold}")

    # Stratified by source dataset. An unstratified partition of 699 questions left
    # one fold with 31 gsm8k questions and another with 14, which moves an arm's
    # score by more than the differences between arms are worth - the folds would
    # have been measuring their own composition. Each source is shuffled and dealt
    # out separately, so every fold gets the same mix to within one question.
    strata = defaultdict(list)
    column = dataset["dataset"] if "dataset" in dataset.column_names else [""] * len(dataset)
    for index, source in enumerate(column):
        strata[source].append(index)

    test_indices = []
    for source in sorted(strata):
        members = strata[source]
        # Seeded per source so a stratum's deal does not depend on the others.
        random.Random(f"{args.split_seed}:{source}").shuffle(members)
        test_indices.extend(members[fold::n_folds])
    test_indices = sorted(test_indices)
    test_set = set(test_indices)
    train_indices = [i for i in range(len(dataset)) if i not in test_set]
    return dataset.select(train_indices), dataset.select(test_indices)


def split_label(args):
    """How this split should be described in output, for the record."""
    if getattr(args, "n_folds", 0):
        return f"fold {args.fold} of {args.n_folds} (split_seed {args.split_seed})"
    return f"test_fraction {args.test_fraction} (split_seed {args.split_seed})"
