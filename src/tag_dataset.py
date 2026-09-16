import argparse
import json
import os
from collections import Counter
from datasets import Dataset
import pandas as pd

from collections import Counter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_tag_frequencies(
    tags_data,
    top_n=30,
    title="Top Tag Frequencies",
    save_path="tag_counts.png",
    figsize=(10, 6),
):
    """Plots a horizontal bar chart of tag frequencies.

    Parameters:
    - tags_data: Iterable of tag lists (e.g., df['tags'] or dataset['tags'])
                 or a Counter / dictionary of {tag: count}.
    - top_n: Number of top tags to display (default: 20).
    - title: Title of the chart.
    - save_path: Optional path string to save the output plot (e.g.,
    'tag_counts.png').
    - figsize: Tuple indicating figure dimensions (width, height).
    """
    # Parse input data into a Counter object
    if isinstance(tags_data, (Counter, dict)):
        counts = Counter(tags_data)
    else:
        # Flatten nested list of tags
        all_tags = [tag for sublist in tags_data for tag in sublist]
        counts = Counter(all_tags)

    if not counts:
        print("No tags available to plot.")
        return

    # Get the top N most common tags and reverse for bottom-to-top rendering
    most_common = counts.most_common(top_n)
    tags, freqs = zip(*reversed(most_common))

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(tags, freqs, color="#3498db", edgecolor="#2980b9")

    # Add numeric count labels on the bars
    ax.bar_label(bars, padding=3, fontsize=9)

    # Labels & Title
    ax.set_xlabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tags", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{title} (Top {min(top_n, len(counts))})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # Clean up borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Graph saved to: {save_path}")

    plt.show()

# Dictionary mapping redundant or overlapping tags to unified primary tags
TAG_MAPPING = {
    # Deduplications & Variants
    "common sense": "commonsense reasoning",
    "common sense reasoning": "commonsense reasoning",
    "fact verification": "fact-checking",
    "factual verification": "fact-checking",
    "historical fact-checking": "fact-checking",
    "myth debunking": "fact-checking",
    "factual retrieval": "factual recall",
    "causal inference": "causal reasoning",
    "conceptual reasoning": "conceptual understanding",
    "critical reasoning": "critical thinking",
    "critical evaluation": "critical thinking",
    "clinical vignette analysis": "clinical reasoning",
    "medical vignette analysis": "clinical reasoning",
    "medical expertise": "medical knowledge",
    "formal reasoning": "formal logic",
    "inference rules": "formal logic",
    "logical syntax": "formal logic",
    "logical coherence": "logical consistency",
    "logical validity": "logical consistency",
    "text coherence": "logical consistency",
    "contextual coherence": "logical consistency",
    "structural consistency": "logical consistency",
    "logical deduction": "deductive reasoning",
    "step-by-step deduction": "deductive reasoning",
    "symbolic logic": "symbolic reasoning",
    "symbolic translation": "symbolic reasoning",
    "logical translation": "symbolic reasoning",
    "language understanding": "natural language understanding",
    "contextual understanding": "natural language understanding",
    "contextual inference": "natural language understanding",
    "contextual reasoning": "natural language understanding",
    "multiple choice reasoning": "multiple-choice evaluation",
    "multiple-choice reasoning": "multiple-choice evaluation",
    "option evaluation": "multiple-choice evaluation",
    "pronoun resolution": "coreference resolution",
    "arithmetic reasoning": "arithmetic",
    "percentage calculation": "arithmetic",
    "proportional reasoning": "arithmetic",
    "word problem solving": "word problem",
    "text continuation": "text completion",
    "logical continuation": "text completion",
    "sequential reasoning": "step-by-step reasoning",
    "scientific knowledge": "scientific reasoning",
    "scientific literacy": "scientific reasoning",
    "truth table construction": "truth table analysis",
    "indirect truth tables": "truth table analysis",
    "quantifier analysis": "predicate logic",
    "quantifier reasoning": "predicate logic",
    "logical connectives": "propositional logic",
    "medical diagnosis": "differential diagnosis",
}


def read_jsonl(file_path):
    """Read JSONL file and return as list of dictionaries"""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_tag_mapping(path):
    """Read the raw-tag -> canonical-tag mapping written by canonicalise_tags.py.

    Applied before the hand-written TAG_MAPPING, which then still catches any
    synonym pair the canonicaliser left apart.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[warn] no tag mapping at {path}; using the built-in TAG_MAPPING only")
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    mapping = payload.get("mapping", payload) if isinstance(payload, dict) else {}
    print(f"Loaded {len(mapping)} tag mappings from {path}")
    return mapping


def standardize_tags(tags_list, mapping=None):
    """Maps tags to unified categories and removes duplicates per item."""
    unified_tags = set()
    for tag in tags_list:
        # Data-derived mapping first, then the hand-written one
        mapped_tag = mapping.get(tag, tag) if mapping else tag
        mapped_tag = TAG_MAPPING.get(mapped_tag, mapped_tag)
        unified_tags.add(mapped_tag)
    return list(unified_tags)


def clean_tags(tags, threshold=5):
    """Only get tags that meet a threshold count"""
    all_tags = []
    for tags_list in tags:
        all_tags.extend(tags_list)

    tag_frequency = Counter(all_tags)
    valid_tags = {
        tag for tag, count in tag_frequency.items() if count >= threshold
    }

    print(f"Original unique tag count (post-mapping): {len(tag_frequency)}")
    print(
        f"Tags removed (frequency < {threshold}): {len(tag_frequency) - len(valid_tags)}"
    )
    print(f"Tags after frequency filtering: {len(valid_tags)}")

    return valid_tags


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unify tag synonyms, drop rare tags, and save the tagged HuggingFace dataset"
    )
    parser.add_argument(
        "--tags_file",
        default="data-claude/question_tags/gsm8k_arc_hellaswag_truthfulqa_winogrande_pro_medicine_formal_logic_test_100_tags.jsonl",
        help="JSONL written by tag_questions.py",
    )
    parser.add_argument(
        "--out_dir",
        default="data-claude/tagged_dataset",
        help="Where save_to_disk writes the dataset",
    )
    parser.add_argument(
        "--tag_mapping",
        default="data-claude/tag_mapping.json",
        help="JSON from canonicalise_tags.py; empty string uses only the built-in TAG_MAPPING",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Drop tags occurring fewer than this many times",
    )
    parser.add_argument(
        "--plot_path",
        default="data-claude/tag_frequencies.png",
        help="Where to write the tag frequency chart; empty string skips the plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data = read_jsonl(args.tags_file)

    # Create pandas DataFrame
    df = pd.DataFrame(data)
    df["answer"] = df["answer"].astype(str)

    # Step 1: Map raw tags to unified categories & remove duplicates within rows
    tag_mapping = load_tag_mapping(args.tag_mapping)
    df["tags"] = df["tags"].apply(lambda tags: standardize_tags(tags, tag_mapping))

    # Step 2: Filter tags by threshold frequency
    valid_tags = clean_tags(df["tags"], threshold=args.threshold)
    df["tags"] = df["tags"].apply(
        lambda tag_list: [tag for tag in tag_list if tag in valid_tags]
    )

    # Select only needed columns
    df_clean = df[["dataset", "question", "answer", "tags"]].copy()

    # Create datasets.Dataset from dataframe
    dataset = Dataset.from_pandas(df_clean)

    # Save locally
    os.makedirs(os.path.dirname(args.out_dir) or ".", exist_ok=True)
    dataset.save_to_disk(args.out_dir)
    print(f"\u2713 Dataset saved locally to: {args.out_dir}")

    # ==================== Dataset Statistics ====================

    print("\n" + "=" * 60)
    print("DATASET DESCRIPTION")
    print("=" * 60)

    print(f"\nTotal number of questions: {len(dataset)}")

    all_tags = []
    for tags_list in dataset["tags"]:
        all_tags.extend(tags_list)

    unique_tags = sorted(set(all_tags))
    print(f"Unique tags: {len(unique_tags)}")
    print(f"\nTag list:\n{unique_tags}")

    # Tag frequency
    tag_frequency = Counter(all_tags)

    print("\nTag frequency:")
    for tag, count in tag_frequency.most_common():
        print(f"  - {tag}: {count}")

    print(f"\nDatasets included: {set(dataset['dataset'])}")
    print("Questions per dataset:")
    for ds_name in set(dataset["dataset"]):
        count = dataset["dataset"].count(ds_name)
        print(f"  - {ds_name}: {count}")

    print("\nAnswer type distribution:")
    print(
        f"  - Numeric answers: {sum(isinstance(x, (int, float)) for x in dataset['answer'])}"
    )
    print(f"  - String answers: {sum(isinstance(x, str) for x in dataset['answer'])}")

    print("\n" + "=" * 60)
    print("Dataset Info:")
    print("=" * 60)
    print(dataset)
    print("\n")

    if args.plot_path:
        os.makedirs(os.path.dirname(args.plot_path) or ".", exist_ok=True)
        plot_tag_frequencies(
            df["tags"],
            top_n=25,
            title="Dataset Tag Distribution",
            save_path=args.plot_path,
        )


if __name__ == "__main__":
    main()
