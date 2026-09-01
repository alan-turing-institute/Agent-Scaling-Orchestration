import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SELECTIONS_DIR = os.path.join(REPO_ROOT, "results", "agent-selection")
PERSONAS_JSON = os.path.join(REPO_ROOT, "configs", "personas.json")
OUT_DIR = os.path.join(REPO_ROOT, "results", "overlap")
CSV_DIR = os.path.join(OUT_DIR, "csv")
VIZ_DIR = os.path.join(OUT_DIR, "viz")
PNG_DIR = os.path.join(OUT_DIR, "png")


# ============================================================================
# Utilities
# ============================================================================

def load_dicts(first_dict_path: str, canonical_dict_path: str):
    """Load dictionaries from JSON files"""
    with open(first_dict_path, 'r') as f:
        first_dict = json.load(f)

    with open(canonical_dict_path, 'r') as f:
        canonical_dict = json.load(f)

    return first_dict, canonical_dict


def ensure_dirs():
    for d in (CSV_DIR, VIZ_DIR, PNG_DIR):
        os.makedirs(d, exist_ok=True)


def save_csv(rows, headers, filepath):
    """Save rows to CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"✓ Saved CSV: {filepath}")


# ============================================================================
# Metrics
# ============================================================================

def calculate_overlap_metrics(first_dict: Dict, canonical_dict: Dict):
    """Calculate overlap metrics for each task"""
    metrics = {}

    for task, attempts in first_dict.items():
        canonical_agents = set(canonical_dict[task])

        precisions = []
        recalls = []

        for attempt in attempts:
            attempt_agents = set(attempt)

            # Precision
            if len(attempt_agents) > 0:
                precision = len(attempt_agents & canonical_agents) / len(attempt_agents)
            else:
                precision = 0

            # Recall
            if len(canonical_agents) > 0:
                recall = len(attempt_agents & canonical_agents) / len(canonical_agents)
            else:
                recall = 0

            precisions.append(precision)
            recalls.append(recall)

        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        metrics[task] = {
            'precision': mean_precision,
            'recall': mean_recall,
            'f1': (
                2 * mean_precision * mean_recall /
                (mean_precision + mean_recall + 1e-6)
            )
        }

    return metrics


# ============================================================================
# Visualization 1: Overlap Metrics
# ============================================================================

def visualize_overlap_metrics(first_dict: Dict, canonical_dict: Dict, fname):
    metrics = calculate_overlap_metrics(first_dict, canonical_dict)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    metric_rows = []

    for task, task_metrics in metrics.items():
        metric_rows.append([
            task,
            task_metrics["precision"],
            task_metrics["recall"],
            task_metrics["f1"]
        ])

    save_csv(
        metric_rows,
        ["task", "precision", "recall", "f1"],
        f"{CSV_DIR}/{fname}_overlap_metrics.csv"
    )

    # ------------------------------------------------------------------
    # Per-task plots
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(1, len(first_dict), figsize=(4 * len(first_dict), 4))

    if len(first_dict) == 1:
        axes = [axes]

    for idx, (task, task_metrics) in enumerate(metrics.items()):
        ax = axes[idx]

        labels = ['Precision', 'Recall', 'F1']
        values = [
            task_metrics['precision'],
            task_metrics['recall'],
            task_metrics['f1']
        ]

        bars = ax.bar(
            labels,
            values,
            color=['#3498db', '#2ecc71', '#e74c3c']
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title(f'{fname} - {task}')

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )

    plt.tight_layout()
    plt.savefig(
        f"{PNG_DIR}/1_per_task_overlap_metrics.png",
        dpi=300,
        bbox_inches='tight'
    )
    print("✓ Saved: 1_per_task_overlap_metrics.png")
    plt.close()

    # ------------------------------------------------------------------
    # Global plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = list(metrics.keys())
    precisions = [metrics[t]['precision'] for t in tasks]
    recalls = [metrics[t]['recall'] for t in tasks]
    f1s = [metrics[t]['f1'] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.25

    ax.bar(x - width, precisions, width,
           label='Precision', color='#3498db')
    ax.bar(x, recalls, width,
           label='Recall', color='#2ecc71')
    ax.bar(x + width, f1s, width,
           label='F1', color='#e74c3c')

    ax.set_ylabel('Score')
    ax.set_title(f'{fname} - Agent Selection Overlap Across All Tasks')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        f"{VIZ_DIR}/{fname}_1_global_overlap_metrics.pdf",
        bbox_inches='tight'
    )
    print(f"✓ Saved: {fname}_1_global_overlap_metrics.pdf")
    plt.close()


# ============================================================================
# Visualization 2: Agent Selection Counts
# ============================================================================

def visualize_agent_selection_barchart(first_dict: Dict,
                                       canonical_dict: Dict,
                                       fname):

    selection_rows = []

    num_tasks = len(first_dict)
    num_cols = (num_tasks + 1) // 2

    fig, axes = plt.subplots(
        2,
        num_cols,
        figsize=(6 * num_cols, 10)
    )

    axes = axes.flatten()

    for idx, (task, attempts) in enumerate(first_dict.items()):

        canonical_agents_set = set(canonical_dict[task])

        all_picked_agents = []
        for attempt in attempts:
            all_picked_agents.extend(attempt)

        agent_counts = Counter(all_picked_agents)

        agents = sorted(agent_counts.keys())
        counts = [agent_counts[a] for a in agents]

        # CSV rows
        for agent in agents:
            selection_rows.append([
                task,
                agent,
                agent_counts[agent],
                agent in canonical_agents_set
            ])

        colors = [
            '#2ecc71' if agent in canonical_agents_set
            else '#e74c3c'
            for agent in agents
        ]

        ax = axes[idx]

        bars = ax.bar(
            agents,
            counts,
            color=colors,
            edgecolor='black',
            alpha=0.8
        )

        ax.set_ylabel('Number of Times Picked')
        ax.set_title(f'{fname} - {task}')
        ax.set_xlabel('Agent')

        if len(agents) > 5:
            plt.setp(
                ax.xaxis.get_majorticklabels(),
                rotation=45,
                ha='right'
            )

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    save_csv(
        selection_rows,
        ["task", "agent", "count", "is_canonical"],
        f"{CSV_DIR}/{fname}_agent_selection_counts.csv"
    )

    for idx in range(num_tasks, len(axes)):
        axes[idx].set_visible(False)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='#2ecc71',
              edgecolor='black',
              label='Canonical'),
        Patch(facecolor='#e74c3c',
              edgecolor='black',
              label='Non-canonical')
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        f"{VIZ_DIR}/{fname}_2_per_task_agent_selection_barchart.pdf",
        bbox_inches='tight'
    )

    print(f"✓ Saved: {fname}_2_per_task_agent_selection_barchart.pdf")
    plt.close()

    # ------------------------------------------------------------------
    # Global counts
    # ------------------------------------------------------------------

    all_agents_global = set()

    for task, attempts in first_dict.items():
        all_agents_global.update(canonical_dict[task])

        for attempt in attempts:
            all_agents_global.update(attempt)

    all_agents_global = sorted(all_agents_global)

    correct_counts = {agent: 0 for agent in all_agents_global}
    incorrect_counts = {agent: 0 for agent in all_agents_global}

    for task, attempts in first_dict.items():

        canonical_agents_set = set(canonical_dict[task])

        for attempt in attempts:
            for agent in attempt:

                if agent in canonical_agents_set:
                    correct_counts[agent] += 1
                else:
                    incorrect_counts[agent] += 1

    # CSV export

    global_rows = []

    for agent in all_agents_global:
        global_rows.append([
            agent,
            correct_counts[agent],
            incorrect_counts[agent],
            correct_counts[agent] + incorrect_counts[agent]
        ])

    save_csv(
        global_rows,
        ["agent", "correct_count", "incorrect_count", "total_count"],
        f"{CSV_DIR}/{fname}_global_agent_counts.csv"
    )

    correct = [correct_counts[a] for a in all_agents_global]
    incorrect = [incorrect_counts[a] for a in all_agents_global]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_agents_global))

    ax.bar(
        x,
        correct,
        0.8,
        label='Picked Correctly',
        color='#2ecc71',
        edgecolor='black',
        alpha=0.8
    )

    ax.bar(
        x,
        incorrect,
        0.8,
        bottom=correct,
        label='Picked Incorrectly',
        color='#e74c3c',
        edgecolor='black',
        alpha=0.8
    )

    ax.set_ylabel('Number of Times Picked')
    ax.set_xlabel('Agent')
    ax.set_title(
        f'{fname} - Global Agent Selection Count '
        f'(Stacked by Correctness)'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        all_agents_global,
        rotation=45,
        ha='right'
    )

    ax.legend(loc='upper right')

    for i, agent in enumerate(all_agents_global):
        total = correct[i] + incorrect[i]

        ax.text(
            i,
            total,
            f'{int(total)}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    plt.tight_layout()

    plt.savefig(
        f"{VIZ_DIR}/{fname}_2_global_agent_selection_barchart.pdf",
        bbox_inches='tight'
    )

    print(f"✓ Saved: {fname}_2_global_agent_selection_barchart.pdf")
    plt.close()


# ============================================================================
# Visualization 3: Overlap Distribution
# ============================================================================

def visualize_overlap_distribution(first_dict: Dict,
                                   canonical_dict: Dict,
                                   fname):

    all_overlaps_per_task = {}

    max_overlap = 0
    max_frequency = 0

    for task, attempts in first_dict.items():

        canonical_agents = set(canonical_dict[task])

        overlaps = [
            len(set(attempt) & canonical_agents)
            for attempt in attempts
        ]

        all_overlaps_per_task[task] = overlaps

        max_overlap = max(
            max_overlap,
            max(overlaps) if overlaps else 0
        )

        max_frequency = max(
            max_frequency,
            max(Counter(overlaps).values()) if overlaps else 0
        )

    # ------------------------------------------------------------------
    # Raw overlap CSV
    # ------------------------------------------------------------------

    raw_rows = []

    for task, overlaps in all_overlaps_per_task.items():
        for attempt_idx, overlap in enumerate(overlaps):
            raw_rows.append([
                task,
                attempt_idx,
                overlap
            ])

    save_csv(
        raw_rows,
        ["task", "attempt_id", "correct_agent_overlap"],
        f"{CSV_DIR}/{fname}_overlaps_raw.csv"
    )

    # ------------------------------------------------------------------
    # Histogram CSV
    # ------------------------------------------------------------------

    histogram_rows = []

    for task, overlaps in all_overlaps_per_task.items():

        overlap_counts = Counter(overlaps)

        for overlap_size in range(max_overlap + 1):
            histogram_rows.append([
                task,
                overlap_size,
                overlap_counts.get(overlap_size, 0)
            ])

    save_csv(
        histogram_rows,
        ["task", "overlap_size", "frequency"],
        f"{CSV_DIR}/{fname}_overlaps_histogram.csv"
    )

    # ------------------------------------------------------------------
    # Per-task histograms
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(first_dict),
        figsize=(5 * len(first_dict), 4)
    )

    if len(first_dict) == 1:
        axes = [axes]

    for idx, (task, overlaps) in enumerate(all_overlaps_per_task.items()):

        ax = axes[idx]

        ax.hist(
            overlaps,
            bins=range(0, max_overlap + 2),
            color='#3498db',
            edgecolor='black',
            alpha=0.7
        )

        ax.set_xlabel('Number of Correct Agents')
        ax.set_ylabel('Frequency')

        ax.set_title(
            f'{fname} - {task} '
            f'(Canonical: {len(canonical_dict[task])} agents)'
        )

        ax.set_xlim(-0.5, max_overlap + 0.5)
        ax.set_ylim(0, max_frequency * 1.1)

    plt.tight_layout()

    plt.savefig(
        f"{VIZ_DIR}/{fname}_3_per_task_overlap_distribution.pdf",
        bbox_inches='tight'
    )

    print(f"✓ Saved: {fname}_3_per_task_overlap_distribution.pdf")
    plt.close()

    # ------------------------------------------------------------------
    # Global comparison histogram
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(max_overlap + 1)
    width = 0.15

    tasks = list(all_overlaps_per_task.keys())

    for task_idx, task in enumerate(tasks):

        overlaps = all_overlaps_per_task[task]

        overlap_counts = Counter(overlaps)

        counts = [
            overlap_counts.get(i, 0)
            for i in range(max_overlap + 1)
        ]

        ax.bar(
            x + task_idx * width,
            counts,
            width,
            label=task
        )

    ax.set_xlabel('Number of Correct Agents')
    ax.set_ylabel('Frequency')

    ax.set_title(
        f'{fname} - Agent Selection Overlap Distribution Across All Tasks'
    )

    ax.set_xticks(
        x + width * (len(tasks) - 1) / 2
    )

    ax.set_xticklabels(
        [str(i) for i in range(max_overlap + 1)]
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        f"{PNG_DIR}/3_global_overlap_distribution.png",
        dpi=300,
        bbox_inches='tight'
    )

    print("✓ Saved: 3_global_overlap_distribution.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():

    ensure_dirs()

    model_name = "gpt-4.1"
    agent_desc = "single"

    fname = f"{model_name}_{agent_desc}"

    first_dict, canonical_dict = load_dicts(
        f"{SELECTIONS_DIR}/{model_name}-agents=4-{agent_desc}-chosen-agents.json",
        PERSONAS_JSON,
    )

    print("Loaded dictionaries:")
    print(f"  first_dict: {len(first_dict)} tasks")
    print(f"  canonical_dict: {len(canonical_dict)} tasks")
    print()

    print("Generating visualizations and CSVs...")

    visualize_overlap_metrics(
        first_dict,
        canonical_dict,
        fname
    )

    visualize_agent_selection_barchart(
        first_dict,
        canonical_dict,
        fname
    )

    visualize_overlap_distribution(
        first_dict,
        canonical_dict,
        fname
    )

    print("\nAll visualizations complete!")
    print(f"CSV files written to {CSV_DIR}")
    print(f"Plots written to {VIZ_DIR}")


if __name__ == "__main__":
    main()
