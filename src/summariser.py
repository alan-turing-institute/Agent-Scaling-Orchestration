import json
import os
import tempfile
from pathlib import Path



def summariser_prompt(existing_md, summary_items):
    if existing_md:
        system_prompt = (
            "You are an assistant that updates an existing Markdown report. "
            "Update or insert ONLY machine-readable numeric results: per-agent accuracies by tag. "
            "Add in brackets the number of questions per tag if available. "
            "Do NOT add recommendations, prose, or summaries — output only the markdown that contains tables or compact numeric blocks mapping each tag to agents and their accuracies. "
            "Return ONLY the full updated markdown document (no explanation)."
        )

        user_prompt_parts = [
            "NEW EVALUATION RESULTS (JSON):",
            json.dumps(summary_items, indent=2, default=str),
            "\n---\n",
            "EXISTING MARKDOWN:",
            existing_md,
            "\n---\n",
            "Please update the existing markdown to include the new numeric results. Average any overlapping results. Add to the existing number of questions per tag in brackets if available. Do not add any prose or recommendations. Return only the updated markdown document."
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_parts)},
        ]
    else:
        # No existing markdown: generate a fresh, well-structured markdown report
        system_prompt = (
            "You are an assistant that converts evaluation results into compact numeric-only markdown. "
            "For each tag, produce a minimal table or code block that lists each agent and their numeric accuracy (as a decimal or percentage). "
            "Do NOT include recommendations, summaries, or extra prose — output only the markdown containing the numeric results."
        )

        user_prompt_parts = [
            "EVALUATION RESULTS (JSON):",
            json.dumps(summary_items, indent=2, default=str),
            "\n---\n",
            "Please generate a new markdown document containing only numeric tables or compact blocks mapping tags to per-agent accuracies. Return only the markdown document."
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_parts)},
        ]

    return messages

def save_evaluation_summary_with_llm(orchestrator, evaluations, out_md_path="out/agent_performance_by_tag.md"):
    """Use the orchestrator's LLM client to synthesize or update a markdown
    summary of which agents performed well on which tags, then write it
    atomically to the specified path. If a markdown file already exists,
    include its current content and ask the LLM to update it rather than
    always creating a fresh file.
    """
    summary_items = []
    for ev in evaluations:
        item = {
            "chosen_tag": ev.get("chosen_tag"),
            "tag_profile": ev.get("tag_profile"),
            "selected_team": ev.get("selected_team"),
            "team_accuracy": ev.get("report", {}).get("team_accuracy"),
            "per_agent_accuracy": ev.get("report", {}).get("per_agent_accuracy", {}),
            "per_agent_accuracy_by_tag": ev.get("report", {}).get("per_agent_accuracy_by_tag", {}),
        }
        summary_items.append(item)

    out_path = Path(out_md_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_md = None
    if out_path.exists():
        try:
            existing_md = out_path.read_text(encoding="utf-8")
        except Exception:
            existing_md = None
    messages = summariser_prompt(existing_md, summary_items)

    print(f"\n--- SUMMARISER MESSAGES ---\n")
    for msg in messages:
        print(f"{msg['role']}: {msg['content']}")

    md_content = None
    try:
        response = orchestrator.client.chat.completions.create(
            model=orchestrator.model_name,
            messages=messages,
            max_tokens=4096,
            temperature=0.5,
        )

        md_content = response.choices[0].message.content
    except Exception:
        # If LLM call fails, fall back to appending a simple programmatic section
        fallback = []
        fallback.append("# Agent performance summary (auto-generated fallback)")
        if existing_md:
            fallback.append(existing_md)
        fallback.append("\n## Recent evaluation additions\n")
        fallback.append("```json\n" + json.dumps(summary_items, indent=2, default=str) + "\n```")
        md_content = "\n\n".join(fallback)

    if not isinstance(md_content, str):
        md_content = str(md_content)

    # Atomic write: write to a temp file then replace
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md", dir=str(out_path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(md_content)
        os.replace(tmp_path, str(out_path))
    except Exception:
        # Best-effort fallback write
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(md_content)

    print(f"Saved agent performance summary to {out_path}")


# ---------------------------------------------------------------------------
# Deterministic scoreboard
#
# The LLM rewrite above is the loop's only memory, which makes that memory
# non-deterministic: a rewrite can drop a tag, round a number, or lose the
# question counts the numbers mean nothing without. The functions below keep
# the counts in a JSON state file and render the markdown from it, so the
# scoreboard is reproducible and the model only ever reads it.
# ---------------------------------------------------------------------------

EMPTY_STATE = {"iterations": 0, "agents": {}, "tags": {}, "teams": []}


def load_state(state_path):
    path = Path(state_path)
    if not path.exists():
        return json.loads(json.dumps(EMPTY_STATE))
    try:
        with path.open(encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception:
        print(f"[warn] could not read {state_path}; starting a fresh scoreboard")
        return json.loads(json.dumps(EMPTY_STATE))
    for key, default in EMPTY_STATE.items():
        state.setdefault(key, json.loads(json.dumps(default)))
    return state


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=path.suffix, dir=str(path.parent))
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp_path, str(path))


def update_state(state, evaluation):
    """Fold one iteration's counts into the running scoreboard state."""
    report = evaluation.get("report", {}) or {}
    team = evaluation.get("selected_team", []) or []
    n_samples = report.get("n_samples", 0) or 0

    state["iterations"] = state.get("iterations", 0) + 1

    per_agent_correct = report.get("per_agent_correct", {}) or {}
    for name in team:
        agent = state["agents"].setdefault(
            name, {"selected": 0, "questions": 0, "correct": 0}
        )
        agent["selected"] += 1
        agent["questions"] += n_samples
        agent["correct"] += per_agent_correct.get(name, 0)

    per_tag_counts = report.get("per_tag_counts", {}) or {}
    per_agent_correct_by_tag = report.get("per_agent_correct_by_tag", {}) or {}
    for tag, count in per_tag_counts.items():
        tag_state = state["tags"].setdefault(tag, {"questions": 0, "teams": 0, "agents": {}})
        tag_state["questions"] += count
        tag_state["teams"] += 1
        for name in team:
            agent = tag_state["agents"].setdefault(name, {"questions": 0, "correct": 0})
            agent["questions"] += count
            agent["correct"] += per_agent_correct_by_tag.get(tag, {}).get(name, 0)

    state["teams"].append(
        {
            "iteration": state["iterations"],
            "chosen_tag": evaluation.get("chosen_tag"),
            "team": team,
            "n_samples": n_samples,
            "team_correct": report.get("team_correct"),
            "team_accuracy": report.get("team_accuracy"),
        }
    )
    return state


def _rate(correct, seen):
    return (correct / seen) if seen else 0.0


def render_scoreboard(state, pool_names=None, recent=8, min_questions=1):
    """Render the state as the markdown the orchestrator reads next iteration."""
    lines = []
    total_questions = sum(team["n_samples"] or 0 for team in state["teams"])

    lines.append("# Agent performance by tag")
    lines.append("")
    lines.append(
        "Generated after every iteration from recorded counts. Accuracy is "
        "`correct / questions seen`; an agent only sees the questions of the "
        "batches it was selected for, so compare accuracies with the counts in view."
    )
    lines.append("")
    lines.append(
        f"Iterations: {state.get('iterations', 0)} | question-answer pairs scored: "
        f"{total_questions} | agents tried: {len(state['agents'])}"
    )
    lines.append("")

    # --- Per-agent overall -------------------------------------------------
    lines.append("## Agent totals")
    lines.append("")
    lines.append("| agent | teams | questions | correct | accuracy |")
    lines.append("|---|---:|---:|---:|---:|")
    ranked = sorted(
        state["agents"].items(),
        key=lambda kv: (-_rate(kv[1]["correct"], kv[1]["questions"]), -kv[1]["questions"], kv[0]),
    )
    for name, agent in ranked:
        lines.append(
            f"| {name} | {agent['selected']} | {agent['questions']} | {agent['correct']} | "
            f"{_rate(agent['correct'], agent['questions']):.2f} |"
        )
    if not ranked:
        lines.append("| _none yet_ | 0 | 0 | 0 | - |")
    lines.append("")

    # --- Per-tag -----------------------------------------------------------
    lines.append("## Accuracy by tag")
    lines.append("")
    lines.append("| tag | questions | agent | correct/seen | accuracy |")
    lines.append("|---|---:|---|---:|---:|")
    for tag, tag_state in sorted(
        state["tags"].items(), key=lambda kv: (-kv[1]["questions"], kv[0])
    ):
        agents = sorted(
            tag_state["agents"].items(),
            key=lambda kv: (-_rate(kv[1]["correct"], kv[1]["questions"]), -kv[1]["questions"], kv[0]),
        )
        agents = [a for a in agents if a[1]["questions"] >= min_questions]
        if not agents:
            continue
        first = True
        for name, agent in agents:
            tag_cell = f"{tag} | {tag_state['questions']}" if first else " | "
            lines.append(
                f"| {tag_cell} | {name} | {agent['correct']}/{agent['questions']} | "
                f"{_rate(agent['correct'], agent['questions']):.2f} |"
            )
            first = False
    if not state["tags"]:
        lines.append("| _none yet_ | 0 | - | - | - |")
    lines.append("")

    # --- Recent selections -------------------------------------------------
    lines.append("## Recent team selections")
    lines.append("")
    lines.append("| iteration | tag | team | team accuracy |")
    lines.append("|---:|---|---|---:|")
    for team in state["teams"][-recent:]:
        accuracy = team.get("team_accuracy")
        accuracy_cell = f"{accuracy:.2f}" if isinstance(accuracy, (int, float)) else "-"
        lines.append(
            f"| {team['iteration']} | {team.get('chosen_tag')} | "
            f"{', '.join(team.get('team', []))} | {accuracy_cell} |"
        )
    if not state["teams"]:
        lines.append("| - | - | _none yet_ | - |")
    lines.append("")

    # --- Untried agents ----------------------------------------------------
    if pool_names:
        untried = [name for name in pool_names if name not in state["agents"]]
        lines.append("## Agents not yet tried")
        lines.append("")
        lines.append(
            ", ".join(untried) if untried else "_none - every agent in the pool has been used_"
        )
        lines.append("")

    return "\n".join(lines)


def save_evaluation_summary(evaluations, out_md_path, state_path, pool_names=None, recent=8):
    """Fold one or more evaluations into the scoreboard and rewrite the markdown.

    Takes a list when the loop batches its updates, so several iterations are
    folded in order under one rewrite.
    """
    if isinstance(evaluations, dict):
        evaluations = [evaluations]

    state = load_state(state_path)
    for evaluation in evaluations:
        state = update_state(state, evaluation)

    _atomic_write(state_path, json.dumps(state, indent=2))
    markdown = render_scoreboard(state, pool_names=pool_names, recent=recent)
    _atomic_write(out_md_path, markdown)

    print(f"Scoreboard written to {out_md_path} (state: {state_path})")
    return state
