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
            max_tokens=8192,
            extra_body={"thinking_token_budget": 2048},
            temperature=0.5,
        )

        print(f"\n--- SUMMARISER RESPONSE ---\n{response}\n")

        md_content = response.choices[0].message.content
    except Exception as e:
        # If LLM call fails, fall back to appending a simple programmatic section
        print(f"Summariser call failed: {e}")
        # fallback = []
        # fallback.append("# Agent performance summary (auto-generated fallback)")
        # if existing_md:
        #     fallback.append(existing_md)
        # fallback.append("\n## Recent evaluation additions\n")
        # fallback.append("```json\n" + json.dumps(summary_items, indent=2, default=str) + "\n```")
        # md_content = "\n\n".join(fallback)

    if not isinstance(md_content, str):
        md_content = str(md_content)

    # Atomic write: write to a temp file then replace
    if md_content is None:
        print(f"Failed to generate markdown content. Writing fallback content to {out_path}")
    else:
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
