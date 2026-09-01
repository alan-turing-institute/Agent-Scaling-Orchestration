import json
import os
import tempfile
from pathlib import Path

from defaults import (
    SUMMARISER_MAX_ATTEMPTS,
    SUMMARISER_MAX_TOKENS,
    SUMMARISER_TEMPERATURE,
    THINKING_TOKEN_BUDGET,
)



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

    md_content = None
    last_error = None
    for attempt in range(1, SUMMARISER_MAX_ATTEMPTS + 1):
        try:
            response = orchestrator.client.chat.completions.create(
                model=orchestrator.model_name,
                messages=messages,
                max_tokens=SUMMARISER_MAX_TOKENS,
                extra_body={"thinking_token_budget": THINKING_TOKEN_BUDGET},
                temperature=SUMMARISER_TEMPERATURE,
            )
            content = response.choices[0].message.content
            if isinstance(content, str) and content.strip():
                md_content = content
                break
            last_error = "empty response"
        except Exception as e:
            last_error = e
        print(
            f"Summariser attempt {attempt}/{SUMMARISER_MAX_ATTEMPTS} failed: {last_error}"
        )

    # The scoreboard is the loop's only memory. Leave the previous version in
    # place rather than replacing it with a failed generation.
    if md_content is None:
        raise RuntimeError(
            f"Summariser failed after {SUMMARISER_MAX_ATTEMPTS} attempts; "
            f"{out_path} left unchanged. Last error: {last_error}"
        )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md", dir=str(out_path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(md_content)
        os.replace(tmp_path, str(out_path))
    except Exception:
        os.unlink(tmp_path)
        raise

    print(f"Saved agent performance summary to {out_path}")
