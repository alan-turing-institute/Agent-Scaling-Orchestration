"""What a stage is asked to do, and how earlier stages' work reaches it.

A persona says *how* an agent thinks; a role says *what job* it has in the team.
The two compose: a stage's prompt is its persona, the question, the work it was
handed, then its role instruction and the answer-format suffix. Every role ends
by asking for an answer in the usual format, so every stage can be scored, which
is what lets a pipeline credit the stage that fixed an answer or broke one.

A solver with nothing handed to it gets exactly the prompt `run_team_evaluation`
has always built - persona, blank line, question and suffix - so a vote of
solvers reproduces every team run on disk. Anything else changes the prompt, so
the text below is versioned: bump `ROLE_TEMPLATES_VERSION` whenever a template,
the layout or a handoff format changes, and every record carries it.
"""

from __future__ import annotations

ROLE_TEMPLATES_VERSION = "1"

ROLES = {
    "solver": "Solve the question. The work above is for reference only; reach your own answer.",
    "debater": ("Above are answers to the same question from the team, including your own earlier "
                "answer if you gave one. Weigh their reasoning against yours, then give your "
                "updated answer."),
    "critic": ("Check the work above step by step against the question. Say plainly where it is "
               "wrong, if anywhere, then give the answer you believe is correct."),
    "reviser": ("Revise the solution above in light of the review. Keep what is right, fix what is "
                "wrong, and give the final answer."),
    "planner": ("Do not write out a full solution. Break the question into the steps needed to "
                "answer it and note what could go wrong at each, then give the answer your plan "
                "points to."),
    "checker": ("Check the solution above against the question and the plan, then give the answer "
                "you believe is correct."),
    "hub": ("You lead this team. Compare the team's answers above, check the reasoning behind "
            "them, and decide the team's answer."),
    "synthesiser": ("Combine the team's work above into one answer. Where the team disagrees, "
                    "decide whose reasoning holds."),
}

# How a stage's output is passed to the stages that read it.
HANDOFFS = ("answer", "rationale", "full")

# "rationale" keeps the end of the response, where the conclusion and the final
# answer sit; about 150 tokens of English.
RATIONALE_CHARS = 600


def render_handoff(text, prediction, handoff):
    """The part of one stage's output that the next stage gets to see."""
    if handoff == "answer":
        if prediction is not None and prediction.parsed:
            return f"Answer: {prediction.value}"
        return "Answer: (no readable answer)"
    text = text or ""
    if handoff == "rationale" and len(text) > RATIONALE_CHARS:
        return "..." + text[-RATIONALE_CHARS:]
    return text


def render_prompt(persona_text, question, suffix, role="solver", inputs=()):
    """The user message for one stage.

    `inputs` is a sequence of `(label, rendered_text)` pairs, in the order the
    stage lists them. Labels name the source's role and position, never its
    persona, so a stage judges work rather than a reputation.
    """
    if role not in ROLES:
        raise KeyError(f"unknown role {role!r}; known: {sorted(ROLES)}")

    if role == "solver" and not inputs:
        # Byte-identical to the prompt every team run on disk was given.
        if persona_text:
            return f"{persona_text}\n\n{question + suffix}"
        return question + suffix

    parts = []
    if persona_text:
        parts.append(persona_text)
    parts.append(f"Question:\n{question}")
    if inputs:
        handed = "\n\n".join(f"--- {label} ---\n{text}" for label, text in inputs)
        parts.append(f"Work from your team:\n\n{handed}")
    parts.append(ROLES[role] + suffix)
    return "\n\n".join(parts)
