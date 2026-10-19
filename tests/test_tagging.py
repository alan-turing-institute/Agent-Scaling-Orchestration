"""The tagger never turns a reply that ran out of tokens into tags.

    PYTHONPATH=src python tests/test_tagging.py

A reasoning model that spends its whole budget thinking returns an empty answer,
and `Completion.text` falls back to the reasoning. Parsed as tags, that prose
became ~50 fragments per question (the first hard-pool build, 2026-10-19).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.openai_compat import Completion
from tag_questions import TRUNCATED_RETRY_FACTOR, ask_for_tags, parse_tags

FAILURES = []
ARGS = SimpleNamespace(temperature=0.0, top_p=0.9)
THINKING = "Here's a thinking process:\n\n1. **Analyze User Input:**\n   - Task: assign tags, short labels"
ANSWER = '\n\n{"tags": ["math", "arithmetic"]}'


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


class Budgeted:
    """Thinks for `needs` tokens, then answers; cut off below that."""

    def __init__(self, needs):
        self.needs, self.budgets = needs, []

    def generate(self, messages, max_tokens=None, **_):
        self.budgets.append(max_tokens)
        if max_tokens < self.needs:
            return Completion(content="", reasoning=THINKING, finish_reason="length")
        return Completion(content=ANSWER, reasoning=THINKING, finish_reason="stop")


agent = Budgeted(needs=1000)
check("enough budget: answer", ask_for_tags(agent, "q", 4096, ARGS), (ANSWER, False))

agent = Budgeted(needs=1000)
text, truncated = ask_for_tags(agent, "q", 512, ARGS)
check("cut off: no text", text, "")
check("cut off: flagged", truncated, True)
check("cut off: never parsed as tags", parse_tags(text), [])
check("reasoning alone would have parsed as junk", len(parse_tags(THINKING)) > 2, True)

# tag_one_question retries a cut-off reply at TRUNCATED_RETRY_FACTOR times the budget.
import asyncio
from tag_questions import tag_one_question

agent = Budgeted(needs=1500)
instance = SimpleNamespace(question="What is 2+2?", answer="4", id="x", tags=[], metadata={})
args = SimpleNamespace(temperature=0.0, top_p=0.9, max_new_tokens=512)
record = asyncio.run(tag_one_question(agent, "gsm8k", 0, instance, args, asyncio.Semaphore(1)))
check("retry budgets", agent.budgets, [512, 512 * TRUNCATED_RETRY_FACTOR])
check("retry: tags", record["tags"], ["math", "arithmetic"])
check("retry: not truncated", record["truncated"], False)

agent = Budgeted(needs=10**6)
record = asyncio.run(tag_one_question(agent, "gsm8k", 0, instance, args, asyncio.Semaphore(1)))
check("still cut off: no tags", record["tags"], [])
check("still cut off: flagged", record["truncated"], True)

if FAILURES:
    print("FAILED:\n  " + "\n  ".join(FAILURES))
    raise SystemExit(1)
print("ok - tagging truncation tests passed")
