"""An LLM judge, for answers that neither a parser nor an equivalence check can grade.

BrowseComp-Plus answers are short facts in free text ("the 1987 Cannes jury",
"Marie Curie"), and Finance-Agent answers are paragraphs graded against a
rubric. Both need a model to decide whether an answer matches. The judge is a
separate model key (`--judge_model`, normally the 35B on 8002), never the agent
being graded, and its calls are counted apart from the agents' budget.

Two prompts, versioned together as `JUDGE_PROMPTS_VERSION` and recorded with
every verdict:

- `answer_matches`: BrowseComp-Plus's own grader (GRADER_TEMPLATE in
  texttron/BrowseComp-Plus scripts_evaluation/evaluate_run.py, MIT), parsed
  with its own pattern, which tolerates markdown bold.
- `criterion`: one rubric line at a time. A `correctness` criterion passes when
  the response states the fact; a `contradiction` criterion passes when the
  response does *not* contradict the reference. The paper harness asked
  "does it match?" for both, which passes a contradiction check exactly when
  the answer is wrong; this asks the question the operator names.

Verdicts are cached per prompt for the life of the process, so the same answer
judged for two stages costs one call. Temperature 0: a verdict should not
depend on a draw.
"""

from __future__ import annotations

import hashlib
import re
import threading
from functools import lru_cache
from typing import Optional, Tuple

JUDGE_PROMPTS_VERSION = "1"

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()

CRITERION_TEMPLATES = {
    "correctness": """Judge whether the following [response] to a finance [question] states the fact in [criterion].

[question]: {question}

[response]: {response}

[criterion]: {criterion}

The response passes if it states or clearly implies the criterion's fact. Numbers match if they agree to two significant figures or within 1%, whichever is more lenient. Wording may differ.

reasoning: Explain briefly whether the response states the criterion's fact.

correct: Answer 'yes' if it does. Answer 'no' otherwise.""",
    "contradiction": """Judge whether the following [response] to a finance [question] contradicts the [reference] answer.

[question]: {question}

[response]: {response}

[reference]: {criterion}

A response contradicts the reference if it asserts something incompatible with it: a different number beyond a 1% tolerance, the opposite conclusion, a different entity or date. Leaving out details is not a contradiction.

reasoning: Explain briefly whether anything in the response is incompatible with the reference.

correct: Answer 'yes' if the response does NOT contradict the reference. Answer 'no' if it does.""",
}

_VERDICT = [re.compile(r"\*\*correct:\*\*\s*(yes|no)", re.IGNORECASE),
            re.compile(r"\*\*correct\*\*:\s*(yes|no)", re.IGNORECASE),
            re.compile(r"correct:\s*(yes|no)", re.IGNORECASE)]


def parse_verdict(text: str) -> Optional[bool]:
    """`True`/`False` from a judge's reply, or `None` when it gave no verdict."""
    for pattern in _VERDICT:
        match = pattern.search(text or "")
        if match:
            return match.group(1).lower() == "yes"
    return None


class Judge:
    """Grades free-text answers with a model. Thread-safe; verdicts cached per prompt."""

    def __init__(self, client, max_tokens: int = 2048):
        self.client = client
        self.max_tokens = max_tokens
        self._cache = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def ask(self, prompt: str) -> Tuple[bool, str]:
        key = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        completion = self.client.generate([{"role": "user", "content": prompt}],
                                          max_tokens=self.max_tokens, temperature=0, top_p=1.0)
        text = completion.text
        verdict = parse_verdict(text)
        result = (bool(verdict), text)  # no verdict counts as not correct
        with self._lock:
            self.calls += 1
            self.prompt_tokens += completion.prompt_tokens or 0
            self.completion_tokens += completion.completion_tokens or 0
            self._cache[key] = result
        return result

    def answer_matches(self, question: str, response: str, correct_answer: str) -> Tuple[bool, str]:
        return self.ask(GRADER_TEMPLATE.format(question=question, response=response,
                                               correct_answer=correct_answer))

    def criterion(self, question: str, response: str, criterion: str, operator: str) -> Tuple[bool, str]:
        template = CRITERION_TEMPLATES.get(operator)
        if template is None:
            raise ValueError(f"unknown rubric operator {operator!r}; known: {sorted(CRITERION_TEMPLATES)}")
        return self.ask(template.format(question=question, response=response, criterion=criterion))


@lru_cache(maxsize=4)
def _judge(model_name: str, base_url: str, api_key: str) -> Judge:
    from model.openai_compat import OpenAICompatChatWrapper
    return Judge(OpenAICompatChatWrapper(base_url=base_url, model_name=model_name, api_key=api_key))


def judge_from_args(args) -> Optional[Judge]:
    """The run's judge, from `--judge_model` and `--judge_api_base_url`; None when not given."""
    model = getattr(args, "judge_model", None)
    if not model:
        return None
    base_url = getattr(args, "judge_api_base_url", None) or getattr(args, "api_base_url", None) \
        or "http://127.0.0.1:8002/v1"
    return _judge(model, base_url, getattr(args, "api_key", "EMPTY") or "EMPTY")
