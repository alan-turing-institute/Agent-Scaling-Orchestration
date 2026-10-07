"""Read the text out of whatever `engine` handed back.

`get_agents` returns wrappers that are not related by a base class and do not
agree on a return type: `OpenAICompatChatWrapper.complete` returns a string,
`AzureOpenAIWrapper.complete` returns the raw SDK response object, and the local
HuggingFace wrappers return a decoded string.

`main.py` used to assume the response object unconditionally
(`resp.choices[0].message.content`), so the benchmark harness crashed against a
vLLM server - the only way models are served here. `team_evaluation` had a
private helper that handled all three shapes; this is that helper, shared, so
the two halves of the repository cannot disagree about it again.
"""


def response_text(resp) -> str:
    """Best-effort text from a response object, a completion, or a string."""
    if isinstance(resp, str):
        return resp
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        pass
    try:
        return resp.choices[0].text or ""
    except Exception:
        pass
    return str(resp)
