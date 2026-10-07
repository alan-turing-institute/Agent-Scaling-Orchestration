"""A stand-in OpenAI-compatible server, for running the entry points offline.

The real servers are often busy with a sweep, and a sweep's answers change with
whatever else is in the batch, so tests must never send them load. This server
answers `/v1/chat/completions` with plausible, deterministic replies chosen by
what the prompt is asking for:

    tagging            {"tags": [...]}, drawn from a small vocabulary
    tag clustering     {"groups": [...]}, merging tags that share a first word
    tag assignment     {"assignments": {...}}, onto an existing tag sharing a word
    team selection     {"selected_agents": [...]}, names copied from the pool listed
    an agent's answer  reasoning ending in the format its suffix asks for
    a request with     two tool calls with arguments made up from each tool's schema
      tools            (a `delegate` first when offered), then the tool that ends the
                       task (impossible, submit, done, ...), or a text reply if none

Answers are a hash of the prompt, so a rerun gives the same text and a test can
compare two runs. Accuracy is whatever the hash lands on: these runs check that
the plumbing works and the records add up, not that anything is right.

    python tests/mock_openai_server.py --port 18001

Every request is appended to `--log` (one JSON line each) when given, so a test
can count calls or inspect prompts.
"""

import argparse
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TAG_VOCABULARY = [
    "step-by-step reasoning", "arithmetic", "algebra", "formal reasoning", "physics",
    "chemistry", "quantum mechanics", "organic chemistry", "legal reasoning",
    "engineering analysis", "number theory", "combinatorics", "geometry",
    "scientific reasoning", "domain knowledge", "unit conversion",
]


def digest(text, salt=""):
    return int(hashlib.sha1((salt + text).encode("utf-8")).hexdigest(), 16)


def pick(options, text, salt="", k=1):
    """`k` distinct options chosen by hashing `text`."""
    options = list(options)
    chosen = []
    for i in range(k):
        if not options:
            break
        chosen.append(options.pop(digest(text, f"{salt}{i}") % len(options)))
    return chosen


def tag_reply(prompt):
    question = prompt.split("Question:", 1)[-1]
    return json.dumps({"tags": pick(TAG_VOCABULARY, question, "tags", k=3)})


def cluster_reply(prompt):
    tags = re.findall(r"^- (.+?) \(\d+\)$", prompt, re.M)
    by_word = {}
    for tag in tags:
        by_word.setdefault(tag.split()[0], []).append(tag)
    groups = [{"canonical": members[0], "members": members}
              for members in by_word.values() if len(members) > 1]
    return json.dumps({"groups": groups})


def assign_reply(prompt):
    existing_part, _, new_part = prompt.partition("New tags:")
    existing = re.findall(r"^- (.+)$", existing_part.split("Existing tags:", 1)[-1], re.M)
    new = re.findall(r"^- (.+)$", new_part.split("Return only", 1)[0], re.M)
    assignments = {}
    for tag in new:
        words = set(tag.split())
        match = next((e for e in existing if words & set(e.split())), None)
        assignments[tag] = match
    return json.dumps({"assignments": assignments})


def selection_reply(prompt):
    pool = re.findall(r"^\s*• ([A-Za-z0-9_]+):", prompt, re.M)
    size = re.search(r"best team of (\d+) agents", prompt)
    size = int(size.group(1)) if size else 4
    return json.dumps({"selected_agents": pick(pool, prompt, "team", k=size),
                       "reasoning": "mock selection"})


def answer_reply(prompt):
    """Reasoning, then the answer in whatever format the suffix asked for."""
    if "{final answer: (A)}" in prompt:
        letters = re.findall(r"^\(([A-J])\) ", prompt, re.M) or list("ABCD")
        answer = "{final answer: (%s)}" % pick(sorted(set(letters)), prompt, "mcq")[0]
    elif "\\boxed{" in prompt:
        answer = "\\boxed{%d}" % (digest(prompt, "boxed") % 20)
    elif "{final answer: 123}" in prompt:
        answer = "{final answer: %d}" % (digest(prompt, "num") % 1000)
    else:
        answer = "I am not sure."
    return f"Let me work through this step by step.\nSo the answer follows.\n{answer}"


FINISHING_TOOLS = ("impossible", "submit", "submit_final_result", "done", "finish", "submit_answer")


def _argument(name, schema, salt):
    if schema.get("enum"):
        return pick(schema["enum"], salt, name)[0]
    if schema.get("type") == "integer":
        return 1 + digest(salt, name) % 3
    if schema.get("type") == "number":
        return 1.0
    if schema.get("type") == "array":
        return []
    if schema.get("type") == "object":
        return {}
    if "slot" in name:
        return pick(["[I1]", "[I2]", "[A1]", "[B2]"], salt, name)[0]
    return "mock"


def tool_reply(messages, tools):
    """A tool call chosen from the conversation so far: explore twice, then finish."""
    functions = {t["function"]["name"]: t["function"] for t in tools if t.get("type") == "function"}
    made = [m for m in messages if m.get("role") == "tool"]
    delegated = any("delegate" in json.dumps(m.get("tool_calls") or "") for m in messages if m.get("role") == "assistant")
    salt = json.dumps(messages[1:2]) + str(len(made))
    if "delegate" in functions and not delegated:
        name = "delegate"
    elif len(made) < 2:
        explore = [n for n in functions if n not in FINISHING_TOOLS and n != "delegate"] or list(functions)
        name = pick(explore, salt, "explore")[0]
    else:
        finishing = [n for n in FINISHING_TOOLS if n in functions]
        if not finishing:
            return {"role": "assistant", "content": "I have finished the task."}, "stop"
        name = finishing[0]
    properties = functions[name].get("parameters", {}).get("properties", {})
    arguments = {k: _argument(k, v, salt) for k, v in properties.items()}
    call = {"id": "call_%d" % (digest(salt, name) % 10**8), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}
    return {"role": "assistant", "content": "", "tool_calls": [call]}, "tool_calls"


def reply_for(messages):
    system = " ".join(m.get("content") or "" for m in messages if m.get("role") == "system")
    prompt = (messages[-1].get("content") or "") if messages else ""
    everything = system + "\n" + prompt
    if "You are assigning tags" in everything:
        return tag_reply(prompt)
    if "Group tags that name the same capability" in everything:
        return cluster_reply(prompt)
    if "EXISTING tags" in everything:
        return assign_reply(prompt)
    if "orchestrator agent responsible for assembling teams" in everything:
        return selection_reply(prompt)
    return answer_reply(prompt)


class Handler(BaseHTTPRequestHandler):
    log_path = None
    lock = threading.Lock()

    def log_message(self, *args):  # keep test output readable
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send({"object": "list", "data": [{"id": "mock", "object": "model"}]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send({"error": "not found"}, 404)
        messages = request.get("messages", [])
        if request.get("tools"):
            message, finish = tool_reply(messages, request["tools"])
        else:
            message, finish = {"role": "assistant", "content": reply_for(messages)}, "stop"
        content = message.get("content") or json.dumps(message.get("tool_calls"))
        if Handler.log_path:
            with Handler.lock, open(Handler.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"request": request, "reply": content}) + "\n")
        prompt_tokens = sum(len(str(m.get("content") or "").split()) for m in messages)
        self._send({
            "id": "mock-%d" % digest(json.dumps(messages)),
            "object": "chat.completion",
            "model": request.get("model", "mock"),
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(content.split()),
                      "total_tokens": prompt_tokens + len(content.split())},
        })


def serve(port, log_path=None):
    Handler.log_path = log_path
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--log", default=None, help="Append every request and reply here as JSON lines")
    args = parser.parse_args()
    if args.port in (8000, 8001, 8002):
        raise SystemExit(f"port {args.port} is a real model server's; pick another")
    server = serve(args.port, args.log)
    print(f"mock OpenAI server on http://127.0.0.1:{args.port}/v1", flush=True)
    server.serve_forever()
