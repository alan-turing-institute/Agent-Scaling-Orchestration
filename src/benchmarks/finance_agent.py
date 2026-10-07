"""Finance-Agent: analyst questions answered from SEC filings, graded against an expert rubric.

Vals AI's Finance Agent benchmark (vals-ai/finance-agent, MIT; its 50 public
questions), one of the six of Kim et al. (arXiv:2512.08296), and the one where
a centralized team clearly beats a single agent. The agent searches EDGAR,
reads filings, computes, and submits an answer; an LLM judge grades it one
rubric line at a time (`judge.Judge.criterion`).

External calls are allowed (decided 2026-10-07); `FINANCE_AGENT_ONLINE=0`
turns every tool that leaves the machine off, for an offline run. Online:

- `edgar_search` uses EDGAR's free full-text search (efts.sec.gov), not
  upstream's paid sec-api.io. The SEC requires a declared contact in the
  User-Agent: set `SEC_USER_AGENT` ("Org name contact@example.org").
  Requests are kept under 5 per second, as the SEC's 10/s limit asks.
- `parse_html_page` fetches a page and stores its text under a key, as
  upstream's does; `read_page` and `search_page` then read slices of it.
  Upstream reads stored pages with an LLM call (`retrieve_information`); here
  the agent reads them itself, so no model call hides outside its own budget.
- `web_search` queries a local SearXNG (`scripts/searxng/run.sh`, at
  `SEARXNG_URL`, default http://127.0.0.1:8888), a self-hosted metasearch
  engine with no key and no quota, standing in for upstream's Tavily. The
  tool is offered only when that server answers. SearXNG cannot cap results
  at a date, so unlike EDGAR, web results can postdate 2025-04-07.

Every response from the network is cached on disk under
`<data_dir>/finance-agent/http-cache/`, so a rerun asks the SEC nothing new.
`python` runs code in a separate interpreter with a 10 s CPU limit, 1 GB of
memory and an empty working directory; it is not network-isolated. Dates are
capped at 2025-04-07, upstream's "today".

Grading: each `correctness` criterion passes when the answer states it; the
one `contradiction` criterion per question passes when the answer does not
contradict the reference. Correct when at least half pass, the paper's rule.
The paper's grader asked "does it match?" of the contradiction line too, which
passes it exactly when the answer is wrong (so every two-line rubric scored
0.5, correct); this asks what the operator means. Each criterion's verdict is
kept in the outcome, so another threshold can be computed afterwards.
"""
NAME = 'finance_agent'
ANSWER_TYPE = 'outcome'
# The paper assigned no persona set to this benchmark; see gpqa_diamond.py.
PERSONA_SET = []
JUDGED = True
# Upstream allows 50 turns.
MAX_STEPS = 40
PASS_FRACTION = 0.5
TODAY = "2025-04-07"
PAGE_SLICE = 6000

DATA_URL = ("https://raw.githubusercontent.com/vals-ai/finance-agent/"
            "8ba65f81ab759a8e0d44e72aabc5a47cf839d563/data/public.csv")
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from benchmarks.base import Instance
from benchmarks.environment import Outcome, function_tool

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data-claude" / "benchmarks" / "finance-agent"
_RATE_LOCK = threading.Lock()
_LAST_REQUEST = [0.0]


def root_dir(data_dir=None):
    return Path(data_dir) / "finance-agent" if data_dir else _DEFAULT_ROOT


def fetch(args):
    root = root_dir(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "public.csv").exists():
        urllib.request.urlretrieve(DATA_URL, root / "public.csv")


def online():
    return os.environ.get("FINANCE_AGENT_ONLINE", "1") != "0"


OFFLINE = ("ERROR: external access is off for this run (FINANCE_AGENT_ONLINE=0). Answer from what "
           "you know.")


def sec_user_agent(root):
    """The declared contact the SEC requires: `SEC_USER_AGENT`, else `<root>/sec_user_agent.txt`.

    The file lives under the git-ignored data directory, so a contact address
    is set once per machine and never committed.
    """
    if os.environ.get("SEC_USER_AGENT"):
        return os.environ["SEC_USER_AGENT"]
    path = Path(root) / "sec_user_agent.txt"
    return path.read_text().strip() if path.exists() else None


SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888").rstrip("/")
_SEARXNG = {}


def searxng_available():
    """Whether the local SearXNG answers; asked once per process."""
    if "up" not in _SEARXNG:
        try:
            with urllib.request.urlopen(f"{SEARXNG_URL}/healthz", timeout=3) as response:
                _SEARXNG["up"] = response.status == 200
        except Exception:
            _SEARXNG["up"] = False
    return _SEARXNG["up"]


def http_get(url, root, headers=None, needs_contact=True):
    """GET with an on-disk cache and the SEC's rate limit. Returns (status, text).

    `needs_contact`: the request goes to the SEC, which requires a declared
    contact in the User-Agent; the local SearXNG does not.
    """
    cache = Path(root) / "http-cache" / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".json")
    if cache.exists():
        cached = json.loads(cache.read_text())
        return cached["status"], cached["text"]
    agent = sec_user_agent(root) or ("" if needs_contact else "agent-scaling-orchestration")
    if not agent:
        return 0, ("ERROR: set SEC_USER_AGENT (or write <data_dir>/finance-agent/sec_user_agent.txt) "
                   "to an organisation and contact email; the SEC requires it")
    with _RATE_LOCK:
        wait = 0.2 - (time.monotonic() - _LAST_REQUEST[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST[0] = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": agent, **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status, text = response.status, response.read().decode("utf-8", errors="replace")
    except Exception as error:
        return 0, f"ERROR: {type(error).__name__}: {error}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"url": url, "status": status, "text": text}))
    return status, text


class _Text(HTMLParser):
    """The visible text of an HTML page: no scripts, no styles, whitespace collapsed."""

    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_text(html):
    parser = _Text()
    parser.feed(html)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def run_python(code, timeout=10):
    """Run `code` in a fresh interpreter: CPU and memory capped, empty working directory."""
    def limits():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    with tempfile.TemporaryDirectory() as workdir:
        try:
            done = subprocess.run([sys.executable, "-I", "-c", code], cwd=workdir, capture_output=True,
                                  text=True, timeout=timeout + 2, preexec_fn=limits,
                                  env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return "ERROR: timed out"
    output = (done.stdout + (("\n" + done.stderr) if done.stderr else "")).strip()
    if done.returncode < 0:
        output += f"\nERROR: stopped by signal {-done.returncode} (the CPU or memory limit)"
    return output.strip()[:4000] or f"(no output; exit code {done.returncode})"


class FinanceTask:
    """One Finance-Agent question as an `Environment` (see benchmarks.environment)."""

    def __init__(self, metadata, judge):
        if judge is None:
            raise ValueError("finance_agent is graded by an LLM judge: pass --judge_model")
        self.metadata = metadata
        self.judge = judge
        self.root = metadata.get("root") or str(_DEFAULT_ROOT)
        self.pages = {}
        self.final = None
        self.done = False
        self.counts = {"edgar_search": 0, "parse_html_page": 0}

    def task_prompt(self):
        return ("You are a financial agent. Answer the question using the tools provided. You cannot "
                "ask for clarification. Answer as if the current date is April 07, 2025. Store pages "
                "with parse_html_page and read them with read_page or search_page. Give calculated "
                "answers to at least two decimal places and do not round intermediate steps. When you "
                "have the final answer, call submit_final_result with it, including your reasoning "
                "and sources.\n\n"
                f"Question:\n{self.metadata['question']}")

    def tools(self):
        tools = [
            function_tool("edgar_search", "Full-text search of SEC EDGAR filings (2001 onwards). Returns "
                                          "matching filings with their form, date, company and document URL.",
                          {"query": {"type": "string", "description": "Words or a quoted phrase"},
                           "forms": {"type": "string", "description": "Comma-separated form types, e.g. 10-K,10-Q"},
                           "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                           "end_date": {"type": "string", "description": "YYYY-MM-DD, at most 2025-04-07"}},
                          ["query"]),
            function_tool("parse_html_page", "Fetch a web page or filing and store its text under a key.",
                          {"url": {"type": "string"}, "key": {"type": "string"}}, ["url", "key"]),
            function_tool("read_page", f"Read up to {PAGE_SLICE} characters of a stored page from a position.",
                          {"key": {"type": "string"}, "start": {"type": "integer"}}, ["key"]),
            function_tool("search_page", "Find a word or phrase in a stored page; returns up to 5 passages "
                                         "around the matches with their positions.",
                          {"key": {"type": "string"}, "query": {"type": "string"}}, ["key", "query"]),
            function_tool("python", "Run Python code and return what it prints. Use it for calculations.",
                          {"code": {"type": "string"}}, ["code"]),
            function_tool("submit_final_result", "Submit your final answer, with reasoning and sources, and "
                                                 "end the task.",
                          {"final_result": {"type": "string"}}, ["final_result"]),
        ]
        if online() and searxng_available():
            tools.insert(0, function_tool("web_search", "Search the web. Returns titles, URLs and snippets; "
                                                        "store a page with parse_html_page to read it.",
                                          {"query": {"type": "string"}}, ["query"]))
        return tools

    def call(self, name, arguments):
        if self.done:
            return "The task is over."
        args = arguments or {}
        if name == "submit_final_result":
            self.final, self.done = str(args.get("final_result", "")).strip(), True
            return "Final result submitted."
        if name == "python":
            return run_python(str(args.get("code", "")))
        if name == "read_page":
            text = self.pages.get(str(args.get("key")))
            if text is None:
                return f"ERROR: no stored page {args.get('key')!r}; stored: {sorted(self.pages)}"
            start = max(0, int(args.get("start") or 0))
            return f"[{start}-{min(len(text), start + PAGE_SLICE)} of {len(text)}]\n{text[start:start + PAGE_SLICE]}"
        if name == "search_page":
            text = self.pages.get(str(args.get("key")))
            if text is None:
                return f"ERROR: no stored page {args.get('key')!r}; stored: {sorted(self.pages)}"
            query = str(args.get("query", "")).strip()
            hits = [m.start() for m in re.finditer(re.escape(query), text, re.IGNORECASE)][:5] if query else []
            return "\n\n".join(f"[{h}] ...{text[max(0, h - 300):h + 300]}..." for h in hits) or "No matches."
        if name in ("edgar_search", "parse_html_page", "web_search"):
            if not online():
                return OFFLINE
            self.counts[name] = self.counts.get(name, 0) + 1
            return getattr(self, "_" + name)(args)
        return f"ERROR: unknown tool {name!r}"

    def _edgar_search(self, args):
        end = min(str(args.get("end_date") or TODAY), TODAY)
        params = {"q": str(args.get("query", "")), "dateRange": "custom",
                  "startdt": str(args.get("start_date") or "2001-01-01"), "enddt": end}
        if args.get("forms"):
            params["forms"] = str(args["forms"])
        status, text = http_get(f"{EDGAR_SEARCH}?{urllib.parse.urlencode(params)}", self.root)
        if status != 200:
            return text if text.startswith("ERROR") else f"ERROR: EDGAR returned {status}"
        hits = json.loads(text).get("hits", {}).get("hits", [])[:10]
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            adsh, _, filename = hit.get("_id", "").partition(":")
            cik = (source.get("ciks") or ["0"])[0].lstrip("0") or "0"
            results.append({"company": (source.get("display_names") or [""])[0], "form": source.get("form"),
                            "filed": source.get("file_date"), "period": source.get("period_ending"),
                            "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{filename}"})
        return json.dumps(results, indent=1) if results else "No filings matched."

    def _parse_html_page(self, args):
        status, text = http_get(str(args.get("url", "")), self.root)
        if status != 200:
            return text if text.startswith("ERROR") else f"ERROR: the page returned {status}"
        key = str(args.get("key") or f"page{len(self.pages) + 1}")
        self.pages[key] = html_text(text)
        return f"Stored {len(self.pages[key])} characters under {key!r}. Stored pages: {sorted(self.pages)}"

    def _web_search(self, args):
        query = urllib.parse.urlencode({"q": str(args.get("query", "")), "format": "json", "language": "en"})
        status, text = http_get(f"{SEARXNG_URL}/search?{query}", self.root, needs_contact=False)
        if status != 200:
            return text if text.startswith("ERROR") else f"ERROR: web search returned {status}"
        results = [{"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:300],
                    "published": r.get("publishedDate")} for r in json.loads(text).get("results", [])[:8]]
        return json.dumps(results, indent=1, ensure_ascii=False) if results else "No results."

    def resume(self):
        """The next agent may revise a submitted answer; stored pages stay."""
        self.final, self.done = None, False

    def outcome(self):
        if not self.final:
            return Outcome(success=False, fingerprint="no_answer", detail=dict(self.counts))
        verdicts = []
        for item in self.metadata["rubric"]:
            passed, _ = self.judge.criterion(self.metadata["question"], self.final, item["criteria"],
                                             item["operator"])
            verdicts.append({"operator": item["operator"], "passed": passed})
        score = sum(v["passed"] for v in verdicts) / len(verdicts) if verdicts else 0.0
        digest = hashlib.sha1(re.sub(r"\s+", " ", self.final.lower()).encode("utf-8")).hexdigest()[:10]
        return Outcome(success=score >= PASS_FRACTION, fingerprint=f"answer:{digest}",
                       detail={"rubric_score": round(score, 3), "criteria": verdicts, **self.counts,
                               "answer": self.final[:2000]})

    def fork(self):
        twin = FinanceTask(self.metadata, self.judge)
        twin.pages, twin.final, twin.done, twin.counts = dict(self.pages), self.final, self.done, dict(self.counts)
        return twin


def ENVIRONMENT(instance, judge=None):
    return FinanceTask(instance.metadata, judge)


def load_instances(args, split='test'):
    import pandas as pd
    root = root_dir(args.data_dir)
    rows = pd.read_csv(root / "public.csv")
    if args.data_size and args.data_size > 0:
        rows = rows.head(args.data_size)
    instances = []
    for index, row in rows.iterrows():
        kind = re.sub(r"\s+", " ", str(row["Question Type"])).strip()
        instances.append(Instance(
            question=str(row["Question"]).strip(),
            answer=str(row["Answer"]).strip(),
            id=f"{NAME}:{index:03d}",
            tags=[f"benchmark: {NAME}", f"question type: {kind.lower()}", "tools: 6"],
            metadata={"question": str(row["Question"]).strip(), "answer": str(row["Answer"]).strip(),
                      "question_type": kind, "rubric": json.loads(row["Rubric"]),
                      "expert_minutes": float(row["Expert time (mins)"]), "root": str(root.resolve())},
        ))
    return instances
