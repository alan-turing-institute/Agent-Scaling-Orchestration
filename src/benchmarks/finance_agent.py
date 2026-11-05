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
- `web_search` searches Wikipedia (the MediaWiki Action API's full-text
  search) and Wikidata (the same search over its items), standing in for upstream's
  Tavily. Since 2026-11-05 these are the only search sources: scraped web
  engines (the old SearXNG setup, `scripts/searxng/`) broke their operators'
  rules. Wikipedia cannot cap results at a date, so unlike EDGAR, web results
  can postdate 2025-04-07.
- Pages can be fetched only from the SEC, Wikipedia and Wikidata
  (`ALLOWED_HOSTS`); every other host is refused before any request.

Rate limits hold across every process on the machine (lock files under
`RATE_DIR`, default `~/.cache/agent-scaling-rate`), so parallel runs and
parallel experiments share one budget:

- SEC: at most 5 requests a second (the SEC allows 10).
- Wikimedia (Wikipedia and Wikidata together): one request at a time and at
  least 1 s between them, so at most 60 a minute. Wikimedia's published rules
  (checked 2026-11-05): unauthenticated Action API clients get concurrency 1,
  under 5/s and 200 requests a minute with an identifying User-Agent. The
  User-Agent carries the SEC contact; responses are gzip; `maxlag=5` is sent;
  a 429 or maxlag reply pauses all Wikimedia requests for its Retry-After.

Every successful response is cached on disk under
`<data_dir>/finance-agent/http-cache/`, so a rerun asks nothing new.
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

import fcntl
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path

from benchmarks.base import Instance, local_root
from benchmarks.environment import Outcome, function_tool

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data-claude" / "benchmarks" / "finance-agent"


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


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
RATE_DIR = Path(os.environ.get("FINANCE_RATE_DIR", Path.home() / ".cache" / "agent-scaling-rate"))
SEC_GAP = 0.2         # seconds between request starts, machine-wide
WIKIMEDIA_GAP = 1.0   # and one Wikimedia request in flight at a time
RETRY_AFTER_CAP = 300


def service(host):
    """Which rate budget a host draws on: 'sec', 'wikimedia', or None (not allowed)."""
    host = (host or "").lower()
    if host == "sec.gov" or host.endswith(".sec.gov"):
        return "sec"
    if host.endswith(".wikipedia.org") or host in ("wikidata.org", "www.wikidata.org", "m.wikidata.org"):
        return "wikimedia"
    return None


ALLOWED_HOSTS = "sec.gov, *.wikipedia.org, www.wikidata.org"


@contextmanager
def throttled(name, gap, hold):
    """Wait until `gap` seconds after the last request start on budget `name`, across processes.

    An exclusive lock on `<RATE_DIR>/<name>.lock` orders every process on the machine; the
    file holds the last start time. `hold`: keep the lock for the whole request (one in flight).
    The caller may push the next start later through the yielded dict ('pause_until').
    """
    RATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(RATE_DIR / f"{name}.lock", "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            try:
                next_start = float(handle.read().strip() or 0)
            except ValueError:
                next_start = 0.0
            wait = next_start - time.time()
            if wait > 0:
                time.sleep(min(wait, RETRY_AFTER_CAP))
            state = {"pause_until": 0.0}

            def stamp():
                handle.seek(0)
                handle.truncate()
                handle.write(repr(max(time.time() + gap, state["pause_until"])))
                handle.flush()

            stamp()
            if not hold:
                fcntl.flock(handle, fcntl.LOCK_UN)
            yield state
            if state["pause_until"]:
                if not hold:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                stamp()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def retry_after(headers):
    try:
        return min(float(headers.get("Retry-After") or 30), RETRY_AFTER_CAP)
    except (TypeError, ValueError):
        return 30.0


def http_get(url, root, headers=None):
    """GET from an allowed host, with an on-disk cache and machine-wide rate limits. Returns (status, text)."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return 0, f"ERROR: not a web address: {url[:200]!r}; give a full http(s):// URL"
    budget = service(parts.hostname)
    if budget is None:
        return 0, f"ERROR: only these sites can be reached: {ALLOWED_HOSTS}"
    cache = Path(root) / "http-cache" / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".json")
    if cache.exists():
        cached = json.loads(cache.read_text())
        return cached["status"], cached["text"]
    contact = sec_user_agent(root)
    if not contact:
        return 0, ("ERROR: set SEC_USER_AGENT (or write <data_dir>/finance-agent/sec_user_agent.txt) "
                   "to an organisation and contact email; the SEC and Wikimedia require it")
    # The SEC asks for "Org contact@example.org"; Wikimedia for "client/version (contact) library".
    agent = contact if budget == "sec" else f"agent-scaling-orchestration/1.0 ({contact}) python-urllib"
    request = urllib.request.Request(url, headers={"User-Agent": agent, "Accept-Encoding": "gzip",
                                                   **(headers or {})})
    gap, hold = (SEC_GAP, False) if budget == "sec" else (WIKIMEDIA_GAP, True)
    with throttled(budget, gap, hold) as state:
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                status, raw, reply = response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            if error.code in (429, 503):
                state["pause_until"] = time.time() + retry_after(error.headers)
            return 0, f"ERROR: the site returned {error.code}; try again later or another way"
        except Exception as error:
            return 0, f"ERROR: {type(error).__name__}: {error}"
        if reply.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        text = raw.decode("utf-8", errors="replace")
        if budget == "wikimedia" and reply.get("MediaWiki-API-Error") == "maxlag":
            state["pause_until"] = time.time() + retry_after(reply)
            return 0, "ERROR: Wikipedia is busy (replication lag); try again later"
    if status == 200:
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
        self.root = str(local_root(metadata.get("root"), _DEFAULT_ROOT))
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
        if online():
            tools.insert(0, function_tool("web_search", "Search Wikipedia and Wikidata. Returns titles, URLs and "
                                                        "snippets; store a page with parse_html_page to read it.",
                                          {"query": {"type": "string"}}, ["query"]))
        return tools

    def call(self, name, arguments):
        """Run one tool. Any failure comes back as text the agent can read, never as an exception:
        a malformed argument from the model must not cost the whole episode."""
        try:
            return self._call(name, arguments)
        except Exception as error:
            return f"ERROR: {name} failed: {type(error).__name__}: {str(error)[:300]}"

    def _call(self, name, arguments):
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
        """Wikipedia full-text search (8 hits), then Wikidata's (3 items): two requests, one after the other."""
        query = str(args.get("query", "")).strip()
        if not query:
            return "ERROR: give a query"
        results = []
        status, text = http_get(WIKIPEDIA_API + "?" + urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": query, "srlimit": 8, "srprop": "snippet|timestamp",
            "format": "json", "formatversion": 2, "maxlag": 5}), self.root)
        if status != 200:
            return text if text.startswith("ERROR") else f"ERROR: Wikipedia search returned {status}"
        for hit in json.loads(text).get("query", {}).get("search", []):
            title = hit.get("title", "")
            results.append({"title": title, "source": "Wikipedia",
                            "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                            "snippet": html_text(hit.get("snippet") or "")[:300], "edited": hit.get("timestamp")})
        # Wikidata's full-text search: its label lookup (wbsearchentities) finds nothing for multi-word queries.
        status, text = http_get(WIKIDATA_API + "?" + urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": query, "srnamespace": 0, "srlimit": 3,
            "srprop": "snippet", "format": "json", "formatversion": 2, "maxlag": 5}), self.root)
        if status == 200:
            for hit in json.loads(text).get("query", {}).get("search", []):
                results.append({"title": hit.get("title"), "source": "Wikidata",
                                "url": f"https://www.wikidata.org/wiki/{hit.get('title')}",
                                "snippet": html_text(hit.get("snippet") or "")[:300]})
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
