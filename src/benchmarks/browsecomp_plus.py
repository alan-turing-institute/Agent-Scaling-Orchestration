"""BrowseComp-Plus: hard-to-find facts, answered by searching a fixed corpus of 100,195 web pages.

Chen et al., 2025 (texttron/BrowseComp-Plus, MIT), one of the six benchmarks of
Kim et al. (arXiv:2512.08296), where a decentralized team does best. The corpus
is fixed, so runs are reproducible and need no web access: the agent searches
it, reads documents, and submits an answer, which an LLM judge compares with
the gold answer using BrowseComp-Plus's own grader (`judge.Judge.answer_matches`).

Questions: the 100 the paper sampled (10 per category; its
`datasets/browsecomp_plus_sampled_100.json`, ybkim95/agent-scaling, MIT), in
plain text. `index` there is BrowseComp-Plus's query id.

Retrieval is BM25 (`bm25s`, pure Python, with a Snowball stemmer), built by
`scripts/build_browsecomp_index.py`. The paper's harness used a dense
retriever (Qwen3-Embedding-4B with FAISS), which needs `flash-attn` and has no
ARM build, so numbers here are BM25 numbers and not directly comparable with
the paper's. BrowseComp-Plus reports BM25 as one of its baselines. Search
returns the top 5 with the first ~2,000 characters of each, standing in for the
official 512-token snippet; `get_document` returns a document, clipped to
`DOCUMENT_CHARS` so a page cannot fill a small model's context.

Files under `<data_dir>/browsecomp-plus/` (`fetch` downloads the first three;
the index is built separately, since a full build needs 8-10 GB of memory):

    queries.json, qrel_evidence.txt, qrel_golds.txt
    corpus/data/train-0000?-of-00007.parquet     docid, text, url
    bm25/                                        the index, its docids and doc map

`BROWSECOMP_INDEX` points at another index directory (e.g. the small
evidence-only index tests use).
"""
NAME = 'browsecomp_plus'
ANSWER_TYPE = 'outcome'
# The paper assigned no persona set to this benchmark; see gpqa_diamond.py.
PERSONA_SET = []
# Needs an LLM judge (`--judge_model`).
JUDGED = True
# Turns per agent. BrowseComp-Plus's own agent allows 100; most questions need
# a handful of searches and reads, and a small model rarely uses more.
MAX_STEPS = 30
SNIPPET_CHARS = 2000
DOCUMENT_CHARS = 12000
TOP_K = 5

QUERIES_URL = ("https://raw.githubusercontent.com/ybkim95/agent-scaling/"
               "6f3bfb78a6481c1098d182680f39b0f904b292a2/datasets/browsecomp_plus_sampled_100.json")
QRELS_URL = ("https://raw.githubusercontent.com/texttron/BrowseComp-Plus/"
             "046949032b0328319cc9a02663a759ec601d9402/topics-qrels/{name}")
CORPUS_REPO = "Tevatron/browsecomp-plus-corpus"

import json
import os
import re
import threading
import urllib.request
from pathlib import Path

from benchmarks.base import Instance
from benchmarks.environment import Outcome, function_tool

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data-claude" / "benchmarks" / "browsecomp-plus"


def root_dir(data_dir=None):
    return Path(data_dir) / "browsecomp-plus" if data_dir else _DEFAULT_ROOT


def fetch(args):
    """Download the questions, the relevance judgements and the corpus. Not the index."""
    root = root_dir(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "queries.json").exists():
        urllib.request.urlretrieve(QUERIES_URL, root / "queries.json")
    for name in ("qrel_evidence.txt", "qrel_golds.txt"):
        if not (root / name).exists():
            urllib.request.urlretrieve(QRELS_URL.format(name=name), root / name)
    if len(list((root / "corpus" / "data").glob("*.parquet"))) < 7:
        from huggingface_hub import snapshot_download
        snapshot_download(CORPUS_REPO, repo_type="dataset", allow_patterns=["data/*.parquet"],
                          local_dir=str(root / "corpus"))


def qrels(root, name="qrel_evidence.txt"):
    """{query id: [docid, ...]} from a TREC qrels file."""
    found = {}
    for line in (Path(root) / name).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            found.setdefault(parts[0], []).append(parts[2])
    return found


class Retriever:
    """BM25 over the corpus, with documents read from the parquet files on demand.

    Holds the index (memory-mapped) and a map from docid to (shard, row group,
    row), never the 3.3 GB of text. One per index directory per process.
    """

    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, index_dir):
        index_dir = str(Path(index_dir).resolve())
        with cls._lock:
            if index_dir not in cls._instances:
                cls._instances[index_dir] = cls(index_dir)
            return cls._instances[index_dir]

    def __init__(self, index_dir):
        import bm25s
        import Stemmer
        index_dir = Path(index_dir)
        if not (index_dir / "docmap.json").exists():
            raise FileNotFoundError(f"no BrowseComp-Plus index at {index_dir}; build one with "
                                    f"scripts/build_browsecomp_index.py")
        self.index_dir = index_dir
        self.model = bm25s.BM25.load(str(index_dir / "bm25"), mmap=True)
        meta = json.loads((index_dir / "docmap.json").read_text())
        self.corpus_dir = Path(meta["corpus_dir"])
        self.shards = meta["shards"]
        self.docids = meta["docids"]                 # position in the index -> docid
        self.where = {d: tuple(w) for d, w in zip(meta["docids"], meta["where"])}
        self.stemmer = Stemmer.Stemmer("english")
        self._files = {}
        self._read_lock = threading.Lock()

    def search(self, query, k=TOP_K):
        import bm25s
        # Token strings, which the index maps through its own vocabulary; words it
        # has never seen score zero and are dropped below.
        tokens = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, return_ids=False,
                                show_progress=False)
        if not tokens or not tokens[0]:
            return []
        results, scores = self.model.retrieve(tokens, k=min(k, len(self.docids)), show_progress=False)
        return [(self.docids[int(i)], float(s)) for i, s in zip(results[0], scores[0]) if s > 0]

    def text(self, docid):
        import pyarrow.parquet as pq
        where = self.where.get(str(docid))
        if where is None:
            return None
        shard, group, row = where
        with self._read_lock:
            handle = self._files.get(shard)
            if handle is None:
                handle = self._files[shard] = pq.ParquetFile(self.corpus_dir / self.shards[shard])
            table = handle.read_row_group(group, columns=["text", "url"])
        return table.column("text")[row].as_py(), table.column("url")[row].as_py()


def _index_dir(root):
    return Path(os.environ.get("BROWSECOMP_INDEX") or (Path(root) / "bm25"))


def _normalise(answer):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.%-]", "", str(answer).lower())).strip()


class BrowseCompTask:
    """One BrowseComp-Plus question as an `Environment` (see benchmarks.environment)."""

    TOOLS = [
        function_tool("search", "Search the document collection. Returns the top 5 hits with their docid, "
                                "score and the opening of each document.",
                      {"query": {"type": "string", "description": "The search query"}}, ["query"]),
        function_tool("get_document", "Retrieve a document by its docid.",
                      {"docid": {"type": "string"}}, ["docid"]),
        function_tool("done", "Submit your final answer and end the task.",
                      {"answer": {"type": "string", "description": "Your succinct, exact final answer"},
                       "confidence": {"type": "integer", "description": "Your confidence, 0-100"}},
                      ["answer"]),
    ]

    def __init__(self, metadata, judge, retriever=None):
        if judge is None:
            raise ValueError("browsecomp_plus is graded by an LLM judge: pass --judge_model")
        self.metadata = metadata
        self.judge = judge
        self.retriever = retriever or Retriever.get(_index_dir(metadata.get("root") or _DEFAULT_ROOT))
        self.done = False
        self.answer = None
        self.confidence = None
        self.searches = 0

    def task_prompt(self):
        return ("The task is to answer the given question accurately by searching a collection of "
                "documents with the tools provided. When you are confident, call done with your "
                "succinct, exact answer.\n\n"
                f"Question: {self.metadata['question']}")

    def tools(self):
        return self.TOOLS

    def call(self, name, arguments):
        if self.done:
            return "The task is over."
        if name == "search":
            self.searches += 1
            hits = self.retriever.search(str(arguments.get("query", "")))
            if not hits:
                return "No documents matched."
            return json.dumps([{"docid": d, "score": round(s, 2), "snippet": self.retriever.text(d)[0][:SNIPPET_CHARS]}
                               for d, s in hits], ensure_ascii=False, indent=1)
        if name == "get_document":
            found = self.retriever.text(str(arguments.get("docid", "")).strip())
            if found is None:
                return f"ERROR: no document with docid {arguments.get('docid')!r}"
            text, url = found
            clipped = text if len(text) <= DOCUMENT_CHARS else (
                text[:DOCUMENT_CHARS] + f"\n...[{len(text) - DOCUMENT_CHARS} more characters not shown]")
            return f"url: {url}\n{clipped}"
        if name == "done":
            self.answer = str(arguments.get("answer", "")).strip()
            self.confidence = arguments.get("confidence")
            self.done = True
            return "Answer submitted."
        return f"ERROR: unknown tool {name!r}; use search, get_document or done"

    def resume(self):
        """The next agent may revise a submitted answer."""
        self.done, self.answer, self.confidence = False, None, None

    def outcome(self):
        if not self.answer:
            return Outcome(success=False, fingerprint="no_answer", detail={"searches": self.searches})
        confidence = f"{self.confidence}%" if self.confidence is not None else "100%"
        correct, verdict = self.judge.answer_matches(
            self.metadata["question"], f"Exact Answer: {self.answer}\nConfidence: {confidence}",
            self.metadata["answer"])
        return Outcome(success=correct, fingerprint="answer:" + _normalise(self.answer)[:200],
                       detail={"answer": self.answer, "confidence": self.confidence, "searches": self.searches,
                               "judge": verdict[-600:], "index": str(self.retriever.index_dir)})

    def fork(self):
        twin = BrowseCompTask(self.metadata, self.judge, self.retriever)
        twin.done, twin.answer, twin.confidence, twin.searches = self.done, self.answer, self.confidence, self.searches
        return twin


def ENVIRONMENT(instance, judge=None):
    return BrowseCompTask(instance.metadata, judge)


def load_instances(args, split='test'):
    root = root_dir(args.data_dir)
    queries = json.loads((root / "queries.json").read_text())["instances"]
    if args.data_size and args.data_size > 0:
        queries = queries[:args.data_size]
    return [
        Instance(
            question=q["problem"],
            answer=q["answer"],
            id=f"{NAME}:{q['index']}",
            tags=[f"benchmark: {NAME}", f"category: {q['category'].lower()}", "tools: 3"],
            metadata={"query_id": str(q["index"]), "question": q["problem"], "answer": q["answer"],
                      "category": q["category"], "root": str(root.resolve())},
        )
        for q in queries
    ]
