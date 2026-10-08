"""Build the BM25 index BrowseComp-Plus searches (src/benchmarks/browsecomp_plus.py).

    python scripts/build_browsecomp_index.py                    # all 100,195 documents
    python scripts/build_browsecomp_index.py --subset evidence  # a small index for tests

The full build tokenizes the corpus one parquet row group at a time into a
shared vocabulary, so the text is never all in memory at once, but the token
lists are: expect 8-10 GB at peak, about 5 minutes, and ~1 GB on disk. On the
DGX Spark that memory is the same memory the vLLM servers use, so do not run a
full build while a sweep is running.

`--subset evidence` indexes only the evidence and gold documents of the
sampled questions plus `--negatives` random others: a few thousand documents,
built in seconds, for tests and smoke runs. It writes to `bm25-evidence/`;
point `BROWSECOMP_INDEX` at it to use it.

Writes `<out>/bm25/` (the bm25s index) and `<out>/docmap.json` (docid order and
each docid's shard, row group and row, so documents are read from the parquet
files on demand).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmarks.browsecomp_plus import qrels, root_dir  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", default="data-claude/benchmarks")
    parser.add_argument("--subset", choices=["full", "evidence"], default="full")
    parser.add_argument("--negatives", type=int, default=2000, help="--subset evidence: random extra documents")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Default: <root>/bm25, or <root>/bm25-evidence for the subset")
    args = parser.parse_args()

    import bm25s
    import pyarrow.parquet as pq
    import Stemmer
    from bm25s.tokenization import Tokenizer

    root = root_dir(args.data_dir)
    corpus_dir = root / "corpus" / "data"
    shards = sorted(p.name for p in corpus_dir.glob("*.parquet"))
    if len(shards) != 7:
        raise SystemExit(f"expected 7 corpus shards in {corpus_dir}, found {len(shards)}; "
                         f"run scripts/fetch_benchmarks.py browsecomp_plus")
    out = Path(args.out or (root / ("bm25" if args.subset == "full" else "bm25-evidence")))
    out.mkdir(parents=True, exist_ok=True)

    wanted = None
    if args.subset == "evidence":
        queries = {str(q["index"]) for q in json.loads((root / "queries.json").read_text())["instances"]}
        wanted = set()
        for name in ("qrel_evidence.txt", "qrel_golds.txt"):
            for qid, docids in qrels(root, name).items():
                if qid in queries:
                    wanted.update(docids)
        print(f"{len(wanted)} evidence and gold documents for {len(queries)} questions, "
              f"plus {args.negatives} random others")
        rng = random.Random(args.seed)

    tokenizer = Tokenizer(stemmer=Stemmer.Stemmer("english"), stopwords="en")
    token_ids, docids, where = [], [], []
    started = time.time()
    for shard_index, shard in enumerate(shards):
        handle = pq.ParquetFile(corpus_dir / shard)
        for group in range(handle.num_row_groups):
            table = handle.read_row_group(group, columns=["docid", "text"])
            ids, texts = table.column("docid").to_pylist(), table.column("text").to_pylist()
            keep = [i for i, d in enumerate(ids)
                    if wanted is None or d in wanted or rng.random() < args.negatives / 100195]
            if not keep:
                continue
            token_ids.extend(tokenizer.tokenize([texts[i] for i in keep], return_as="ids",
                                                update_vocab=True, show_progress=False))
            docids.extend(ids[i] for i in keep)
            where.extend([shard_index, group, i] for i in keep)
            del table, texts
        print(f"  {shard}: {len(docids)} documents so far ({time.time() - started:.0f}s)")

    model = bm25s.BM25()
    model.index((token_ids, tokenizer.get_vocab_dict()), show_progress=False)
    model.save(str(out / "bm25"))
    (out / "docmap.json").write_text(json.dumps({
        "corpus_dir": str(corpus_dir.resolve()), "shards": shards, "docids": docids, "where": where,
        "subset": args.subset, "negatives": args.negatives if wanted is not None else None,
    }))
    print(f"indexed {len(docids)} documents into {out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
