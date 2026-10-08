"""Download every registered benchmark's data into `--data_dir`, and say what arrived.

Run once on a new machine, or after adding a benchmark, before tagging or any
experiment, so that a missing download or an unaccepted licence shows up here
rather than halfway through a sweep. A benchmark module may define
`fetch(args)` for data its loader does not pull itself (a repository, a corpus,
an index); otherwise its `load` is called, which fills the Hugging Face cache.

    python scripts/fetch_benchmarks.py                       # everything registered
    python scripts/fetch_benchmarks.py gpqa_diamond aime     # just these

Gated sets read the token from the local Hugging Face login or `HF_TOKEN`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmarks  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="Benchmarks to fetch (default: all registered)")
    parser.add_argument("--data_dir", default="data-claude/benchmarks")
    parser.add_argument("--data_size", type=int, default=0)
    parser.add_argument("--sub_data", default="")
    args = parser.parse_args()

    names = args.names or benchmarks.list_names()
    failed = []
    for name in names:
        try:
            benchmark = benchmarks.get(name)
            fetch = getattr(benchmark, "fetch", None)
            if fetch is not None:
                fetch(args)
            questions, labels = benchmark.load(args, split="test")
            if not questions:
                # gsm8k's loader takes head(data_size), so it returns nothing at 0;
                # the 699-question pool was built with --data_size 100.
                print(f"  warn {name:16s}     0 questions: this loader needs --data_size")
                continue
            print(f"  ok   {name:16s} {len(questions):5d} questions  ({benchmark.answer_type})")
        except Exception as error:  # report every failure, not just the first
            failed.append(name)
            print(f"  FAIL {name:16s} {type(error).__name__}: {str(error)[:160]}")
    if failed:
        raise SystemExit(f"{len(failed)} benchmark(s) failed: {failed}")


if __name__ == "__main__":
    main()
