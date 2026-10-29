"""X1 stage 1: every worker run the teams need, as bare_model_baseline.py commands.

X1's teams (configs/x1_teams.json) are fixed, and every worker answers alone, so stage 1
is a set of independent runs: each persona and each specialist, in each of its thinking
modes, over its teams' questions, for every fold and sample. One model is served at a time,
so runs are listed per model. A specialist on several teams of one pool runs once over all
their questions (BigBang answers physics, chemistry, biology and law in one run); the hub
(scripts/x1_hub.py) picks out each team's.

    python scripts/x1_plan.py models                       # serving order, with run counts
    python scripts/x1_plan.py commands --model bigbang     # one command per line
    python scripts/x1_plan.py estimate                     # calls and episodes per model

Results go to <root>/stage1/<pool>/<worker>-<mode>/fold<f>/s<k>/, one bare-model run each;
a run with a holdout_summary.json is finished (the driver skips it).
"""

import argparse
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CONFIG = "configs/x1_teams.json"
ROOT = "data-claude/x1-teams"
POOLS = {"hard": "data-claude/tagged_hard", "agentic": "data-claude/tagged_agentic"}
JUDGED = ("browsecomp_plus", "finance_agent")
PERSONAS = "personas"  # the model key persona workers run on


def load_config(path=CONFIG):
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def run_dir(root, pool, worker, mode, fold, sample):
    return Path(root) / "stage1" / pool / f"{worker}-{mode}" / f"fold{fold}" / f"s{sample}"


def model_entry(config, key):
    return config["persona_model"] if key == PERSONAS else config["specialists"][key]


def _union(sources):
    seen = []
    for group in sources:
        for item in group.split(","):
            if item not in seen:
                seen.append(item)
    return ",".join(seen)


def workers(config):
    """Every (model key, worker, mode, pool, sources, persona) stage 1 runs, before folds and samples."""
    out = []
    by_specialist = {}
    for team in config["teams"]:
        for persona in team["personas"]:
            out.append({"model": PERSONAS, "worker": persona, "mode": config["persona_model"]["mode"],
                        "sampling": config["persona_model"]["sampling"], "pool": team["pool"],
                        "sources": team["sources"], "persona": persona})
        for key in team["specialists"]:
            by_specialist.setdefault((key, team["pool"]), []).append(team["sources"])
    for (key, pool), groups in by_specialist.items():
        spec = config["specialists"][key]
        for mode, sampling in spec["modes"].items():
            out.append({"model": key, "worker": key, "mode": mode, "sampling": sampling, "pool": pool,
                        "sources": _union(groups), "persona": None, "system_file": spec.get("system_file")})
    return out


def runs(config, model=None, folds=None, samples=None, root=ROOT):
    settings = config["settings"]
    folds = range(settings["folds"]) if folds is None else folds
    samples = range(settings["samples"]) if samples is None else samples
    out = []
    # Fold-major: a finished fold is a full replicate of every worker.
    for fold in folds:
        for sample in samples:
            for w in workers(config):
                if model and w["model"] != model:
                    continue
                out.append({**w, "fold": fold, "sample": sample,
                            "out_dir": str(run_dir(root, w["pool"], w["worker"], w["mode"], fold, sample))})
    return out


def command(config, run, api, python="./env/bin/python", limit=0):
    """The bare_model_baseline.py command for one run, against the model served at `api`."""
    settings = config["settings"]
    pool = settings[run["pool"]]
    entry = model_entry(config, run["model"])
    thinking = {"on": "on", "off": "off", "always": "server"}[run["mode"]]
    args = [python, "-u", "scripts/bare_model_baseline.py",
            "--dataset_path", POOLS[run["pool"]], "--model_name", entry["handle"], "--api_base_url", api,
            "--split_seed", str(settings["split_seed"]), "--n_folds", str(settings["folds"]),
            "--fold", str(run["fold"]), "--sample", str(run["sample"]),
            "--workers", str(pool["workers"]), "--test_batch_size", str(entry.get("batch_size", pool["workers"] * 3)),
            "--max_tokens", str(pool["max_tokens"]), "--request_timeout", str(settings["request_timeout"]),
            "--thinking", thinking, "--sources", run["sources"],
            "--agent_name", f"{run['worker']}-{run['mode']}", "--resume", "--out_dir", run["out_dir"]]
    args += shlex.split(config["sampling"][run["sampling"]])
    # Ministral's budget is never enforced (smoke tests), so it gets only the total cap.
    if run["mode"] == "on":
        args += ["--thinking_budget", str(pool["thinking_budget"])]
    if run["persona"]:
        args += ["--persona", run["persona"]]
    if run.get("system_file"):
        args += ["--system_file", run["system_file"]]
    if run["pool"] == "hard":
        args += ["--summary"]
    if limit:
        args += ["--limit", str(limit)]  # tests only
    if any(s.split(":")[0] in JUDGED for s in run["sources"].split(",")):
        args += ["--defer_judge"]
    return args


def serving_order(config):
    """Personas first (the most runs, on the smallest model), then specialists small to large."""
    sizes = {"qwen4b": 0, "ministral": 0, "gemma": 1, "qwen35b": 2}
    keys = sorted(config["specialists"], key=lambda k: (sizes[config["specialists"][k]["profile"]], k))
    return [PERSONAS] + keys


def question_counts(config):
    """Held-out questions per team source over all folds: the whole pool, once."""
    from datasets import load_from_disk
    sys.path.insert(0, str(REPO / "scripts"))
    from bare_model_baseline import matches_sources
    counts = {}
    for pool, path in POOLS.items():
        rows = list(load_from_disk(str(REPO / path)))
        for w in workers(config):
            if w["pool"] == pool:
                counts[(w["worker"], w["mode"], pool)] = sum(matches_sources(r, w["sources"]) for r in rows)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["models", "commands", "estimate"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--folds", default=None, help="Comma-separated; default all")
    parser.add_argument("--samples", default=None, help="Comma-separated; default all")
    parser.add_argument("--pending", action="store_true", help="commands: only runs without a summary yet")
    parser.add_argument("--limit", type=int, default=0, help="commands: first N questions per run (tests)")
    parser.add_argument("--teams", default=None, help="Comma-separated team ids; default all (tests)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.teams:
        config["teams"] = [t for t in config["teams"] if t["id"] in args.teams.split(",")]
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    samples = [int(s) for s in args.samples.split(",")] if args.samples else None

    if args.action == "models":
        tally = Counter(r["model"] for r in runs(config, folds=folds, samples=samples, root=args.root))
        for key in serving_order(config):
            entry = model_entry(config, key)
            print(f"{key}\t{entry['profile']}\t{entry.get('quant') or '-'}\t{entry['handle']}\t{tally[key]} runs")
    elif args.action == "commands":
        for run in runs(config, args.model, folds, samples, args.root):
            if args.pending and (REPO / run["out_dir"] / "holdout_summary.json").exists():
                continue
            print(shlex.join(command(config, run, args.api, limit=args.limit)))
    else:
        counts = question_counts(config)
        k = config["settings"]["samples"]
        per_model = Counter()
        for (worker, mode, pool), n in counts.items():
            key = next(w["model"] for w in workers(config) if w["worker"] == worker and w["mode"] == mode)
            per_model[(key, pool)] += n * k
            print(f"{worker}-{mode}\t{pool}\t{n} questions x {k} samples = {n * k}")
        print()
        for (key, pool), n in sorted(per_model.items(), key=lambda kv: serving_order(config).index(kv[0][0])):
            print(f"{key}\t{pool}\t{n} {'calls' if pool == 'hard' else 'episodes'}")


if __name__ == "__main__":
    main()
