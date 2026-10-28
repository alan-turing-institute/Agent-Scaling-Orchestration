"""Summarise the agentic baselines (each model alone on the agentic pool) into tables, as JSON and an HTML tab.

Reads data-claude/agentic-baselines/<agent>/fold<k>/bare_model/holdout_predictions.jsonl for every
fold whose holdout_summary.json exists, plus the 35B's 131k-context reruns of BrowseComp-Plus and
Finance-Agent under data-claude/reruns/agentic-131k/ when they exist, and writes:

  <root>/results.json   per-agent, oracle and rerun numbers
  <root>/tab.html       the "Agentic baselines" tab for the results artifact, between
                        <!-- AGENTIC:START --> and <!-- AGENTIC:END -->; --findings FILE inserts an
                        HTML <ul> per section, keyed by section id

Usage: python scripts/summarise_agentic.py [--root data-claude/agentic-baselines] [--findings f.json]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from summarise_x0 import cell, pct, section

AGENTS = [("qwen3.5-4b-off", "4B · off"), ("qwen3.5-4b-on", "4B · on"),
          ("qwen3.6-35b-off", "35B · off"), ("qwen3.6-35b-on", "35B · on")]
BENCHMARKS = [("workbench", "WorkBench"), ("plancraft", "Plancraft"),
              ("finance_agent", "Finance-Agent"), ("browsecomp_plus", "BrowseComp-Plus")]
RERUN = ["qwen3.6-35b-off", "qwen3.6-35b-on"]
RERUN_BENCHMARKS = ["finance_agent", "browsecomp_plus"]
NAME = dict(AGENTS)


def load(root, labels):
    """{label: {fold: {question_id: row}}} for finished folds only."""
    runs = {}
    for label in labels:
        for fold in range(5):
            d = root / label / f"fold{fold}" / "bare_model"
            if not (d / "holdout_summary.json").exists():
                continue
            rows = {}
            for line in (d / "holdout_predictions.jsonl").open():
                r = json.loads(line)
                rows[r["question_id"]] = {
                    "dataset": r["dataset"], "correct": bool(r["team_correct"]),
                    "tokens": r.get("completion_tokens") or 0, "calls": r.get("calls") or 0,
                    "context": r.get("limit") == "context",
                }
            runs.setdefault(label, {})[fold] = rows
    return runs


def flat(folds):
    return {q: r for f in folds.values() for q, r in f.items()}


def stats(rows):
    n = len(rows)
    return {"n": n, "correct": sum(r["correct"] for r in rows), "context": sum(r["context"] for r in rows),
            "mean_tokens": sum(r["tokens"] for r in rows) / n if n else 0,
            "mean_calls": sum(r["calls"] for r in rows) / n if n else 0}


def per_agent(runs):
    out = {}
    for label, folds in runs.items():
        rows = list(flat(folds).values())
        out[label] = {"folds": sorted(folds), **stats(rows),
                      "by": {k: stats([r for r in rows if r["dataset"] == k]) for k, _ in BENCHMARKS}}
    return out


def oracle(runs):
    """On the folds all four agents have finished: each agent's count, the oracle and the gap."""
    if not all(a in runs for a, _ in AGENTS):
        return {}
    folds = sorted(set.intersection(*[set(runs[a]) for a, _ in AGENTS]))
    rows = {a: {q: r for f in folds for q, r in runs[a][f].items()} for a, _ in AGENTS}
    qs = set(rows[AGENTS[0][0]])
    out = {"folds": folds, "by": {}}
    for key in [k for k, _ in BENCHMARKS] + ["all"]:
        sel = [q for q in qs if key == "all" or rows[AGENTS[0][0]][q]["dataset"] == key]
        acc = {a: sum(rows[a][q]["correct"] for q in sel) for a, _ in AGENTS}
        orc = sum(any(rows[a][q]["correct"] for a, _ in AGENTS) for q in sel)
        out["by"][key] = {"n": len(sel), "acc": acc, "oracle": orc}
    return out


def rerun(base, long):
    """The 35B on BrowseComp-Plus and Finance-Agent: 32k context against 131k, on the folds the rerun finished."""
    out = {}
    for label in RERUN:
        folds = sorted(long.get(label, {}))
        for key in RERUN_BENCHMARKS:
            old = [r for f in folds for r in base.get(label, {}).get(f, {}).values() if r["dataset"] == key]
            new = [r for f in folds for r in long[label][f].values() if r["dataset"] == key] if folds else []
            out[f"{label}/{key}"] = {"folds": folds, "32k": stats(old), "131k": stats(new)}
    return out


def render(agents, orc, rr, findings):
    def f(k):
        return findings.get(k, "")

    head = "<th>Agent</th><th class=\"num\">Folds</th>" + "".join(f'<th class="num">{n}</th>' for _, n in BENCHMARKS) \
        + '<th class="num">All</th>'
    body = ""
    for label, name in AGENTS:
        a = agents.get(label)
        if not a:
            continue
        body += f'<tr><td class="arm">{name}</td><td class="num">{len(a["folds"])}</td>' \
            + "".join(cell(a["by"][k]["correct"], a["by"][k]["n"]) for k, _ in BENCHMARKS) \
            + cell(a["correct"], a["n"], ci=True) + "</tr>"
    out = section("Agentic pool · each agent alone · 350 tasks", "Success rate by agent",
                  "Tasks solved per benchmark, pooled over the folds each agent has finished, with a 95% "
                  "Wilson interval on the total.", head, body, f("accuracy"))

    head = "<th>Agent</th>" + "".join(f'<th class="num">{n}</th>' for _, n in BENCHMARKS) + '<th class="num">All</th>'
    body = ""
    for label, name in AGENTS:
        a = agents.get(label)
        if not a:
            continue
        cells = [a["by"][k] for k, _ in BENCHMARKS] + [a]
        body += f'<tr><td class="arm">{name}</td>' + "".join(
            f'<td class="num">{c["mean_tokens"]:,.0f}<span class="sub">{c["mean_calls"]:.1f} calls · '
            f'{c["context"]} out of context</span></td>' for c in cells) + "</tr>"
    out += section("Cost per task", "Tokens, model calls and context limits",
                   "Mean completion tokens and model calls per task, and the number of tasks that ran out of "
                   "context (graded on the state they reached).", head, body, f("cost"))

    if orc:
        head = '<th>Benchmark</th>' + "".join(f'<th class="num">{n}</th>' for _, n in AGENTS) \
            + '<th class="num">Oracle</th><th class="num">Gap</th>'
        body = ""
        for key, name in BENCHMARKS + [("all", "All")]:
            b = orc["by"][key]
            best = max(b["acc"].values())
            body += f'<tr{" class=\"headline\"" if key == "all" else ""}><td>{name}</td>' \
                + "".join(cell(b["acc"][a], b["n"]) for a, _ in AGENTS) + cell(b["oracle"], b["n"]) \
                + f'<td class="num">+{pct(b["oracle"] - best, b["n"]):.1f}</td></tr>'
        out += section("Headroom for choosing an agent per task", "Oracle against the best single agent",
                       "Success of each agent and of the oracle (solved if any of the four solves it); the gap "
                       "is oracle minus the best agent on that row.", head, body, f("headroom"))

    if any(v["folds"] for v in rr.values()):
        head = '<th>35B</th><th class="num">Folds</th><th class="num">32k context</th>' \
               '<th class="num">131k context</th><th class="num">Out of context, 32k → 131k</th>'
        body = ""
        for label in RERUN:
            for key in RERUN_BENCHMARKS:
                v = rr[f"{label}/{key}"]
                if not v["folds"]:
                    continue
                body += f'<tr><td class="arm">{NAME[label]} · {dict(BENCHMARKS)[key]}</td>' \
                    f'<td class="num">{len(v["folds"])}</td>' + cell(v["32k"]["correct"], v["32k"]["n"]) \
                    + cell(v["131k"]["correct"], v["131k"]["n"]) \
                    + f'<td class="num">{v["32k"]["context"]} → {v["131k"]["context"]}</td></tr>'
        out += section("The 35B with more context", "BrowseComp-Plus and Finance-Agent at 131k context",
                       "Tasks solved by the 35B on the same folds with a 32k-context server (above) and a "
                       "131k-context rerun.", head, body, f("rerun"))
    return out


def page(body, findings, now, status_lines, done):
    pill, status = ("done", "complete") if done else ("running", "running")
    return f'''<!-- AGENTIC:START -->
  <div id="agentic-baselines" role="tabpanel" aria-labelledby="tabbtn-agentic-baselines" hidden>

  <header class="masthead">
    <div class="eyebrow">Agentic baselines · bare model · from 23 Oct 2026 · updated {now}</div>
    <h1>Agentic baselines</h1>
    <dl class="runmeta">
      <div><dt>Agents</dt><dd>Qwen3.5-4B · Qwen3.6-35B-A3B-NVFP4, thinking off and on</dd></div>
      <div><dt>Pool</dt><dd>agentic, 350 tasks</dd></div>
      <div><dt>Folds</dt><dd>5, stratified, split seed 0</dd></div>
      <div><dt>Per turn</dt><dd>8,192 tokens</dd></div>
      <div><dt>Sampling</dt><dd>Qwen's recommended, as X0</dd></div>
    </dl>
  </header>

{body}
  <section>
    <div class="sec-head">
      <div class="kicker">Read the numbers against these</div>
      <h2>How the baselines ran</h2>
    </div>
    <ul class="caveats">
      <li>Same sampling as X0: thinking on at temperature 1.0, top_p 0.95; thinking off at temperature 0.7, top_p 0.8; top_k 20 and presence penalty 1.5 for both. The judge (the 35B, for BrowseComp-Plus and Finance-Agent) runs at temperature 0.</li>
      <li>Context: the 4B's server allows 131k tokens; the 35B's allowed 32k for the main run. A task that runs out of context is graded on the state it reached.</li>
      <li>BrowseComp-Plus searches its fixed corpus with BM25, not the paper's dense retriever. Finance-Agent searches the live web and EDGAR, with responses cached.</li>
{findings.get("caveats", "")}    </ul>
  </section>

  <div class="status">
    <span class="label">Status <span class="pill {pill}">{status}</span></span>
{"".join(f"    <p>{s}</p>" + chr(10) for s in status_lines)}    <p>Results on disk: <code>data-claude/agentic-baselines/</code> and <code>data-claude/reruns/agentic-131k/</code>. Code on branch <code>experiments/x0-headroom</code>.</p>
  </div>

  </div>
<!-- AGENTIC:END -->
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data-claude/agentic-baselines")
    ap.add_argument("--rerun_root", default="data-claude/reruns/agentic-131k")
    ap.add_argument("--findings", default=None, help="JSON {section id: [bullet html, ...]}")
    args = ap.parse_args()
    root = Path(args.root)
    runs = load(root, [a for a, _ in AGENTS])
    long = load(Path(args.rerun_root), RERUN)
    agents, orc, rr = per_agent(runs), oracle(runs), rerun(runs, long)
    raw = json.loads(Path(args.findings).read_text()) if args.findings else {}
    findings = {k: ("    <ul class=\"findings\">\n" + "".join(f"      <li>{b}</li>\n" for b in v) + "    </ul>\n")
                if k != "caveats" else "".join(f"      <li>{b}</li>\n" for b in v) for k, v in raw.items()}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rerun_folds = {a: len(long.get(a, {})) for a in RERUN}
    main_folds = {n: len(agents[a]["folds"]) if a in agents else 0 for a, n in AGENTS}
    status = ["Main run: " + "; ".join(f"{n} {k}" for n, k in main_folds.items()) + " of 5 folds.",
              "35B at 131k context, BrowseComp-Plus and Finance-Agent: "
              + "; ".join(f"{NAME[a]} {k}" for a, k in rerun_folds.items()) + " of 5 folds."]
    done = all(k == 5 for k in main_folds.values()) and all(k == 5 for k in rerun_folds.values())
    (root / "results.json").write_text(json.dumps({"agents": agents, "oracle": orc, "rerun": rr}, indent=2))
    (root / "tab.html").write_text(page(render(agents, orc, rr, findings), findings, now, status, done))
    print(f"{root}/tab.html, {root}/results.json · " + " ".join(status))


if __name__ == "__main__":
    main()
