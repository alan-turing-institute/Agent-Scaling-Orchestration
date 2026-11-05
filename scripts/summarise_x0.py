"""Summarise X0 (headroom on the hard pool) into tables, as JSON and as an HTML tab fragment.

Reads data-claude/x0-headroom/<agent>/fold<k>/bare_model/holdout_predictions.jsonl for every
fold whose holdout_summary.json exists (a fold without one is still running), and writes:

  <root>/results.json   per-agent and headroom numbers
  <root>/tab.html       the "Headroom (X0)" tab body for the results artifact, between
                        <!-- X0:START --> and <!-- X0:END -->; findings are left to the author
                        (--findings FILE inserts an HTML <ul> per section, keyed by section id)

Usage: python scripts/summarise_x0.py [--root data-claude/x0-headroom] [--findings f.json]
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

AGENTS = [  # label, short name, cost weight (active parameters, billions)
    ("qwen3.5-4b-off-s0", "4B · off", 4),
    ("qwen3.5-4b-off-s1", "4B · off, sample 2", 4),
    ("qwen3.5-4b-off-s2", "4B · off, sample 3", 4),
    ("qwen3.5-4b-on", "4B · on", 4),
    ("qwen3.6-35b-off", "35B · off", 3),
    ("qwen3.6-35b-on", "35B · on", 3),
]
# The four agents the headroom tables compare; the extra 4B samples feed the agreement table.
CORE = ["qwen3.5-4b-off-s0", "qwen3.5-4b-on", "qwen3.6-35b-off", "qwen3.6-35b-on"]
BENCHMARKS = [("gpqa_diamond", "GPQA Diamond"), ("mmlu_pro", "MMLU-Pro (hard slice)"),
              ("aime", "AIME"), ("math500", "MATH-500, levels 4–5")]
NAME = {label: name for label, name, _ in AGENTS}
LONG = ["qwen3.5-4b-on", "qwen3.6-35b-on"]  # rerun at 81,920 tokens where cut off at 30,720


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(root, labels=None):
    """{label: {fold: {question_id: row}}} for finished folds only."""
    runs = {}
    for label in labels or [l for l, _, _ in AGENTS]:
        for fold in range(5):
            d = root / label / f"fold{fold}" / "bare_model"
            if not (d / "holdout_summary.json").exists():
                continue
            rows = {}
            for line in (d / "holdout_predictions.jsonl").open():
                r = json.loads(line)
                s = r["stages"][0]
                rows[r["question_id"]] = {
                    "dataset": r["dataset"], "correct": bool(s["correct"]),
                    "tokens": s.get("completion_tokens") or 0, "cut": s.get("limit") == "max_tokens",
                    "answer": s.get("prediction"), "sampling": s.get("sampling"),
                }
            runs.setdefault(label, {})[fold] = rows
    return runs


def pct(k, n):
    return 100 * k / n if n else float("nan")


def cell(k, n, ci=False):
    if not n:
        return '<td class="num">—</td>'
    sub = f"{k}/{n}"
    if ci:
        lo, hi = wilson(k, n)
        sub += f" · {lo:.0f}–{hi:.0f}"
    return f'<td class="num">{pct(k, n):.1f}%<span class="sub">{sub}</span></td>'


def per_agent(runs):
    out = {}
    for label, folds in runs.items():
        rows = [r for f in folds.values() for r in f.values()]
        entry = {"folds": sorted(folds), "n": len(rows), "correct": sum(r["correct"] for r in rows),
                 "cut": sum(r["cut"] for r in rows),
                 "mean_tokens": sum(r["tokens"] for r in rows) / len(rows) if rows else 0,
                 "sampling": rows[0]["sampling"] if rows else None, "by": {}}
        for key, _ in BENCHMARKS:
            b = [r for r in rows if r["dataset"] == key]
            entry["by"][key] = {"n": len(b), "correct": sum(r["correct"] for r in b),
                                "cut": sum(r["cut"] for r in b),
                                "mean_tokens": sum(r["tokens"] for r in b) / len(b) if b else 0}
        out[label] = entry
    return out


def headroom(runs):
    """On the folds every core agent has finished: best single, oracle, pairwise wins."""
    folds = sorted(set.intersection(*[set(runs.get(a, {})) for a in CORE])) if all(a in runs for a in CORE) else []
    if not folds:
        return None
    qs = [(f, q) for f in folds for q in runs[CORE[0]][f]]
    right = {a: {(f, q): runs[a][f][q]["correct"] for f, q in qs} for a in CORE}
    tokens = {a: {(f, q): runs[a][f][q]["tokens"] for f, q in qs} for a in CORE}
    dataset = {(f, q): runs[CORE[0]][f][q]["dataset"] for f, q in qs}
    res = {"folds": folds, "n": len(qs), "by": {}}
    for key, _ in [("all", "All")] + BENCHMARKS:
        sel = [x for x in qs if key == "all" or dataset[x] == key]
        acc = {a: sum(right[a][x] for x in sel) for a in CORE}
        any_right = sum(any(right[a][x] for a in CORE) for x in sel)
        # The oracle's cost: the cheapest correct agent's tokens, or the cheapest agent's when none is right.
        oracle_tokens = sum(min([tokens[a][x] for a in CORE if right[a][x]] or [min(tokens[a][x] for a in CORE)])
                            for x in sel)
        res["by"][key] = {"n": len(sel), "acc": acc, "oracle": any_right,
                          "none": len(sel) - any_right, "oracle_mean_tokens": oracle_tokens / len(sel) if sel else 0}
    res["pairwise"] = {a: {b: sum(right[a][x] and not right[b][x] for x in qs) for b in CORE if b != a} for a in CORE}
    return res


def agreement(runs):
    """4B thinking off, three samples: majority, unanimity, and accuracy split by agreement."""
    labels = [f"qwen3.5-4b-off-s{i}" for i in range(3)]
    if not all(l in runs for l in labels):
        return None
    folds = sorted(set.intersection(*[set(runs[l]) for l in labels]))
    if not folds:
        return None
    qs = [(f, q) for f in folds for q in runs[labels[0]][f]]
    unan = split = 0
    unan_right = split_right_s0 = maj_right = any_right = 0
    for f, q in qs:
        rows = [runs[l][f][q] for l in labels]
        answers = [r["answer"] for r in rows]
        top = max(set(answers), key=answers.count)
        top_right = next(r["correct"] for r in rows if r["answer"] == top)
        maj_right += top_right if answers.count(top) >= 2 else rows[0]["correct"]
        any_right += any(r["correct"] for r in rows)
        if len(set(answers)) == 1:
            unan += 1
            unan_right += rows[0]["correct"]
        else:
            split += 1
            split_right_s0 += rows[0]["correct"]
    return {"folds": folds, "n": len(qs), "unanimous": unan, "unanimous_right": unan_right,
            "split": split, "split_right_s0": split_right_s0, "majority_right": maj_right,
            "any_right": any_right, "s0_right": sum(runs[labels[0]][f][q]["correct"] for f, q in qs)}


def long_rerun(runs, long):
    """Thinking on with the longer limit: answers cut off at 30,720 tokens, rerun at 81,920, on finished folds."""
    out = {}
    for label in LONG:
        folds = sorted(long.get(label, {}))
        if not folds:
            continue
        off = label.replace("-on", "-off-s0" if "4b" in label else "-off")
        old = {q: r for f in folds for q, r in runs[label][f].items()}
        new = {q: r for f in folds for q, r in long[label][f].items()}
        assert all(old[q]["cut"] for q in new), f"{label}: rerun has a question that was not cut off"
        merged = {q: new.get(q, r) for q, r in old.items()}
        out[label] = {"folds": folds, "n": len(old), "rerun": len(new),
                      "rerun_right": sum(r["correct"] for r in new.values()),
                      "still_cut": sum(r["cut"] for r in new.values()),
                      "short": sum(r["correct"] for r in old.values()),
                      "long": sum(r["correct"] for r in merged.values()),
                      "off": sum(runs[off][f][q]["correct"] for f in folds for q in runs[off][f]),
                      "off_label": off}
    return out


def section(kicker, title, caption, head, body, findings_html=""):
    return f'''  <section>
    <div class="sec-head">
      <div class="kicker">{kicker}</div>
      <h2>{title}</h2>
    </div>
    <div class="tablewrap"><table><caption>{caption}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
{findings_html}  </section>
'''


def render(agents, head, agree, findings, now, longr=None):
    th = lambda *cols: "".join(f'<th class="num">{c}</th>' if i else f"<th>{c}</th>" for i, c in enumerate(cols))
    present = [l for l, _, _ in AGENTS if l in agents]
    done = {l: len(agents[l]["folds"]) for l in present}
    fold_note = ", ".join(f"{NAME[l]} {done[l]}" for l in present) or "none yet"

    # Table 1: accuracy by agent and benchmark.
    body = ""
    for l in present:
        a = agents[l]
        body += f'<tr><td class="arm">{NAME[l]}</td><td class="num">{len(a["folds"])}</td>'
        body += "".join(cell(a["by"][k]["correct"], a["by"][k]["n"]) for k, _ in BENCHMARKS)
        body += cell(a["correct"], a["n"], ci=True) + "</tr>"
    t1 = section("Hard pool · each agent alone · pooled over its finished folds",
                 "Accuracy by agent", "Bare-model accuracy per benchmark, pooled over the folds each agent has "
                 "finished (about 156 questions per fold), with a 95% Wilson interval on the total.",
                 th("Agent", "Folds", *[n for _, n in BENCHMARKS], "All · 95% CI"), body,
                 findings.get("accuracy", ""))

    # Table 2: cost and cut-offs.
    body = ""
    for l in present:
        a = agents[l]
        body += f'<tr><td class="arm">{NAME[l]}</td>'
        body += "".join(f'<td class="num">{a["by"][k]["mean_tokens"]:,.0f}</td>' for k, _ in BENCHMARKS)
        body += f'<td class="num">{a["mean_tokens"]:,.0f}</td>'
        body += f'<td class="num">{pct(a["cut"], a["n"]):.1f}%<span class="sub">{a["cut"]}/{a["n"]}</span></td></tr>'
    t2 = section("Hard pool · cost", "Tokens per answer and cut-offs",
                 "Mean completion tokens per answer, thinking included, and the share of answers cut off at "
                 "the 30,720-token limit (scored wrong).",
                 th("Agent", *[n for _, n in BENCHMARKS], "All", "Cut off"), body, findings.get("cost", ""))

    out = t1 + t2
    if longr:
        body = ""
        for l, v in longr.items():
            body += (f'<tr><td class="arm">{NAME[l]}</td><td class="num">{", ".join(map(str, v["folds"]))}</td>'
                     f'{cell(v["rerun_right"], v["rerun"])}<td class="num">{v["still_cut"]}/{v["rerun"]}</td>'
                     f'{cell(v["short"], v["n"])}{cell(v["long"], v["n"])}{cell(v["off"], v["n"])}</tr>')
        out += section("Hard pool · thinking on with a longer limit",
                       "Cut-off answers rerun at 81,920 tokens",
                       "Answers cut off at 30,720 tokens, rerun at 81,920 on the folds the rerun has finished: "
                       "how many are now right, how many are still cut off, and the thinking-on accuracy on those "
                       "folds before and after, against the same model with thinking off.",
                       th("Agent", "Folds", "Rerun right", "Still cut off", "Thinking on, 30,720",
                          "Thinking on, 81,920", "Thinking off"), body, findings.get("long", ""))
    if head:
        h = head
        body = ""
        for key, name in [("all", "All")] + BENCHMARKS:
            b = h["by"][key]
            best = max(CORE, key=lambda a: b["acc"][a])
            cls = ' class="headline"' if key == "all" else ""
            body += f'<tr{cls}><td>{name}</td>' + "".join(cell(b["acc"][a], b["n"]) for a in CORE)
            body += cell(b["oracle"], b["n"]) + f'<td class="num">{pct(b["oracle"] - b["acc"][best], b["n"]):+.1f}</td>'
            body += f'<td class="num">{b["none"]}</td></tr>'
        out += section(f"Hard pool · four agents on the same questions · folds {', '.join(map(str, h['folds']))}",
                       "Oracle against the best single agent",
                       "Accuracy of each agent and of the oracle (right if any of the four is right) on the "
                       "folds all four have finished; the gap is oracle minus the best agent on that row.",
                       th("Benchmark", *[NAME[a] for a in CORE], "Oracle", "Gap", "None right"), body,
                       findings.get("headroom", ""))
        body = ""
        for a in CORE:
            body += f'<tr><td class="arm">{NAME[a]}</td>' + "".join(
                f'<td class="num">{h["pairwise"][a][b]}</td>' if b != a else '<td class="num">—</td>' for b in CORE) + "</tr>"
        out += section("Hard pool · pairwise", "Right where another agent is wrong",
                       f"Questions the row agent answers correctly and the column agent does not, out of {h['n']}.",
                       th("Right ↓ / wrong →", *[NAME[a] for a in CORE]), body, findings.get("pairwise", ""))
    if agree:
        g = agree
        body = (f'<tr><td>Unanimous (all three give the same answer)</td>{cell(g["unanimous"], g["n"])}'
                f'{cell(g["unanimous_right"], g["unanimous"])}</tr>'
                f'<tr><td>Split (at least two answers)</td>{cell(g["split"], g["n"])}'
                f'{cell(g["split_right_s0"], g["split"])}</tr>'
                f'<tr><td>Majority of three</td><td class="num">—</td>{cell(g["majority_right"], g["n"])}</tr>'
                f'<tr><td>Any of three right</td><td class="num">—</td>{cell(g["any_right"], g["n"])}</tr>'
                f'<tr class="headline"><td>First sample alone</td><td class="num">—</td>{cell(g["s0_right"], g["n"])}</tr>')
        out += section(f"Hard pool · 4B thinking off, three samples · folds {', '.join(map(str, g['folds']))}",
                       "Agreement among the 4B's own samples",
                       "Share of questions in each case, and accuracy of the first sample there (or of the "
                       "majority / any sample).",
                       th("Case", "Share", "Accuracy"), body, findings.get("agreement", ""))
    return out, fold_note


def page(tab_body, fold_note, findings, now, all_done):
    status = "complete" if all_done else "running"
    pill = "done" if all_done else "running"
    caveats = findings.get("caveats", "")
    return f'''<!-- X0:START -->
  <div id="x0-headroom" role="tabpanel" aria-labelledby="tabbtn-x0-headroom" hidden>

  <header class="masthead">
    <div class="eyebrow">X0 · headroom on the hard pool · bare model · from 21 Oct 2026 · updated {now}</div>
    <h1>Headroom on the hard pool</h1>
    <dl class="runmeta">
      <div><dt>Agents</dt><dd>Qwen3.5-4B · Qwen3.6-35B-A3B-NVFP4, thinking off and on</dd></div>
      <div><dt>Pool</dt><dd>hard, 780 questions</dd></div>
      <div><dt>Folds</dt><dd>5, stratified, split seed 0</dd></div>
      <div><dt>Answer limit</dt><dd>30,720 tokens</dd></div>
      <div><dt>Sampling</dt><dd>Qwen's recommended, not temperature 0</dd></div>
    </dl>
  </header>

{tab_body}
  <section>
    <div class="sec-head">
      <div class="kicker">Read the numbers against these</div>
      <h2>How X0 ran</h2>
    </div>
    <ul class="caveats">
      <li>Thinking on: temperature 1.0, top_p 0.95, top_k 20, presence penalty 1.5. Thinking off: temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5. Each result row records its settings.</li>
      <li>Not comparable with the temperature-0, 4,096-token screens in "Harder benchmarks": sampling and answer limit both differ.</li>
      <li>Thinking is switched per request on one server per model; the two models' servers ran side by side on one DGX Spark.</li>
      <li>The 4B with thinking off is sampled three times; the first sample is the one compared with the other agents.</li>
{caveats}    </ul>
  </section>

  <div class="status">
    <span class="label">Status <span class="pill {pill}">{status}</span></span>
    <p>Folds finished: {fold_note} (of 5 each).</p>
    <p>Results on disk: <code>data-claude/x0-headroom/</code>. Code on branch <code>experiments/x0-headroom</code>; issue #33.</p>
  </div>

  </div>
<!-- X0:END -->
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data-claude/x0-headroom")
    ap.add_argument("--long_root", default="data-claude/reruns/x0-long")
    ap.add_argument("--findings", default=None, help="JSON {section id: [bullet html, ...]}")
    args = ap.parse_args()
    root = Path(args.root)
    runs = load(root)
    agents, head, agree = per_agent(runs), headroom(runs), agreement(runs)
    longr = long_rerun(runs, load(Path(args.long_root), LONG))
    raw = json.loads(Path(args.findings).read_text()) if args.findings else {}
    findings = {k: ("    <ul class=\"findings\">\n" + "".join(f"      <li>{b}</li>\n" for b in v) + "    </ul>\n")
                if k != "caveats" else "".join(f"      <li>{b}</li>\n" for b in v) for k, v in raw.items()}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body, fold_note = render(agents, head, agree, findings, now, longr)
    if longr:
        fold_note += "; rerun at 81,920 tokens: " + ", ".join(f"{NAME[l]} {len(v['folds'])}" for l, v in longr.items())
    all_done = all(len(agents.get(l, {}).get("folds", [])) == 5 for l, _, _ in AGENTS)
    all_done = all_done and all(len(longr.get(l, {}).get("folds", [])) == 5 for l in LONG)
    (root / "results.json").write_text(json.dumps({"agents": agents, "headroom": head, "agreement": agree, "long": longr}, indent=2))
    (root / "tab.html").write_text(page(body, fold_note, findings, now, all_done))
    print(f"{root}/tab.html, {root}/results.json · folds: {fold_note}")


if __name__ == "__main__":
    main()
