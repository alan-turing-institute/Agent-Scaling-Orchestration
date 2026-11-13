"""Results page: the "Frontier model" tab (Claude via headless Claude Code), from frontier_summary.py's JSON.

    python3 scripts/frontier_tab.py --summary data-claude/frontier-sonnet/summary_fold0_s0.json \
        --windows data-claude/frontier-sonnet/usage_windows.json --notes data-claude/frontier-sonnet/tab_notes.json \
        LIVE.html OUT.html

Replaces the block between <!-- FRONTIER:START --> and <!-- FRONTIER:END --> (or adds it after the
last tab, with its nav button at the end of the "Current" group). The bullets and status lines
are hand-written in the notes file ("status", "accuracy", "cost", "windows", "eyebrow"), so recheck
them against the numbers on every rebuild.
"""
import argparse
import html
import json
import re
from collections import defaultdict

TAB_ID = "frontier-sonnet"
START, END = "<!-- FRONTIER:START -->", "<!-- FRONTIER:END -->"
NAMES = {"aime": "AIME", "math500": "MATH-500", "gpqa_diamond": "GPQA Diamond", "mmlu_pro": "MMLU-Pro",
         "browsecomp_plus": "BrowseComp-Plus", "finance_agent": "Finance-Agent", "workbench": "WorkBench",
         "plancraft": "Plancraft"}
HARD = ("aime", "math500", "gpqa_diamond", "mmlu_pro")
AGENTIC = ("browsecomp_plus", "finance_agent", "workbench", "plancraft")


def pct(c, n):
    return f'{100 * c / n:.1f}%<span class="sub">{c}/{n}</span>' if n else "—"


def bullets(items):
    return '    <ul class="findings">\n' + "".join(f"      <li><span>{b}</span></li>\n" for b in items) + "    </ul>\n"


def accuracy_table(b):
    rows, groups = [], []
    for label, keys in (("Hard questions", HARD), ("Tool tasks", AGENTIC)):
        tot = {"n": 0, "c": 0, "a": 0, "an": 0, "o": 0, "on": 0}
        first = True
        for k in keys:
            if k not in b:
                continue
            r = b[k]
            n, c = r["n"], r["claude_correct"]
            a, an, o, on = r["alone_correct"], r["alone_n"], r["orch_correct"], r["orch_n"]
            for key, v in zip("n c a an o on".split(), (n, c, a, an, o, on)):
                tot[key] += v
            diff = (c / n - o / on) * 100 if on else None
            cls = ' class="group-gap"' if first and groups else ""
            first = False
            dcls = " delta-pos" if diff is not None and diff >= 5 else (" delta-neg" if diff is not None and diff <= -5 else "")
            rows.append(f'          <tr{cls}><td>{NAMES[k]}</td><td class="num">{n}</td><td class="num">{pct(c, n)}</td>'
                        f'<td class="num">{pct(a, an)}</td><td class="num">{pct(o, on)}</td>'
                        f'<td class="num{dcls}">{"—" if diff is None else f"{diff:+.1f}"}</td></tr>')
        diff = (tot["c"] / tot["n"] - tot["o"] / tot["on"]) * 100 if tot["on"] else None
        rows.append(f'          <tr class="headline"><td>{label}</td><td class="num">{tot["n"]}</td>'
                    f'<td class="num">{pct(tot["c"], tot["n"])}</td><td class="num">{pct(tot["a"], tot["an"])}</td>'
                    f'<td class="num">{pct(tot["o"], tot["on"])}</td><td class="num">{"—" if diff is None else f"{diff:+.1f}"}</td></tr>')
        groups.append(label)
    return ('    <div class="tablewrap">\n      <table>\n'
            '        <caption>Fold 0, sample 0: accuracy on the same questions for Claude Sonnet alone, the 35B alone '
            '(thinking off) and X1’s orchestrator with its team (headline modes); last column is Sonnet minus '
            'the orchestrator, in points.</caption>\n'
            '        <thead><tr><th>Benchmark</th><th class="num">Questions</th><th class="num">Sonnet</th>'
            '<th class="num">35B alone</th><th class="num">Orchestrator + team</th><th class="num">Sonnet − orch.</th></tr></thead>\n'
            '        <tbody>\n' + "\n".join(rows) + '\n        </tbody>\n      </table>\n    </div>\n')


def combined_table(b, spend):
    """Accuracy and output tokens per question side by side, per system."""
    def cells(c, n, tok, tn):
        return f'<td class="num">{pct(c, n)}</td><td class="num">{tok / tn:,.0f}</td>' if tn else '<td class="num">—</td><td class="num">—</td>'
    rows = []
    for label, keys in (("Hard questions", HARD), ("Tool tasks", AGENTIC)):
        g = defaultdict(float)
        first = True
        for k in keys:
            if k not in b:
                continue
            r, t = b[k], b[k]["totals"]
            cost = spend.get(k, {}).get("cost_usd", 0)
            for key, v in (("n", r["n"]), ("c", r["claude_correct"]), ("ct", t.get("claude_completion_tokens", 0)),
                           ("cost", cost), ("a", r["alone_correct"]), ("an", r["alone_n"]),
                           ("at", t.get("alone_completion_tokens", 0)), ("o", r["orch_correct"]), ("on", r["orch_n"]),
                           ("ot", t.get("orch_completion_tokens", 0))):
                g[key] += v
            gap = ' class="group-gap"' if first and rows else ""
            first = False
            rows.append(f'          <tr{gap}><td>{NAMES[k]}</td><td class="num">{r["n"]}</td>'
                        + cells(r["claude_correct"], r["n"], t.get("claude_completion_tokens", 0), r["n"])
                        + cells(r["alone_correct"], r["alone_n"], t.get("alone_completion_tokens", 0), r["alone_n"])
                        + cells(r["orch_correct"], r["orch_n"], t.get("orch_completion_tokens", 0), r["orch_n"]) + "</tr>")
        n = int(g["n"])
        rows.append(f'          <tr class="headline"><td>{label}</td><td class="num">{n}</td>'
                    + cells(int(g["c"]), n, g["ct"], n)
                    + cells(int(g["a"]), int(g["an"]), g["at"], g["an"])
                    + cells(int(g["o"]), int(g["on"]), g["ot"], g["on"]) + "</tr>")
    return ('    <div class="tablewrap">\n      <table>\n'
            '        <caption>Fold 0, sample 0, same questions: accuracy and output tokens per question (thinking included) '
            'for Claude Sonnet alone, the 35B alone (thinking off) and X1’s orchestrator with its team (headline modes, '
            'every member’s tokens).</caption>\n'
            '        <thead><tr><th rowspan="2">Benchmark</th><th class="num" rowspan="2">Questions</th>'
            '<th class="num" colspan="2" style="text-align:center">Sonnet</th>'
            '<th class="num" colspan="2" style="text-align:center">35B alone</th>'
            '<th class="num" colspan="2" style="text-align:center">Orchestrator + team</th></tr>\n'
            '        <tr><th class="num">Accuracy</th><th class="num">Tokens</th>'
            '<th class="num">Accuracy</th><th class="num">Tokens</th>'
            '<th class="num">Accuracy</th><th class="num">Tokens</th></tr></thead>\n'
            '        <tbody>\n' + "\n".join(rows) + '\n        </tbody>\n      </table>\n    </div>\n')


def cost_table(b, spend):
    rows, tot = [], {"calls": 0, "in": 0, "out": 0, "think": 0, "cost": 0.0, "n": 0}
    for k in HARD + AGENTIC:
        if k not in spend:
            continue
        s, n = spend[k], b[k]["n"]
        tot["calls"] += s["calls"]; tot["in"] += s["input_tokens"]; tot["out"] += s["output_tokens"]
        tot["think"] += s["thinking_tokens"]; tot["cost"] += s["cost_usd"]; tot["n"] += n
        rows.append(f'          <tr><td>{NAMES[k]}</td><td class="num">{n}</td><td class="num">{s["calls"]:,.0f}</td>'
                    f'<td class="num">{s["input_tokens"]:,.0f}</td><td class="num">{s["output_tokens"]:,.0f}</td>'
                    f'<td class="num">{s["thinking_tokens"]:,.0f}</td><td class="num">${s["cost_usd"]:.2f}</td>'
                    f'<td class="num">${s["cost_usd"] / n:.3f}</td></tr>')
    rows.append(f'          <tr class="headline"><td>All</td><td class="num">{tot["n"]}</td><td class="num">{tot["calls"]:,.0f}</td>'
                f'<td class="num">{tot["in"]:,.0f}</td><td class="num">{tot["out"]:,.0f}</td><td class="num">{tot["think"]:,.0f}</td>'
                f'<td class="num">${tot["cost"]:.2f}</td><td class="num">${tot["cost"] / tot["n"]:.3f}</td></tr>')
    return ('    <div class="tablewrap">\n      <table>\n'
            '        <caption>Sonnet’s usage for the fold, every call included (also those lost to the usage limit '
            'and rerun); input counts cache writes and reads; cost is Claude Code’s list-price figure.</caption>\n'
            '        <thead><tr><th>Benchmark</th><th class="num">Tasks</th><th class="num">Calls</th>'
            '<th class="num">Input tokens</th><th class="num">Output tokens</th><th class="num">of which thinking</th>'
            '<th class="num">Cost</th><th class="num">Per task</th></tr></thead>\n'
            '        <tbody>\n' + "\n".join(rows) + '\n        </tbody>\n      </table>\n    </div>\n')


def token_tables(b):
    cols = ("claude_completion_tokens", "alone_completion_tokens", "orch_completion_tokens", "hub_completion_tokens",
            "large_worker_completion_tokens", "small_worker_completion_tokens")
    ins = ("claude_prompt_tokens", "alone_prompt_tokens", "orch_prompt_tokens")
    out_rows, in_rows = [], []
    for label, keys in (("Hard questions", HARD), ("Tool tasks", AGENTIC)):
        g = {}
        first = True
        for k in keys:
            if k not in b:
                continue
            t = b[k]["totals"]
            for c in cols + ins + ("n",):
                g[c] = g.get(c, 0) + t.get(c, 0)
            gap = ' class="group-gap"' if first and out_rows else ""
            first = False
            out_rows.append(f'          <tr{gap}><td>{NAMES[k]}</td>' + "".join(
                f'<td class="num">{t.get(c, 0) / t["n"]:,.0f}</td>' for c in cols) + "</tr>")
            in_rows.append(f'          <tr{gap}><td>{NAMES[k]}</td>' + "".join(
                f'<td class="num">{t.get(c, 0) / t["n"]:,.0f}</td>' for c in ins) + "</tr>")
        out_rows.append(f'          <tr class="headline"><td>{label}</td>' + "".join(
            f'<td class="num">{g[c] / g["n"]:,.0f}</td>' for c in cols) + "</tr>")
        in_rows.append(f'          <tr class="headline"><td>{label}</td>' + "".join(
            f'<td class="num">{g[c] / g["n"]:,.0f}</td>' for c in ins) + "</tr>")
    out = ('    <div class="tablewrap">\n      <table>\n'
           '        <caption>Output tokens per question (thinking included), fold 0, sample 0; orchestrator + team is '
           'every token the team spent (4 workers and the orchestrator, whole tool episodes), split in the last three '
           'columns.</caption>\n'
           '        <thead><tr><th>Benchmark</th><th class="num">Sonnet</th><th class="num">35B alone</th>'
           '<th class="num">Orchestrator + team</th><th class="num">of which orchestrator</th>'
           '<th class="num">35B-class workers</th><th class="num">3–4B workers</th></tr></thead>\n'
           '        <tbody>\n' + "\n".join(out_rows) + '\n        </tbody>\n      </table>\n    </div>\n')
    inp = ('    <div class="tablewrap">\n      <table>\n'
           '        <caption>Input tokens per question, summed over every call (tool tasks resend the conversation each '
           'turn); Sonnet’s include cache reads and writes.</caption>\n'
           '        <thead><tr><th>Benchmark</th><th class="num">Sonnet</th><th class="num">35B alone</th>'
           '<th class="num">Orchestrator + team</th></tr></thead>\n'
           '        <tbody>\n' + "\n".join(in_rows) + '\n        </tbody>\n      </table>\n    </div>\n')
    return out, inp


def windows_table(windows):
    rows = [f'          <tr><td>{html.escape(w["window"])}</td><td class="num">{w["calls"]}</td>'
            f'<td class="num">${w["cost_usd"]:.2f}</td><td>{html.escape(w["ended"])}</td></tr>' for w in windows]
    return ('    <div class="tablewrap">\n      <table>\n'
            '        <caption>Claude subscription usage by 5-hour window (UTC): calls and list-price cost in each, '
            'and how the window ended.</caption>\n'
            '        <thead><tr><th>Window</th><th class="num">Calls</th><th class="num">Cost</th><th>Ended</th></tr></thead>\n'
            '        <tbody>\n' + "\n".join(rows) + '\n        </tbody>\n      </table>\n    </div>\n')


def build(summary, windows, notes):
    b, spend = summary["benchmarks"], summary["spend"]
    out = [START, f'  <div id="{TAB_ID}" role="tabpanel" aria-labelledby="tabbtn-{TAB_ID}" hidden>\n',
           '  <header class="masthead">',
           f'    <div class="eyebrow">{notes["eyebrow"]}</div>',
           '    <h1>Frontier model: Claude Sonnet</h1>',
           '    <dl class="runmeta">',
           '      <div><dt>Model</dt><dd>claude-sonnet-5-5, Claude Code default effort</dd></div>',
           '      <div><dt>Calls</dt><dd>headless claude -p, a fresh session per call, no tools of its own</dd></div>',
           '      <div><dt>Questions</dt><dd>X1’s, fold 0, sample 0</dd></div>',
           '      <div><dt>Limits</dt><dd>30,720 tokens (hard), 8,192 per turn (tools)</dd></div>',
           '      <div><dt>Judge</dt><dd>Qwen3.6-35B, as for every other arm</dd></div>',
           '    </dl>', '  </header>\n',
           '  <div class="status">',
           f'    <span class="label">Status <span class="pill">{notes.get("pill", "Preliminary")}</span></span>',
           bullets(notes["status"]).rstrip("\n"), '  </div>\n',
           '  <section>',
           '    <div class="sec-head"><div class="kicker">Fold 0 · sample 0 · same questions as X1</div>'
           '<h2>Sonnet against the 35B and the orchestrator: accuracy and tokens</h2></div>',
           combined_table(b, spend).rstrip("\n"), bullets(notes["accuracy"]).rstrip("\n"), '  </section>\n',
           '  <section>',
           '    <div class="sec-head"><div class="kicker">Tokens and cost · fold 0</div><h2>What the fold used</h2></div>',
           cost_table(b, spend).rstrip("\n"), bullets(notes["cost"]).rstrip("\n"), '  </section>\n',
           '  <section>',
           '    <div class="sec-head"><div class="kicker">Tokens per question · fold 0</div>'
           '<h2>Sonnet against the 35B and the team, in tokens</h2></div>',
           token_tables(b)[0].rstrip("\n"), bullets(notes["tokens"]).rstrip("\n"),
           token_tables(b)[1].rstrip("\n"), bullets(notes["tokens_in"]).rstrip("\n"), '  </section>\n',
           '  <section>',
           '    <div class="sec-head"><div class="kicker">Subscription usage</div><h2>How much of the usage limit it took</h2></div>',
           windows_table(windows).rstrip("\n"), bullets(notes["windows"]).rstrip("\n"), '  </section>\n',
           '  </div>', END]
    return "\n".join(out)


def splice(page, block):
    button = (f'    <button type="button" role="tab" id="tabbtn-{TAB_ID}" aria-controls="{TAB_ID}" '
              f'aria-selected="false" tabindex="-1">Frontier model</button>\n')
    if START in page:
        page = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, page, count=1, flags=re.S)
    else:
        last_end = max(m.end() for m in re.finditer(r"<!-- [A-Z0-9-]+:END -->", page))
        page = page[:last_end] + "\n" + block + page[last_end:]
    if f'id="tabbtn-{TAB_ID}"' not in page:
        marker = '    <span class="tabgroup" aria-hidden="true">Earlier'
        assert page.count(marker) == 1, "nav group marker not found"
        page = page.replace(marker, button + marker)
    return page


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--windows", required=True)
    p.add_argument("--notes", required=True)
    p.add_argument("live")
    p.add_argument("out")
    args = p.parse_args()
    block = build(json.load(open(args.summary)), json.load(open(args.windows)), json.load(open(args.notes)))
    page = splice(open(args.live, encoding="utf-8").read(), block)
    open(args.out, "w", encoding="utf-8").write(page)


if __name__ == "__main__":
    main()
