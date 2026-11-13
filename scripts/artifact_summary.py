"""Summary tab and layout of the results page ("Does Orchestrator Memory Help?").

    python scripts/artifact_summary.py LIVE.html OUT.html
    python scripts/artifact_summary.py LIVE.html OUT.html \
        --x1-stage1 spark.json local.json --x1-stage2 stage2.json [--x1-notes notes.json]

Always run on a fresh read of the live page, then publish OUT to the same URL. Every step is
idempotent, so it can run after any other tab's splice script:

- The Summary tab (docs/artifact/summary_tab.html, between <!-- SUMMARY:START/END -->) goes in
  first. Its X1 part (<!-- SUMMARY-X1:START/END -->) is kept from the live page unless the --x1-*
  inputs are given, in which case it is rebuilt from them (same inputs as
  ../Agent-Scaling-Orchestration-x1/scripts/x1_stage1_tab.py, whose helpers it imports).
- The nav is split into "Current" and "Earlier" groups; a tab this script does not know goes at
  the end of "Current". The Summary tab is the one shown on load.
- The X1 tab's design sections (Teams ... Expert persona prompts) are folded into one <details>.

--x1-stage2-datasets (optional) takes x1_stage2_by_dataset.py output and adds a by-dataset table after
the stage 2 table, with the notes' "stage2_datasets" bullets.

--x1-notes (optional) is a JSON object with lists "stage2" and "stage1" of extra finding bullets
(HTML) added after the computed ones, plus optional "asof" (time of the X1 numbers, for the
eyebrow) and "status" (list of HTML bullets for the Summary status box). Without "asof"/"status",
the live page's eyebrow and status box are kept, never reset to summary_tab.html. Optional
"x1_kicker" sets the kicker of the Summary's X1 section, and "pill" / "pill_class" ("running" or
"done") the Summary status pill; without them the live page's are kept.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs" / "artifact" / "summary_tab.html"
X1_SCRIPTS = REPO.parent / "Agent-Scaling-Orchestration-x1" / "scripts"

CURRENT = ["summary", "x1-teams", "x0-headroom", "agentic-baselines", "harder"]
EARLIER = ["e2-topology-4b", "e2-topology", "team-of-one", "cv-small", "all-runs", "writeup"]
LABELS = {"summary": "Summary", "e2-topology": "Selection × topology, 0.8B"}
GROUPS = ("Current · hard and agentic pools", "Earlier · easy pool, 50-persona bank")
BUTTON = re.compile(r'[ \t]*<button type="button" role="tab" id="tabbtn-([\w-]+)"[^>]*>([^<]*)</button>[ \t]*\n?')
X1_RE = re.compile(r"\n  <!-- SUMMARY-X1:START -->.*?<!-- SUMMARY-X1:END -->", re.S)
EYEBROW_RE = re.compile(r'(<div class="eyebrow">)Summary · [^<]*(</div>)')
STATUS_RE = re.compile(r'(<div class="status">\s*<span class="label">.*?</span></span>\s*<ul class="findings">).*?(</ul>)', re.S)
KICKER_RE = re.compile(r'(<!-- SUMMARY-X1:START -->\s*<section>\s*<div class="sec-head"><div class="kicker">)[^<]*(</div>)')
PILL_RE = re.compile(r'(<div class="status">\s*<span class="label">Status )<span class="pill [\w-]+">[^<]*</span>')
X1_KICKER = "X1 · fixed expert teams with a 35B orchestrator · running"
DESIGN_START = '  <section>\n    <div class="sec-head"><div class="kicker">Design</div><h2>Teams</h2>'
DESIGN_SUMMARY = ("Design and setup: teams, settled decisions, ablations, smoke tests, "
                  "thinking on vs off, model cards, persona prompts")


def pct(num, den):
    return f"{100 * num / den:.0f}%" if den else "—"


def cell(num, den, sub=True):
    s = f"<span class=\"sub\">{num}/{den}</span>" if sub and den else ""
    return f"<td class=\"num\">{pct(num, den)}{s}</td>"


def bullets(items):
    return "\n".join(["    <ul class=\"findings\">", *(f"      <li><span>{b}</span></li>" for b in items), "    </ul>"])


def x1_block(stage1_paths, stage2_path, notes, datasets_path=None):
    sys.path.insert(0, str(X1_SCRIPTS))
    from x1_stage1_tab import DATASETS, SPECIALISTS, TEAMS, dataset_rows, merge  # noqa: E402

    label = lambda w: SPECIALISTS[w][0] if w in SPECIALISTS else w
    out = ["  <!-- SUMMARY-X1:START -->", "  <section>",
           f"    <div class=\"sec-head\"><div class=\"kicker\">{notes.get('x1_kicker', X1_KICKER)}</div>"
           "<h2>Orchestrator with its team, and workers alone</h2>"
           "<button type=\"button\" class=\"linkbtn\" data-goto=\"x1-teams\">Full X1 tab</button></div>"]

    if stage2_path:
        groups = {}
        for r in json.load(open(stage2_path)):
            if r["modes"] == "headline":
                groups.setdefault(r["team"], []).append(r)
        rows, beat_vote, beat_best, fixed, broke, runs = [], 0, 0, 0, 0, set()
        for team in (t for t in TEAMS if t in groups):
            rs = groups[team]
            runs.add(len(rs))
            q = sum(r["questions"] for r in rs)
            ws = {}
            for r in rs:
                for w, c in r["workers_correct"].items():
                    ws[w] = ws.get(w, 0) + c
            best = max(ws, key=ws.get)
            hub, vote = sum(r["hub_correct"] for r in rs), sum(r["vote_correct"] for r in rs)
            f, b = (sum(r["hub_transitions"][k] for r in rs) for k in ("fixed", "broke"))
            beat_vote += hub > vote
            beat_best += hub > ws[best]
            fixed, broke = fixed + f, broke + b
            rows.append(f"          <tr><td>{team} {TEAMS[team]}</td><td>{label(best)}</td><td class=\"num\">{len(rs)}</td>"
                        f"{cell(hub, q)}{cell(vote, q)}{cell(ws[best], q)}"
                        f"<td class=\"num\">{f}</td><td class=\"num\">{b}</td></tr>")
        n = len(rows)
        out += ["    <div class=\"tablewrap\">", "      <table>",
                f"        <caption>Stage 2{'' if runs == {15} else ' so far'}, headline thinking modes: the orchestrator (35B, thinking off) "
                "against its team's majority vote and its best worker (picked in hindsight), over "
                f"{'all 15 fold/sample runs' if runs == {15} else 'the fold/sample runs finished (of 15)'}; "
                "fixed / broke = orchestrator right where the vote is wrong, and the reverse.</caption>",
                "        <thead><tr><th>Team</th><th>Best worker</th><th class=\"num\">Runs</th><th class=\"num\">Orchestrator</th>"
                "<th class=\"num\">Vote</th><th class=\"num\">Best worker</th><th class=\"num\">Fixed</th>"
                "<th class=\"num\">Broke</th></tr></thead>", "        <tbody>", *rows,
                "        </tbody>", "      </table>", "    </div>",
                bullets([f"The orchestrator beats the majority vote on {beat_vote} of {n} teams and the best worker on {beat_best} of {n}.",
                         f"Over these teams it fixes {fixed} answers the vote got wrong and breaks {broke} it got right.",
                         *notes.get("stage2", [])])]

    if datasets_path:
        rows, last = [], None
        for r in dataset_rows(json.load(open(datasets_path))):
            name = DATASETS[r["dataset"]] if r["dataset"] != last else ""
            gap = " class=\"group-gap\"" if r["dataset"] != last and last is not None else ""
            last = r["dataset"]
            if r["team"] is None:
                who, best = f"all {r['teams']} teams", "<td>—</td>"
                bestc = "<td class=\"num\">—</td>"
            else:
                who, best = f"{r['team']} {TEAMS[r['team']]}", f"<td>{label(r['best'])}</td>"
                bestc = cell(r["best_c"], r["q"])
            rows.append(f"          <tr{gap}><td>{name}</td><td>{who}</td>{best}<td class=\"num\">{r['distinct']}</td>"
                        f"{cell(r['hub'], r['q'])}{cell(r['vote'], r['q'])}{bestc}"
                        f"<td class=\"num\">{r['fixed']}</td><td class=\"num\">{r['broke']}</td></tr>")
        out += ["    <div class=\"tablewrap\">", "      <table>",
                "        <caption>Stage 2 by source dataset, headline thinking modes: orchestrator, majority vote and best worker "
                "on that dataset; questions = different questions, each answered in 3 samples.</caption>",
                "        <thead><tr><th>Dataset</th><th>Team</th><th>Best worker</th><th class=\"num\">Questions</th>"
                "<th class=\"num\">Orchestrator</th><th class=\"num\">Vote</th><th class=\"num\">Best worker</th>"
                "<th class=\"num\">Fixed</th><th class=\"num\">Broke</th></tr></thead>", "        <tbody>", *rows,
                "        </tbody>", "      </table>", "    </div>",
                bullets(notes.get("stage2_datasets", []))]

    if stage1_paths:
        merged = merge(stage1_paths)
        rows, wins, big = [], 0, []
        for team in TEAMS:
            members = [r for r in merged if r["team"] == team and not r["judged"]]
            if not members:
                continue
            shared = set.intersection(*(set(r["by_run"]) for r in members))
            if not shared:
                continue

            def matched(r):
                q = sum(r["by_run"][k]["questions"] for k in shared)
                c = sum(r["by_run"][k]["correct"] for k in shared)
                return c / q if q else 0.0

            personas = [r for r in members if r["persona"]]
            specs = [r for r in members if not r["persona"]]
            if not personas or not specs:
                continue
            bp = max(personas, key=matched)
            bs = max(specs, key=matched)
            p, s = round(100 * matched(bp)), round(100 * matched(bs))
            ties = [r for r in personas if round(100 * matched(r)) == p]
            pname = "both personas" if len(ties) > 1 and len(ties) == len(personas) else bp["worker"]
            size = " (35B)" if bs["worker"] in SPECIALISTS and SPECIALISTS[bs["worker"]][2].startswith("35B") else ""
            d = s - p
            wins += d > 0
            if d >= 10:
                big.append(label(bs["worker"]) + size)
            cls = " delta-pos" if d >= 5 else (" delta-neg" if d <= -5 else "")
            rows.append(f"          <tr><td>{team} {TEAMS[team]}</td><td>{pname}</td>"
                        f"<td>{label(bs['worker'])}{size} · {bs['mode']}</td><td class=\"num\">{p}%</td>"
                        f"<td class=\"num\">{s}%</td><td class=\"num{cls}\">{f'{d:+d}'.replace('-', '−')}</td>"
                        f"<td class=\"num\">{len(shared)}</td></tr>")
        computed = [f"The best specialist beats the best persona on {wins} of {len(rows)} teams."]
        if big:
            computed.append("Gains of 10 points or more: " + ", ".join(dict.fromkeys(big)) + ".")
        out += ["    <div class=\"tablewrap\">", "      <table>",
                "        <caption>Stage 1: each worker alone on its team's questions, on the runs every member of the team "
                "has finished; personas on the 4B with thinking off, specialist in its better mode.</caption>",
                "        <thead><tr><th>Team</th><th>Best persona</th><th>Best specialist</th><th class=\"num\">Persona</th>"
                "<th class=\"num\">Specialist</th><th class=\"num\">Difference</th><th class=\"num\">Runs</th></tr></thead>",
                "        <tbody>", *rows, "        </tbody>", "      </table>", "    </div>",
                bullets(computed + notes.get("stage1", []))]

    out += ["  </section>", "  <!-- SUMMARY-X1:END -->"]
    return "\n".join(out)


def nav(page):
    a = page.index('<nav class="tabs"')
    a = page.index(">", a) + 1
    b = page.index("</nav>", a)
    found = {m[1]: m[2] for m in BUTTON.finditer(page[a:b])}
    found.setdefault("summary", "Summary")
    extra = [t for t in found if t not in CURRENT + EARLIER]
    lines = []
    for title, ids in zip(GROUPS, (CURRENT + extra, EARLIER)):
        lines.append(f'    <span class="tabgroup" aria-hidden="true">{title}</span>')
        for t in ids:
            if t not in found:
                continue
            on = t == "summary"
            lines.append(f'    <button type="button" role="tab" id="tabbtn-{t}" aria-controls="{t}" '
                         f'aria-selected="{"true" if on else "false"}"{"" if on else " tabindex=\"-1\""}>'
                         f'{LABELS.get(t, found[t])}</button>')
    return page[:a] + "\n" + "\n".join(lines) + "\n  " + page[b:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("live")
    ap.add_argument("out")
    ap.add_argument("--x1-stage1", nargs="+", help="x1_stage1_summary.py output from each box")
    ap.add_argument("--x1-stage2", help="JSON list of stage 2 holdout_summary.json records")
    ap.add_argument("--x1-stage2-datasets", help="x1_stage2_by_dataset.py output: adds a by-dataset table")
    ap.add_argument("--x1-notes", help="JSON with extra bullets: {\"stage2\": [...], \"stage1\": [...]}")
    args = ap.parse_args()

    page = Path(args.live).read_text()
    block = SOURCE.read_text().rstrip("\n") + "\n"

    # X1 part of the summary: rebuilt from the inputs, else kept from the live page.
    if args.x1_stage1 or args.x1_stage2:
        notes = json.load(open(args.x1_notes)) if args.x1_notes else {}
        new_x1 = "\n" + x1_block(args.x1_stage1, args.x1_stage2, notes, args.x1_stage2_datasets)
        block = X1_RE.sub(lambda _: new_x1, block, count=1)
    elif "<!-- SUMMARY-X1:START -->" in page:
        live_x1 = X1_RE.search(page)[0]
        block = X1_RE.sub(lambda _: live_x1, block, count=1)

    # Eyebrow and status list: from the notes ("asof", "status") if given, else kept from the live page.
    notes = json.load(open(args.x1_notes)) if args.x1_notes else {}
    live = page[page.index("<!-- SUMMARY:START -->"):page.index("<!-- SUMMARY:END -->")] if "<!-- SUMMARY:START -->" in page else ""
    if "asof" in notes:
        block = EYEBROW_RE.sub(lambda m: m[1] + f"Summary · all experiments · X1 numbers as of {notes['asof']}" + m[2], block, count=1)
    elif live:
        block = EYEBROW_RE.sub(lambda _: EYEBROW_RE.search(live)[0], block, count=1)
    if "status" in notes:
        block = STATUS_RE.sub(lambda m: m[1] + "\n" + "\n".join(f"      <li><span>{s}</span></li>" for s in notes["status"]) + "\n    " + m[2], block, count=1)
    elif live:
        block = STATUS_RE.sub(lambda _: STATUS_RE.search(live)[0], block, count=1)

    if "x1_kicker" in notes:
        block = KICKER_RE.sub(lambda m: m[1] + notes["x1_kicker"] + m[2], block, count=1)
    elif live and KICKER_RE.search(live):
        block = KICKER_RE.sub(lambda _: KICKER_RE.search(live)[0], block, count=1)
    if "pill" in notes:
        block = PILL_RE.sub(lambda m: m[1] + f'<span class="pill {notes.get("pill_class", "running")}">{notes["pill"]}</span>', block, count=1)
    elif live and PILL_RE.search(live):
        block = PILL_RE.sub(lambda _: PILL_RE.search(live)[0], block, count=1)

    if "<!-- SUMMARY:START -->" in page:
        a = page.index("  <!-- SUMMARY:START -->")
        b = page.index("<!-- SUMMARY:END -->") + len("<!-- SUMMARY:END -->\n")
        page = page[:a] + block + page[b:]
    else:
        a = page.index("</nav>") + len("</nav>\n")
        page = page[:a] + "\n" + block + page[a:]

    page = nav(page)

    # Only the Summary panel is visible on load.
    def panel(m):
        tag = m[0].replace(" hidden>", ">")
        return tag if m[1] == "summary" else tag[:-1] + " hidden>"
    page = re.sub(r'<div id="([\w-]+)" role="tabpanel"[^>]*>', panel, page)

    # Open any tab named in the URL hash, not only the ones listed when the page was first written.
    page = re.sub(r'if \(h === "all-runs"[^\n]*\) showTab\(h\);',
                  'if (h && document.getElementById(h) && document.getElementById(h).getAttribute("role") === "tabpanel") showTab(h);',
                  page, count=1)

    # Fold the X1 tab's design sections into one collapsed block.
    a, b = page.index("<!-- X1:START -->"), page.index("<!-- X1:END -->")
    tab = page[a:b]
    if "design-block" not in tab and DESIGN_START in tab:
        s = tab.index(DESIGN_START)
        e = tab.index('\n  <div class="status">')
        tab = (tab[:s] + f'  <details class="design-block">\n  <summary>{DESIGN_SUMMARY}</summary>\n'
               + tab[s:e].rstrip() + "\n  </details>\n" + tab[e:])
        page = page[:a] + tab + page[b:]

    for t in CURRENT + EARLIER:
        assert page.count(f'id="tabbtn-{t}"') == 1, t
        assert page.count(f'id="{t}" role="tabpanel"') == 1, t
    Path(args.out).write_text(page)


if __name__ == "__main__":
    main()
