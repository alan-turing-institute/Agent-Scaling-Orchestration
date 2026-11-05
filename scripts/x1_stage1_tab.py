"""Splice X1 stage 1 results into the X1 tab of the results page ("Does Orchestrator Memory Help?").

    python scripts/x1_stage1_tab.py --summary spark.json local.json --notes notes.json LIVE.html OUT.html

`--stage2` (optional) takes a JSON list of stage 2 holdout_summary.json records; with it, a
"Orchestrator with its team" section (notes "stage2_kicker" and "stage2" bullets) goes above stage 1.
`--stage2-datasets` (optional, with `--stage2`) takes x1_stage2_by_dataset.py output and adds a "By
dataset" part to that section: a table, a chart of every dataset and team, per-worker charts for the
teams that draw on more than one dataset, and notes "stage2_datasets" bullets.

`--summary` takes x1_stage1_summary.py output from each box running stage 1; a run present in
more than one is counted once. In the notes, "{done}" and "{planned}" become the counts of finished
and planned runs. `--notes` is a JSON file of the hand-written parts: "eyebrow",
"kicker", "pill", "hard" and "agentic" (finding bullets, HTML) and "status" (paragraphs, HTML), plus
optional "pill_class" ("running", the default, or "done") and "stage1_title" / "stage2_title" (section
headings; defaults "Workers answering alone, so far" and "Orchestrator with its team, so far"). The stage 1
section, the eyebrow and the status box of the tab between <!-- X1:START --> and <!-- X1:END -->
are replaced; everything else on the page is kept.

Each row shows every finished run, and "matched" restricts the team to the runs all its rows
have finished, so workers are compared on the same questions.
"""

import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from x1_plan import load_config, workers

TEAMS = {"T1": "Maths", "T2": "Physics", "T3": "Chemistry", "T4": "Biology", "T5": "Law", "T6": "Engineering",
         "T7": "BrowseComp-Plus", "T8": "Finance-Agent", "T9": "WorkBench", "T10": "Plancraft"}
SPECIALISTS = {  # model key: label, base model, size
    "rl-math": ("RL-MATH", "Qwen3.5-4B", "4B dense"),
    "agentmercury": ("AgentMercury-4B", "Qwen3.5-4B", "4B dense"),
    "agents-a1": ("Agents-A1-4B", "Qwen3.5-4B", "4B dense"),
    "r2s": ("R2S ALFWorld student", "Qwen3.5-4B", "4B dense"),
    "skillgym": ("SkillGym-4B", "Qwen3.5-4B", "4B dense"),
    "ministral-r": ("Ministral-3B-Reasoning", "Ministral-3-3B-Base-2512", "3B dense"),
    "shiningvaliant3": ("ShiningValiant3", "gemma-4-E4B-it", "4B effective (E4B)"),
    "bigbang": ("BigBang", "Qwen3.6-35B-A3B", "35B MoE, 3B active"),
    "fx-work": ("Fx-Work", "Qwen3.6-35B-A3B", "35B MoE, 3B active"),
    "occamy": ("Occamy-1.0", "Qwen3.6-35B-A3B", "35B MoE, 3B active"),
    "xyz-aquila": ("XYZ-Aquila-mini", "Qwen3.6-35B-A3B", "35B MoE, 3B active"),
}
DATASETS = {"math500": "MATH-500", "aime": "AIME", "gpqa_diamond": "GPQA Diamond", "mmlu_pro": "MMLU-Pro",
            "browsecomp_plus": "BrowseComp-Plus", "finance_agent": "Finance-Agent", "workbench": "WorkBench",
            "plancraft": "Plancraft"}
COUNTS = ("questions", "correct", "answered", "cut_off", "completion_tokens")


def merge(paths):
    rows = {}
    for path in paths:
        for r in json.load(open(path)):
            key = (r["team"], r["worker"], r["mode"])
            if key not in rows:
                rows[key] = {**r, "by_run": {}}
            rows[key]["by_run"].update(r["by_run"])
    for r in rows.values():
        r["runs"] = len(r["by_run"])
        for k in COUNTS:
            r[k] = sum(run[k] for run in r["by_run"].values())
    return list(rows.values())


def pct(num, den, sub=True, bold=False):
    if not den:
        return "<td class='num' style='color:var(--faint)'>—</td>"
    s = f"<span class='sub'>{num}/{den}</span>" if sub else ""
    v = f"<strong>{100 * num / den:.0f}%</strong>" if bold else f"{100 * num / den:.0f}%"
    return f"<td class='num'>{v}{s}</td>"


def matched_stats(members):
    """Per row: correct and answers over the runs every member of the team has finished, and the number of
    distinct questions behind them (the samples of a fold repeat the same questions)."""
    shared = set.intersection(*(set(r["by_run"]) for r in members))
    for r in members:
        runs = [r["by_run"][k] for k in shared]
        r["m_q"] = sum(m["questions"] for m in runs)
        r["m_c"] = sum(m["correct"] for m in runs)
        r["m_a"] = sum(m["answered"] for m in runs)
        folds = {}
        for k in shared:
            fold = k.split("s")[0]
            folds[fold] = max(folds.get(fold, 0), r["by_run"][k]["questions"])
        r["m_distinct"] = sum(folds.values())
    scored = [r for r in members if not r["judged"] and r["m_q"]]
    top = max((r["m_c"] / r["m_q"] for r in scored), default=None)
    for r in members:
        r["best"] = top is not None and not r["judged"] and bool(r["m_q"]) and r["m_c"] / r["m_q"] == top


def wilson(p, n, z=1.96):
    if not n:
        return 0.0, 0.0
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def label_of(r):
    if r["persona"]:
        return r["worker"], "Qwen3.5-4B"
    return SPECIALISTS[r["worker"]][0], SPECIALISTS[r["worker"]][1]


W, LX, X0, X1, VX, RH, TOP = 520, 272, 278, 462, 518, 24, 6


def bar_svg(items, aria):
    """Horizontal bars from 0 to 100% with a 95% Wilson interval. Each item: label, colour, c, q, distinct, bold, tip."""
    sx = lambda v: X0 + (X1 - X0) * v
    h = TOP + RH * len(items) + 22
    svg = [f"<svg viewBox='0 0 {W} {h}' role='img' aria-label='{html.escape(aria, quote=True)}' "
           "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='11'>"]
    for t in (0, 0.25, 0.5, 0.75, 1):
        x = sx(t)
        svg.append(f"<line x1='{x:.1f}' y1='{TOP - 2}' x2='{x:.1f}' y2='{h - 20}' stroke='var(--rule)' stroke-width='1'/>")
        svg.append(f"<text x='{x:.1f}' y='{h - 6}' text-anchor='middle' fill='var(--faint)' font-size='10'>{int(t * 100)}%</text>")
    for i, it in enumerate(items):
        y = TOP + RH * i
        cy = y + RH / 2
        weight = " font-weight='700'" if it["bold"] else ""
        ink = "var(--ink)" if it["bold"] else "var(--muted)"
        label = it["label"] if len(it["label"]) <= 42 else it["label"][:41] + "…"
        svg.append(f"<text x='{LX}' y='{cy + 4:.1f}' text-anchor='end' fill='{ink}'{weight}>{html.escape(label)}</text>")
        if not it["q"]:
            continue
        p = it["c"] / it["q"]
        lo, hi = wilson(p, it["distinct"])
        bw = max(sx(p) - X0, 0.5)
        bh = 14
        rad = min(4, bw)
        top = cy - bh / 2
        path = (f"M{X0},{top} h{bw - rad:.1f} a{rad},{rad} 0 0 1 {rad},{rad} v{bh - 2 * rad} "
                f"a{rad},{rad} 0 0 1 -{rad},{rad} h-{bw - rad:.1f} z")
        tip = (f"{it['tip']}<br>{100 * p:.1f}% ({it['c']}/{it['q']} answers)<br>"
               f"95% interval {100 * lo:.0f}–{100 * hi:.0f}% over {it['distinct']} {it.get('unit', 'questions')}")
        svg.append(f"<g class='x1-bar' tabindex='0' data-tip='{html.escape(tip, quote=True)}'>"
                   f"<rect x='0' y='{y}' width='{W}' height='{RH}' fill='transparent'/>"
                   f"<path d='{path}' fill='{it['colour']}'/>"
                   f"<line x1='{sx(lo):.1f}' y1='{cy}' x2='{sx(hi):.1f}' y2='{cy}' stroke='var(--ink)' stroke-width='1.5'/>"
                   f"<line x1='{sx(lo):.1f}' y1='{cy - 4}' x2='{sx(lo):.1f}' y2='{cy + 4}' stroke='var(--ink)' stroke-width='1.5'/>"
                   f"<line x1='{sx(hi):.1f}' y1='{cy - 4}' x2='{sx(hi):.1f}' y2='{cy + 4}' stroke='var(--ink)' stroke-width='1.5'/>"
                   f"<text x='{VX}' y='{cy + 4:.1f}' text-anchor='end' fill='{ink}'{weight}>{100 * p:.0f}%</text></g>")
    svg.append("</svg>")
    return "".join(svg)


PERSONA_COLOUR, SPECIALIST_COLOUR = "var(--s-random)", "var(--s-no_memory)"
HUB_COLOUR, VOTE_COLOUR = "var(--s-continual)", "var(--faint)"
LEGEND_WORKERS = ("<span><i class='swatch sq' style='background:var(--s-random)'></i> persona (Qwen3.5-4B)</span>"
                  "<span><i class='swatch sq' style='background:var(--s-no_memory)'></i> specialist</span>")


def charts(rows, pool):
    """One horizontal bar chart per scored team: matched accuracy per worker, with a 95% interval."""
    teams = [t for t in TEAMS if any(r["team"] == t and r["pool"] == pool and not r["judged"] for r in rows)]
    if not teams:
        return ""
    unit = "questions" if pool == "hard" else "tasks"
    figs = []
    for team in teams:
        members = [r for r in rows if r["team"] == team]
        items = []
        for r in members:
            name, base = label_of(r)
            kind = "persona" if r["persona"] else "specialist"
            items.append({"label": f"{name} · {r['mode']}", "bold": r["best"], "c": r["m_c"], "q": r["m_q"],
                          "distinct": r["m_distinct"], "unit": unit,
                          "colour": PERSONA_COLOUR if r["persona"] else SPECIALIST_COLOUR,
                          "tip": f"<b>{name}</b> · thinking {r['mode']}<br><span class=m>{kind} · {base}</span>"})
        n = members[0]["m_distinct"]
        svg = bar_svg(items, f"{team} {TEAMS[team]}: accuracy of each worker alone")
        figs.append(f"      <div class='x1-chart'><h3>{team} {TEAMS[team]} <span>{n} {unit} × {members[0]['m_q'] // max(n, 1)} samples</span></h3>"
                    f"<div class='x1-svg'>{svg}</div></div>")
    return "\n".join([
        "    <figure>",
        f"      <figcaption>Accuracy of each worker alone on its team's {unit}, over the runs every worker of the team has "
        f"finished; whiskers = 95% Wilson interval over the team's distinct {unit}; best worker in bold.</figcaption>",
        f"      <div class='legend'>{LEGEND_WORKERS}</div>",
        "      <div class='x1-charts'>", *figs, "      </div>",
        "    </figure>"])


def worker_mode(config, worker, modes):
    """The thinking mode a worker answered in under a stage 2 mode setting (as x1_hub.team_workers picks it)."""
    spec = config["specialists"].get(worker)
    if spec is None:
        return config["persona_model"]["mode"]
    return spec["headline"] if modes == "headline" else (modes if modes in spec["modes"] else spec["headline"])


def stage2_charts(records, config, split=False):
    """Per team, one chart per thinking mode (switched by buttons): orchestrator, vote and each worker, with 95% intervals.

    With `split`, `records` are x1_stage2_by_dataset.py records and there is one chart per dataset of each
    team that draws on more than one dataset."""
    groups = {}
    for r in records:
        groups.setdefault(((r["team"], r["dataset"]) if split else r["team"], r["modes"]), []).append(r)
    if split:
        multi = {t for t in TEAMS if len({k[0][1] for k in groups if k[0][0] == t}) > 1}
        teams = [(t, d) for d in DATASETS for t in TEAMS if t in multi and any(k[0] == (t, d) for k in groups)]
    else:
        teams = [t for t in TEAMS if any(k[0] == t for k in groups)]
    pools = {t["id"]: t["pool"] for t in config["teams"]}
    figs = []
    for team in teams:
        title = f"{DATASETS[team[1]]} · {team[0]} {TEAMS[team[0]]}" if split else f"{team} {TEAMS[team]}"
        unit = "questions" if pools[team[0] if split else team] == "hard" else "tasks"
        panes = []
        for mode in MODES:
            rs = groups.get((team, mode))
            hidden = "" if mode == "headline" else " hidden"
            if not rs:
                panes.append(f"<div class='x1-mode' data-mode='{mode}'{hidden}><p class='x1-none'>No {MODES[mode]} runs yet.</p></div>")
                continue
            q = sum(r["questions"] for r in rs)
            folds = {}
            for r in rs:
                folds[r["fold"]] = max(folds.get(r["fold"], 0), r["questions"])
            distinct = sum(folds.values())
            ws = {}
            for r in rs:
                for w, c in r["workers_correct"].items():
                    ws[w] = ws.get(w, 0) + c
            top = max(ws.values())
            trans = lambda k: sum(r["hub_transitions"][k] for r in rs)
            hub_c, vote_c = sum(r["hub_correct"] for r in rs), sum(r["vote_correct"] for r in rs)
            items = [
                {"label": "Orchestrator (35B) · off", "bold": True, "c": hub_c, "q": q, "distinct": distinct, "unit": unit,
                 "colour": HUB_COLOUR,
                 "tip": f"<b>Orchestrator</b> · Qwen3.6-35B, thinking off<br><span class=m>fixed {trans('fixed')}, broke {trans('broke')} against the vote</span>"},
                {"label": "Majority vote", "bold": False, "c": vote_c, "q": q, "distinct": distinct, "unit": unit,
                 "colour": VOTE_COLOUR, "tip": "<b>Majority vote</b> over the four workers' answers"}]
            for w, c in ws.items():
                persona = w not in SPECIALISTS
                name = w if persona else SPECIALISTS[w][0]
                base = "Qwen3.5-4B" if persona else SPECIALISTS[w][1]
                wm = worker_mode(config, w, mode)
                items.append({"label": f"{name} · {wm}", "bold": c == top, "c": c, "q": q, "distinct": distinct, "unit": unit,
                              "colour": PERSONA_COLOUR if persona else SPECIALIST_COLOUR,
                              "tip": f"<b>{name}</b> · thinking {wm}<br><span class=m>{'persona' if persona else 'specialist'} · {base}</span>"})
            svg = bar_svg(items, f"{title}, {MODES[mode]}: orchestrator, vote and each worker")
            panes.append(f"<div class='x1-mode' data-mode='{mode}'{hidden}>"
                         f"<div class='x1-sub'>{len(rs)} of 15 runs · {distinct} {unit}</div><div class='x1-svg'>{svg}</div></div>")
        figs.append(f"      <div class='x1-chart'><h3>{title}</h3>{''.join(panes)}</div>")
    seg = "".join(f"<button type='button' data-mode='{m}' aria-pressed='{'true' if m == 'headline' else 'false'}'>{MODES[m]}</button>"
                  for m in MODES)
    caption = ("Accuracy of the orchestrator, the majority vote and each worker on the same runs; whiskers = 95% Wilson "
               "interval over the team's distinct questions or tasks; best worker in bold.")
    if split:
        caption = ("Teams that draw on more than one dataset, one chart per dataset: accuracy of the orchestrator, the majority "
                   "vote and each worker on the same runs; whiskers = 95% Wilson interval over the distinct questions; best worker in bold.")
    return "\n".join([
        "    <figure class='x1-hubfig'>",
        f"      <figcaption>{caption}</figcaption>",
        f"      <div class='controls'><div class='control'><span class='label'>Specialists' thinking</span><div class='seg x1-seg'>{seg}</div></div></div>",
        "      <div class='legend'><span><i class='swatch sq' style='background:var(--s-continual)'></i> orchestrator</span>"
        f"<span><i class='swatch sq' style='background:var(--faint)'></i> majority vote</span>{LEGEND_WORKERS}</div>",
        "      <div class='x1-charts'>", *figs, "      </div>",
        "    </figure>"])


HUB_EXPLAINER = """    <h3 class="x1-what">What the orchestrator does</h3>
    <ul class="findings">
      <li>Each team is fixed: four workers (approach personas on Qwen3.5-4B plus fine-tuned specialists) and an orchestrator, Qwen3.6-35B-A3B with thinking off. The orchestrator does not choose the team.</li>
      <li>Stage 1: every worker answers each question alone, and its answer is saved. Stage 2 replays those saved answers to the orchestrator.</li>
      <li>Hard questions: the orchestrator gets the question and the four answers, labelled Agent 1–4 rather than by name (each worker's closing summary and final answer, or its last 600 characters if it wrote no summary). It is told to compare the answers, check the reasoning and decide the team's answer, in one call with no tools.</li>
      <li>Agentic tasks: the orchestrator reads each worker's actions and final message, then does the task itself with the same tools in a fresh environment.</li>
      <li>It is scored against the majority vote over the same four answers and against the best single worker, picked in hindsight. Fixed = orchestrator right where the vote is wrong; broke = the reverse.</li>
      <li>Thinking: headline = each specialist in its better stage 1 mode; all on / all off = every specialist in that mode where it has one. Personas and the orchestrator always have thinking off.</li>
    </ul>"""


CHART_STYLE = """  <style>
    #x1-teams table.x1-workers { min-width: 760px; }
    #x1-teams table.x1-workers th, #x1-teams table.x1-workers td { white-space: nowrap; padding-right: 10px; }
    #x1-teams table.x1-workers td.num, #x1-teams table.x1-workers th.num { padding-left: 14px; padding-right: 0; }
    #x1-teams .x1-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 24px 36px; }
    #x1-teams .x1-chart h3 { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 0.8rem; font-weight: 500; letter-spacing: 0.04em; margin-bottom: 6px; }
    #x1-teams .x1-chart h3 span { color: var(--faint); font-weight: 400; margin-left: 6px; }
    #x1-teams .x1-svg { overflow-x: auto; }
    #x1-teams .x1-svg svg { min-width: 420px; width: 100%; height: auto; display: block; }
    #x1-teams .x1-bar { cursor: default; outline: none; }
    #x1-teams .x1-bar:hover rect, #x1-teams .x1-bar:focus rect { fill: var(--accent-soft); }
    #x1-teams .x1-sub, #x1-teams .x1-none { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 0.74rem; color: var(--faint); margin-bottom: 4px; }
    #x1-teams h3.x1-what { font-size: var(--step-1); }
  </style>"""

CHART_SCRIPT = """  <script>
  (function () {
    var panel = document.getElementById('x1-teams');
    if (!panel) return;
    var tip = document.createElement('div');
    tip.className = 'tip'; tip.hidden = true; document.body.appendChild(tip);
    function show(g, x, y) {
      tip.innerHTML = g.getAttribute('data-tip'); tip.hidden = false;
      var w = tip.offsetWidth, hgt = tip.offsetHeight;
      tip.style.left = Math.min(x + 14, window.innerWidth - w - 8) + 'px';
      tip.style.top = Math.max(8, y - hgt - 10) + 'px';
    }
    panel.querySelectorAll('.x1-bar').forEach(function (g) {
      g.addEventListener('mousemove', function (e) { show(g, e.clientX, e.clientY); });
      g.addEventListener('mouseleave', function () { tip.hidden = true; });
      g.addEventListener('focus', function () { var b = g.getBoundingClientRect(); show(g, b.left + b.width / 2, b.top); });
      g.addEventListener('blur', function () { tip.hidden = true; });
    });
    panel.querySelectorAll('.x1-seg').forEach(function (seg) {
      var fig = seg.closest('figure');
      seg.querySelectorAll('button').forEach(function (b) {
        b.addEventListener('click', function () {
          seg.querySelectorAll('button').forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
          fig.querySelectorAll('.x1-mode').forEach(function (m) { m.hidden = m.getAttribute('data-mode') !== b.getAttribute('data-mode'); });
        });
      });
    });
  })();
  </script>"""


def table(rows, pool, total_runs):
    judged_note = "; answered = episodes that submitted an answer (BrowseComp-Plus and Finance-Agent are judged by the 35B in stage 2)"
    if pool == "hard":
        caption = ("Hard questions: each worker alone on its team's held-out questions, over the runs finished "
                   f"(of {total_runs}: 5 folds × 3 samples); matched = only the runs every row of the team has finished; "
                   "cut off = hit 30,720 tokens; tokens = mean completion per question.")
        last = ("Cut off", "Tokens")
    else:
        caption = ("Agentic tasks: success rate over the runs finished "
                   f"(of {total_runs}); matched = only the runs every row of the team has finished{judged_note}.")
        last = ("Answered", "Tokens")
    head = "".join(f"<th>{h}</th>" for h in ("Team", "Worker", "Model", "Thinking"))
    head += "".join(f"<th class='num'>{h}</th>" for h in ("Runs", "Correct", "Matched", *last))
    out = [f"    <div class=\"tablewrap\"><table class=\"x1-workers\"><caption>{caption}</caption>",
           f"    <thead><tr>{head}</tr></thead><tbody>"]
    teams = [t for t in TEAMS if any(r["team"] == t and r["pool"] == pool for r in rows)]
    for i, team in enumerate(teams):
        members = [r for r in rows if r["team"] == team]
        for j, r in enumerate(members):
            label, base = label_of(r)
            cls = " class='group-gap'" if j == 0 and i else ""
            first = f"<strong>{team}</strong> {TEAMS[team]}" if j == 0 else ""
            name = f"<strong>{label}</strong>" if r["best"] else label
            model = "persona" if r["persona"] else base.replace("-Base-2512", "-Base")
            cells = [first, f"<span class='arm'>{name}</span>", model, r["mode"]]
            tds = "".join(f"<td>{c}</td>" for c in cells)
            tds += f"<td class='num'>{r['runs']}</td>"
            if r["judged"]:
                stage2 = "<td class='num' style='color:var(--faint)'>stage 2</td>"
                tds += stage2 + stage2 + pct(r["answered"], r["questions"])
            else:
                tds += pct(r["correct"], r["questions"], bold=r["best"]) + pct(r["m_c"], r["m_q"], sub=False, bold=r["best"])
                if pool == "hard":
                    tds += pct(r["cut_off"], r["questions"], sub=False) if r["questions"] else pct(0, 0)
                else:
                    tds += pct(r["answered"], r["questions"])
            tok = f"{r['completion_tokens'] / r['questions'] / 1000:.1f}k" if r["questions"] else "—"
            tds += f"<td class='num'>{tok}</td>"
            out.append(f"<tr{cls}>{tds}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


MODES = {"headline": "headline", "on": "all on", "off": "all off"}


def stage2_table(records):
    groups = {}
    for r in records:
        groups.setdefault((r["team"], r["modes"]), []).append(r)
    complete = all(len(rs) == 15 for rs in groups.values())
    runs = "all 15 fold/sample runs" if complete else "the fold/sample runs finished (of 15)"
    caption = (f"Orchestrator (35B, thinking off) reading its team's four answers, summed over {runs}; specialists' thinking: headline = each one's better stage 1 mode, personas always off; "
               "best worker = the team member with most correct over the same runs, picked in hindsight; "
               "fixed / broke = questions where the orchestrator is right and the majority vote wrong, and the reverse.")
    head = "".join(f"<th>{h}</th>" for h in ("Team", "Thinking", "Best worker"))
    head += "".join(f"<th class='num'>{h}</th>" for h in ("Runs", "Orchestrator", "Vote", "Best worker", "Fixed", "Broke"))
    out = [f"    <div class=\"tablewrap\"><table class=\"x1-workers\"><caption>{caption}</caption>",
           f"    <thead><tr>{head}</tr></thead><tbody>"]
    teams = [t for t in TEAMS if any(k[0] == t for k in groups)]
    for i, team in enumerate(teams):
        for j, mode in enumerate(m for m in MODES if (team, m) in groups):
            rs = groups[(team, mode)]
            q = sum(r["questions"] for r in rs)
            ws = {}
            for r in rs:
                for w, c in r["workers_correct"].items():
                    ws[w] = ws.get(w, 0) + c
            best = max(ws, key=ws.get)
            label = SPECIALISTS[best][0] if best in SPECIALISTS else best
            sumk = lambda k: sum(r[k] for r in rs)
            trans = lambda k: sum(r["hub_transitions"][k] for r in rs)
            cls = " class='group-gap'" if j == 0 and i else ""
            first = f"<strong>{team}</strong> {TEAMS[team]}" if j == 0 else ""
            tds = f"<td>{first}</td><td>{MODES[mode]}</td><td><span class='arm'>{label}</span></td>"
            tds += f"<td class='num'>{len(rs)}</td>"
            tds += pct(sumk("hub_correct"), q) + pct(sumk("vote_correct"), q) + pct(ws[best], q)
            tds += f"<td class='num'>{trans('fixed')}</td><td class='num'>{trans('broke')}</td>"
            out.append(f"<tr{cls}>{tds}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def dataset_rows(records, mode="headline"):
    """Rows of the by-dataset view for one thinking mode: per dataset, an "all teams" row where several
    teams draw on it, then one row per team. Counts are answers (questions × samples); "distinct" is the
    number of different questions."""
    groups = {}
    for r in records:
        if r["modes"] == mode:
            groups.setdefault((r["dataset"], r["team"]), []).append(r)
    rows = []
    for dataset in DATASETS:
        teams = [t for t in TEAMS if (dataset, t) in groups]
        members = []
        for team in teams:
            rs = groups[(dataset, team)]
            folds = {}
            for r in rs:
                folds[r["fold"]] = max(folds.get(r["fold"], 0), r["questions"])
            ws = {}
            for r in rs:
                for w, c in r["workers_correct"].items():
                    ws[w] = ws.get(w, 0) + c
            best = max(ws, key=ws.get)
            members.append({"dataset": dataset, "team": team, "runs": len(rs), "distinct": sum(folds.values()),
                            "q": sum(r["questions"] for r in rs), "hub": sum(r["hub_correct"] for r in rs),
                            "vote": sum(r["vote_correct"] for r in rs), "best": best, "best_c": ws[best],
                            "persona": best not in SPECIALISTS,
                            "fixed": sum(r["hub_transitions"]["fixed"] for r in rs),
                            "broke": sum(r["hub_transitions"]["broke"] for r in rs)})
        if len(members) > 1:
            total = lambda k: sum(m[k] for m in members)
            rows.append({"dataset": dataset, "team": None, "teams": len(members), "distinct": total("distinct"),
                         "q": total("q"), "hub": total("hub"), "vote": total("vote"), "best": None,
                         "fixed": total("fixed"), "broke": total("broke")})
        rows.extend(members)
    return rows


def worker_label(worker):
    return SPECIALISTS[worker][0] if worker in SPECIALISTS else worker


def dataset_table(records):
    rows = dataset_rows(records)
    caption = ("Stage 2 by source dataset, headline thinking modes: the orchestrator (35B, thinking off) against its team's majority "
               "vote and its best worker on that dataset (picked in hindsight), over all fold/sample runs; questions = different "
               "questions, each answered in 3 samples; fixed / broke = orchestrator right where the vote is wrong, and the reverse.")
    head = "".join(f"<th>{h}</th>" for h in ("Dataset", "Team", "Best worker"))
    head += "".join(f"<th class='num'>{h}</th>" for h in ("Questions", "Orchestrator", "Vote", "Best worker", "Fixed", "Broke"))
    out = [f"    <div class=\"tablewrap\"><table class=\"x1-workers\"><caption>{caption}</caption>",
           f"    <thead><tr>{head}</tr></thead><tbody>"]
    last = None
    for r in rows:
        first = r["dataset"] != last
        cls = " class='group-gap'" if first and last is not None else ""
        name = f"<strong>{DATASETS[r['dataset']]}</strong>" if first else ""
        last = r["dataset"]
        if r["team"] is None:
            tds = f"<td>{name}</td><td>all {r['teams']} teams</td><td style='color:var(--faint)'>—</td>"
            best = "<td class='num' style='color:var(--faint)'>—</td>"
        else:
            tds = (f"<td>{name}</td><td><strong>{r['team']}</strong> {TEAMS[r['team']]}</td>"
                   f"<td><span class='arm'>{worker_label(r['best'])}</span></td>")
            best = pct(r["best_c"], r["q"])
        tds += f"<td class='num'>{r['distinct']}</td>" + pct(r["hub"], r["q"]) + pct(r["vote"], r["q"]) + best
        tds += f"<td class='num'>{r['fixed']}</td><td class='num'>{r['broke']}</td>"
        out.append(f"<tr{cls}>{tds}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


DW, DLX, DX0, DX1, DVX = 520, 196, 204, 470, 518


def dataset_dots_svg(rows, aria):
    """One line per dataset and team: majority vote (hollow), best worker (square) and orchestrator (filled, with 95% interval)."""
    rows = [r for r in rows if r["team"] is not None]
    sx = lambda v: DX0 + (DX1 - DX0) * v
    gaps = sum(1 for a, b in zip(rows, rows[1:]) if a["dataset"] != b["dataset"])
    h = TOP + RH * len(rows) + 8 * gaps + 22
    svg = [f"<svg viewBox='0 0 {DW} {h}' role='img' aria-label='{html.escape(aria, quote=True)}' "
           "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='11'>"]
    for t in (0, 0.25, 0.5, 0.75, 1):
        x = sx(t)
        svg.append(f"<line x1='{x:.1f}' y1='{TOP - 2}' x2='{x:.1f}' y2='{h - 20}' stroke='var(--rule)' stroke-width='1'/>")
        svg.append(f"<text x='{x:.1f}' y='{h - 6}' text-anchor='middle' fill='var(--faint)' font-size='10'>{int(t * 100)}%</text>")
    y = TOP
    for i, r in enumerate(rows):
        if i and r["dataset"] != rows[i - 1]["dataset"]:
            y += 8
        cy = y + RH / 2
        hub, vote, best = r["hub"] / r["q"], r["vote"] / r["q"], r["best_c"] / r["q"]
        lo, hi = wilson(hub, r["distinct"])
        name, subject = DATASETS[r["dataset"]], TEAMS[r["team"]]
        label = name if name == subject else f"{name} · {subject}"
        colour = PERSONA_COLOUR if r["persona"] else SPECIALIST_COLOUR
        tip = (f"<b>{DATASETS[r['dataset']]}</b> · {r['team']} {TEAMS[r['team']]}<br>"
               f"orchestrator {100 * hub:.1f}% ({r['hub']}/{r['q']}), 95% interval {100 * lo:.0f}–{100 * hi:.0f}%<br>"
               f"majority vote {100 * vote:.1f}%<br>best worker {100 * best:.1f}%: {html.escape(worker_label(r['best']))}<br>"
               f"<span class=m>{r['distinct']} questions × 3 samples · fixed {r['fixed']}, broke {r['broke']}</span>")
        lo_x, hi_x = sx(min(hub, vote, best)), sx(max(hub, vote, best))
        svg.append(f"<g class='x1-bar' tabindex='0' data-tip='{html.escape(tip, quote=True)}'>"
                   f"<rect x='0' y='{y}' width='{DW}' height='{RH}' fill='transparent'/>"
                   f"<text x='{DLX}' y='{cy + 4:.1f}' text-anchor='end' fill='var(--ink)'>{html.escape(label)}</text>"
                   f"<line x1='{lo_x:.1f}' y1='{cy}' x2='{hi_x:.1f}' y2='{cy}' stroke='var(--rule-strong)' stroke-width='2'/>"
                   f"<line x1='{sx(lo):.1f}' y1='{cy}' x2='{sx(hi):.1f}' y2='{cy}' stroke='{HUB_COLOUR}' stroke-width='1.5' opacity='0.6'/>"
                   f"<circle cx='{sx(vote):.1f}' cy='{cy}' r='4.5' fill='var(--surface)' stroke='var(--muted)' stroke-width='1.5'/>"
                   f"<rect x='{sx(best) - 4:.1f}' y='{cy - 4}' width='8' height='8' fill='{colour}'/>"
                   f"<circle cx='{sx(hub):.1f}' cy='{cy}' r='5.5' fill='{HUB_COLOUR}'/>"
                   f"<text x='{DVX}' y='{cy + 4:.1f}' text-anchor='end' fill='var(--ink)' font-weight='700'>{100 * hub:.0f}%</text></g>")
        y += RH
    svg.append("</svg>")
    return "".join(svg)


def dataset_dots(records):
    panes = []
    for mode in MODES:
        rows = dataset_rows(records, mode)
        hidden = "" if mode == "headline" else " hidden"
        if not rows:
            panes.append(f"<div class='x1-mode' data-mode='{mode}'{hidden}><p class='x1-none'>No {MODES[mode]} runs yet.</p></div>")
            continue
        svg = dataset_dots_svg(rows, f"Stage 2 by dataset, {MODES[mode]}: orchestrator, majority vote and best worker")
        panes.append(f"<div class='x1-mode' data-mode='{mode}'{hidden}><div class='x1-svg'>{svg}</div></div>")
    seg = "".join(f"<button type='button' data-mode='{m}' aria-pressed='{'true' if m == 'headline' else 'false'}'>{MODES[m]}</button>"
                  for m in MODES)
    return "\n".join([
        "    <figure class='x1-hubfig'>",
        "      <figcaption>Each dataset and team: the orchestrator (with its 95% Wilson interval over the distinct questions), the majority "
        "vote and the best worker on that dataset, picked in hindsight; value = orchestrator.</figcaption>",
        f"      <div class='controls'><div class='control'><span class='label'>Specialists' thinking</span><div class='seg x1-seg'>{seg}</div></div></div>",
        "      <div class='legend'><span><i class='swatch' style='background:var(--s-continual)'></i> orchestrator</span>"
        "<span><i class='swatch hollow'></i> majority vote</span>"
        "<span><i class='swatch sq' style='background:var(--s-random)'></i> best worker, persona</span>"
        "<span><i class='swatch sq' style='background:var(--s-no_memory)'></i> best worker, specialist</span></div>",
        "      <div class='x1-dots'>", *panes, "      </div>",
        "    </figure>"])


def bullets(items):
    return "\n".join(["    <ul class=\"findings\">", *(f"      <li>{b}</li>" for b in items), "    </ul>"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", nargs="+", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--stage2")
    parser.add_argument("--stage2-datasets")
    parser.add_argument("live")
    parser.add_argument("out")
    args = parser.parse_args()
    config = load_config()
    pools = {t["id"]: t["pool"] for t in config["teams"]}
    rows = merge(args.summary)
    for r in rows:
        r["pool"] = pools[r["team"]]
    for team in TEAMS:
        members = [r for r in rows if r["team"] == team]
        if members:
            matched_stats(members)
    total = config["settings"]["folds"] * config["settings"]["samples"]
    # A run answers every team that shares its pool, so count each (pool, worker, mode, run) once.
    done = len({(r["pool"], r["worker"], r["mode"], k) for r in rows for k in r["by_run"]})
    planned = len({(w["pool"], w["worker"], w["mode"]) for w in workers(config)}) * total
    fill = lambda s: s.replace("{done}", str(done)).replace("{planned}", str(planned))
    notes = json.loads(fill(Path(args.notes).read_text()))
    section = "\n".join([
        "  <section>",
        f"    <div class=\"sec-head\"><div class=\"kicker\">{notes['kicker']}</div><h2>{notes.get('stage1_title', 'Workers answering alone, so far')}</h2></div>",
        CHART_STYLE,
        charts(rows, "hard"), table(rows, "hard", total), bullets(notes["hard"]),
        charts(rows, "agentic"), table(rows, "agentic", total), bullets(notes["agentic"]),
        CHART_SCRIPT,
        "  </section>"])
    status = "\n".join([
        "  <div class=\"status\">",
        f"    <span class=\"label\">Status <span class=\"pill {notes.get('pill_class', 'running')}\">{notes['pill']}</span></span>",
        *(f"    <p>{p}</p>" for p in notes["status"]),
        "  </div>"])

    page = Path(args.live).read_text()
    start, end = page.index("<!-- X1:START -->"), page.index("<!-- X1:END -->")
    tab = page[start:end]
    tab, n1 = re.subn(r"  <section>\n    <div class=\"sec-head\"><div class=\"kicker\">[^<]*</div><h2>Workers answering alone[^<]*</h2>.*?\n  </section>",
                      lambda _: section, tab, count=1, flags=re.S)
    tab, n2 = re.subn(r"  <div class=\"status\">.*?\n  </div>", lambda _: status, tab, count=1, flags=re.S)
    tab, n3 = re.subn(r"(<div class=\"eyebrow\">)[^<]*(</div>)", lambda m: m[1] + notes["eyebrow"] + m[2], tab, count=1)
    if (n1, n2, n3) != (1, 1, 1):
        sys.exit(f"X1 tab layout changed: stage 1 section {n1}, status {n2}, eyebrow {n3}")
    if args.stage2:
        records2 = json.load(open(args.stage2))
        by_dataset = []
        if args.stage2_datasets:
            records3 = json.load(open(args.stage2_datasets))
            by_dataset = ["    <h3 class=\"x1-what\">By dataset</h3>", dataset_table(records3), bullets(notes.get("stage2_datasets", [])),
                          dataset_dots(records3), stage2_charts(records3, config, split=True)]
        hub = "\n".join([
            "  <section>",
            f"    <div class=\"sec-head\"><div class=\"kicker\">{notes['stage2_kicker']}</div><h2>{notes.get('stage2_title', 'Orchestrator with its team, so far')}</h2></div>",
            HUB_EXPLAINER, stage2_charts(records2, config),
            stage2_table(records2), bullets(notes["stage2"]), *by_dataset,
            "  </section>"])
        pattern = r"  <section>\n    <div class=\"sec-head\"><div class=\"kicker\">[^<]*</div><h2>(?:Hub|Orchestrator) with its team[^<]*</h2>.*?\n  </section>"
        tab, n4 = re.subn(pattern, lambda _: hub, tab, count=1, flags=re.S)
        if not n4:
            at = tab.index(section)
            tab = tab[:at] + hub + "\n\n" + tab[at:]
    Path(args.out).write_text(page[:start] + tab + page[end:])


if __name__ == "__main__":
    main()
