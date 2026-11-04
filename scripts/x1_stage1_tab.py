"""Splice X1 stage 1 results into the X1 tab of the results page ("Does Orchestrator Memory Help?").

    python scripts/x1_stage1_tab.py --summary spark.json local.json --notes notes.json LIVE.html OUT.html

`--stage2` (optional) takes a JSON list of stage 2 holdout_summary.json records; with it, a
"Hub with its team" section (notes "stage2_kicker" and "stage2" bullets) goes above stage 1.

`--summary` takes x1_stage1_summary.py output from each box running stage 1; a run present in
more than one is counted once. In the notes, "{done}" and "{planned}" become the counts of finished
and planned runs. `--notes` is a JSON file of the hand-written parts: "eyebrow",
"kicker", "hard" and "agentic" (finding bullets, HTML) and "status" (paragraphs, HTML). The stage 1
section, the eyebrow and the status box of the tab between <!-- X1:START --> and <!-- X1:END -->
are replaced; everything else on the page is kept.

Each row shows every finished run, and "matched" restricts the team to the runs all its rows
have finished, so workers are compared on the same questions.
"""

import argparse
import json
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


def pct(num, den, sub=True):
    if not den:
        return "<td class='num' style='color:var(--faint)'>—</td>"
    s = f"<span class='sub'>{num}/{den}</span>" if sub else ""
    return f"<td class='num'>{100 * num / den:.0f}%{s}</td>"


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
    head = "".join(f"<th>{h}</th>" for h in ("Team", "Worker", "Kind", "Base model", "Size", "Thinking"))
    head += "".join(f"<th class='num'>{h}</th>" for h in ("Runs", "Correct", "Matched", *last))
    out = [f"    <div class=\"tablewrap\"><table><caption>{caption}</caption>",
           f"    <thead><tr>{head}</tr></thead><tbody>"]
    teams = [t for t in TEAMS if any(r["team"] == t and r["pool"] == pool for r in rows)]
    for i, team in enumerate(teams):
        members = [r for r in rows if r["team"] == team]
        shared = set.intersection(*(set(r["by_run"]) for r in members))
        for j, r in enumerate(members):
            if r["persona"]:
                label, base, size = r["worker"], "Qwen3.5-4B", "4B dense"
            else:
                label, base, size = SPECIALISTS[r["worker"]]
            cls = " class='group-gap'" if j == 0 and i else ""
            first = f"<strong>{team}</strong> {TEAMS[team]}" if j == 0 else ""
            cells = [first, f"<span class='arm'>{label}</span>", "persona" if r["persona"] else "specialist", base, size, r["mode"]]
            tds = "".join(f"<td>{c}</td>" for c in cells)
            tds += f"<td class='num'>{r['runs']}</td>"
            matched = [r["by_run"][k] for k in shared]
            mq, mc, ma = (sum(m[k] for m in matched) for k in ("questions", "correct", "answered"))
            if r["judged"]:
                stage2 = "<td class='num' style='color:var(--faint)'>stage 2</td>"
                tds += stage2 + stage2 + pct(r["answered"], r["questions"])
            else:
                tds += pct(r["correct"], r["questions"]) + pct(mc, mq, sub=False)
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
    caption = ("Hub (35B, thinking off) reading its team's four answers, summed over the fold/sample runs finished "
               "(of 15); specialists' thinking: headline = each one's better stage 1 mode, personas always off; "
               "best worker = the team member with most correct over the same runs, picked in hindsight; "
               "fixed / broke = questions where the hub is right and the majority vote wrong, and the reverse.")
    head = "".join(f"<th>{h}</th>" for h in ("Team", "Thinking", "Best worker"))
    head += "".join(f"<th class='num'>{h}</th>" for h in ("Runs", "Hub", "Vote", "Best worker", "Fixed", "Broke"))
    out = [f"    <div class=\"tablewrap\"><table><caption>{caption}</caption>",
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


def bullets(items):
    return "\n".join(["    <ul class=\"findings\">", *(f"      <li>{b}</li>" for b in items), "    </ul>"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", nargs="+", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--stage2")
    parser.add_argument("live")
    parser.add_argument("out")
    args = parser.parse_args()
    config = load_config()
    pools = {t["id"]: t["pool"] for t in config["teams"]}
    rows = merge(args.summary)
    for r in rows:
        r["pool"] = pools[r["team"]]
    total = config["settings"]["folds"] * config["settings"]["samples"]
    # A run answers every team that shares its pool, so count each (pool, worker, mode, run) once.
    done = len({(r["pool"], r["worker"], r["mode"], k) for r in rows for k in r["by_run"]})
    planned = len({(w["pool"], w["worker"], w["mode"]) for w in workers(config)}) * total
    fill = lambda s: s.replace("{done}", str(done)).replace("{planned}", str(planned))
    notes = json.loads(fill(Path(args.notes).read_text()))
    section = "\n".join([
        "  <section>",
        f"    <div class=\"sec-head\"><div class=\"kicker\">{notes['kicker']}</div><h2>Workers answering alone, so far</h2></div>",
        table(rows, "hard", total), bullets(notes["hard"]),
        table(rows, "agentic", total), bullets(notes["agentic"]),
        "  </section>"])
    status = "\n".join([
        "  <div class=\"status\">",
        f"    <span class=\"label\">Status <span class=\"pill running\">{notes['pill']}</span></span>",
        *(f"    <p>{p}</p>" for p in notes["status"]),
        "  </div>"])

    page = Path(args.live).read_text()
    start, end = page.index("<!-- X1:START -->"), page.index("<!-- X1:END -->")
    tab = page[start:end]
    tab, n1 = re.subn(r"  <section>\n    <div class=\"sec-head\"><div class=\"kicker\">[^<]*</div><h2>Workers answering alone, so far</h2>.*?\n  </section>",
                      lambda _: section, tab, count=1, flags=re.S)
    tab, n2 = re.subn(r"  <div class=\"status\">.*?\n  </div>", lambda _: status, tab, count=1, flags=re.S)
    tab, n3 = re.subn(r"(<div class=\"eyebrow\">)[^<]*(</div>)", lambda m: m[1] + notes["eyebrow"] + m[2], tab, count=1)
    if (n1, n2, n3) != (1, 1, 1):
        sys.exit(f"X1 tab layout changed: stage 1 section {n1}, status {n2}, eyebrow {n3}")
    if args.stage2:
        hub = "\n".join([
            "  <section>",
            f"    <div class=\"sec-head\"><div class=\"kicker\">{notes['stage2_kicker']}</div><h2>Hub with its team, so far</h2></div>",
            stage2_table(json.load(open(args.stage2))), bullets(notes["stage2"]),
            "  </section>"])
        pattern = r"  <section>\n    <div class=\"sec-head\"><div class=\"kicker\">[^<]*</div><h2>Hub with its team, so far</h2>.*?\n  </section>"
        tab, n4 = re.subn(pattern, lambda _: hub, tab, count=1, flags=re.S)
        if not n4:
            at = tab.index(section)
            tab = tab[:at] + hub + "\n\n" + tab[at:]
    Path(args.out).write_text(page[:start] + tab + page[end:])


if __name__ == "__main__":
    main()
