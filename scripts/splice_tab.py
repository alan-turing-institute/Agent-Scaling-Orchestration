"""Splice a tab block into a saved copy of the results artifact.

The block runs from <!-- NAME:START --> to <!-- NAME:END --> and holds one tabpanel. If the page
already has the block it is replaced; otherwise it goes in after the last tab block and its button
after the last tab button. Always splice into a fresh read of the live page: other sessions update
other tabs.

Usage: python scripts/splice_tab.py LIVE.html OUT.html --tab TAB.html --name AGENTIC \
           --tab_id agentic-baselines --label "Agentic baselines"
"""

import argparse
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("live")
    ap.add_argument("out")
    ap.add_argument("--tab", required=True)
    ap.add_argument("--name", required=True, help="marker name, e.g. AGENTIC for <!-- AGENTIC:START -->")
    ap.add_argument("--tab_id", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    page, tab = Path(args.live).read_text(), Path(args.tab).read_text().rstrip("\n")
    start, end = f"<!-- {args.name}:START -->", f"<!-- {args.name}:END -->"
    if start in page:
        a, b = page.index(start), page.index(end) + len(end)
        page = page[:a] + tab + page[b:]
    else:
        last_end = list(re.finditer(r"<!-- [A-Z0-9_]+:END -->", page))[-1].end()
        page = page[:last_end] + "\n" + tab + page[last_end:]
        last_button = list(re.finditer(r'<button type="button" role="tab"[^\n]*\n', page))[-1].end()
        button = (f'    <button type="button" role="tab" id="tabbtn-{args.tab_id}" aria-controls="{args.tab_id}" '
                  f'aria-selected="false" tabindex="-1">{args.label}</button>\n')
        page = page[:last_button] + button + page[last_button:]
    assert page.count(f'id="{args.tab_id}"') == 1 and page.count(f'id="tabbtn-{args.tab_id}"') == 1
    Path(args.out).write_text(page)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
