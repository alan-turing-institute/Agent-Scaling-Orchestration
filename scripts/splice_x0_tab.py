"""Splice the X0 tab (data-claude/x0-headroom/tab.html) into a saved copy of the results artifact.

Replaces the block between <!-- X0:START --> and <!-- X0:END --> if the page has one; otherwise
inserts it after <!-- HARD:END --> and adds its tab button after the "Harder benchmarks" button.
Always splice into a fresh read of the live page: other sessions update other tabs.

Usage: python scripts/splice_x0_tab.py LIVE.html OUT.html [--tab data-claude/x0-headroom/tab.html]
"""

import argparse
from pathlib import Path

BUTTON = ('    <button type="button" role="tab" id="tabbtn-x0-headroom" aria-controls="x0-headroom" '
          'aria-selected="false" tabindex="-1">Headroom (X0)</button>\n')
HARDER_BUTTON = 'id="tabbtn-harder"'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("live")
    ap.add_argument("out")
    ap.add_argument("--tab", default="data-claude/x0-headroom/tab.html")
    args = ap.parse_args()
    page, tab = Path(args.live).read_text(), Path(args.tab).read_text().rstrip("\n")
    if "<!-- X0:START -->" in page:
        a = page.index("<!-- X0:START -->")
        b = page.index("<!-- X0:END -->") + len("<!-- X0:END -->")
        page = page[:a] + tab + page[b:]
    else:
        end = page.index("<!-- HARD:END -->") + len("<!-- HARD:END -->")
        page = page[:end] + "\n\n" + tab + page[end:]
        i = page.index(HARDER_BUTTON)
        line_end = page.index("\n", i) + 1
        page = page[:line_end] + BUTTON + page[line_end:]
    assert page.count('id="x0-headroom"') == 1 and page.count('id="tabbtn-x0-headroom"') == 1
    Path(args.out).write_text(page)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
