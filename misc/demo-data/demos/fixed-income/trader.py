#!/usr/bin/env python3
# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

"""A trader's live pivot screen, redrawn four times a second.

The screen is a SUBSCRIBE to this trader's rows of the shared `screens` view.
Expanding or collapsing a node writes to the `expanded` table, and the new rows
arrive through the same subscription. Nothing about this trader is a separate
dataflow: their state is rows in two tables.

    pip install 'psycopg[binary]'
    python3 trader.py alice                 # interactive (curses)
    python3 trader.py alice --frames 3      # print three frames and exit

Keys: up/down select, enter expand/collapse, s cycle sort, v tree/2-D, q quit.
"""

import argparse
import curses
import os
import threading
import time

import psycopg

DSN = os.environ.get(
    "MZ_DSN", "host=localhost port=6875 user=materialize dbname=materialize"
)
# Everything the demo creates lives in the mz-demo-data skill's schema.
SEARCH_PATH = "-c search_path=materialize_demo"

TREE_COLS = [
    "level", "path", "parent", "label", "n", "market_value", "dv01",
    "avg_price", "min_price", "max_price", "median_mv",
]
CCYS = ["usd", "eur", "gbp", "jpy", "cad", "aud", "chf", "sek"]
SORTS = ["market_value", "dv01", "n", "label"]


def now_ms():
    return time.time() * 1000.0


class Subscription:
    """Maintains the current contents of a SUBSCRIBE, applied at progress boundaries.

    Updates are buffered until a progress message says their timestamp is
    complete, so the displayed state is always a consistent snapshot.
    """

    def __init__(self, query, columns):
        self.query = query
        self.columns = columns
        self.rows = {}  # row tuple -> multiplicity
        self.lock = threading.Lock()
        self.frontier = None
        self.changes = 0
        self.error = None
        self.conn = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        pending = []
        sql = f"COPY (SUBSCRIBE ({self.query}) WITH (PROGRESS)) TO STDOUT"
        try:
            with self.conn.cursor().copy(sql) as cp:
                for r in cp.rows():
                    ts, progressed = int(r[0]), r[1] == "t"
                    if not progressed:
                        pending.append((ts, int(r[2]), tuple(r[3:])))
                        continue
                    ready = [p for p in pending if p[0] < ts]
                    pending = [p for p in pending if p[0] >= ts]
                    with self.lock:
                        for _, diff, row in ready:
                            count = self.rows.get(row, 0) + diff
                            if count:
                                self.rows[row] = count
                            else:
                                self.rows.pop(row, None)
                        self.changes += len(ready)
                        self.frontier = ts
        except Exception as e:  # surfaced in the header
            self.error = e

    def snapshot(self):
        with self.lock:
            rows = [dict(zip(self.columns, r)) for r in self.rows]
            return rows, self.frontier, self.changes


def num(v):
    return float(v) if v not in (None, "") else float("nan")


def fmt_mm(v):
    return f"{num(v) / 1e6:10.1f}"


def fmt_k(v):
    return f"{num(v) / 1e3:9.1f}"


class Screen:
    def __init__(self, trader):
        self.trader = trader
        self.sort = 0
        self.view = "tree"
        self.selected = 0
        self.writer = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
        if not self.writer.execute(
            "SELECT 1 FROM traders WHERE trader = %s", (trader,)
        ).fetchone():
            self.writer.execute("INSERT INTO traders VALUES (%s)", (trader,))
        self.tree = Subscription(
            f"SELECT {', '.join(TREE_COLS)} FROM screens WHERE trader = '{trader}'",
            TREE_COLS,
        )
        self.ccy = Subscription(
            f"SELECT business_group, {', '.join(CCYS)}, total FROM pivot_group_by_ccy",
            ["business_group"] + CCYS + ["total"],
        )
        self.last_changes = (time.time(), 0)
        self.rate = 0.0
        self.visible = []

    def expanded(self):
        return {
            r[0]
            for r in self.writer.execute(
                "SELECT path FROM expanded WHERE trader = %s", (self.trader,)
            ).fetchall()
        }

    def toggle(self):
        if not self.visible or self.view != "tree":
            return
        node = self.visible[self.selected]
        if int(node["level"]) >= 3 or node["path"] == "All":
            return
        if node["path"] in self.expanded():
            self.writer.execute(
                "DELETE FROM expanded WHERE trader = %s AND path = %s",
                (self.trader, node["path"]),
            )
        else:
            self.writer.execute(
                "INSERT INTO expanded VALUES (%s, %s)", (self.trader, node["path"])
            )

    def header(self, sub):
        _, frontier, changes = sub.snapshot()
        t, c = self.last_changes
        if time.time() - t >= 1.0:
            self.rate = (changes - c) / (time.time() - t)
            self.last_changes = (time.time(), changes)
        if sub.error:
            return f"trader {self.trader} | subscription error: {sub.error}"
        if frontier is None:
            return f"trader {self.trader} | waiting for first snapshot..."
        asof = time.strftime("%H:%M:%S", time.localtime(frontier / 1000))
        asof += f".{frontier % 1000:03d}"
        return (
            f"trader {self.trader} | as of {asof} | lag {now_ms() - frontier:4.0f}ms"
            f" | {self.rate:6.0f} row changes/s | redraw 4Hz"
            f" | sort {SORTS[self.sort]} | view {self.view}"
        )

    def tree_lines(self):
        rows, _, _ = self.tree.snapshot()
        children = {}
        for r in rows:
            children.setdefault(r["parent"], []).append(r)
        key = SORTS[self.sort]

        def order(rs):
            if key == "label":
                return sorted(rs, key=lambda r: r["label"])
            return sorted(rs, key=lambda r: -abs(num(r[key])))

        visible = []

        def walk(parent):
            for r in order(children.get(parent, [])):
                visible.append(r)
                walk(r["path"])

        walk(None)
        self.visible = visible
        self.selected = min(self.selected, max(0, len(visible) - 1))
        head = (
            f"{'':32} {'rows':>7} {'MV (mm)':>10} {'DV01 (k)':>9} {'avg px':>8}"
            f" {'min px':>8} {'max px':>8} {'median MV (k)':>13}"
        )
        lines = [head]
        for r in visible:
            level = int(r["level"])
            has_kids = level < 3 and r["path"] != "All"
            mark = ("-" if any(c["parent"] == r["path"] for c in rows) else "+") if has_kids else " "
            label = f"{'  ' * level}{mark} {r['label']}"
            lines.append(
                f"{label:32.32} {int(r['n']):7d} {fmt_mm(r['market_value'])}"
                f" {fmt_k(r['dv01'])} {num(r['avg_price']):8.3f}"
                f" {num(r['min_price']):8.3f} {num(r['max_price']):8.3f}"
                f" {fmt_k(r['median_mv']):>13}"
            )
        return lines

    def ccy_lines(self):
        rows, _, _ = self.ccy.snapshot()
        head = f"{'MV (mm)':14}" + "".join(f"{c.upper():>9}" for c in CCYS) + f"{'TOTAL':>10}"
        lines = [head]
        for r in sorted(rows, key=lambda r: r["business_group"]):
            lines.append(
                f"{r['business_group']:14}"
                + "".join(f"{num(r[c]) / 1e6:9.1f}" for c in CCYS)
                + f"{num(r['total']) / 1e6:10.1f}"
            )
        return lines

    def frame(self):
        sub = self.tree if self.view == "tree" else self.ccy
        body = self.tree_lines() if self.view == "tree" else self.ccy_lines()
        return [self.header(sub), ""] + body


def plain(screen, frames, interval):
    # Wait for the first snapshot, which takes a while right after load.sh.
    deadline = time.time() + 120
    while not screen.tree.snapshot()[0] and time.time() < deadline:
        time.sleep(0.25)
    for i in range(frames):
        print("\n".join(screen.frame()))
        print()
        if i + 1 < frames:
            time.sleep(interval)


def interactive(stdscr, screen, interval):
    curses.curs_set(0)
    stdscr.nodelay(True)
    while True:
        key = stdscr.getch()
        while key != -1:
            if key in (ord("q"), 27):
                return
            if key == curses.KEY_UP:
                screen.selected = max(0, screen.selected - 1)
            elif key == curses.KEY_DOWN:
                screen.selected += 1
            elif key in (10, 13, curses.KEY_ENTER, ord(" ")):
                screen.toggle()
            elif key == ord("s"):
                screen.sort = (screen.sort + 1) % len(SORTS)
            elif key == ord("v"):
                screen.view = "ccy" if screen.view == "tree" else "tree"
            key = stdscr.getch()
        lines = screen.frame()
        height, width = stdscr.getmaxyx()
        stdscr.erase()
        for y, line in enumerate(lines[: height - 1]):
            attr = curses.A_REVERSE if screen.view == "tree" and y - 3 == screen.selected else 0
            stdscr.addnstr(y, 0, line, width - 1, attr)
        stdscr.refresh()
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trader")
    ap.add_argument("--frames", type=int, help="print this many frames and exit")
    ap.add_argument("--interval", type=float, default=0.25, help="redraw period (s)")
    ap.add_argument("--view", choices=["tree", "ccy"], default="tree")
    args = ap.parse_args()
    screen = Screen(args.trader)
    screen.view = args.view
    if args.frames:
        plain(screen, args.frames, args.interval)
    else:
        curses.wrapper(interactive, screen, args.interval)


if __name__ == "__main__":
    main()
