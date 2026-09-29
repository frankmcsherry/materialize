#!/usr/bin/env python3
# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

"""A live risk board in the browser, one screen per trader.

Serves board.html and streams each trader's screen to it four times a second
over server-sent events. Two ways to serve a layout:

  hot        Every row and across dimension is one of the five cube
             dimensions. The layout is rows in board_layout / board_across /
             board_expanded / board_measures, and the screen is one long-lived
             SUBSCRIBE to `board WHERE trader = ...`. Expanding a node is an
             INSERT. No per-trader dataflow.
  on demand  Any other column is involved (issuer, book, seniority, ...). The
             server writes a GROUP BY over the blotter index for exactly the
             visible nodes, and restarts that SUBSCRIBE on every click. The
             board shows how long the click took to reach numbers.

The positions panel is a SUBSCRIBE per selected node, the trade tape one
SUBSCRIBE shared by everyone.

    PIVOTS=board ./load.sh
    pip install 'psycopg[binary]'
    python3 board.py                  # then open http://localhost:8765/?trader=alice

Env: MZ_DSN (default localhost:6875 as materialize), BOARD_PORT (8765).
"""

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import psycopg

DSN = os.environ.get(
    "MZ_DSN", "host=localhost port=6875 user=materialize dbname=materialize"
)
# Everything the demo creates lives in the mz-demo-data skill's schema.
SEARCH_PATH = "-c search_path=materialize_demo"
HERE = os.path.dirname(os.path.abspath(__file__))

# Dimensions the cube pre-maintains, with their SQL over the blotter.
HOT = {
    "business_group": "business_group",
    "currency": "currency",
    "sector": "sector",
    "grade": "grade",
    "tenor": "benchmark",
}
BITS = {"business_group": 1, "currency": 2, "sector": 4, "grade": 8, "tenor": 16}
# Everything else a trader can pivot on, computed on demand. Reference fields
# come from bonds and books and rarely change.
REFERENCE = {
    "book": "book",
    "issuer": "issuer",
    "bond": "bond",
    "isin": "isin",
    "rating": "rating",
    "seniority": "seniority",
    "day_count": "day_count",
    "coupon_freq": "CASE coupon_freq WHEN 1 THEN 'Annual' ELSE 'Semi-annual' END",
    "coupon_band": "floor(coupon)::int::text || '-' || (floor(coupon)::int + 1)::text || '%'",
    "maturity_year": "extract(year FROM maturity)::int::text",
    "duration_band": "(floor(duration / 5) * 5)::int::text || '-' || (floor(duration / 5) * 5 + 5)::int::text || 'y'",
    "issue_size": "(issue_size / 1000000)::text || 'mm'",
    "structure": "CASE WHEN callable THEN 'Callable' ELSE 'Bullet' END",
}
# Fields computed from the position or its price, so a row can move between
# groups on any trade or price tick.
TICKING = {
    "side": "CASE WHEN quantity > 0 THEN 'Long' ELSE 'Short' END",
    "size_bucket": ("CASE WHEN abs(quantity) < 1000000 THEN '<1mm' WHEN abs(quantity) < 5000000"
                    " THEN '1-5mm' WHEN abs(quantity) < 10000000 THEN '5-10mm' ELSE '10mm+' END"),
    "price_band": "(floor(price / 5) * 5)::int::text || '-' || (floor(price / 5) * 5 + 5)::int::text",
}
ON_DEMAND = {**REFERENCE, **TICKING}
DIM_SQL = {**HOT, **ON_DEMAND}
# A trader can also type a SQL expression over the blotter's columns.
CUSTOM = "expr:"


def valid_dim(d):
    if d in DIM_SQL:
        return True
    expr = d[len(CUSTOM):] if isinstance(d, str) and d.startswith(CUSTOM) else ""
    # A local demo, not a security boundary: keep it to one expression.
    return 0 < len(expr.strip()) <= 300 and not any(t in expr for t in (";", "--", "/*"))


def dim_sql(d):
    return DIM_SQL[d] if d in DIM_SQL else d[len(CUSTOM):]
HEAVY = ("min_price", "max_price", "median_mv")
MEASURES = ["n", "face", "market_value", "dv01", "avg_price", "min_price", "max_price", "median_mv"]
HOT_COLS = ["level", "mask", "business_group", "currency", "sector", "grade", "tenor", "across"] + MEASURES
MAX_DEPTH = 5
# What one frame carries, so a high-cardinality pivot (bonds down, issuers
# across) cannot wedge the browser: the largest children per parent and the
# largest across values, by the first displayed measure.
CHILD_MAX = 300
ACROSS_MAX = 20


# The blotter without prices, for ranking positions by size.
DRILL_BASE = """
    SELECT bk.business_group, bk.book, b.bond, b.isin, b.issuer, b.currency, b.sector,
           r.rating, r.grade, b.seniority, b.coupon, b.day_count, b.maturity, b.benchmark,
           b.duration, b.callable, p.quantity, p.bond_id
    FROM positions p
    JOIN books bk ON bk.id = p.book_id
    JOIN bonds b ON b.id = p.bond_id
    JOIN bond_ratings r ON r.bond_id = p.bond_id
    WHERE p.quantity <> 0"""


def lit(v):
    return "'" + str(v).replace("'", "''") + "'"


def now_ms():
    return time.time() * 1000.0


def num(v):
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return v


# Connection id -> what its SUBSCRIBE is for, so the stats can name dataflows.
CONNECTIONS = {}


class Subscription:
    """The current contents of a SUBSCRIBE, updated at progress boundaries.

    `on_ready` is called once, after the first complete timestamp, with the
    milliseconds from start to that point.
    """

    def __init__(self, query, columns, kind, on_ready=None, on_update=None):
        self.query = query
        self.columns = columns
        self.kind = kind
        self.rows = {}
        self.lock = threading.Lock()
        self.frontier = None
        self.error = None
        self.ready = False
        self.version = 0
        self.started = now_ms()
        self.ready_ms = None
        self.on_ready = on_ready
        self.on_update = on_update
        self.closed = False
        self.conn = None
        self.pid = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        pending = []
        sql = f"COPY (SUBSCRIBE ({self.query}) WITH (PROGRESS)) TO STDOUT"
        try:
            self.conn = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
            if self.closed:
                return self.conn.close()
            self.pid = int(self.conn.execute("SELECT pg_backend_pid()").fetchone()[0])
            CONNECTIONS[self.pid] = self.kind
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
                        self.frontier = ts
                        if ready:
                            self.version += 1
                        first = not self.ready and self.frontier is not None
                        self.ready = True
                    if first:
                        self.ready_ms = now_ms() - self.started
                        if self.on_ready:
                            self.on_ready(self)
                    if ready and self.on_update:
                        self.on_update(self)
        except Exception as e:
            if not self.closed:
                self.error = str(e).splitlines()[0]

    def snapshot(self):
        with self.lock:
            return [dict(zip(self.columns, r)) for r in self.rows], self.frontier

    def close(self):
        self.closed = True
        CONNECTIONS.pop(self.pid, None)
        conn = self.conn
        if conn is not None:
            try:
                conn.cancel()
            except Exception:
                pass
            threading.Thread(target=conn.close, daemon=True).start()


def node_filter(dims, path):
    return " AND ".join(f"({dim_sql(d)})::text = {lit(v)}" for d, v in zip(dims, path)) or "true"


class Session:
    """One trader's screen: layout, subscriptions, and the last click's timing."""

    def __init__(self, trader):
        self.trader = trader
        self.lock = threading.RLock()
        self.writer = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
        self.rows = []
        self.across = None
        self.measures = []
        self.expanded = set()  # tuples of values along self.rows
        self.selected = ()
        self.sub = None  # hot or on-demand screen
        self.stale = None  # previous on-demand screen, shown until the new one is ready
        self.drill = None
        self.action = None  # {"what", "label", "via", "detail", "t0", "ms"}
        self.history = []  # finished actions, newest first
        self.force_live = False
        self.pending = None  # (action, predicate, fence) for hot-mode clicks
        self.clients = 0
        self.last_seen = time.time()
        # Whether this trader has rows in the board_* tables. A fresh session
        # clears leftovers from a previous run of the server.
        self.in_tables = True

    # -- layout ----------------------------------------------------------------

    def mode(self):
        dims = self.rows + ([self.across] if self.across else [])
        return "hot" if self.rows and not self.force_live and all(d in HOT for d in dims) else "on_demand"

    def set_layout(self, rows, across, measures, expanded=None, force_live=False, label=None):
        rows = [d for d in rows if valid_dim(d)][:MAX_DEPTH]
        across = across if across and valid_dim(across) and across not in rows else None
        measures = [m for m in measures if m in MEASURES]
        with self.lock:
            self.rows, self.across, self.measures = rows, across, measures
            self.force_live = bool(force_live)
            self.expanded = {tuple(p) for p in (expanded or []) if 0 < len(p) < len(rows)}
            self.selected = ()
            self._start_action("layout", label)
            if self.mode() == "hot" and self._hot_sub(self.sub):
                self.pending = (self.action, lambda rows: any(r["level"] == 1 for r in rows), None)
            self._write_layout()
            self._fence()
            self._restart_screen()
            self._restart_drill()

    def _write_layout(self):
        w, t = self.writer, self.trader
        if self.in_tables:
            for table in ("board_layout", "board_across", "board_expanded", "board_measures"):
                w.execute(f"DELETE FROM {table} WHERE trader = %s", (t,))
        self.in_tables = self.mode() == "hot"
        if not self.in_tables:
            return
        with w.cursor() as c:
            c.executemany(
                "INSERT INTO board_layout VALUES (%s, %s, %s)",
                [(t, i + 1, d) for i, d in enumerate(self.rows)],
            )
            if self.across:
                c.execute("INSERT INTO board_across VALUES (%s, %s)", (t, self.across))
            heavy = [m for m in self.measures if m in HEAVY]
            if heavy:
                c.executemany("INSERT INTO board_measures VALUES (%s, %s)", [(t, m) for m in heavy])
            c.executemany(
                "INSERT INTO board_expanded VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [self._expanded_row(p) for p in [()] + sorted(self.expanded)],
            )

    def _expanded_row(self, path):
        key = {d: "*" for d in HOT}
        mask = 0
        for d, v in zip(self.rows, path):
            key[d] = v
            mask |= BITS[d]
        return (self.trader, mask, key["business_group"], key["currency"],
                key["sector"], key["grade"], key["tenor"])

    def toggle(self, path):
        path = tuple(path)
        if not path or len(path) >= len(self.rows):
            return
        with self.lock:
            opening = path not in self.expanded
            if opening:
                self.expanded.add(path)
            else:
                self.expanded = {p for p in self.expanded if p[: len(path)] != path}
            self._start_action("expand" if opening else "collapse", " › ".join(path))
            if self.mode() == "hot":
                # Arm the check before writing, so no update can slip past it.
                shows_children = lambda rows: any(
                    r["level"] == len(path) + 1 and tuple(r["path"][: len(path)]) == path
                    for r in rows)
                if opening:
                    self.pending = (self.action, shows_children, None)
                    self.writer.execute(
                        "INSERT INTO board_expanded VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        self._expanded_row(path),
                    )
                else:
                    self.pending = (self.action, lambda rows: not shows_children(rows), None)
                    conds = " AND ".join(f"{d} = %s" for d in self.rows[: len(path)])
                    self.writer.execute(
                        f"DELETE FROM board_expanded WHERE trader = %s AND {conds}",
                        (self.trader, *path),
                    )
                self._fence()
            else:
                self._restart_screen(keep_old=True)

    def _fence(self):
        """Stamp the pending click with a timestamp at or after its writes.

        A strict serializable read happens after every write that finished
        before it, so a subscription whose frontier passes this timestamp is
        showing the click. Without it, a click that changes nothing visible
        (re-opening the same layout) would look instant.
        """
        if self.pending and self.pending[2] is None:
            ts = int(self.writer.execute("SELECT mz_now()::text").fetchone()[0])
            self.pending = (self.pending[0], self.pending[1], ts)

    def select(self, path):
        with self.lock:
            self.selected = tuple(path)
            self._restart_drill()

    def _start_action(self, what, label=None):
        hot = self.mode() == "hot"
        if what == "layout":
            detail = ("rewrote this trader's rows in board_layout / board_expanded; the shared"
                      " board view picks them up" if hot else
                      "started a new SUBSCRIBE: GROUP BY over the blotter index")
        else:
            detail = ("wrote one row to board_expanded" if hot else
                      "restarted the SUBSCRIBE with the new visible nodes")
        self.action = {"what": what, "label": label or "", "via": self.mode(), "detail": detail,
                       "t0": now_ms(), "ms": None}

    def _finish_action(self, action):
        if action is not None and action["ms"] is None:
            action["ms"] = round(now_ms() - action["t0"])
            self.history = [dict(action)] + self.history[:19]

    # -- subscriptions ---------------------------------------------------------

    @staticmethod
    def _hot_sub(sub):
        return sub is not None and sub.query.startswith("SELECT * FROM board WHERE")

    def _restart_screen(self, keep_old=False):
        old = self.sub
        if self.mode() == "hot":
            if self.stale:
                self.stale.close()
                self.stale = None
            if self._hot_sub(old):
                return  # Same long-lived subscription; the layout rows changed underneath.
            if old:
                old.close()
            action = self.action
            self.sub = Subscription(
                f"SELECT * FROM board WHERE trader = {lit(self.trader)}",
                ["trader"] + HOT_COLS, "hot screens",
                on_ready=lambda s: self._finish_action(action),
                on_update=self._check_pending,
            )
        else:
            # On an expand, keep showing the old screen until the new one has
            # numbers. On a new layout the old rows mean something else: blank it.
            if self.stale:
                self.stale.close()
            self.stale = old if keep_old and old and old.ready else None
            if old and old is not self.stale:
                old.close()
            action = self.action
            self.sub = Subscription(
                self._on_demand_sql(), ["level"] + [f"p{i}" for i in range(MAX_DEPTH)]
                + ["across"] + MEASURES, "on-demand pivots",
                on_ready=lambda s: self._on_demand_ready(s, action),
            )

    def _on_demand_ready(self, sub, action):
        self._finish_action(action)
        with self.lock:
            if self.sub is sub and self.stale:
                self.stale.close()
                self.stale = None

    def _on_demand_sql(self):
        rows = self.rows
        # Only what is displayed: min and max hold every value underneath.
        agg_sql = {
            "face": "SUM(quantity)", "market_value": "SUM(market_value)", "dv01": "SUM(dv01)",
            "avg_price": "SUM(price) / COUNT(*)", "min_price": "MIN(price)", "max_price": "MAX(price)",
        }
        aggs = ", ".join(["COUNT(*) AS n"] + [
            f"{agg_sql[m] if m in self.measures else 'NULL::numeric'} AS {m}"
            for m in MEASURES if m in agg_sql] + ["NULL::float8 AS median_mv"])
        parts = []
        for level in range(len(rows) + 1):
            if level >= 2:
                parents = [p for p in self.expanded if len(p) == level - 1]
                if not parents:
                    break
                where = " OR ".join(f"({node_filter(rows, p)})" for p in parents)
            else:
                where = "true"
            keys = [f"({dim_sql(d)})::text" for d in rows[:level]]
            cols = keys + ["NULL::text"] * (MAX_DEPTH - level)
            group = ", ".join(str(i + 2) for i in range(level))
            across_opts = [("''", group)]
            if self.across:
                # Column 1 is the level, 2.. the path, MAX_DEPTH + 2 the across value.
                across_opts.append((f"({dim_sql(self.across)})::text",
                                    ", ".join(filter(None, [group, str(MAX_DEPTH + 2)]))))
            for across_sql, grp in across_opts:
                # Unused path slots and measures are NULL so every part has one shape.
                sel = ",\n       ".join([", ".join([str(level) + " AS level"] + [
                    f"{c} AS p{i}" for i, c in enumerate(cols)]), f"{across_sql} AS across", aggs])
                parts.append(f"SELECT {sel}\n  FROM blotter WHERE {where}"
                             + (f"\n  GROUP BY {grp}" if grp else ""))
        return "\nUNION ALL\n".join(parts)

    def _restart_drill(self):
        if self.drill:
            self.drill.close()
        dims = self.rows[: len(self.selected)]
        cols = ["book", "bond", "issuer", "rating", "tenor", "quantity", "price", "market_value", "dv01"]
        if any(d in TICKING or d not in DIM_SQL for d in dims):
            # The filter may read prices or quantities, so rank the blotter itself.
            q = ("SELECT book, bond, issuer, rating, benchmark, quantity, price, market_value, dv01"
                 f" FROM blotter WHERE {node_filter(dims, self.selected)}"
                 " ORDER BY abs(quantity) DESC LIMIT 30")
            self.drill = Subscription(q, cols, "position panes")
            return
        # Rank on face, which only trades move, then join prices. Ranking on
        # market value would re-rank every price tick.
        q = (
            "SELECT s.book, s.bond, s.issuer, s.rating, s.benchmark, s.quantity, pr.price,"
            " s.quantity * pr.price / 100 AS market_value,"
            " round(s.quantity * pr.price / 100 * s.duration / 10000, 2) AS dv01"
            f" FROM (SELECT * FROM ({DRILL_BASE}) WHERE {node_filter(dims, self.selected)}"
            "       ORDER BY abs(quantity) DESC LIMIT 30) s"
            " JOIN prices pr ON pr.bond_id = s.bond_id"
        )
        self.drill = Subscription(
            q, ["book", "bond", "issuer", "rating", "tenor", "quantity", "price", "market_value", "dv01"],
            "position panes",
        )

    # -- frames ----------------------------------------------------------------

    def frame(self):
        with self.lock:
            sub, stale, drill = self.sub, self.stale, self.drill
            rows_dims, across = list(self.rows), self.across
            expanded = [list(p) for p in self.expanded]
            selected = list(self.selected)
            mode = self.mode()
        shown = sub if (sub and sub.ready) or not stale else stale
        out, frontier = self._rows(shown, rows_dims, across)
        out, limits = self._limit(out, (self.measures or ["n"])[0])
        drill_rows, _ = drill.snapshot() if drill else ([], None)
        action = dict(self.action) if self.action else None
        error = sub.error if sub else None
        if action and action["ms"] is None:
            action["waiting_ms"] = round(now_ms() - action["t0"])
            if error:
                action["error"] = error
        return {
            "trader": self.trader,
            "mode": mode,
            "layout": {"rows": rows_dims, "across": across, "measures": self.measures,
                       "expanded": expanded, "selected": selected},
            "frontier": frontier,
            "lag": round(now_ms() - frontier) if frontier else None,
            "loading": bool(sub and not sub.ready and not error),
            "error": error,
            "force_live": self.force_live,
            "sql": sub.query if sub else None,
            "query_mb": STATS.get("by_pid", {}).get(sub.pid) if sub else None,
            "rows": out,
            "action": action,
            "history": self.history[:12],
            "limits": limits,
            "drill": {
                "path": selected,
                "ready_ms": round(drill.ready_ms) if drill and drill.ready_ms else None,
                "rows": [{k: num(v) if k in ("quantity", "price", "market_value", "dv01") else v
                          for k, v in d.items()} for d in drill_rows],
            },
        }

    def _check_pending(self, sub):
        """Stamp a hot-mode click once its effect is visible in the subscription."""
        pending = self.pending
        if not pending or sub is not self.sub or pending[2] is None:
            return
        if sub.frontier is None or sub.frontier <= pending[2]:
            return
        rows, _ = self._rows(sub, list(self.rows), self.across)
        if pending[0] is self.action and pending[1](rows):
            self._finish_action(pending[0])
            self.pending = None

    @staticmethod
    def _limit(out, measure):
        size = lambda r: abs(r[measure] or 0)
        limits = {"across_values": 0, "across_shown": 0, "hidden": {}}
        # Across: keep the values largest at the root.
        root_cells = [r for r in out if r["level"] == 0 and r["across"]]
        limits["across_values"] = len(root_cells)
        if len(root_cells) > ACROSS_MAX:
            keep = {r["across"] for r in sorted(root_cells, key=size, reverse=True)[:ACROSS_MAX]}
            out = [r for r in out if not r["across"] or r["across"] in keep]
        limits["across_shown"] = min(len(root_cells), ACROSS_MAX)
        # Rows: keep the largest children of each parent.
        kids = {}
        for r in out:
            if not r["across"] and r["level"] > 0:
                kids.setdefault(tuple(r["path"][:-1]), []).append(r)
        dropped = set()
        for parent, rs in kids.items():
            if len(rs) > CHILD_MAX:
                rs.sort(key=size, reverse=True)
                dropped.update(tuple(r["path"]) for r in rs[CHILD_MAX:])
                limits["hidden"]["\u0001".join(parent)] = len(rs) - CHILD_MAX
        if dropped:
            out = [r for r in out if tuple(r["path"]) not in dropped]
        return out, limits

    @staticmethod
    def _rows(shown, rows_dims, across):
        rows, frontier = shown.snapshot() if shown else ([], None)
        masks = [0]
        for d in rows_dims:
            masks.append(masks[-1] | BITS.get(d, 0))
        out = []
        for r in rows:
            level = int(r["level"])
            if "mask" in r:
                # Rows from a layout this frame no longer describes are in flight. Skip
                # them. Cells carry the mask of their row node.
                if level >= len(masks) or int(r["mask"]) != masks[level]:
                    continue
                path = [r[d] for d in rows_dims[:level]]
            else:
                path = [r[f"p{i}"] for i in range(level)]
            item = {"level": level, "path": path, "across": r["across"] or ""}
            for m in MEASURES:
                item[m] = num(r[m])
            out.append(item)
        return out, frontier

    def close(self):
        with self.lock:
            for s in (self.sub, self.stale, self.drill):
                if s:
                    s.close()
            self.sub = self.stale = self.drill = None
            for table in ("board_layout", "board_across", "board_expanded", "board_measures"):
                self.writer.execute(f"DELETE FROM {table} WHERE trader = %s", (self.trader,))
            self.writer.close()


# -- shared state --------------------------------------------------------------

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


def session(trader):
    with SESSIONS_LOCK:
        s = SESSIONS.get(trader)
        if s is None:
            s = SESSIONS[trader] = Session(trader)
        s.last_seen = time.time()
        return s


def reaper():
    """Drop a trader's layout a while after their last browser tab closes."""
    while True:
        time.sleep(5)
        with SESSIONS_LOCK:
            idle = [t for t, s in SESSIONS.items() if s.clients == 0 and time.time() - s.last_seen > 30]
            gone = [SESSIONS.pop(t) for t in idle]
        for s in gone:
            try:
                s.close()
            except Exception:
                pass


TAPE = None
STATS = {"arrangements": [], "cores": None, "traders": 0, "at": None}


def tape():
    global TAPE
    TAPE = Subscription(
        "SELECT t.traded_at, b.bond, b.issuer, b.currency, bk.book, t.side, t.quantity"
        " FROM trades t JOIN bonds b ON b.id = t.bond_id JOIN books bk ON bk.id = t.book_id"
        # Only the last few seconds, so the top-k holds tens of trades, not the window.
        " WHERE mz_now() < t.traded_at + INTERVAL '5 seconds'"
        " ORDER BY t.traded_at DESC LIMIT 14",
        ["traded_at", "bond", "issuer", "currency", "book", "side", "quantity"],
        "trade tape",
    )


def classify(name, subscribes):
    n = name.replace("Dataflow: ", "").replace("materialize.materialize_demo.", "")
    if n in subscribes:
        return subscribes[n]
    if n.startswith("board_by_trader"):
        return "board view (all traders)"
    if n.startswith("board_cube"):
        return "cube roll-ups"
    if n.startswith("board_leaf") or n.startswith("board_hist_leaf"):
        return "cube leaf + histograms"
    if n.startswith(("blotter", "positions", "prices", "bond_ratings", "bonds", "books")):
        return "blotter pipeline"
    return None


def stats_loop():
    conn = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
    cpu_sql = """
        SELECT d.name, SUM(s.elapsed_ns)::float8
        FROM mz_introspection.mz_scheduling_elapsed s
        JOIN mz_introspection.mz_dataflow_addresses a ON a.id = s.id
        JOIN mz_introspection.mz_dataflows d ON d.id = a.address[1]
        WHERE NOT EXISTS (SELECT 1 FROM mz_introspection.mz_dataflow_operator_parents p
                          WHERE p.parent_id = s.id)
        GROUP BY d.name"""
    prev = None
    while True:
        try:
            sizes = conn.execute(
                "SELECT name, size FROM mz_introspection.mz_dataflow_arrangement_sizes"
            ).fetchall()
            cpu = dict(conn.execute(cpu_sql).fetchall())
            t = time.time()
            pids = {
                f"subscribe-{sid}": int(cid)
                for sid, cid in conn.execute(
                    "SELECT s.id, se.connection_id FROM mz_internal.mz_subscriptions s"
                    " JOIN mz_internal.mz_sessions se ON se.id = s.session_id"
                ).fetchall()
            }
            subscribes = {name: CONNECTIONS.get(pid) for name, pid in pids.items()}
            groups, by_pid = {}, {}
            for name, size in sizes:
                g = classify(name, subscribes)
                if g:
                    groups.setdefault(g, [0, 0.0])[0] += size or 0
                pid = pids.get(name.replace("Dataflow: ", ""))
                if pid is not None:
                    by_pid[pid] = round(by_pid.get(pid, 0) + (size or 0) / 1e6, 2)
            if prev:
                dt = t - prev[0]
                for name, ns in cpu.items():
                    g = classify(name, subscribes)
                    if g:
                        groups.setdefault(g, [0, 0.0])[1] += max(0.0, ns - prev[1].get(name, 0)) / 1e9 / dt
            prev = (t, cpu)
            traders = conn.execute("SELECT COUNT(DISTINCT trader) FROM board_layout").fetchone()[0]
            layout_rows = conn.execute(
                "SELECT (SELECT COUNT(*) FROM board_layout) + (SELECT COUNT(*) FROM board_across)"
                " + (SELECT COUNT(*) FROM board_expanded) + (SELECT COUNT(*) FROM board_measures)"
            ).fetchone()[0]
            with SESSIONS_LOCK:
                on_demand = sum(1 for s in SESSIONS.values() if s.mode() != "hot")
            STATS.update(
                arrangements=[{"group": g, "mb": round(v[0] / 1e6, 1), "cores": round(v[1], 2)}
                              for g, v in sorted(groups.items(), key=lambda kv: -kv[1][0])],
                traders=traders, layout_rows=layout_rows, on_demand=on_demand,
                sessions=len(SESSIONS), at=t, by_pid=by_pid,
            )
        except Exception as e:
            STATS["error"] = str(e).splitlines()[0]
            try:
                conn = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
            except Exception:
                pass
        time.sleep(3)


# -- HTTP ----------------------------------------------------------------------

FIELDS = {
    "hot": list(HOT),
    "reference": list(REFERENCE),
    "ticking": list(TICKING),
    "measures": MEASURES,
    "heavy": list(HEAVY),
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            body = open(os.path.join(HERE, "board.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/fields":
            self._json(FIELDS)
        elif url.path == "/events":
            self._events(q.get("trader", ["alice"])[0], float(q.get("hz", ["4"])[0]))
        else:
            self.send_error(404)

    def _events(self, trader, hz):
        s = session(trader)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        s.clients += 1
        try:
            while True:
                s.last_seen = time.time()
                f = s.frame()
                f["tape"] = TAPE.snapshot()[0] if TAPE else []
                f["stats"] = STATS
                self.wfile.write(b"data: " + json.dumps(f).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(1.0 / hz)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            s.clients -= 1

    def do_POST(self):
        url = urlparse(self.path)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        s = session(body.get("trader", "alice"))
        try:
            if url.path == "/layout":
                s.set_layout(body.get("rows", []), body.get("across"),
                             body.get("measures", []), body.get("expanded"),
                             body.get("force_live", False), body.get("label"))
            elif url.path == "/toggle":
                s.toggle(body["path"])
            elif url.path == "/select":
                s.select(body["path"])
            else:
                return self.send_error(404)
            self._json({"ok": True})
        except Exception as e:
            self._json({"ok": False, "error": str(e)}, 500)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("BOARD_PORT", 8765)))
    args = ap.parse_args()
    # The board tables hold only this server's sessions. Clear any left by a
    # previous run, or their traders' screens keep being computed.
    with psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH) as c:
        for table in ("board_layout", "board_across", "board_expanded", "board_measures"):
            c.execute(f"DELETE FROM {table}")
    tape()
    threading.Thread(target=stats_loop, daemon=True).start()
    threading.Thread(target=reaper, daemon=True).start()
    print(f"Board on http://localhost:{args.port}/?trader=alice  (Materialize: {DSN})")
    ThreadingHTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
