#!/usr/bin/env python3
# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

"""What does each additional trader cost?

Adds N traders' customized views in four ways, measures the extra arrangement
memory and CPU they bring, then removes them again:

  grid    each trader keeps their own sorted copy of the wide blotter
  direct  each trader's tree pivot is computed from the blotter rows
  cube    each trader's tree pivot is re-aggregated from the shared cube
  data    each trader is rows in `traders` and `expanded`, served by `screens`

Every trader gets a different tree (a different ordering of business group,
currency, sector and grade), so no two views are identical.

    python3 memory.py                 # default N = 1 4 8
    python3 memory.py --n 1 8 32 --modes cube data
"""

import argparse
import itertools
import os
import random
import statistics
import time

import psycopg

DSN = os.environ.get(
    "MZ_DSN", "host=localhost port=6875 user=materialize dbname=materialize"
)
# Everything the demo creates lives in the mz-demo-data skill's schema.
SEARCH_PATH = "-c search_path=materialize_demo"

DIMS = ["business_group", "currency", "sector", "grade"]
ORDERS = list(itertools.permutations(DIMS, 3))


def tree_sql(dims, source):
    """A three-level tree pivot over `source` (either blotter or pivot_leaf)."""
    if source == "blotter":
        aggs = ("COUNT(*)", "SUM(market_value)", "SUM(dv01)", "SUM(price)",
                "MIN(price)", "MAX(price)")
    else:
        aggs = ("SUM(n)", "SUM(market_value)", "SUM(dv01)", "SUM(sum_price)",
                "MIN(min_price)", "MAX(max_price)")
    cols = "n, market_value, dv01, sum_price, min_price, max_price"
    parts = []
    for depth in (1, 2, 3):
        keys = dims[:depth]
        path = " || '/' || ".join(keys)
        parts.append(
            f"SELECT {depth} AS level, {path} AS path, {', '.join(aggs)} "
            f"FROM {source} GROUP BY {', '.join(keys)}"
        )
    return f"SELECT level, path, {cols} FROM (" + " UNION ALL ".join(
        f"SELECT * FROM ({p}) AS t{i}(level, path, {cols})" for i, p in enumerate(parts)
    ) + ")"


class Mz:
    def __init__(self):
        self.conn = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)

    def x(self, sql, args=None):
        return self.conn.execute(sql, args)

    def arrangement_bytes(self):
        rows = self.x(
            "SELECT name, size FROM mz_introspection.mz_dataflow_arrangement_sizes"
        ).fetchall()
        return {n.replace("Dataflow: materialize.materialize_demo.", ""): s or 0 for n, s in rows}

    def operator_seconds(self):
        # Leaf operators only: nested scopes also report their children's time.
        return self.x(
            """SELECT sum(s.elapsed_ns)::float8 / 1e9
               FROM mz_introspection.mz_scheduling_elapsed s
               WHERE NOT EXISTS (SELECT 1 FROM mz_introspection.mz_dataflow_operator_parents c
                                 WHERE c.parent_id = s.id)"""
        ).fetchone()[0]

    def settle(self, check_sql, seconds=8):
        """Wait until `check_sql` returns, then let arrangements compact."""
        self.x(check_sql).fetchall()
        time.sleep(seconds)

    def sample(self, match, samples=3):
        """Median total arrangement bytes over dataflows whose name matches."""
        totals = []
        for _ in range(samples):
            sizes = self.arrangement_bytes()
            totals.append(sum(v for k, v in sizes.items() if match(k)))
            time.sleep(1)
        return statistics.median(totals)

    def cpu(self, seconds=10):
        a = self.operator_seconds()
        time.sleep(seconds)
        return (self.operator_seconds() - a) / seconds


def run(mz, mode, n):
    names = [f"t{mode}_{i}" for i in range(n)]
    base_cpu = mz.cpu()
    if mode == "data":
        base = mz.sample(lambda k: k.startswith("screens_by_trader"))
        rng = random.Random(n)
        groups = [r[0] for r in mz.x("SELECT DISTINCT business_group FROM books").fetchall()]
        ccys = ["USD", "EUR", "GBP", "JPY"]
        with mz.conn.transaction():
            for name in names:
                mz.x("INSERT INTO traders VALUES (%s)", (name,))
                for g in rng.sample(groups, 2):
                    mz.x("INSERT INTO expanded VALUES (%s, %s)", (name, g))
                    mz.x("INSERT INTO expanded VALUES (%s, %s)", (name, f"{g}/{rng.choice(ccys)}"))
        mz.settle("SELECT count(*) FROM screens")
        after = mz.sample(lambda k: k.startswith("screens_by_trader"))
        added = after - base
        visible = mz.x(
            "SELECT count(*) FROM screens WHERE trader LIKE 'tdata_%'"
        ).fetchone()[0]
        cpu = mz.cpu() - base_cpu
        mz.x("DELETE FROM expanded WHERE trader LIKE 'tdata_%'")
        mz.x("DELETE FROM traders WHERE trader LIKE 'tdata_%'")
        return added, cpu, f"{visible} visible rows"

    for i, name in enumerate(names):
        if mode == "grid":
            # A trader's own copy of the blotter, sorted their way.
            key = ["currency", "sector", "issuer", "book", "maturity", "rating"][i % 6]
            mz.x(f"CREATE VIEW {name} AS SELECT * FROM blotter")
            mz.x(f"CREATE INDEX {name}_idx ON {name} ({key}, bond, book)")
        else:
            source = "blotter" if mode == "direct" else "pivot_leaf"
            mz.x(f"CREATE VIEW {name} AS {tree_sql(ORDERS[i % len(ORDERS)], source)}")
            mz.x(f"CREATE DEFAULT INDEX {name}_idx ON {name}")
    for name in names:
        mz.x(f"SELECT count(*) FROM {name}").fetchall()
    mz.settle("SELECT 1")
    added = mz.sample(lambda k: any(k.startswith(f"{name}_idx") for name in names))
    rows = sum(mz.x(f"SELECT count(*) FROM {name}").fetchone()[0] for name in names)
    cpu = mz.cpu() - base_cpu
    for name in names:
        mz.x(f"DROP VIEW {name} CASCADE")
    return added, cpu, f"{rows} rows held"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--modes", nargs="+", default=["grid", "direct", "cube", "data"])
    args = ap.parse_args()
    mz = Mz()

    shared = mz.arrangement_bytes()
    print("Shared arrangements (MB):")
    for k in ["blotter_primary_idx", "pivot_leaf_primary_idx", "mv_hist_leaf_primary_idx",
              "screens_by_trader", "positions_primary_idx", "prices_primary_idx"]:
        print(f"  {k:28} {shared.get(k, 0) / 1e6:8.1f}")
    print()
    print(f"{'mode':8} {'N':>4} {'added MB':>9} {'MB/trader':>10} {'added cores':>12}  detail")
    for mode in args.modes:
        for n in args.n:
            added, cpu, detail = run(mz, mode, n)
            print(f"{mode:8} {n:4d} {added / 1e6:9.2f} {added / 1e6 / n:10.3f} {cpu:12.2f}  {detail}",
                  flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
