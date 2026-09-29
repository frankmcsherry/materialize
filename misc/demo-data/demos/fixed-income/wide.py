#!/usr/bin/env python3
# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

"""What do very wide rows cost, and where?

Adds W extra reference columns per bond (a mix of float, integer and short
text, hashed from the bond) and measures two ways of holding them:

  reference     the W columns indexed once per bond (4,096 rows), joined to
                positions only when someone looks at a row
  denormalized  the W columns joined into the ticking blotter and indexed,
                the single combined wide table. Every price tick re-emits
                the whole row.

For each width, reports arrangement sizes, how long the wide table took to
hydrate, and the lag of the shared cube and of one trader's page of the wide
table while it is maintained.

    python3 wide.py --widths 150 600
"""

import argparse
import os
import threading
import time

import psycopg

import freshness

DSN = os.environ.get(
    "MZ_DSN", "host=localhost port=6875 user=materialize dbname=materialize"
)
# Everything the demo creates lives in the mz-demo-data skill's schema.
SEARCH_PATH = "-c search_path=materialize_demo"


def attrs_sql(width):
    cols = []
    for k in range(width):
        byte = f"get_byte(h{k % 2}, {k % 16})"
        if k % 4 in (0, 1):
            cols.append(f"(mod({byte} * {k + 7}, 10007))::float8 / 100 AS c{k}")
        elif k % 4 == 2:
            cols.append(f"{byte} + {k} AS c{k}")
        else:
            cols.append(f"'v{k}_' || {byte} AS c{k}")
    return (
        f"SELECT bond, {', '.join(cols)} FROM ("
        # From a table, not generate_series: a view over constants is folded
        # into a constant and never appears as an arrangement.
        "SELECT 'BND-' || lpad(id::text, 4, '0') AS bond,"
        " digest('attr0:' || id::text, 'md5') AS h0,"
        " digest('attr1:' || id::text, 'md5') AS h1 FROM wide_ids)"
    )


def size_of(conn, pattern):
    vals = []
    for _ in range(3):
        vals.append(conn.execute(
            "SELECT coalesce(sum(size), 0), coalesce(sum(records), 0)"
            " FROM mz_introspection.mz_dataflow_arrangement_sizes WHERE name LIKE %s",
            (pattern,),
        ).fetchone())
        time.sleep(1)
    vals.sort()
    return float(vals[1][0]), int(vals[1][1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--widths", type=int, nargs="+", default=[150, 600])
    ap.add_argument("--hydrate-timeout", type=int, default=600)
    args = ap.parse_args()
    conn = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS wide_ids (id int)")
    if not conn.execute("SELECT count(*) FROM wide_ids").fetchone()[0]:
        conn.execute("INSERT INTO wide_ids SELECT generate_series(0, 4095)")

    res0 = {}
    freshness.measure("SELECT * FROM pivot_leaf", 10, res0)
    print(f"cube lag with narrow blotter only: p50 {res0['p50']:.0f}ms p99 {res0['p99']:.0f}ms")
    base, _ = size_of(conn, "%blotter_primary_idx%")
    print(f"narrow blotter index (22 columns): {base / 1e6:.1f} MB")
    print(f"{'W':>5} {'reference MB':>13} {'denormalized MB':>16} {'B/row':>7} {'hydrate s':>10}"
          f" {'cube lag p50/p99':>17} {'page lag p50/p99':>17}")
    for w in args.widths:
        conn.execute(f"CREATE VIEW bond_attrs_{w} AS {attrs_sql(w)}")
        conn.execute(f"CREATE INDEX bond_attrs_{w}_idx ON bond_attrs_{w} (bond)")
        conn.execute(f"SELECT count(*) FROM bond_attrs_{w}").fetchall()
        time.sleep(5)
        ref, _ = size_of(conn, f"%bond_attrs_{w}_idx%")
        conn.execute(
            f"CREATE VIEW blotter_wide_{w} AS SELECT b.*, "
            + ", ".join(f"a.c{k}" for k in range(w)) + " FROM blotter b"
            f" JOIN bond_attrs_{w} a ON a.bond = b.bond"
        )
        conn.execute(f"CREATE INDEX blotter_wide_{w}_idx ON blotter_wide_{w} (book, bond)")
        t0 = time.time()
        conn.execute(f"SET statement_timeout = '{args.hydrate_timeout}s'")
        try:
            rows = conn.execute(f"SELECT count(*) FROM blotter_wide_{w}").fetchone()[0]
        except psycopg.errors.QueryCanceled:
            print(f"{w:5d} {ref / 1e6:13.1f}  did not hydrate within {args.hydrate_timeout}s", flush=True)
            conn.execute(f"DROP VIEW bond_attrs_{w} CASCADE")
            continue
        hydrate = time.time() - t0
        time.sleep(15)
        wide, _ = size_of(conn, f"%blotter_wide_{w}_idx%")
        cube, page = {}, {}
        ts = [threading.Thread(target=freshness.measure, args=("SELECT * FROM pivot_leaf", 15, cube)),
              threading.Thread(target=freshness.measure, args=(
                  f"SELECT * FROM blotter_wide_{w} WHERE book = 'BOOK-00'", 15, page))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        print(f"{w:5d} {ref / 1e6:13.1f} {wide / 1e6:16.1f} {wide / rows:7.0f} {hydrate:10.1f}"
              f" {cube['p50']:8.0f}/{cube['p99']:<8.0f} {page['p50']:8.0f}/{page['p99']:<8.0f}", flush=True)
        conn.execute(f"DROP VIEW bond_attrs_{w} CASCADE")
        time.sleep(10)


if __name__ == "__main__":
    main()
