#!/usr/bin/env python3
# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

"""Reload the demo under several knob settings and record cost and freshness.

For each (retention, price slots) configuration: tear down, reload, add four
traders, wait for hydration, then record arrangement memory and operator CPU
per dataflow, blotter churn, and screen/cube lag at each timestamp interval.
Writes one JSON object per configuration to stdout.

Needs PSQL and SYS as for load.sh, plus MZ_DSN / MZ_SYS_DSN as for
freshness.py. Set CONTAINER to also sample `docker stats` for that container.

    python3 sweep.py --config "3 hours:50" --config "3 hours:10"
"""

import argparse
import json
import os
import subprocess
import threading
import time

import psycopg

import freshness

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "..", "assets")
DSN = freshness.DSN
SEARCH_PATH = freshness.SEARCH_PATH
PSQL = os.environ.get("PSQL", "psql -h localhost -p 6875 -U materialize")


def psql_file(path, extra=()):
    with open(path) as f:
        subprocess.run(PSQL.split() + ["-X", "-q", *extra], stdin=f, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def per_dataflow(conn):
    sizes = dict(conn.execute(
        "SELECT name, size FROM mz_introspection.mz_dataflow_arrangement_sizes"
    ).fetchall())
    records = dict(conn.execute(
        "SELECT name, records FROM mz_introspection.mz_dataflow_arrangement_sizes"
    ).fetchall())
    return sizes, records


def operator_seconds(conn):
    # Leaf operators only: nested scopes also report their children's time.
    return dict(conn.execute(
        """SELECT d.dataflow_name, sum(s.elapsed_ns)::float8 / 1e9
           FROM mz_introspection.mz_scheduling_elapsed s
           JOIN mz_introspection.mz_dataflow_operator_dataflows d ON d.id = s.id
           WHERE NOT EXISTS (SELECT 1 FROM mz_introspection.mz_dataflow_operator_parents c
                             WHERE c.parent_id = s.id)
           GROUP BY 1"""
    ).fetchall())


def churn(seconds=5):
    """Blotter row updates per second (each change is a retraction plus an insertion)."""
    conn = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
    timer = threading.Timer(seconds + 1, conn.cancel)
    timer.start()
    n, start = 0, None
    try:
        with conn.cursor().copy(
            "COPY (SUBSCRIBE (SELECT book, bond FROM blotter) WITH (PROGRESS)) TO STDOUT"
        ) as cp:
            for r in cp.rows():
                if r[1] == "t" and start is None:
                    start = time.time()  # snapshot complete
                elif r[1] == "f" and start is not None:
                    n += abs(int(r[2]))
    except psycopg.errors.QueryCanceled:
        pass
    finally:
        timer.cancel()
    return n / (time.time() - start) if start else float("nan")


def docker_stats():
    c = os.environ.get("CONTAINER")
    if not c:
        return None
    out = subprocess.run(["docker", "stats", "--no-stream", "--format",
                          "{{.CPUPerc}}|{{.MemUsage}}", c], capture_output=True, text=True)
    return out.stdout.strip()


def run(retention, slots, intervals, seconds):
    psql_file(os.path.join(ASSETS, "teardown.sql"))
    env = dict(os.environ, FI_RETENTION=retention, FI_PRICE_SLOTS=str(slots))
    subprocess.run(["sh", os.path.join(HERE, "load.sh")], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    conn = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
    t0 = time.time()
    conn.execute("INSERT INTO traders VALUES ('alice'), ('bob'), ('carol'), ('dave')")
    conn.execute("""INSERT INTO expanded VALUES
        ('alice', 'Rates'), ('alice', 'Rates/USD'), ('bob', 'HY Credit'), ('bob', 'EM'),
        ('carol', 'IG Credit'), ('carol', 'IG Credit/EUR'), ('dave', 'Munis')""")
    rows = conn.execute("SELECT count(*) FROM screens").fetchone()[0]
    hydrate = time.time() - t0
    time.sleep(20)

    cpu_a, ta = operator_seconds(conn), time.time()
    stats = [docker_stats() for _ in range(3)]
    time.sleep(max(0, 10 - (time.time() - ta)))
    cpu_b, tb = operator_seconds(conn), time.time()
    cpu = {k: (cpu_b[k] - cpu_a.get(k, 0)) / (tb - ta) for k in cpu_b}
    sizes, records = per_dataflow(conn)

    def short(k):
        return k.replace("Dataflow: materialize.materialize_demo.", "")

    result = {
        "retention": retention,
        "price_slots": slots,
        "blotter_rows": conn.execute("SELECT count(*) FROM blotter").fetchone()[0],
        "cube_cells": conn.execute("SELECT count(*) FROM pivot_leaf").fetchone()[0],
        "screen_rows": rows,
        "hydrate_s": round(hydrate, 1),
        "blotter_updates_per_s": round(churn()),
        "arrangement_mb": {short(k): round((v or 0) / 1e6, 1) for k, v in sizes.items()
                           if "introspection" not in k},
        "arrangement_records": {short(k): v for k, v in records.items()
                                if "introspection" not in k},
        "cores": {short(k): round(v, 2) for k, v in cpu.items() if v >= 0.005},
        "docker": stats,
        "freshness": {},
    }
    sys_conn = psycopg.connect(freshness.SYS_DSN, autocommit=True)
    for interval in intervals:
        sys_conn.execute(f"ALTER SYSTEM SET default_timestamp_interval = '{interval}'")
        time.sleep(3)
        res = {"screen": {}, "cube": {}}
        before, t = freshness.counters(), time.time()
        threads = [
            threading.Thread(target=freshness.measure, args=(
                "SELECT * FROM screens WHERE trader = 'alice'", seconds, res["screen"])),
            threading.Thread(target=freshness.measure, args=(
                "SELECT * FROM pivot_leaf", seconds, res["cube"])),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        after, dt = freshness.counters(), time.time() - t
        cas = after.get("cas", 0) - before.get("cas", 0)
        res["oracle_writes_per_s"] = round((after.get("oracle", 0) - before.get("oracle", 0)) / dt, 1)
        res["appends_per_s"] = round((after.get("append", 0) - before.get("append", 0)) / dt, 1)
        res["cas_per_s"] = round(cas / dt, 1)
        res["cas_ms"] = round(1000 * (after.get("cas_s", 0) - before.get("cas_s", 0)) / cas, 2) if cas else None
        for k in ("screen", "cube"):
            res[k] = {m: round(v, 1) for m, v in res[k].items()}
        result["freshness"][interval] = res
    sys_conn.execute("ALTER SYSTEM SET default_timestamp_interval = '100ms'")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", action="append", required=True,
                    help="RETENTION:PRICE_SLOTS, e.g. '3 hours:50'")
    ap.add_argument("--intervals", nargs="+", default=["1s", "250ms", "100ms"])
    ap.add_argument("--seconds", type=int, default=15)
    args = ap.parse_args()
    for c in args.config:
        retention, slots = c.rsplit(":", 1)
        print(json.dumps(run(retention, int(slots), args.intervals, args.seconds)), flush=True)


if __name__ == "__main__":
    main()
