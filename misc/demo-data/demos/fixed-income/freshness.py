#!/usr/bin/env python3
# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

"""How stale is a trader's screen when an update reaches it?

For each `default_timestamp_interval` in the sweep, subscribes to a trader's
screen and to the shared cube, and records, for every timestamp that carries a
change, the wall-clock arrival time minus that timestamp. Also counts persist
and timestamp-oracle operations, since each tick costs round trips to both.

    python3 freshness.py                          # sweep 1s 250ms 100ms
    python3 freshness.py --intervals 100ms 50ms --seconds 30

NOTE: stream SUBSCRIBE with COPY (or FETCH n). `FETCH ALL ... WITH (timeout)`
waits out the whole timeout before returning, which looks exactly like a
view that only updates once per timeout.
"""

import argparse
import os
import re
import threading
import time
import urllib.request

import psycopg

DSN = os.environ.get(
    "MZ_DSN", "host=localhost port=6875 user=materialize dbname=materialize"
)
# Everything the demo creates lives in the mz-demo-data skill's schema.
SEARCH_PATH = "-c search_path=materialize_demo"
SYS_DSN = os.environ.get(
    "MZ_SYS_DSN", "host=localhost port=6877 user=mz_system dbname=materialize"
)
METRICS = os.environ.get("MZ_METRICS", "http://localhost:6878/metrics")


def now_ms():
    return time.time() * 1000.0


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100.0 * len(xs)))] if xs else float("nan")


def measure(query, seconds, out):
    """Arrival lag of each changed timestamp, skipping the initial snapshot."""
    conn = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
    timer = threading.Timer(seconds, conn.cancel)
    timer.start()
    seen, lags, start = set(), [], now_ms()
    try:
        with conn.cursor().copy(f"COPY (SUBSCRIBE ({query}) WITH (PROGRESS)) TO STDOUT") as cp:
            for r in cp.rows():
                ts = int(r[0])
                if r[1] == "f" and ts not in seen and ts > start + 1000:
                    seen.add(ts)
                    lags.append(now_ms() - ts)
    except psycopg.errors.QueryCanceled:
        pass
    finally:
        timer.cancel()
        conn.close()
    out.update(p50=pct(lags, 50), p99=pct(lags, 99), per_s=len(lags) / (seconds - 1))


def counters():
    try:
        txt = urllib.request.urlopen(METRICS, timeout=2).read().decode()
    except OSError:
        return {}
    out = {}
    for pat, name in [
        (r'^mz_persist_external_started_count\{op="consensus_cas"\} (\d+)', "cas"),
        (r'^mz_persist_external_op_latency_sum\{op="consensus_cas"\} ([\d.e+-]+)', "cas_s"),
        (r'^mz_persist_cmd_succeeded_count\{cmd="compare_and_append"\} (\d+)', "append"),
        (r'^mz_ts_oracle_retry_started_count\{op="write_ts"\} (\d+)', "oracle"),
    ]:
        m = re.search(pat, txt, re.M)
        if m:
            out[name] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--intervals", nargs="+", default=["1s", "250ms", "100ms"])
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--trader", default="alice")
    args = ap.parse_args()
    queries = {
        "screen": f"SELECT * FROM screens WHERE trader = '{args.trader}'",
        "cube": "SELECT * FROM pivot_leaf",
    }
    sys_conn = psycopg.connect(SYS_DSN, autocommit=True)
    print(f"{'interval':>9} {'view':>7} {'lag p50':>8} {'lag p99':>8} {'updates/s':>10}"
          f" {'oracle w/s':>11} {'appends/s':>10} {'CAS/s':>7} {'CAS ms':>7}")
    for interval in args.intervals:
        sys_conn.execute(f"ALTER SYSTEM SET default_timestamp_interval = '{interval}'")
        time.sleep(3)
        results = {k: {} for k in queries}
        before, t0 = counters(), time.time()
        threads = [threading.Thread(target=measure, args=(q, args.seconds, results[k]))
                   for k, q in queries.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        after, dt = counters(), time.time() - t0

        def rate(k):
            return (after[k] - before[k]) / dt if k in after and k in before else float("nan")

        cas_n = after.get("cas", 0) - before.get("cas", 0)
        cas_ms = 1000 * (after.get("cas_s", 0) - before.get("cas_s", 0)) / cas_n if cas_n else float("nan")
        for k, r in results.items():
            print(f"{interval:>9} {k:>7} {r['p50']:7.0f}ms {r['p99']:7.0f}ms {r['per_s']:10.1f}"
                  f" {rate('oracle'):11.1f} {rate('append'):10.1f} {rate('cas'):7.1f} {cas_ms:7.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
