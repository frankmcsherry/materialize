#!/usr/bin/env python3
# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

"""Click-to-numbers latency through a running board.py, both ways of serving.

Drives the board's HTTP API as a browser would and reads the timing the
server stamps on each action: from receiving the click to the first
subscription update that shows its effect.

  cube       expand/collapse on a cube layout: a write to board_expanded,
             seen through the trader's long-lived SUBSCRIBE
  cube+med   the same with min/max/median shown
  on demand  expand/collapse on an issuer > bond layout: a new SUBSCRIBE
             over the blotter index per click
  layout     switching to a new on-demand layout: a new SUBSCRIBE

    python3 board.py &
    python3 clicks.py --n 20
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request

BASE = "http://localhost:8765"


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


class Stream:
    def __init__(self, trader):
        self.last = None

        def run():
            for line in urllib.request.urlopen(f"{BASE}/events?trader={trader}&hz=20"):
                if line.startswith(b"data: "):
                    self.last = json.loads(line[6:])

        threading.Thread(target=run, daemon=True).start()

    def wait_action(self, t0, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            a = self.last and self.last["action"]
            if a and a["t0"] >= t0 and a["ms"] is not None:
                return a["ms"]
            time.sleep(0.02)
        return None


def run(stream, trader, n, rows, measures, paths, label):
    post("/layout", {"trader": trader, "rows": rows, "measures": measures, "expanded": []})
    time.sleep(3)
    out = []
    for i in range(n):
        path = paths[i % len(paths)]
        for _ in range(2):  # expand, then collapse
            t0 = time.time() * 1000
            post("/toggle", {"trader": trader, "path": path})
            ms = stream.wait_action(t0)
            if ms is not None:
                out.append(ms)
            time.sleep(0.3)
    report(label, out, stream)


def report(label, out, stream):
    if not out:
        print(f"{label:10} no samples")
        return
    out.sort()
    p90 = out[int(0.9 * (len(out) - 1))]
    print(f"{label:10} n={len(out):3}  p50 {statistics.median(out):5.0f} ms  p90 {p90:5.0f} ms"
          f"  max {out[-1]:5.0f} ms   (screen lag now {stream.last['lag']} ms)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--trader", default="clicks")
    args = ap.parse_args()
    s = Stream(args.trader)
    time.sleep(0.5)
    groups = [["Rates"], ["EM"], ["Munis"], ["Covered"], ["HY Credit"], ["IG Credit"]]
    run(s, args.trader, args.n, ["business_group", "currency", "sector"],
        ["market_value", "dv01"], groups, "cube")
    run(s, args.trader, args.n, ["business_group", "currency", "sector"],
        ["market_value", "dv01", "min_price", "max_price", "median_mv"], groups, "cube+med")
    issuers = [[f"Issuer {i:04d}"] for i in (0, 7, 13, 21, 42, 99)]
    run(s, args.trader, args.n, ["issuer", "bond"], ["market_value", "dv01"], issuers, "on demand")
    out = []
    layouts = [["seniority", "business_group"], ["book"], ["maturity_year", "currency"], ["coupon_band"]]
    for i in range(args.n):
        t0 = time.time() * 1000
        post("/layout", {"trader": args.trader, "rows": layouts[i % len(layouts)],
                         "measures": ["market_value", "dv01"], "expanded": []})
        ms = s.wait_action(t0)
        if ms is not None:
            out.append(ms)
        time.sleep(0.5)
    report("layout", out, s)


if __name__ == "__main__":
    main()
