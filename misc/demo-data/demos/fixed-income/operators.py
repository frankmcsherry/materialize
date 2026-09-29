#!/usr/bin/env python3
# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

"""Where scheduling and progress time goes, per operator, from mz_introspection.

    python3 operators.py snap 20 ops.json     # two snapshots 20s apart
    python3 operators.py report ops.json 16   # 16 = workers in the replica

A subgraph's (region's or dataflow's) "self" time is its scheduling time minus
its children's: the progress tracking and child scheduling it does itself.
"Sched" counts are operator schedulings summed over workers, from
mz_compute_operator_durations_histogram. Records are what arrive on an
operator's input channels, from mz_message_counts.
"""

import collections
import json
import os
import re
import sys
import time

import psycopg

DSN = os.environ.get("MZ_DSN", "host=127.0.0.1 port=16875 user=materialize dbname=materialize")
# Everything the demo creates lives in the mz-demo-data skill's schema.
SEARCH_PATH = "-c search_path=materialize_demo"


def snapshot():
    c = psycopg.connect(DSN, autocommit=True, options=SEARCH_PATH)
    c.execute("SET cluster = quickstart")
    def q(sql): return c.execute(sql).fetchall()
    def snap():
        return {
            "elapsed": {int(i): int(e) for i, e in q("SELECT id, elapsed_ns FROM mz_introspection.mz_scheduling_elapsed")},
            "inv": {int(i): int(n) for i, n in q("SELECT id, SUM(count) FROM mz_introspection.mz_compute_operator_durations_histogram GROUP BY id")},
            "msg": {int(i): (int(s), int(r), int(bs), int(br)) for i, s, r, bs, br in q("SELECT channel_id, sent, received, batch_sent, batch_received FROM mz_introspection.mz_message_counts")},
            "t": time.time(),
        }
    ops = {int(i): (n, [int(x) for x in a.strip("{}").split(",")]) for i, n, a in
           q("SELECT o.id, o.name, a.address::text FROM mz_introspection.mz_dataflow_operators o JOIN mz_introspection.mz_dataflow_addresses a USING (id)")}
    parent = {int(i): int(p) for i, p in q("SELECT id, parent_id FROM mz_introspection.mz_dataflow_operator_parents")}
    flows = {int(i): n.replace("Dataflow: ", "") for i, n in q("SELECT id, name FROM mz_introspection.mz_dataflows")}
    chans = {int(i): (int(f) if f is not None else None, int(t) if t is not None else None, ty) for i, f, t, ty in
             q("SELECT id, from_operator_id, to_operator_id, type FROM mz_introspection.mz_dataflow_channel_operators")}
    dt = float(sys.argv[2])
    a = snap(); time.sleep(dt); b = snap(); dt = b["t"] - a["t"]
    d = lambda k, i: (b[k].get(i, 0) - a[k].get(i, 0))
    children = collections.defaultdict(list)
    for i, p in parent.items(): children[p].append(i)
    out = {}
    for i, (name, addr) in ops.items():
        el = d("elapsed", i) / 1e9 / dt
        kids = children.get(i, [])
        self_ = el - sum(d("elapsed", k) / 1e9 / dt for k in kids)
        out[i] = dict(name=name, addr=addr, flow=flows.get(addr[0], str(addr[0])), leaf=not kids,
                      cores=el, self=self_, inv=d("inv", i) / dt, recv=0.0, batches=0.0)
    for ch, (f, t, ty) in chans.items():
        if t in out and ch in b["msg"]:
            s0 = a["msg"].get(ch, (0, 0, 0, 0)); s1 = b["msg"][ch]
            out[t]["recv"] += (s1[1] - s0[1]) / dt; out[t]["batches"] += (s1[3] - s0[3]) / dt
    json.dump({"dt": dt, "ops": out}, open(sys.argv[3], "w"))
    print(f"saved {len(out)} operators over {dt:.1f}s")


def report_dataflows():
    D = json.load(open(sys.argv[2])); ops = D["ops"]
    leaf = [o for o in ops.values() if o["leaf"]]; sub = [o for o in ops.values() if not o["leaf"]]
    C = lambda xs, k="self": sum(o[k] for o in xs)
    print(f"operators {len(ops)}: {len(leaf)} leaf, {len(sub)} subgraphs (regions + dataflow roots)")
    print(f"leaf operator time {C(leaf):.2f} cores; subgraph self time (progress + scheduling children) {C(sub):.2f} cores")
    print(f"leaf schedulings {C(leaf,'inv'):,.0f}/s; subgraph schedulings {C(sub,'inv'):,.0f}/s  (summed over workers)")
    # per dataflow
    fl = collections.defaultdict(lambda: dict(n=0, leaf=0.0, sub=0.0, inv=0.0, recv=0.0, subn=0, sinv=0.0))
    for o in ops.values():
        f = fl[o["flow"]]; f["n"] += 1
        if o["leaf"]: f["leaf"] += o["self"]; f["inv"] += o["inv"]; f["recv"] += o["recv"]
        else: f["sub"] += o["self"]; f["subn"] += 1; f["sinv"] += o["inv"]
    print(f"\n{'dataflow':44} {'ops':>5} {'regions':>7} {'leaf cores':>10} {'progress cores':>14} {'ratio':>6} {'leaf sched/s':>12} {'records in/s':>12}")
    for n, f in sorted(fl.items(), key=lambda kv: -(kv[1]["sub"] + kv[1]["leaf"])):
        r = f["sub"] / f["leaf"] if f["leaf"] > 1e-4 else float("inf")
        print(f"{n[:44]:44} {f['n']:5} {f['subn']:7} {f['leaf']:10.3f} {f['sub']:14.3f} {r:6.1f} {f['inv']:12,.0f} {f['recv']:12,.0f}")


def report_categories():
    D = json.load(open(sys.argv[2])); ops = D["ops"]; flow = None
    def cat(n):
        if re.search(r'ArrangementSize|LogOperatorHydration|LogDataflowErrors|StartSignal|Probe|InspectBatch|LogImport|Logging|shutdown|Shutdown|token', n): return "introspection / lifecycle"
        if re.search(r'err|Err|Fallibl|ErrorCheck', n): return "error path"
        if re.search(r'Arrange|Reduce|Join|TopK|Threshold|Distinct|Mins|Maxes|Hierarchical|Accumulable|Basic|Bucket|Consolidate|consolidat|Collation', n): return "stateful: arrange / reduce / join / topk"
        if re.search(r'persist|txns|Persist|shard|Source|source|Import|import', n): return "sources / persist"
        if re.search(r'subscribe|Subscribe|Sink|sink', n): return "sink / subscribe"
        return "stateless plumbing (map, flat_map, concat, enter/leave, ...)"
    g = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0]); names = collections.defaultdict(collections.Counter)
    for o in ops.values():
        if not o["leaf"] or (flow and flow not in o["flow"]): continue
        c = cat(o["name"]); g[c][0] += 1; g[c][1] += o["self"]; g[c][2] += o["inv"]; g[c][3] += o["recv"]
        names[c][re.sub(r'\d+', 'N', o["name"])[:40]] += 1
    tot = sum(v[0] for v in g.values())
    print(f"{flow or 'all dataflows'}: {tot} leaf operators")
    for c, v in sorted(g.items(), key=lambda kv: -kv[1][0]):
        print(f"  {v[0]:5} ({100*v[0]/tot:4.1f}%)  {v[1]:6.3f} cores  {v[2]:9,.0f} sched/s  {v[3]:10,.0f} rec/s  {c}")
        print("         " + ", ".join(f"{n}×{k}" for n, k in names[c].most_common(6)))


def report_per_tick():
    ops = json.load(open(sys.argv[2]))["ops"]; W = int(sys.argv[3])
    print(f"\nschedulings per worker per 100ms tick ({W} workers), dataflow roots:")
    for o in sorted([o for o in ops.values() if len(o["addr"]) == 1], key=lambda o: -o["inv"]):
        print(f"  {o['inv'] / W / 10:7.1f}  {o['name'][:70]}")


if __name__ == "__main__":
    if sys.argv[1] == "snap":
        snapshot()
    else:
        report_dataflows(); print(); report_categories(); report_per_tick()
