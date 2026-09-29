#!/usr/bin/env python3
# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

"""Bucket a clusterd CPU profile (mzfg) into operator work, progress tracking and logging.

Take the profile from the replica's internal HTTP server, a unix socket named by
clusterd's --internal-http-listen-addr, and pull the `mzfg` string out of the page:

    docker exec mz sh -c 'P=$(pgrep -n clusterd); S=$(tr "\\0" "\\n" < /proc/$P/cmdline \
        | grep internal-http | cut -d= -f2); curl -s --unix-socket $S -X POST \
        -d "action=time_fg&hz=99&time_secs=15&threads=merge" http://x/' > prof.html

    python3 fgbreakdown.py prof.mzfg [frames per bucket]

Buckets are by the frames on the stack: under a leaf OperatorCore is operator
work (merges split out), under Subgraph scheduling but not an operator is
progress tracking, anything touching mz_compute::logging is introspection.
Assumes 99 Hz sampling.
"""

import sys, re, collections
path = sys.argv[1]
lines = open(path).read().split("\n")
secs = float([l for l in lines if l.startswith("Sampling time")][0].split(":")[1]); hz = 99
stacks, syms = [], {}
for l in lines[5:]:
    m = re.match(r'^(\S+) (\d+)$', l)
    if m: stacks.append((m.group(1).split(";"), int(m.group(2))))
    elif l.endswith(";"):
        a, _, n = l.partition(" "); syms[a] = re.sub(r'\[[0-9a-f]{12,}\]', '', n[:-1])
C = lambda w: w / hz / secs
def classify(names):
    # drop the signal-handler frames at the leaf
    while names and (names[-1] in ("", "?", "perf_signal_handler") or names[-1].startswith("0x")):
        names = names[:-1]
    s = " ; ".join(names)
    if "run_worker" not in s:
        return "other threads", names
    if "mz_compute::logging" in s or "TimelyEvent" in s or "logging::" in s and "Logger" in s:
        return "introspection logging (timely/differential events)", names
    if "OperatorCore" in s:
        # Leaf operator work. Split DD merges out.
        if "Merger" in s or "MergeVariant" in s or "Spine" in s and "exert" in s.lower():
            return "operators: arrangement merges", names
        return "operators: other logic", names
    if "Subgraph" in s:
        leaf = "\n".join(names)
        if "propagate_pointstamps" in s or "reachability::Tracker" in s:
            return "progress: pointstamp propagation (Tracker)", names
        if "Progcaster" in s:
            return "progress: broadcast send/recv between workers", names
        if "MutableAntichain" in s or "ChangeBatch" in s:
            return "progress: frontier/changebatch maintenance", names
        return "progress: other subgraph scheduling", names
    if "step_or_park" in s:
        if "park" in names[-1].lower() or "futex" in s or "condvar" in s.lower():
            return "worker: parked/waking", names
        if "allocator" in s or "communication" in s:
            return "worker: communication (receive/flush)", names
        return "worker: other step_or_park", names
    return "worker: outside step", names
cat = collections.Counter(); self_by = collections.defaultdict(collections.Counter)
total = 0
for frames, w in stacks:
    names = [syms.get(a, a) for a in frames]
    c, trimmed = classify(names)
    cat[c] += w; total += w
    self_by[c][(trimmed[-1] if trimmed else "?")[:150]] += w
print(f"total {C(total):.2f} cores")
for c, w in cat.most_common():
    print(f"{C(w):6.2f} cores {100*w/total:5.1f}%  {c}")
    for n, x in self_by[c].most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 3):
        print(f"           {C(x):5.2f}  {n}")
