# Design notes: mz-bridge

A reference design for replicating several Materialize materialized views into a
downstream transactional database with **cross-view-consistent, restartable,
exactly-once** commits, built on `SUBSCRIBE … AS OF … WITH (PROGRESS)` and
`RETAIN HISTORY`.

This is an MVP meant to read like an architecture diagram that happens to run.
The point is the *mechanism*; the Postgres sink is one concrete instantiation.

## Why this is harder than it looks

The user docs and the reference consumer (`console/src/api/materialize/
SubscribeManager.ts`) describe the happy path. Real attempts fail on what the
happy path omits: head-of-line stalls, snapshot bursts, the history-expired
case, multiset semantics, 64-bit timestamp precision, partial-timestamp FETCH
batches, and two-writer races. The value here is confronting those directly.

## Verified Materialize semantics

The correctness argument rests on these behaviors, **verified against the
Materialize source** (paths relative to the repo root), not just the docs:

| Behavior | Source |
|---|---|
| Output is non-decreasing in `mz_timestamp`, and per-timestamp **consolidated** (no `+1/-1` for an unchanged row) | `src/compute/src/sink/subscribe.rs`, `src/adapter/src/active_compute_sink.rs` |
| `FETCH`/batch boundaries do **not** align with timestamps — a batch may straddle timestamps, a timestamp may span batches; the only "time `t` closed" signal is a message with timestamp `> t` | `src/compute-client/src/protocol/response.rs` (`SubscribeBatch { lower, upper }`) |
| First message is a **progress at `AS OF`**; with `SNAPSHOT true`, snapshot rows are emitted **at** `AS OF` (fast-forwarded) | `active_compute_sink.rs::initialize` |
| Progress at `T` ⇒ no more updates at times `< T`. `AS OF` inclusive, `UP TO` exclusive | `doc/user/content/sql/subscribe.md` |
| `SNAPSHOT false AS OF X` emits only timestamps `> X` (so resume uses `AS OF F-1`) | `doc/user/content/transform-data/patterns/durable-subscriptions.md` |
| `mz_diff` is an `i64` **z-set** multiplicity (magnitude can exceed 1) | `src/repr/src/diff.rs` |
| Frontiers advance per-view on a **~1s tick**, including idle views | `default_timestamp_interval` in `src/sql/src/session/vars/definitions.rs`; `src/adapter/src/coord/timeline.rs::advance_timelines` |
| `mz_internal.mz_frontiers(object_id, read_frontier, write_frontier)` in epoch-millis | `src/catalog/src/builtin/mz_internal.rs` |
| `AS OF` below the since/read frontier ⇒ `"could not find a valid timestamp for the query"`; `read_frontier` is the hard floor | `src/adapter/src/coord/timestamp_selection.rs`, `src/adapter/src/error.rs` (`ImpossibleTimestampConstraints`) |
| `RETAIN HISTORY FOR D` is a **duration relative to the write frontier** (`since = upper − D`), not an absolute floor; increasing it does not restore already-compacted history | `src/adapter-types/src/compaction.rs` (`CompactionWindow::lag_from`) |
| `mz_timestamp` (numeric) and `mz_diff` (int8) arrive as **strings** over pgwire → must be parsed as `BigInt`, never JS `number` | node-postgres default parsers |

## Architecture (ports & adapters)

```
   Materialize                 dialect-free core                downstream
  ┌───────────┐   events    ┌──────────────────┐   Sink calls  ┌──────────┐
  │  mz.ts    │ ──────────▶ │  coordinator.ts  │ ────────────▶ │  pg.ts   │
  │ SUBSCRIBE │  onData /   │  min-frontier    │  applyWindow  │ count    │
  │ + parse   │  onProgress │  buffer + commit │  (CAS)        │ mirror   │
  └───────────┘             └──────────────────┘               └──────────┘
        ▲                            ▲                                ▲
        └──────────── index.ts (wiring: fresh vs resume) ─────────────┘
```

- **Source edge (`mz.ts`)** — everything that knows Materialize's dialect. Swap
  this to read a different change source.
- **Core (`coordinator.ts`)** — the dialect-free algorithm. Knows neither SQL
  nor Materialize; just buffers, computes `min(frontier)`, and commits.
- **Sink edge (`pg.ts`)** — everything that knows the downstream's dialect.
  **This is the file you rewrite to port to another transactional store.**
- **Wiring (`index.ts`)** — composition root; no logic of its own.

## Consistency & exactly-once argument

1. **One timeline.** All views in a Materialize environment share a single
   logical clock, so progress timestamps from independent subscriptions are
   directly comparable.
2. **Consistent cut.** `F = min(latest progress across all streams)`. Once every
   stream has progressed to `F`, each has delivered *all* of its updates with
   `timestamp < F` (non-decreasing + consolidated output). So everything `< F`
   is final in every view — a globally consistent cut.
3. **Atomic commit.** Apply all buffered updates with `ts < F` **and** advance
   the checkpoint to `F` in one downstream transaction. We never commit updates
   *at* `F` (a progress at `F` says nothing about `F` itself).
4. **Exactly-once across restarts.** The only durable state is downstream (rows
   + checkpoint, committed atomically). On restart we resume every stream
   `AS OF F-1 WITH (SNAPSHOT false)`, which redelivers exactly the timestamps
   `≥ F` — the uncommitted suffix. No gap, no duplicate.
5. **Buffer drop is safe.** After committing `< F`, no update with `ts < F` can
   still arrive: every stream's frontier is already `≥ F`, and it had delivered
   everything below its frontier before emitting it.

## "Safe to commit" = compare-and-swap (the portable core)

The commit's checkpoint write is a CAS, in the **same transaction** as the data:

```sql
-- once, when a fresh bridge starts:
INSERT INTO _mz_bridge_progress (bridge_id, frontier) VALUES (:id, :T0);
-- every commit:
UPDATE _mz_bridge_progress SET frontier = :F
 WHERE bridge_id = :id AND frontier = :expected;   -- row count MUST be 1
```

If another writer (a zombie or accidental double-start) advanced the checkpoint,
the `UPDATE` matches 0 rows; we raise `CheckpointConflict` and the whole
transaction — including the data changes — rolls back, so nothing is
double-applied. The loser exits; exactly one writer survives. Postgres serializes
the two on the checkpoint row's lock, so there is no deadlock and no corruption.

This is deliberately plain standard SQL (`INSERT` once, then `UPDATE … WHERE …`
+ affected-row count). Porting to another transactional store reduces to those
two primitives — no advisory locks, no DB-specific features. This is the
intended "express *safe to commit* portably" contract.

## Count mirror (default diff envelope)

`mz_diff` is a z-set, so the sink mirrors each view as a **count table**: one row
per distinct content (all columns `TEXT`), plus an integer `mz_diff` multiplicity
and a `__key` content key (`JSON.stringify(values)`, which distinguishes NULL
from `"null"` from `""`). Applying a change is arithmetic — `mz_diff += d` via
`ON CONFLICT (__key)`, then delete rows whose count reaches 0 — so there is no
"remove N physical duplicates" problem, and multiset views (non-unique rows) are
represented faithfully.

## Failure-mode catalog

| Mode | Status in this MVP |
|---|---|
| **Two writers, same `bridgeId`** | **Handled** — checkpoint CAS rejects the loser, which exits (code 2); no corruption. |
| **Independent groups** (distinct `bridgeId` + tables) | **Handled** — fully concurrent, no contention. |
| **History window blown on resume** (`AS OF F-1` < `read_frontier`) | **Fails loudly** — `could not find a valid timestamp`; bridge exits. Remediation is manual: drop the sink tables + checkpoint and re-snapshot. |
| **Multiset / keyless views** | **Handled** — count mirror. |
| **64-bit timestamp precision** | **Handled** — BigInt throughout. |
| **Connection drop / unexpected MZ or PG error** | Partial — the bridge exits; restart resumes from the checkpoint. No in-process retry. |
| **Large initial snapshot** | **Not handled** — buffered in memory; a huge snapshot can OOM. (Future: spill to staging tables.) |
| **Head-of-line stall** (one lagging view holds back `min`) | **Not handled** — buffers grow; no backpressure or metrics yet. |
| **Type fidelity** | All data stored as `TEXT` (faithful wire passthrough); not a typed schema. |
| **Schema drift** (MV columns change) | Not handled. |
| **Dynamic `RETAIN HISTORY` tightening / runtime add-remove of streams** | Not handled. |

## Verification performed

- **`test/sim.ts`** — 16 assertions driving the real `Coordinator` + `Pg` sink
  against a live Postgres with a simulated SUBSCRIBE stream: snapshot, update,
  multiset decrement, checkpoint, resume (no backward commit, no double-apply),
  and the two-writer CAS rejection (second writer rejected, sink uncorrupted).
- **Live Materialize end-to-end** (single-node, `RETAIN HISTORY` enabled):
  fresh-start snapshot consistent with source `AS OF F` for all three views
  (incl. the multiset); live update/delete/multiset burst consistent; crash +
  resume exactly-once (`mz_diff <> 1` count = 0); history-expired negative test
  (loud failure, no silent divergence); duplicate-writer exits via CAS while the
  original stays healthy and uncorrupted.
