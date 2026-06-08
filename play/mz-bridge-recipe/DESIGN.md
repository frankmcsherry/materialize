# mz-bridge-recipe — an idiomatic bridge for a cohort of SUBSCRIBEs

This is a **reference recipe**, not a product. It shows the *mechanism* for
consuming several Materialize `SUBSCRIBE` streams as a single consistent whole,
and it makes the easy-to-get-wrong parts explicit. There is **no downstream** —
no Postgres, no Kafka, nothing. The bridge has exactly **one integration
point**: an upcall fired once per consistent moment. The default upcall writes a
one-line summary to the console; you replace it with your own logic.

> The point is *what to subscribe to, when, and how to resume* — not where the
> bytes go. The bridge does not own a store, does not own durability, and never
> stalls Materialize. Everything past the upcall is yours, and we say exactly
> what you owe.

Three idioms, two layers, one rule:

1. **A cohort is a consistency wrapper over independent SUBSCRIBEs** — it turns
   `N` streams into one stream of consistent moments.
2. **The cohort lifecycle *is* the API** — `fresh`/`resuming`, then
   `add`/`drop`/`merge`/`split`. The interesting decision (which `AS OF`) is the
   wrapper's, not yours.
3. **Durability is the consumer's** — the bridge's only footprint is the
   `resuming(from)` constructor and the rule `resumeAsOf(F) = max(F-1, 0)`.

Layers: a **Subscription** (one managed `SUBSCRIBE`) and a **Cohort** (the
wrapper). The Subscription knows where to start and then only emits; the Cohort
holds the consistency logic and the lifecycle verbs.

---

## Verified Materialize semantics

The correctness argument rests on these behaviors, verified against the
Materialize source (paths relative to the repo root):

| Behavior | Source |
|---|---|
| Output is non-decreasing in `mz_timestamp`, and per-timestamp **consolidated** | `src/compute/src/sink/subscribe.rs`, `src/adapter/src/active_compute_sink.rs` |
| A progress message at `T` ⇒ no more updates at times `< T`; `AS OF` inclusive, `UP TO` exclusive | `doc/user/content/sql/subscribe.md` |
| `FETCH` batch boundaries do **not** align with timestamps; the only "time `t` closed" signal is a later message with timestamp `> t` | `src/compute-client/src/protocol/response.rs` |
| First message is a **progress at `AS OF`**; with `SNAPSHOT true`, snapshot rows are emitted **at** `AS OF` (fast-forwarded) | `active_compute_sink.rs::initialize` |
| `SNAPSHOT false AS OF X` emits only timestamps **strictly greater** than `X` (so resuming through `F` uses `AS OF F-1`) | `doc/user/content/transform-data/patterns/durable-subscriptions.md` |
| `mz_diff` is an `i64` z-set multiplicity (magnitude can exceed 1) | `src/repr/src/diff.rs` |
| Frontiers advance per-object on a **~1s tick**, including idle objects | `default_timestamp_interval` in `src/sql/src/session/vars/definitions.rs` |
| `AS OF` below the read frontier ⇒ `"could not find a valid timestamp for the query"` | `src/adapter/src/coord/timestamp_selection.rs` |
| `mz_timestamp` / `mz_diff` arrive as **strings** over pgwire → parse as `BigInt`, never JS `number` | node-postgres default parsers |

All of it rests on one fact: **every object in one Materialize environment shares
one logical timeline**, so progress timestamps from independent subscriptions are
directly comparable. That is what makes `min(frontier)` meaningful across streams.

---

## Idiom 1 — a cohort is a consistency wrapper

A **subscription** is one `SUBSCRIBE <obj> WITH (PROGRESS) AS OF <t>` stream. It
emits data updates `(ts, diff, row)` and progress messages (`ts`, no row). A
**cohort** is a set of subscriptions observed as one unit. The cohort's
**consistency frontier** is `F = min(frontier)` over its members. When `F`
advances, every member has delivered *all* of its updates at times `< F`, so the
union of those updates is a globally consistent cut — and we fire the upcall:

```
onConsistent(F, batch)   // batch = per-subscription, z-set-consolidated updates with ts < F
```

We never release updates *at* `F` — a progress at `F` closes `(-∞, F)` but says
nothing about `F` itself. Consolidation means a row that churned `+1/-1` within a
window collapses to nothing; a multiset row carries its net `i64` multiplicity.
An **idle advance** (F moved, no data) still fires the upcall with an empty
batch: it is a real commit point, and lets the consumer record a tighter resume
position even when nothing changed.

### Never backpressure Materialize

Each subscription's `FETCH` loop **drains continuously and never pauses for the
consumer**. Not consuming a `SUBSCRIBE` makes *Materialize* hold state and grow
unboundedly — the bridge must be the thing that falls over, not MZ. So the loops
are fully decoupled from the upcall (and from each other). A separate,
**serialized** commit pump notices when `F` advances, slices the batch, and
`await`s the upcall before offering the next one — purely so the consumer's
commit notifications are **ordered and non-overlapping**, *not* to throttle the
drain. A slow consumer grows the cohort's in-memory buffers; under sustained lag
the bridge falls over (OOM, or a configured bound that fails loudly). MZ keeps
being consumed throughout.

So the upcall's whole role is: **communicate one atomic, cross-subscribe,
transactional commit point.** Its return means only "delivered, send the next
when ready" — never "durable" (see Idiom 3).

---

## Idiom 2 — the cohort lifecycle is the API

You manage cohorts; the wrapper translates that into correct `SUBSCRIBE` choices.
The verbs split into two classes:

**Subscription-bearing** (start/stop real streams):

- `Cohort.fresh(conn, views, upcall)` — snapshot every view at one shared time
  and stream forward. Every member: `AS OF mz_now() WITH (SNAPSHOT = TRUE)`, so
  the initial snapshots form one consistent cut.
- `Cohort.resuming(conn, views, from, upcall)` — reconstitute from a durably
  recorded frontier (Idiom 3). Every member: `AS OF resumeAsOf(from) WITH
  (SNAPSHOT = FALSE)`.
- `cohort.add(view) -> handle` — join a brand-new view to an **already-live**
  cohort: `AS OF F_C WITH (SNAPSHOT = TRUE)`, where `F_C` is the cohort's current
  consistency frontier. Its first progress then lands exactly at `F_C`, so it
  neither regresses the cohort nor leaves a gap. (For the initial set, use the
  constructor — that avoids racing a member onto a cohort that has begun
  emitting. `add` deliberately throws if the cohort is not live yet.)
- `cohort.drop(handle)` — stop and remove a member; its undelivered buffer is
  discarded and `min` may jump.

**Membership-only** (no `SUBSCRIBE` is ever re-issued — instant):

- `cohort.merge(other)` — absorb `other`'s members; from here the union is one
  consistency unit governed by **this** cohort's upcall. The merged commit point
  is `max(committedF)` of the two, so the governing upcall sees monotonic `F`
  and no buffered data is lost (the lagging half's backlog flushes once it
  catches up). Each half's *pre-merge* history stays with its old upcall, so for
  a clean handoff point both upcalls at the same store, or make apply idempotent.
- `cohort.split(handles, upcall) -> child` — move some members into a new cohort
  with its own upcall; this cohort keeps the rest. The child inherits this
  cohort's commit point, so it continues cleanly. Use it to **decouple a slow
  member** so the rest can advance and be made durable without waiting.

The single non-trivial decision — which `AS OF` / `SNAPSHOT` — lives entirely
inside the wrapper, as a function of cohort state:

```
if cohort is live (has F_C):     asOf = F_C,              snapshot = TRUE   // join
elif cohort is resuming:         asOf = resumeAsOf(from), snapshot = FALSE  // resume
else (fresh):                    asOf = mz_now(),         snapshot = TRUE   // first snapshot
```

`join` and `fresh` snapshot (you need the contents); resume does not (you have
them). Only `AS OF` differs. The consumer never types an `AS OF`.

---

## Idiom 3 — durability is the consumer's

**The bridge cannot provide durability and takes no action on it.** It owns no
store, cannot confirm your write landed, and your commit may confirm
asynchronously. So there is no `confirmDurable` verb — it would have nothing to
do:

- *Drop buffers?* Already done at **delivery** (recovery is re-`SUBSCRIBE`, never
  replay-from-buffer), so a durability ack buys nothing.
- *Pick the resume point?* Resume only happens in a **new process**; an in-memory
  durable frontier dies with the crash that needed it. The only durable record
  of `F` is the one **you** wrote to your store.
- *Compact upstream?* Different feature, and a footgun (one slow consumer would
  gate history for everyone). Out of scope.

So the contract is **at-least-once delivery of consistent batches**, which you
turn into **exactly-once**:

1. The upcall hands you `(F, batch)`.
2. You durably apply `batch` **and** record `F` — ideally in one transaction in
   your store.
3. On restart, you read `F` back and build the cohort `from: F`.

After a crash you resume from your last durably recorded `F` and **re-receive
everything since**, so your apply must be idempotent. The bridge's entire
durability footprint is two things — no verb, no state:

```ts
Cohort.resuming(conn, views, from, upcall)   // `from` is the F you recorded
const resumeAsOf = (F: bigint) => (F > 0n ? F - 1n : 0n);
```

**The off-by-one is the whole lesson.** To redeliver exactly `ts >= F`, you must
subscribe `AS OF F-1` with `SNAPSHOT = FALSE` (which emits strictly `> AS OF`).
Restarting `AS OF F` — "the next moment I want to hear about" — silently **skips
the boundary `ts = F`**. Restart from the last instant you *have* (`F-1`) and
re-hear it; never from the next instant you *want*. The `max(·, 0)` clamps the
very first instant (timestamps are non-negative).

If `F-1` has fallen below an object's read frontier (its `RETAIN HISTORY` window
expired while you were down), the `SUBSCRIBE` fails loudly with *"could not find
a valid timestamp"* rather than diverging silently — re-snapshot by hand.

---

## Module layout

```
src/types.ts         bigint helpers, content key, batch types, resumeAsOf (the rule)
src/mz.ts            Subscription (managed SUBSCRIBE) + readMzNow + factory seam
src/cohort.ts        the wrapper: min-frontier engine, serialized pump, lifecycle verbs
src/console-sink.ts  the default upcall (the "write to console" placeholder seam)
src/config.ts        tiny config (mzConn + view list)
src/index.ts         demo: a fresh/resuming cohort over three views
test/sim.ts          drives the engine with a fake source (no live MZ): the idioms as assertions
harness/setup.sql    a table + three MVs (keyed set, multiset, aggregate) with RETAIN HISTORY
harness/changes.sql  some changes to watch flow through
```

## What the consumer owns (non-goals here)

- Durable storage and the transaction that makes a recorded `F` true.
- **Idempotent apply** (needed after every resume re-offer, and after a merge).
- Type fidelity — values pass through as raw `TEXT`; you cast.
- Backpressure / spill for very large snapshots (buffered in memory here).
- `RETAIN HISTORY` (set it on the objects; the bridge does not `ALTER`).
- Schema drift, runtime retention changes, and durable bookkeeping of cohort
  *topology* across restarts (which `from` a merged/split cohort resumes with —
  conservatively the `min` over the members you durably committed).
