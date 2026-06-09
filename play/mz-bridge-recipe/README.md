# mz-bridge-recipe

A **reference recipe** for consuming a *cohort* of Materialize `SUBSCRIBE`
streams as one consistent stream. There is no downstream — the bridge's only
output is **one upcall per consistent moment**. The default upcall writes a line
to the console; you replace it with your own logic.

> This project is the part *before* the sink: the mechanism for turning N
> subscriptions into one consistent, resumable stream. There is intentionally no
> worked downstream (Postgres mirror, Kafka, exactly-once checkpoint) — that part
> is yours, plugged in at the upcall. Read **`DESIGN.md`** for the full argument.

## The idea in one screen

```ts
import { Cohort } from "./src/cohort";

// One cohort over three views. The upcall fires each time the cohort's lower
// bound of consistency advances, with everything below it, consolidated.
const cohort = await Cohort.fresh(mzConn, ["mv_accounts", "mv_names", "mv_balances"],
  async (F, batch) => {
    // <-- YOUR LOGIC HERE. Apply `batch`, then durably record `F`.
    //     On restart, build the cohort with Cohort.resuming(..., F, ...).
    console.log(`consistent through ${F}`, batch);
  });

await cohort.failed; // only rejects — fail loudly on a stream error; Ctrl-C exits
```

That's the whole surface. Everything else is the three idioms it encodes.

## Three idioms (see DESIGN.md)

1. **A cohort is a consistency wrapper.** `F = min(frontier)` over the members;
   when it advances, the union of updates below it is a globally consistent cut.
   The bridge **drains every subscribe continuously and never backpressures
   Materialize** — under a slow consumer the bridge falls over, not MZ.
2. **The lifecycle is the API.** `fresh` / `resuming` to start; then `add`,
   `drop`, `merge`, `split`. The one tricky decision — which `AS OF` / `SNAPSHOT`
   — is the wrapper's, computed from cohort state. You never type an `AS OF`.
3. **Durability is yours.** The bridge owns no store and no durable frontier. It
   gives you `(F, batch)`; you commit it and record `F`; on restart you resume
   `from` that `F`. The bridge's whole footprint is `Cohort.resuming(from)` and
   the rule `resumeAsOf(F) = max(F-1, 0)` — restart from the last instant you
   *have*, not the next one you *want*.

## Layout

```
src/types.ts         bigint helpers, content key, resumeAsOf (the durability rule)
src/mz.ts            Subscription: one managed SUBSCRIBE (+ readMzNow)
src/cohort.ts        the wrapper: min-frontier engine, serialized pump, lifecycle verbs
src/console-sink.ts  the default upcall (the seam you replace)
src/index.ts         demo over three views; `--from <F>` to resume
harness/             setup.sql (3 MVs) + changes.sql
RUNNING.md           validate the recipe against a live Materialize (Docker)
```

## Run

```bash
cd play/mz-bridge-recipe
npm install
npm run typecheck     # type-check the recipe
```

To see it actually work, **`RUNNING.md`** walks through bringing up Materialize
in Docker, loading the harness, running the bridge, driving changes, and
resuming — with the consistent-moment output you should expect. The short
version, once MZ is up and `harness/setup.sql` is loaded:

```bash
cp config.example.json config.json
npm run dev                              # fresh: snapshot every view, then stream
npm run dev -- config.json --from <F>    # resume from a recorded F
```

## What you owe (non-goals)

Durable storage and the transaction that makes a recorded `F` true; **idempotent
apply** (delivery is at-least-once; resume re-offers the uncommitted suffix);
type fidelity (values pass through as raw `TEXT`); spill for very large
snapshots (buffered in memory here); and `RETAIN HISTORY` on the objects (set it
in the `CREATE`; the bridge never `ALTER`s it). See `DESIGN.md` for the full list.
