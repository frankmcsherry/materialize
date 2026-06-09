# Running the recipe against a live Materialize

This is the recipe's verification: bring up a real Materialize in Docker, point
the bridge at it, and watch consistent moments stream from the default
console upcall. Everything below has been run end-to-end; the output blocks are
real (only the `F=` timestamps, which are wall-clock logical times, will differ
on your machine).

Prerequisites: Docker, Node 20+, and `psql`.

## 1. Start a local Materialize

```bash
docker run -d --name mz -p 6875:6875 -p 6877:6877 materialize/materialized:latest

# wait until it answers SQL (a few seconds):
until psql "postgres://materialize@localhost:6875/materialize" -tAc "SELECT 1" >/dev/null 2>&1; do sleep 1; done
```

`:6875` is the normal SQL port; `:6877` is the internal admin port (user
`mz_system`), used only for the next step.

> **`RETAIN HISTORY` on the open-source emulator.** The harness views use
> `RETAIN HISTORY` so a cohort can resume after a restart. The emulator gates
> that behind a private-preview flag, so enable it once, as `mz_system`, before
> creating the views:
>
> ```bash
> psql "postgres://mz_system@localhost:6877/materialize" \
>   -c "ALTER SYSTEM SET enable_logical_compaction_window = true"
> ```
>
> On a managed Materialize region `RETAIN HISTORY` is available and this step is
> unnecessary.

## 2. Create the harness schema

```bash
psql "postgres://materialize@localhost:6875/materialize" -f harness/setup.sql
```

This makes a writable `accounts` table and three views that exercise the shapes
the bridge must handle — `mv_accounts` (keyed set), `mv_names` (a multiset, so
`mz_diff > 1`), `mv_balances` (an aggregate) — and seeds three rows.

## 3. Run the bridge (fresh)

```bash
npm install
cp config.example.json config.json
npm run dev
```

You should see a single consistent moment carrying the snapshot as one
cross-view cut, then ~1s idle advances:

```
[demo] fresh start over mv_accounts, mv_names, mv_balances
[demo] streaming consistent moments (Ctrl-C to exit)
[demo] consistent through F=… — mv_accounts: +1×(2,bob,200) +1×(1,alice,100) +1×(3,alice,50) | mv_names: +1×(bob) +2×(alice) | mv_balances: +1×(bob,200) +1×(alice,150)
[demo] consistent through F=… (idle)
```

Note `mv_names: +2×(alice)` — alice appears in two rows, so the multiset carries
multiplicity 2 — and the idle line: a real commit point with no data.

## 4. Drive changes and watch them flow

In a second shell:

```bash
psql "postgres://materialize@localhost:6875/materialize" -f harness/changes.sql
```

The bridge prints a consistent moment per advance, z-set–consolidated:

```
[demo] consistent through F=… — mv_accounts: +1×(4,carol,75) | mv_names: +1×(carol) | mv_balances: +1×(carol,75)
[demo] consistent through F=… — mv_accounts: -1×(2,bob,200) +1×(2,bob,250) -1×(3,alice,50) +1×(100,alice,100) +1×(101,alice,101) +1×(102,alice,102) +1×(103,alice,103) +1×(104,alice,104) +1×(105,alice,105) | mv_names: +5×(alice) | mv_balances: -1×(bob,200) +1×(bob,250) -1×(alice,150) +1×(alice,715)
```

The `UPDATE` arrives as a retract/insert pair (`-1×(2,bob,200) +1×(2,bob,250)`),
and the `DELETE id=3` plus the six-row bulk insert **consolidate to a net
`+5×(alice)`** in `mv_names`. That consolidation is the bridge doing its job:
one atomic, net batch per consistent moment.

## 5. Resume from a recorded `F`

Pick an `F` the bridge printed (in a real consumer this comes back out of *your*
store, committed in the same transaction as the data). Stop the bridge with
Ctrl-C, then:

```bash
npm run dev -- config.json --from <F>
```

The bridge subscribes `AS OF <F>-1 WITH (SNAPSHOT = FALSE)`, so it redelivers
**only** updates with `ts ≥ F` — the suffix you hadn't durably committed — and
does **not** re-snapshot. Resume from the snapshot's `F` and you'll see just the
changes since, e.g.:

```
[demo] resume from F=… over mv_accounts, mv_names, mv_balances
[demo] consistent through F=… — mv_accounts: +1×(4,carol,75) -1×(2,bob,200) +1×(2,bob,250) | …
[demo] consistent through F=… — mv_accounts: -1×(3,alice,50) +1×(100,alice,100) … | mv_names: +5×(alice) | mv_balances: -1×(alice,150) +1×(alice,715)
```

The changes may be **regrouped** into different consistent moments than you saw
the first time (the frontiers tick at different wall-clock instants), but they
converge to the same net state. That is the at-least-once / idempotent-apply
contract in practice: resume re-offers the uncommitted suffix; your apply makes
it exactly-once. (The `-1` in `AS OF F-1` is the whole point — see Idiom 3 in
`DESIGN.md`.)

## 6. Tear down

```bash
docker rm -f mz
```

## What this validates

- **fresh** — every member's snapshot forms one consistent cross-view cut.
- **idle advances** — commit points fire even with no data.
- **consolidation** — across keyed-set, multiset, and aggregate views.
- **resume** — `AS OF F-1`, `SNAPSHOT = FALSE`, suffix-only redelivery (the
  off-by-one), idempotent re-offer.

Not exercised here: the membership verbs `add` / `merge` / `split`, which the
demo wiring (`src/index.ts`) does not call. Their semantics — and the subtleties
in `merge` (inconsistent until the first post-merge upcall) and `add` (stalls,
never regresses) — are documented in `DESIGN.md`.
