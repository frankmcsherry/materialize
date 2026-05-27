# mz-bridge (MVP)

Streams several Materialize materialized views into a downstream Postgres
database with **cross-view-consistent, restartable** commits.

It runs one `SUBSCRIBE ... AS OF <t> WITH (PROGRESS)` per view, buffers the
changes, and commits a consistent cut — everything below the **minimum**
progress timestamp across all streams — to Postgres in a single transaction,
together with that timestamp. On restart it reads the committed timestamp back
out of Postgres and resumes `AS OF F-1`, so nothing is re-snapshotted and
nothing is lost or duplicated. `RETAIN HISTORY` on the views keeps the resume
point readable.

This is a deliberately small MVP. See **Limitations** below for what it does
*not* yet do.

## How it works

- **Consistent cut.** All materialized views in a Materialize environment share
  one logical timeline, so progress timestamps from independent subscriptions
  are directly comparable. Once every stream has reported a progress message at
  or beyond `F`, every stream has delivered *all* of its updates with timestamp
  `< F` (Materialize's output is non-decreasing in time and per-timestamp
  consolidated). `F = min(progress)` is therefore a globally consistent cut.
- **Exactly once.** The only durable state is downstream: the mirrored rows and
  the checkpoint `F`, committed together atomically. A crash just means we
  resume `AS OF F-1` (which redelivers exactly timestamps `>= F`) and carry on.
- **Count mirror.** With the default (diff) envelope, `mz_diff` is a z-set
  multiplicity. Each target table stores one row per distinct content (all
  columns `TEXT`) plus an integer `mz_diff` count and a `__key` content key.
  Applying a change is `mz_diff += d` via `ON CONFLICT`, then deleting rows
  whose count reaches 0 — no "remove N physical duplicates" gymnastics.
- **"Safe to commit" = compare-and-swap.** Each commit advances the checkpoint
  with `UPDATE _mz_bridge_progress SET frontier = :F WHERE bridge_id = :id AND
  frontier = :expected`, and requires the row count to be exactly 1 — in the
  *same transaction* as the data changes. If another writer (a zombie or an
  accidental double-start) advanced the checkpoint, the CAS matches 0 rows and
  the whole transaction rolls back, so nothing is double-applied; that writer
  exits. This is deliberately plain standard SQL (a one-time `INSERT` to
  establish the row, then `UPDATE … WHERE … ` + row-count) so the architecture
  ports to essentially any transactional store, no DB-specific locking needed.

## Concurrency

- **Independent groups are safe.** Run as many bridges as you like as long as
  each has a distinct `bridgeId` *and* distinct target tables; they share the
  same Materialize timeline but never touch each other's state.
- **Two writers with the same `bridgeId` cannot corrupt.** The checkpoint CAS
  serializes them: exactly one wins each commit, the loser rolls back and exits
  (code 2). After one race, a single writer survives. This holds across process
  restarts because the guard lives in durable state, not a session lock.

## Layout

```
src/config.ts       config + validation
src/types.ts        bigint helpers, content key
src/mz.ts           admin setup + per-view SUBSCRIBE/FETCH reader
src/coordinator.ts  min-frontier consistency engine
src/pg.ts           Postgres sink: DDL + one-transaction apply + checkpoint
src/index.ts        wiring; fresh vs. resume
harness/setup.sql   demo table + 3 materialized views (RETAIN HISTORY 1h)
harness/changes.sql some inserts/updates/deletes to watch replicate
```

## Run

Prereqs: a running Materialize (pgwire on `:6875`) and a Postgres (`:5432`),
plus Node 18+.

```bash
cd play/mz-bridge
npm install
cp config.example.json config.json   # edit connection strings if needed

# 1. set up the demo in Materialize
psql "postgres://materialize@localhost:6875/materialize" -f harness/setup.sql

# 2. start the bridge (Ctrl-C to stop)
npm run dev            # or: npm run build && npm start

# 3. in another shell, drive some changes and watch them land in Postgres
psql "postgres://materialize@localhost:6875/materialize" -f harness/changes.sql
psql "postgres://postgres:postgres@localhost:5432/postgres" \
  -c 'SELECT * FROM mv_accounts ORDER BY 1;' \
  -c 'SELECT name, mz_diff FROM mv_names ORDER BY 1;' \
  -c 'SELECT frontier FROM _mz_bridge_progress;'
```

## Verify

**Consistency.** With the bridge caught up, the committed frontier is in
`_mz_bridge_progress.frontier`. The mirror should equal the view as of that
exact time:

```sql
-- in Postgres: current mirror of mv_accounts
SELECT id, name, balance, mz_diff FROM mv_accounts ORDER BY 1;
```
```sql
-- in Materialize, with <F> from _mz_bridge_progress.frontier:
SELECT id, name, balance FROM mv_accounts AS OF <F> ORDER BY 1;
```

For `mv_names` (a multiset) the mirror keeps a count, which should match:

```sql
-- Postgres
SELECT name, mz_diff FROM mv_names ORDER BY 1;
-- Materialize
SELECT name, count(*) FROM mv_names AS OF <F> GROUP BY name ORDER BY 1;
```

**Resume.** Kill the bridge (`kill -9` the `node`/`tsx` process) mid-stream,
make more changes in Materialize, then start it again. It logs
`resume: committed F=...` (not `fresh start`), does not re-snapshot, and the
mirror converges to the same answer exactly once.

**History expired (negative test).** Shrink the window
(`ALTER MATERIALIZED VIEW mv_accounts SET (RETAIN HISTORY FOR '1s')`), stop the
bridge, wait, then restart: the `SUBSCRIBE ... AS OF` fails with *"could not
find a valid timestamp for the query"* and the bridge exits rather than
silently returning a wrong answer. Remediation is to drop the downstream tables
+ checkpoint row and start fresh.

## Limitations (intentional, for the MVP)

- **Snapshots are buffered in memory.** A very large initial snapshot can OOM.
  (Future: spill to Postgres temp/staging tables.)
- **No automatic recovery from an expired history window** — it fails loudly;
  you re-snapshot by hand.
- **Connection drops / unexpected Materialize or Postgres errors are not
  retried** — the bridge exits; restart it (it resumes from the checkpoint).
- **All data columns are stored as `TEXT`.** This is a faithful passthrough of
  the wire format, but it is not a typed schema.
- **The set of views is fixed at startup.** No runtime add/remove.
- **A tiny transaction fires roughly every second even when idle** (to keep the
  checkpoint tight). Throttling the checkpoint flush is future work.
- **`RETAIN HISTORY` is set once** to a fixed window and never relaxed; it is a
  duration relative to now, not an absolute floor.
- **`ENVELOPE` must be the default** (diff) envelope, and `WITH (PROGRESS)` is
  required — both are set by the bridge.
