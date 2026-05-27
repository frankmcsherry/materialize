// Integration test for the consistency engine + Postgres sink, WITHOUT a live
// Materialize. It feeds the Coordinator a hand-built SUBSCRIBE event stream
// that mimics real semantics (snapshot at AS OF, per-timestamp updates,
// multiset multiplicities, min-frontier progress) and asserts the downstream
// Postgres state after each consistent commit, plus resume behavior.
//
// Run: PG=postgres://postgres@127.0.0.1:55432/postgres node_modules/.bin/tsx test/sim.ts

import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "pg";
import { loadConfig } from "../src/config";
import { Coordinator } from "../src/coordinator";
import { CheckpointConflict, Pg } from "../src/pg";

const PG = process.env.PG ?? "postgres://postgres@127.0.0.1:55432/postgres";

let failures = 0;
function check(name: string, actual: unknown, expected: unknown): void {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) {
    console.log(`  ok   ${name}`);
  } else {
    failures++;
    console.log(`  FAIL ${name}\n        expected ${e}\n        actual   ${a}`);
  }
}

/** Like `check`, but order-insensitive — mirror row order is not significant
 * (and `__key` is now a hash, so physical order is arbitrary). */
function checkRows(name: string, actual: unknown[], expected: unknown[]): void {
  const sorted = (xs: unknown[]) =>
    [...xs].sort((a, b) => (JSON.stringify(a) < JSON.stringify(b) ? -1 : 1));
  check(name, sorted(actual), sorted(expected));
}

/** Read a mirror table as a sorted list of {cols..., mz_diff}, dropping __key. */
async function dump(client: Client, table: string): Promise<unknown[]> {
  const r = await client.query(
    `SELECT * FROM "${table}" ORDER BY "__key"`,
  );
  return r.rows.map((row) => {
    const { __key, ...rest } = row;
    return rest;
  });
}

async function checkpoint(client: Client, bridgeId: string): Promise<string | null> {
  const r = await client.query(
    "SELECT frontier::text AS f FROM _mz_bridge_progress WHERE bridge_id = $1",
    [bridgeId],
  );
  return r.rows.length ? r.rows[0].f : null;
}

async function main(): Promise<void> {
  const reader = new Client({ connectionString: PG });
  await reader.connect();

  // Clean slate.
  await reader.query(`DROP TABLE IF EXISTS ta, tb`);
  await reader.query(`DELETE FROM _mz_bridge_progress WHERE bridge_id = 'test'`).catch(() => {});

  // ---- Scenario 1: fresh start (T0 = 100) -------------------------------
  console.log("scenario: fresh start + snapshot");
  const pg = new Pg(PG);
  await pg.ensureProgressTable();
  await reader.query(`DELETE FROM _mz_bridge_progress WHERE bridge_id = 'test'`);

  const targets = [
    { target: "ta", columns: null as string[] | null },
    { target: "tb", columns: null as string[] | null },
  ];
  const coord = new Coordinator(2, targets, pg, "test", 100n);

  coord.setColumns(0, ["id", "val"]);
  coord.setColumns(1, ["name"]);
  await pg.ensureTable("ta", ["id", "val"]);
  await pg.ensureTable("tb", ["name"]);
  await pg.initCheckpoint("test", 100n, "sim"); // establish row for CAS commits

  // First progress is at AS OF for each stream (T0 = 100). min == committedF.
  await coord.onProgress(0, 100n);
  await coord.onProgress(1, 100n);

  // Snapshot rows, all at ts = 100. tb is a multiset: alice appears twice.
  coord.onData(0, 100n, 1n, ["1", "alice"]);
  coord.onData(0, 100n, 1n, ["2", "bob"]);
  coord.onData(1, 100n, 1n, ["alice"]);
  coord.onData(1, 100n, 1n, ["alice"]);
  coord.onData(1, 100n, 1n, ["bob"]);

  // Frontier advances past 100 -> snapshot commits as one consistent cut.
  await coord.onProgress(0, 101n); // min still 100 (stream 1 at 100): no commit
  check("no commit until both advance (frontier still init 100)", await checkpoint(reader, "test"), "100");
  await coord.onProgress(1, 101n); // min -> 101: commit ts < 101

  checkRows("ta after snapshot", await dump(reader, "ta"), [
    { id: "1", val: "alice", mz_diff: "1" },
    { id: "2", val: "bob", mz_diff: "1" },
  ]);
  checkRows("tb after snapshot (alice multiplicity 2)", await dump(reader, "tb"), [
    { name: "alice", mz_diff: "2" },
    { name: "bob", mz_diff: "1" },
  ]);
  check("checkpoint after snapshot", await checkpoint(reader, "test"), "101");

  // ---- incremental updates at ts = 200 ----------------------------------
  console.log("scenario: update + delete + multiset decrement");
  // update id 2 bob->carol (delete old + insert new at same ts)
  coord.onData(0, 200n, -1n, ["2", "bob"]);
  coord.onData(0, 200n, 1n, ["2", "carol"]);
  // remove one alice (count 2 -> 1); add a dave
  coord.onData(1, 200n, -1n, ["alice"]);
  coord.onData(1, 200n, 1n, ["dave"]);

  await coord.onProgress(0, 201n);
  await coord.onProgress(1, 201n);

  checkRows("ta after update (bob row gone, carol present)", await dump(reader, "ta"), [
    { id: "1", val: "alice", mz_diff: "1" },
    { id: "2", val: "carol", mz_diff: "1" },
  ]);
  checkRows("tb after decrement (alice 2->1, dave added)", await dump(reader, "tb"), [
    { name: "alice", mz_diff: "1" },
    { name: "bob", mz_diff: "1" },
    { name: "dave", mz_diff: "1" },
  ]);
  check("checkpoint after updates", await checkpoint(reader, "test"), "201");
  await pg.end();

  // ---- Scenario 2: resume from checkpoint (no backward / no double apply) -
  console.log("scenario: resume from checkpoint F=201, AS OF 200");
  const pg2 = new Pg(PG);
  const cp = await pg2.readCheckpoint("test");
  check("resume reads committed F", cp?.frontier.toString(), "201");

  const targets2 = [
    { target: "ta", columns: ["id", "val"] as string[] | null },
    { target: "tb", columns: ["name"] as string[] | null },
  ];
  const coord2 = new Coordinator(2, targets2, pg2, "test", cp!.frontier);

  // Resume AS OF 200 (= F-1) emits a progress at 200 first. The guard must
  // prevent any commit at/below the already-committed F=201.
  await coord2.onProgress(0, 200n);
  await coord2.onProgress(1, 200n);
  check("no backward commit on resume", await checkpoint(reader, "test"), "201");

  // MZ with SNAPSHOT false AS OF 200 redelivers only ts > 200; ts=200/100 are
  // NOT redelivered (already committed). New change arrives at ts = 300.
  coord2.onData(0, 300n, 1n, ["3", "erin"]);
  await coord2.onProgress(0, 301n);
  await coord2.onProgress(1, 301n);

  checkRows("ta after resume (erin added, no double-apply)", await dump(reader, "ta"), [
    { id: "1", val: "alice", mz_diff: "1" },
    { id: "2", val: "carol", mz_diff: "1" },
    { id: "3", val: "erin", mz_diff: "1" },
  ]);
  checkRows("tb unchanged across resume (alice still 1)", await dump(reader, "tb"), [
    { name: "alice", mz_diff: "1" },
    { name: "bob", mz_diff: "1" },
    { name: "dave", mz_diff: "1" },
  ]);
  check("checkpoint after resume", await checkpoint(reader, "test"), "301");
  await pg2.end();

  // ---- Scenario 3: two writers, same bridgeId -> CAS rejects the second -----
  console.log("scenario: two writers same bridgeId (CAS guard)");
  await reader.query(`DROP TABLE IF EXISTS tc`);
  await reader.query(`DELETE FROM _mz_bridge_progress WHERE bridge_id = 'test_cas'`);
  const pgX = new Pg(PG);
  const pgY = new Pg(PG);
  await pgX.ensureTable("tc", ["k"]);
  await pgX.initCheckpoint("test_cas", 100n, "cas");

  const mkCoord = (pg: Pg) =>
    new Coordinator(1, [{ target: "tc", columns: ["k"] as string[] | null }], pg, "test_cas", 100n);
  const coordX = mkCoord(pgX);
  const coordY = mkCoord(pgY);

  // Both observe the same change at ts=100 and try to commit through 101.
  coordX.onData(0, 100n, 1n, ["x"]);
  coordY.onData(0, 100n, 1n, ["x"]);

  await coordX.onProgress(0, 101n); // wins the CAS 100 -> 101
  checkRows("first writer commits", await dump(reader, "tc"), [{ k: "x", mz_diff: "1" }]);

  let rejected = false;
  try {
    await coordY.onProgress(0, 101n); // expects 100, but it's 101 now
  } catch (e) {
    rejected = e instanceof CheckpointConflict;
  }
  check("second writer rejected with CheckpointConflict", rejected, true);
  checkRows("no corruption: tc still x|1 (Y's diff rolled back)", await dump(reader, "tc"), [
    { k: "x", mz_diff: "1" },
  ]);
  check("checkpoint advanced once", await checkpoint(reader, "test_cas"), "101");
  // The view-set fingerprint must be persisted so a resume can detect a config
  // change (the guard for that lives in index.ts).
  const viewsCol = await reader.query(
    "SELECT views FROM _mz_bridge_progress WHERE bridge_id = 'test_cas'",
  );
  check("checkpoint persists view fingerprint", viewsCol.rows[0]?.views, "cas");
  await pgX.end();
  await pgY.end();

  await reader.end();

  // ---- Scenario 4: config validation guards ------------------------------
  console.log("scenario: config validation");
  const tmp = (name: string, obj: unknown): string => {
    const p = join(tmpdir(), `mzbridge-${name}-${process.pid}.json`);
    writeFileSync(p, JSON.stringify(obj));
    return p;
  };
  const base = {
    mzConn: "x",
    pgConn: "y",
    views: [{ view: "a", target: "ta" }],
  };
  let ok = true;
  try {
    loadConfig(tmp("good", base));
  } catch {
    ok = false;
  }
  check("valid config loads", ok, true);

  const throws = (name: string, obj: unknown): boolean => {
    try {
      loadConfig(tmp(name, obj));
      return false;
    } catch {
      return true;
    }
  };
  check(
    "duplicate target rejected",
    throws("dup", {
      ...base,
      views: [
        { view: "a", target: "t" },
        { view: "b", target: "t" },
      ],
    }),
    true,
  );
  check("fetchBatch < 1 rejected", throws("fb", { ...base, fetchBatch: 0 }), true);

  console.log(failures === 0 ? "\nALL PASSED" : `\n${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error("sim error:", err);
  process.exit(1);
});
