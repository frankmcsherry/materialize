import { loadConfig } from "./config";
import { Coordinator } from "./coordinator";
import { adminSetup, runSubscription } from "./mz";
import { CheckpointConflict, Pg } from "./pg";

async function main(): Promise<void> {
  const configPath = process.argv[2] ?? "config.json";
  const cfg = loadConfig(configPath);

  const pg = new Pg(cfg.pgConn);
  await pg.ensureProgressTable();

  // RETAIN HISTORY (best-effort) and current logical time for a fresh start.
  const { mzNow } = await adminSetup(cfg);

  // Fingerprint of the view set this bridge replicates. Stored with the
  // checkpoint so a resume can refuse a config whose views changed (which would
  // otherwise resume a never-snapshotted view and silently drop its data).
  const fingerprint = JSON.stringify(
    cfg.views.map((v) => [v.view, v.target]).sort(),
  );

  // Fresh vs. resume is decided entirely by the downstream checkpoint.
  const checkpoint = await pg.readCheckpoint(cfg.bridgeId);
  let asOf: bigint;
  let snapshot: boolean;
  let committedF: bigint;
  if (checkpoint === null) {
    // Fresh: pin every stream to the SAME AS OF so the initial snapshots form
    // a single consistent cut. committedF = T0 so the snapshot (emitted AT T0)
    // is flushed once the frontier first advances past T0. Establish the
    // checkpoint row now so every commit can be a pure CAS update; if another
    // writer created it first, initCheckpoint throws CheckpointConflict.
    asOf = mzNow;
    snapshot = true;
    committedF = mzNow;
    await pg.initCheckpoint(cfg.bridgeId, committedF, fingerprint);
    console.log(`[bridge] fresh start: AS OF ${asOf} (SNAPSHOT true)`);
  } else {
    if (checkpoint.views !== null && checkpoint.views !== fingerprint) {
      throw new Error(
        `bridge '${cfg.bridgeId}' checkpoint was created for a different set of ` +
          `views; pick a new bridgeId, or reset the downstream (drop the mirror ` +
          `tables and the _mz_bridge_progress row) before changing views`,
      );
    }
    // Resume: SNAPSHOT false AS OF F-1 redelivers exactly timestamps >= F.
    asOf = checkpoint.frontier - 1n;
    snapshot = false;
    committedF = checkpoint.frontier;
    console.log(
      `[bridge] resume: committed F=${checkpoint.frontier}, AS OF ${asOf} (SNAPSHOT false)`,
    );
  }

  const targets = cfg.views.map((v) => ({
    target: v.target,
    columns: null as string[] | null,
  }));
  // Log every commit that applied data; for idle frontier advances (which
  // happen ~once a second) emit only an occasional heartbeat so the output
  // shows liveness without flooding.
  let lastHeartbeat = Date.now();
  const onCommit = (F: bigint, rowsApplied: number) => {
    if (rowsApplied > 0) {
      const s = rowsApplied === 1 ? "" : "s";
      console.log(`[bridge] committed F=${F} (${rowsApplied} change${s})`);
    } else if (Date.now() - lastHeartbeat >= 10_000) {
      console.log(`[bridge] idle, committed through F=${F}`);
      lastHeartbeat = Date.now();
    }
  };
  const coord = new Coordinator(
    cfg.views.length,
    targets,
    pg,
    cfg.bridgeId,
    committedF,
    onCommit,
  );

  let stop = false;
  const requestStop = (sig: string) => {
    console.log(`[bridge] ${sig} received, stopping...`);
    stop = true;
  };
  process.on("SIGINT", () => requestStop("SIGINT"));
  process.on("SIGTERM", () => requestStop("SIGTERM"));

  const runs = cfg.views.map((v, i) =>
    runSubscription(
      cfg,
      v.view,
      asOf,
      snapshot,
      {
        onSchema: async (columns) => {
          coord.setColumns(i, columns);
          await pg.ensureTable(v.target, columns);
          console.log(
            `[bridge] stream ${i}: ${v.view} -> ${v.target} (${columns.join(", ")})`,
          );
        },
        onData: (ts, diff, values) => coord.onData(i, ts, diff, values),
        onProgress: (ts) => coord.onProgress(i, ts),
      },
      () => stop,
    ),
  );

  // If any single stream fails (e.g. AS OF below the RETAIN HISTORY window),
  // surface it and bring the whole bridge down — we never commit a partial
  // group, so the last durable checkpoint stays consistent.
  try {
    await Promise.all(runs);
  } finally {
    stop = true;
    await pg.end();
  }
  console.log("[bridge] stopped.");
}

main().catch((err) => {
  if (err instanceof CheckpointConflict) {
    // Another writer owns this bridge's checkpoint. Exit so exactly one
    // survives; no partial/duplicate work was committed (the CAS rolled back).
    console.error(`[bridge] ${err.message}; exiting (another writer is active)`);
    process.exit(2);
  }
  console.error("[bridge] fatal:", err);
  process.exit(1);
});
