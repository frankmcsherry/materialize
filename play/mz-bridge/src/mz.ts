import { Client } from "pg";
import type { BridgeConfig } from "./config";

/**
 * Return every column value as the raw text it arrived as on the wire, instead
 * of letting node-postgres parse it into JS types (Date, number, parsed jsonb,
 * ...). This gives us a faithful TEXT passthrough into the downstream TEXT
 * columns, and keeps mz_timestamp / mz_diff as strings we turn into BigInt.
 * SQL NULL is delivered as JS `null` before the parser runs, so NULL is
 * preserved distinctly from the empty string.
 */
const IDENTITY_TYPES = { getTypeParser: () => (v: string) => v } as any;

export interface SubscribeHandlers {
  /** Called once with the data column names (metadata columns stripped). */
  onSchema(columns: string[]): void | Promise<void>;
  onData(ts: bigint, diff: bigint, values: (string | null)[]): void;
  onProgress(ts: bigint): void | Promise<void>;
}

/**
 * What a `FETCH ... ` returns in array row-mode: the per-column descriptors
 * (we only need the names) and rows as positional arrays of raw text / null.
 */
interface FetchResult {
  fields: { name: string }[];
  rows: (string | null)[][];
}

// NB: `view` is operator-provided config (a view name or a parenthesized
// query), trusted and interpolated directly so both forms are allowed. The
// downstream table name is identifier-quoted in pg.ts.

/**
 * One-time admin work on a dedicated connection: set RETAIN HISTORY on each
 * view (best-effort) and read the current logical time for a fresh start.
 */
export async function adminSetup(
  cfg: BridgeConfig,
): Promise<{ mzNow: bigint }> {
  const c = new Client({ connectionString: cfg.mzConn });
  await c.connect();
  try {
    if (cfg.retainHistory) {
      for (const v of cfg.views) {
        try {
          await c.query(
            `ALTER MATERIALIZED VIEW ${v.view} SET (RETAIN HISTORY FOR '${cfg.retainHistory}')`,
          );
        } catch (e) {
          console.warn(
            `[bridge] could not set RETAIN HISTORY on ${v.view} ` +
              `(set it in the CREATE statement instead): ${(e as Error).message}`,
          );
        }
      }
    }
    const r = await c.query("SELECT mz_now()::text AS now");
    return { mzNow: BigInt(r.rows[0].now) };
  } finally {
    await c.end();
  }
}

function expectMetadataColumns(names: string[]): void {
  if (
    names[0] !== "mz_timestamp" ||
    names[1] !== "mz_progressed" ||
    names[2] !== "mz_diff"
  ) {
    throw new Error(
      `unexpected SUBSCRIBE columns [${names.slice(0, 3).join(", ")}]; ` +
        `expected mz_timestamp, mz_progressed, mz_diff. ` +
        `This bridge requires WITH (PROGRESS) and the default envelope.`,
    );
  }
}

/**
 * Open one dedicated connection, declare a SUBSCRIBE cursor, and pump FETCHes
 * into the handlers until `shouldStop()` returns true or an error occurs.
 *
 * The output is parsed by position: [mz_timestamp, mz_progressed, mz_diff,
 * ...data]. We FETCH with a 1s timeout so idle views still surface their
 * periodic progress messages and the loop stays responsive.
 */
export async function runSubscription(
  cfg: BridgeConfig,
  view: string,
  asOf: bigint,
  snapshot: boolean,
  handlers: SubscribeHandlers,
  shouldStop: () => boolean,
): Promise<void> {
  const c = new Client({ connectionString: cfg.mzConn, types: IDENTITY_TYPES });
  await c.connect();
  try {
    await c.query("BEGIN");
    await c.query(
      `DECLARE c CURSOR FOR SUBSCRIBE ${view} ` +
        `WITH (PROGRESS, SNAPSHOT ${snapshot ? "true" : "false"}) ` +
        `AS OF ${asOf.toString()}`,
    );

    let schemaSent = false;
    while (!shouldStop()) {
      const res = (await c.query({
        text: `FETCH ${cfg.fetchBatch} c WITH (timeout='1s')`,
        rowMode: "array",
      })) as unknown as FetchResult;

      if (!schemaSent) {
        const names = res.fields.map((f) => f.name);
        expectMetadataColumns(names);
        await handlers.onSchema(names.slice(3));
        schemaSent = true;
      }

      // Columns are [mz_timestamp, mz_progressed, mz_diff, ...data].
      for (const row of res.rows) {
        const ts = BigInt(row[0] as string);
        const progressed = row[1] === "t" || row[1] === "true";
        if (progressed) {
          await handlers.onProgress(ts);
        } else {
          handlers.onData(ts, BigInt(row[2] as string), row.slice(3));
        }
      }
    }
  } finally {
    await c.end().catch(() => {});
  }
}
