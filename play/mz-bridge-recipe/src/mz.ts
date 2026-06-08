import { Client } from "pg";

/**
 * Return every column value as the raw text it arrived as on the wire, instead
 * of letting node-postgres parse it into JS types (Date, number, parsed jsonb,
 * ...). This is a faithful TEXT passthrough, and keeps mz_timestamp / mz_diff as
 * strings we turn into BigInt. SQL NULL is delivered as JS `null` before the
 * parser runs, so NULL stays distinct from the empty string.
 */
const IDENTITY_TYPES = {
  getTypeParser: () => (v: string) => v,
};

/**
 * Where a subscription's events go. The cohort owns one of these and dispatches
 * by `view`. It is a field on the Subscription (not baked in at construction) so
 * that `merge`/`split` can re-point a running stream at a different cohort.
 */
export interface SubscriptionSink {
  /** Called once with the data column names (metadata columns stripped). */
  onSchema(view: string, columns: string[]): void;
  onData(view: string, ts: bigint, diff: bigint, values: (string | null)[]): void;
  onProgress(view: string, ts: bigint): void;
}

/**
 * The cohort only ever holds a stream through this handle. `Subscription`
 * implements it; tests can supply a fake (see `SubscriptionFactory`).
 */
export interface SubscriptionHandle {
  readonly view: string;
  /** Resolves when the stream ends cleanly; rejects on a fatal error. */
  readonly done: Promise<void>;
  /** Re-pointable by the cohort on merge/split. */
  sink: SubscriptionSink;
  stop(): void;
}

/** How a cohort creates streams. Defaults to a real `Subscription`. */
export type SubscriptionFactory = (
  conn: string,
  view: string,
  asOf: bigint,
  snapshot: boolean,
  sink: SubscriptionSink,
  fetchBatch: number,
) => SubscriptionHandle;

/** What a `FETCH ...` returns in array row-mode. */
interface FetchResult {
  fields: { name: string }[];
  rows: (string | null)[][];
}

function expectMetadataColumns(names: string[], view: string): void {
  if (
    names[0] !== "mz_timestamp" ||
    names[1] !== "mz_progressed" ||
    names[2] !== "mz_diff"
  ) {
    throw new Error(
      `subscribe to '${view}' returned columns [${names.slice(0, 3).join(", ")}]; ` +
        `expected mz_timestamp, mz_progressed, mz_diff. This bridge requires ` +
        `WITH (PROGRESS) and the default envelope.`,
    );
  }
}

/**
 * A single managed SUBSCRIBE. Created knowing *where to start* — `(asOf,
 * snapshot)` — and from then on it only emits. Its FETCH loop drains
 * continuously and never pauses for the consumer: not consuming a SUBSCRIBE
 * makes Materialize hold state, so the bridge, not MZ, must be the thing that
 * falls over under a slow consumer.
 *
 * NB: `view` is operator-provided config (a view name or a parenthesized query),
 * trusted and interpolated directly so both forms are allowed.
 */
export class Subscription implements SubscriptionHandle {
  private client: Client;
  private stopped = false;
  /** Mutable so a cohort can re-point this stream on merge/split. */
  sink: SubscriptionSink;
  readonly done: Promise<void>;

  constructor(
    conn: string,
    readonly view: string,
    private readonly asOf: bigint,
    private readonly snapshot: boolean,
    sink: SubscriptionSink,
    private readonly fetchBatch = 1000,
  ) {
    this.sink = sink;
    this.client = new Client({ connectionString: conn, types: IDENTITY_TYPES });
    // run() yields at its first `await` (connect), so the constructor returns —
    // and the cohort records the handle — before any event is dispatched.
    this.done = this.run();
  }

  stop(): void {
    this.stopped = true;
  }

  private async run(): Promise<void> {
    await this.client.connect();
    try {
      await this.client.query("BEGIN");
      await this.client.query(
        `DECLARE c CURSOR FOR SUBSCRIBE ${this.view} ` +
          `WITH (PROGRESS, SNAPSHOT ${this.snapshot ? "true" : "false"}) ` +
          `AS OF ${this.asOf.toString()}`,
      );

      let schemaSent = false;
      // FETCH with a 1s timeout so idle views still surface their periodic
      // progress messages and the loop stays responsive to stop().
      while (!this.stopped) {
        const res = (await this.client.query({
          text: `FETCH ${this.fetchBatch} c WITH (timeout='1s')`,
          rowMode: "array",
        })) as unknown as FetchResult;

        if (!schemaSent) {
          const names = res.fields.map((f) => f.name);
          expectMetadataColumns(names, this.view);
          this.sink.onSchema(this.view, names.slice(3));
          schemaSent = true;
        }

        // Columns are [mz_timestamp, mz_progressed, mz_diff, ...data].
        for (const row of res.rows) {
          const ts = BigInt(row[0] as string);
          const progressed = row[1] === "t" || row[1] === "true";
          if (progressed) {
            this.sink.onProgress(this.view, ts);
          } else {
            this.sink.onData(this.view, ts, BigInt(row[2] as string), row.slice(3));
          }
        }
      }
    } finally {
      await this.client.end().catch(() => {});
    }
  }
}

/** The default factory: a real Subscription against a live Materialize. */
export const realSubscriptionFactory: SubscriptionFactory = (
  conn,
  view,
  asOf,
  snapshot,
  sink,
  fetchBatch,
) => new Subscription(conn, view, asOf, snapshot, sink, fetchBatch);

/** Read the current logical time, for a fresh cohort's shared `AS OF`. */
export async function readMzNow(conn: string): Promise<bigint> {
  const c = new Client({ connectionString: conn });
  await c.connect();
  try {
    const r = await c.query("SELECT mz_now()::text AS now");
    return BigInt(r.rows[0].now);
  } finally {
    await c.end();
  }
}
