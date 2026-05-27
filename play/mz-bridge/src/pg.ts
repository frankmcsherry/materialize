import { Pool, type PoolClient } from "pg";
import { rowKey } from "./types";

/** A per-target batch of net changes to apply within one commit window. */
export interface TargetGroup {
  target: string;
  /** Data column names (in order), without the metadata columns. */
  columns: string[];
  /** One entry per distinct row content, with the *net* diff over the window. */
  rows: { values: (string | null)[]; diff: bigint }[];
}

/** Quote a SQL identifier for Postgres. */
function q(ident: string): string {
  return '"' + ident.replace(/"/g, '""') + '"';
}

/**
 * Raised when the commit's compare-and-swap on the checkpoint frontier fails,
 * i.e. another writer advanced (or initialized) this bridge's checkpoint. The
 * whole commit transaction is rolled back, so nothing is double-applied. The
 * caller should treat this as "I am not the sole writer" and exit.
 */
export class CheckpointConflict extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CheckpointConflict";
  }
}

/**
 * Downstream Postgres sink. Mirrors each view as a *count* table: one row per
 * distinct content (all columns TEXT), plus an integer `mz_diff` multiplicity
 * and a `__key` content key. Applying a change is pure arithmetic on the count,
 * so there is no "delete N physical duplicates" problem.
 */
export class Pg {
  private pool: Pool;

  constructor(connectionString: string) {
    this.pool = new Pool({ connectionString });
  }

  async ensureProgressTable(): Promise<void> {
    await this.pool.query(
      `CREATE TABLE IF NOT EXISTS _mz_bridge_progress (
         bridge_id  text PRIMARY KEY,
         frontier   numeric(20,0) NOT NULL,
         views      text,
         updated_at timestamptz NOT NULL DEFAULT now()
       )`,
    );
    // Forward-compat: add `views` to a checkpoint table from an older version.
    await this.pool.query(
      "ALTER TABLE _mz_bridge_progress ADD COLUMN IF NOT EXISTS views text",
    );
  }

  /**
   * Read this bridge's checkpoint: the committed frontier F and the view-set
   * fingerprint it was created with (so a resume can refuse a changed config).
   * Returns null if the bridge has never committed.
   */
  async readCheckpoint(
    bridgeId: string,
  ): Promise<{ frontier: bigint; views: string } | null> {
    const r = await this.pool.query(
      "SELECT frontier::text AS f, views FROM _mz_bridge_progress WHERE bridge_id = $1",
      [bridgeId],
    );
    if (r.rows.length === 0) return null;
    return { frontier: BigInt(r.rows[0].f), views: r.rows[0].views };
  }

  /**
   * Establish the checkpoint row for a fresh bridge (recording its view-set
   * fingerprint), so that every subsequent commit is a pure UPDATE
   * compare-and-swap. If the row already exists, another writer initialized this
   * bridge first: raise CheckpointConflict so we exit rather than run a second
   * writer against the same checkpoint.
   */
  async initCheckpoint(
    bridgeId: string,
    frontier: bigint,
    views: string,
  ): Promise<void> {
    try {
      await this.pool.query(
        "INSERT INTO _mz_bridge_progress (bridge_id, frontier, views) VALUES ($1, $2, $3)",
        [bridgeId, frontier.toString(), views],
      );
    } catch (e) {
      if ((e as { code?: string }).code === "23505") {
        throw new CheckpointConflict(
          `checkpoint for bridge '${bridgeId}' already exists — another writer initialized it`,
        );
      }
      throw e;
    }
  }

  /** Create the mirror table for a view if it does not already exist. */
  async ensureTable(target: string, columns: string[]): Promise<void> {
    const cols = columns.map((c) => `${q(c)} text`).join(", ");
    await this.pool.query(
      `CREATE TABLE IF NOT EXISTS ${q(target)} (${cols}, "mz_diff" bigint NOT NULL, "__key" text NOT NULL)`,
    );
    await this.pool.query(
      `CREATE UNIQUE INDEX IF NOT EXISTS ${q(target + "__key_uidx")} ON ${q(target)} ("__key")`,
    );
  }

  /**
   * Apply one consistent commit window in a single transaction: net changes per
   * target, then advance the checkpoint from `expectedPrev` to `F` with a
   * compare-and-swap. The checkpoint is written even when there is no data (an
   * idle frontier advance), so resume stays tight.
   *
   * The CAS is the "safe to commit" guard, and it is intentionally plain
   * standard SQL — `UPDATE ... WHERE id = ? AND frontier = ?` plus a row-count
   * check — so the same architecture ports to any transactional store. If the
   * stored frontier is no longer `expectedPrev` (another writer advanced it),
   * the row count is 0; we raise CheckpointConflict and the whole transaction
   * (including the data changes above) rolls back, so nothing is double-applied.
   */
  async applyWindow(
    groups: TargetGroup[],
    bridgeId: string,
    expectedPrev: bigint,
    F: bigint,
  ): Promise<void> {
    const client: PoolClient = await this.pool.connect();
    try {
      await client.query("BEGIN");
      for (const g of groups) {
        const n = g.columns.length;
        const colList = [...g.columns.map(q), '"mz_diff"', '"__key"'].join(", ");
        const placeholders = Array.from(
          { length: n + 2 },
          (_, i) => `$${i + 1}`,
        ).join(", ");
        const insertSql =
          `INSERT INTO ${q(g.target)} (${colList}) VALUES (${placeholders}) ` +
          `ON CONFLICT ("__key") DO UPDATE SET "mz_diff" = ${q(g.target)}."mz_diff" + EXCLUDED."mz_diff"`;
        for (const row of g.rows) {
          // bigint -> string so node-postgres binds it as numeric/bigint text.
          const params = [...row.values, row.diff.toString(), rowKey(row.values)];
          await client.query(insertSql, params);
        }
        await client.query(`DELETE FROM ${q(g.target)} WHERE "mz_diff" = 0`);
      }
      const cas = await client.query(
        `UPDATE _mz_bridge_progress SET frontier = $2, updated_at = now()
         WHERE bridge_id = $1 AND frontier = $3`,
        [bridgeId, F.toString(), expectedPrev.toString()],
      );
      if (cas.rowCount !== 1) {
        throw new CheckpointConflict(
          `checkpoint CAS failed for bridge '${bridgeId}': expected frontier ` +
            `${expectedPrev} but it was changed by another writer`,
        );
      }
      await client.query("COMMIT");
    } catch (e) {
      await client.query("ROLLBACK").catch(() => {});
      throw e;
    } finally {
      client.release();
    }
  }

  async end(): Promise<void> {
    await this.pool.end();
  }
}
