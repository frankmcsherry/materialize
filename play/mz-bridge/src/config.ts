import { readFileSync } from "node:fs";

/** One replicated view -> downstream table mapping. */
export interface ViewConfig {
  /**
   * What to SUBSCRIBE to. Normally a materialized view *object name* (e.g.
   * "mv_accounts"), which is the cheapest thing to resume against because
   * Materialize reads it straight from storage. May also be a parenthesized
   * query like "(SELECT ...)", but that builds a fresh dataflow on resume.
   * Inserted verbatim after `SUBSCRIBE`, so quote it yourself if it needs it.
   */
  view: string;
  /** Downstream Postgres table name to mirror into. */
  target: string;
}

export interface BridgeConfig {
  /** Materialize connection string (pgwire, default port 6875). */
  mzConn: string;
  /** Downstream Postgres connection string. */
  pgConn: string;
  /**
   * Identifies this bridge's checkpoint row in `_mz_bridge_progress`. All views
   * in one config share a single checkpoint and are committed together as one
   * consistency group. Run the process again with a different bridgeId for an
   * independent group.
   */
  bridgeId: string;
  /**
   * If set, the bridge issues `ALTER MATERIALIZED VIEW <view> SET (RETAIN
   * HISTORY FOR '<retainHistory>')` for each view at startup so the resume
   * point stays readable after a disconnect. Set to null to manage retention
   * yourself (e.g. in the CREATE statement). Note: this is a *duration relative
   * to now*, not an absolute floor.
   */
  retainHistory: string | null;
  /** Rows to pull per FETCH. Larger = fewer round-trips, more memory. */
  fetchBatch: number;
  views: ViewConfig[];
}

export function loadConfig(path: string): BridgeConfig {
  const raw = JSON.parse(readFileSync(path, "utf8"));
  if (typeof raw.mzConn !== "string" || typeof raw.pgConn !== "string") {
    throw new Error("config must set mzConn and pgConn");
  }
  if (!Array.isArray(raw.views) || raw.views.length === 0) {
    throw new Error("config must list at least one view");
  }
  const cfg: BridgeConfig = {
    bridgeId: "default",
    retainHistory: "1h",
    fetchBatch: 1000,
    ...raw,
  };
  if (!(cfg.fetchBatch >= 1)) {
    throw new Error(`fetchBatch must be >= 1 (got ${cfg.fetchBatch})`);
  }
  // Each view needs its own downstream table: the engine commits per-stream
  // groups, so two views sharing a `target` would fight over the same rows.
  const targets = new Set<string>();
  for (const v of cfg.views) {
    if (!v.view || !v.target) {
      throw new Error("each view needs a non-empty `view` and `target`");
    }
    if (targets.has(v.target)) {
      throw new Error(`duplicate target table '${v.target}'; each view needs its own`);
    }
    targets.add(v.target);
  }
  return cfg;
}
