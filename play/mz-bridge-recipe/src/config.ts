import { readFileSync } from "node:fs";

/** Minimal config for the demo: where Materialize is, and what to subscribe to. */
export interface RecipeConfig {
  /** Materialize connection string (pgwire, default port 6875). */
  mzConn: string;
  /**
   * What each subscription reads. Normally a materialized view *object name*
   * (cheapest to resume against). May also be a parenthesized query, which
   * builds a fresh dataflow on resume. Inserted verbatim after `SUBSCRIBE`.
   */
  views: string[];
  /** Rows to pull per FETCH. */
  fetchBatch?: number;
}

export function loadConfig(path: string): RecipeConfig {
  const raw = JSON.parse(readFileSync(path, "utf8"));
  if (typeof raw.mzConn !== "string") {
    throw new Error("config must set mzConn");
  }
  if (!Array.isArray(raw.views) || raw.views.length === 0) {
    throw new Error("config must list at least one view");
  }
  return { mzConn: raw.mzConn, views: raw.views, fetchBatch: raw.fetchBatch };
}
