import { createHash } from "node:crypto";

// Shared types and helpers for the bridge.
//
// Timestamps (`mz_timestamp`) and diffs (`mz_diff`) are 64-bit values that
// arrive from Materialize as decimal *strings* over pgwire. We always carry
// them as `bigint`, never as `number` (a JS `number` loses precision above
// 2^53, and epoch-millis timestamps will eventually exceed that).

/** A single data change from a SUBSCRIBE stream, default ("Diffs") envelope. */
export interface Update {
  /** The logical timestamp at which the change occurs. */
  ts: bigint;
  /** The z-set multiplicity delta (can be any non-zero i64). */
  diff: bigint;
  /** The data column values, as raw text (or null for SQL NULL). */
  values: (string | null)[];
}

/**
 * A content key that uniquely identifies a row's *contents* for the count
 * mirror. We JSON-encode the values (which distinguishes SQL NULL from the
 * string "null" and from "") and then hash to a fixed-size hex digest. The
 * fixed size matters: the sink puts a unique index on this key, and indexing
 * the raw JSON of a wide row would blow past Postgres's btree entry-size limit.
 * Both the coordinator (aggregation) and the sink (the `__key` / ON CONFLICT
 * target) call this, so they agree by construction. Collisions are
 * cryptographically negligible.
 */
export function rowKey(values: (string | null)[]): string {
  return createHash("sha256").update(JSON.stringify(values)).digest("hex");
}
