import { createHash } from "node:crypto";

// Shared types and helpers.
//
// Timestamps (`mz_timestamp`) and diffs (`mz_diff`) are 64-bit values that
// arrive from Materialize as decimal *strings* over pgwire. We always carry
// them as `bigint`, never as `number` (a JS `number` loses precision above
// 2^53, and epoch-millis timestamps will eventually exceed that).

/** A single data change from a SUBSCRIBE stream, default ("Diffs") envelope. */
export interface Update {
  /** The logical timestamp at which the change occurs. */
  ts: bigint;
  /** The z-set multiplicity delta (any non-zero i64). */
  diff: bigint;
  /** The data column values, as raw text (or null for SQL NULL). */
  values: (string | null)[];
}

/** One subscription's net (consolidated) changes within a consistent window. */
export interface ViewChanges {
  /** The view this subscription is reading. */
  view: string;
  /** Data column names, in order (no metadata columns). */
  columns: string[];
  /** One entry per distinct row content, with the net diff over the window. */
  rows: { values: (string | null)[]; diff: bigint }[];
}

/**
 * The batch handed to the upcall at one consistent moment: the consolidated
 * changes for every member that changed. Empty for an idle frontier advance.
 */
export type CohortBatch = ViewChanges[];

/**
 * A content key identifying a row's *contents* for consolidation. We JSON-encode
 * the values (which distinguishes SQL NULL from the string "null" and from "")
 * and hash to a fixed-size hex digest. Both consolidation here and any
 * downstream `ON CONFLICT` target can call this and agree by construction;
 * collisions are cryptographically negligible.
 */
export function rowKey(values: (string | null)[]): string {
  return createHash("sha256").update(JSON.stringify(values)).digest("hex");
}

/**
 * The one durability rule. "I have durably committed everything through frontier
 * `F`" ⇒ on restart, resume every SUBSCRIBE `AS OF resumeAsOf(F)` with
 * `SNAPSHOT = FALSE`, which redelivers exactly `ts >= F`.
 *
 * The `-1` is the whole lesson: `SNAPSHOT = FALSE` emits times *strictly
 * greater* than `AS OF`, so to re-hear `ts = F` you must start at `F-1` — the
 * last instant you already have — not at `F`, the next one you want (which would
 * silently skip the boundary). The `max(·, 0)` clamps the very first instant,
 * since timestamps are non-negative.
 */
export function resumeAsOf(F: bigint): bigint {
  return F > 0n ? F - 1n : 0n;
}
