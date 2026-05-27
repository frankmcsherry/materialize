import type { Pg, TargetGroup } from "./pg";
import { rowKey } from "./types";

interface BufEntry {
  ts: bigint;
  diff: bigint;
  values: (string | null)[];
}

interface Target {
  target: string;
  /** Filled in once the stream reports its schema (its first FETCH result). */
  columns: string[] | null;
}

/**
 * Cross-view consistency engine.
 *
 * Each stream buffers its updates in memory and reports a frontier (its latest
 * progress timestamp). The committable cut is F = min(frontier) across all
 * streams: every stream has, by then, delivered *all* of its updates with
 * timestamp < F (guaranteed by Materialize's non-decreasing, per-timestamp
 * consolidated output). When F advances past the last committed point we flush
 * everything < F to Postgres in one transaction, together with F itself.
 *
 * Correctness notes:
 *  - We only commit once every stream is "live" (has emitted >= 1 progress);
 *    otherwise min is undefined / too low.
 *  - We never commit updates *at* F, only strictly below it: a progress at F
 *    closes times < F but says nothing about F.
 *  - Commits are serialized by `committing`; a progress that arrives mid-commit
 *    sets `recheck` so we loop again and catch up.
 */
export class Coordinator {
  private frontiers: (bigint | null)[];
  private buffers: BufEntry[][];
  private committedF: bigint;
  private committing = false;
  private recheck = false;

  constructor(
    private readonly n: number,
    private readonly targets: Target[],
    private readonly pg: Pg,
    private readonly bridgeId: string,
    committedF: bigint,
    private readonly onCommit?: (F: bigint, rowsApplied: number) => void,
  ) {
    this.frontiers = new Array(n).fill(null);
    this.buffers = Array.from({ length: n }, () => []);
    this.committedF = committedF;
  }

  setColumns(i: number, columns: string[]): void {
    this.targets[i].columns = columns;
  }

  onData(i: number, ts: bigint, diff: bigint, values: (string | null)[]): void {
    this.buffers[i].push({ ts, diff, values });
  }

  /**
   * Record a stream's new frontier and commit if the global cut advanced.
   * Awaitable so the calling FETCH loop applies natural backpressure (it pauses
   * while its own progress drives a commit) and so a failed commit surfaces as
   * a rejection that brings the bridge down rather than an unhandled rejection.
   */
  async onProgress(i: number, ts: bigint): Promise<void> {
    this.frontiers[i] = ts;
    await this.maybeCommit();
  }

  private minFrontier(): bigint | null {
    let m: bigint | null = null;
    for (const f of this.frontiers) {
      if (f === null) return null; // not all streams are live yet
      if (m === null || f < m) m = f;
    }
    return m;
  }

  private async maybeCommit(): Promise<void> {
    if (this.committing) {
      this.recheck = true;
      return;
    }
    this.committing = true;
    try {
      do {
        this.recheck = false;
        const F = this.minFrontier();
        if (F === null || F <= this.committedF) break;
        const rowsApplied = await this.commitThrough(F);
        this.committedF = F;
        this.onCommit?.(F, rowsApplied);
      } while (this.recheck);
    } finally {
      this.committing = false;
    }
  }

  /** Commit everything with `ts < F` in one transaction; returns rows applied. */
  private async commitThrough(F: bigint): Promise<number> {
    const groups: TargetGroup[] = [];
    for (let i = 0; i < this.n; i++) {
      const toCommit = this.buffers[i].filter((e) => e.ts < F);
      if (toCommit.length === 0) continue;
      const cols = this.targets[i].columns;
      if (cols === null) {
        // Unreachable: a stream with buffered data has reported a schema.
        throw new Error(`stream ${i} has data but no schema`);
      }
      const byKey = new Map<string, { values: (string | null)[]; diff: bigint }>();
      for (const e of toCommit) {
        const k = rowKey(e.values);
        const g = byKey.get(k);
        if (g) g.diff += e.diff;
        else byKey.set(k, { values: e.values, diff: e.diff });
      }
      const rows = [...byKey.values()].filter((r) => r.diff !== 0n);
      groups.push({ target: this.targets[i].target, columns: cols, rows });
    }

    // Commit [committedF, F): the CAS expects the checkpoint to still hold our
    // last committed frontier. If a second writer (zombie / double-start) moved
    // it, applyWindow throws CheckpointConflict and the whole batch rolls back.
    await this.pg.applyWindow(groups, this.bridgeId, this.committedF, F);

    // Drop everything we just committed. Nothing with ts < F can arrive after
    // this point (every stream's frontier is >= F), so this is safe.
    for (let i = 0; i < this.n; i++) {
      this.buffers[i] = this.buffers[i].filter((e) => e.ts >= F);
    }

    return groups.reduce((sum, g) => sum + g.rows.length, 0);
  }
}
