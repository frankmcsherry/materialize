import {
  type SubscriptionHandle,
  type SubscriptionSink,
  type SubscriptionFactory,
  realSubscriptionFactory,
  readMzNow,
} from "./mz";
import {
  type Update,
  type CohortBatch,
  rowKey,
  resumeAsOf,
} from "./types";

/**
 * The single integration point. Fired once per consistent moment with the
 * frontier `F` it closes and the cross-subscribe, consolidated batch of updates
 * with `ts < F` (empty for an idle advance).
 *
 * Returning (or resolving) means only "delivered — send the next when ready". It
 * is NOT a durability acknowledgement: making the batch durable, and recording
 * `F` so you can resume from it, is your job (see DESIGN.md, Idiom 3). The cohort
 * keeps no durable state.
 */
export type Upcall = (F: bigint, batch: CohortBatch) => void | Promise<void>;

export interface CohortOptions {
  /** Rows per FETCH. Larger = fewer round trips, more memory. */
  fetchBatch?: number;
  /** Inject a fake stream source for testing; defaults to a real Subscription. */
  factory?: SubscriptionFactory;
}

interface Member {
  sub: SubscriptionHandle;
  /** Latest progress timestamp, or null until the stream is live. */
  frontier: bigint | null;
  /** Data column names, filled in from the stream's first batch. */
  columns: string[] | null;
  /** Updates not yet released in a consistent batch. */
  buffer: Update[];
}

/**
 * A cohort of SUBSCRIBEs observed as one consistent unit.
 *
 * Each member runs its own FETCH loop that drains continuously and never pauses
 * for the consumer. Those loops feed per-member buffers and frontiers. A
 * separate, serialized commit pump watches `F = min(frontier)`: each time it
 * advances, the pump releases one consistent batch (everything with `ts < F`,
 * z-set consolidated) to the upcall, waits for it (to keep commit notifications
 * ordered and non-overlapping — NOT to throttle the drain), then drops the
 * released updates and continues.
 *
 * The cohort keeps NO durable state. Durability is the consumer's; its only
 * footprint here is the `resuming` constructor (see Idiom 3 in DESIGN.md).
 */
export class Cohort {
  private members = new Map<string, Member>();
  private committedF: bigint;
  private pumping = false;
  private failReject!: (e: unknown) => void;

  /**
   * Rejects on the first stream or upcall error (e.g. an `AS OF` below the
   * RETAIN HISTORY window), and never resolves otherwise — await it to fail
   * loudly. There is deliberately no graceful `stop()`: this is a recipe, so the
   * consumer exits the process (on Ctrl-C, or on this rejection) and resumes
   * from its durable frontier. `drop()`/`split()` stop individual streams.
   */
  readonly failed: Promise<void>;

  // One sink shared by every member, dispatching by view name into `members`.
  // Sharing a single sink (rather than a closure per stream) is what lets
  // merge/split move a Subscription between cohorts: re-point `sub.sink` and
  // re-key its Member, and its events land in the new cohort.
  private readonly sink: SubscriptionSink = {
    onSchema: (view, columns) => {
      const m = this.members.get(view);
      if (m) m.columns = columns;
    },
    onData: (view, ts, diff, values) => {
      const m = this.members.get(view);
      if (m) m.buffer.push({ ts, diff, values });
    },
    onProgress: (view, ts) => {
      const m = this.members.get(view);
      if (!m) return;
      m.frontier = ts;
      void this.pump();
    },
  };

  private constructor(
    private readonly conn: string,
    private readonly upcall: Upcall,
    committedF: bigint,
    private readonly fetchBatch: number,
    private readonly factory: SubscriptionFactory,
  ) {
    this.committedF = committedF;
    this.failed = new Promise<void>((_res, rej) => {
      this.failReject = rej;
    });
  }

  /**
   * Start a fresh cohort: snapshot every view at one shared logical time and
   * stream forward. Every member subscribes `AS OF mz_now() WITH (SNAPSHOT =
   * TRUE)`, so their initial snapshots form a single consistent cut.
   */
  static async fresh(
    conn: string,
    views: string[],
    upcall: Upcall,
    opts: CohortOptions = {},
  ): Promise<Cohort> {
    const asOf = await readMzNow(conn);
    // committedF = asOf (= T0): the snapshot is emitted *at* T0, so it is
    // released as soon as the frontier first advances past T0.
    const c = new Cohort(
      conn,
      upcall,
      asOf,
      opts.fetchBatch ?? 1000,
      opts.factory ?? realSubscriptionFactory,
    );
    for (const view of views) c.startMember(view, asOf, true);
    return c;
  }

  /**
   * Resume a cohort from a frontier the consumer durably recorded. Every member
   * subscribes `AS OF resumeAsOf(from) WITH (SNAPSHOT = FALSE)`, which redelivers
   * exactly the updates with `ts >= from` — the suffix not yet made durable. The
   * `-1` inside resumeAsOf is the whole point: restart from the last instant you
   * HAVE, not the next one you want (see DESIGN.md, Idiom 3).
   */
  static async resuming(
    conn: string,
    views: string[],
    from: bigint,
    upcall: Upcall,
    opts: CohortOptions = {},
  ): Promise<Cohort> {
    const asOf = resumeAsOf(from);
    // committedF = from: everything < from is already durable, so the first
    // released batch covers [from, nextF).
    const c = new Cohort(
      conn,
      upcall,
      from,
      opts.fetchBatch ?? 1000,
      opts.factory ?? realSubscriptionFactory,
    );
    for (const view of views) c.startMember(view, asOf, false);
    return c;
  }

  /** The views currently in this cohort. */
  views(): string[] {
    return [...this.members.keys()];
  }

  /** The handle for a member, e.g. to pass to drop()/split(). */
  subscription(view: string): SubscriptionHandle {
    const m = this.members.get(view);
    if (!m) throw new Error(`'${view}' is not a member of this cohort`);
    return m.sub;
  }

  /**
   * Add a brand-new view to an already-live cohort (the "join" case). It
   * subscribes `AS OF F_C WITH (SNAPSHOT = TRUE)`, where F_C is the cohort's
   * current consistency frontier: its first progress then lands exactly at F_C,
   * so it neither regresses the cohort nor leaves a gap. For the initial set of
   * views, pass them to fresh()/resuming() instead — that avoids racing a member
   * onto a cohort that has already begun emitting.
   */
  add(view: string): SubscriptionHandle {
    const fC = this.minFrontier();
    if (fC === null) {
      throw new Error(
        "add(): cohort is not live yet; include this view in fresh()/resuming()",
      );
    }
    if (this.members.has(view)) throw new Error(`'${view}' is already a member`);
    return this.startMember(view, fC, true);
  }

  /** Stop and remove a member. Its undelivered buffer is discarded. */
  drop(sub: SubscriptionHandle): void {
    const m = this.members.get(sub.view);
    if (!m) return;
    m.sub.stop();
    this.members.delete(sub.view);
    // The laggard may be gone; min(frontier) can jump, releasing buffered data.
    void this.pump();
  }

  /**
   * Absorb another cohort's members into this one; from here on the union is one
   * consistency unit governed by THIS cohort's upcall (the other is retired, its
   * subscriptions kept running and re-pointed here). Membership-only: no
   * SUBSCRIBE is re-issued.
   *
   * The merged commit point is max(committedF) of the two, so the governing
   * upcall sees monotonic F and no buffered data is lost (the lagging half's
   * backlog flushes once its frontier passes the merge point). Each half's
   * *pre-merge* history stayed with its old upcall; for a clean handoff, point
   * both upcalls at the same store, or make your apply idempotent. See DESIGN.md.
   */
  merge(other: Cohort): void {
    if (other === this) return;
    for (const [view, m] of other.members) {
      this.members.set(view, m);
      m.sub.sink = this.sink;
    }
    other.members.clear();
    if (other.committedF > this.committedF) this.committedF = other.committedF;
    void this.pump();
  }

  /**
   * Move some members into a new cohort with its own upcall; this cohort keeps
   * the rest and its own upcall. Membership-only: no SUBSCRIBE is re-issued, so
   * the split is instant. The child inherits this cohort's commit point (they
   * were committing together), so its first batch continues cleanly from there.
   * Use this to decouple a slow member so the rest can advance and be made
   * durable without waiting on it.
   */
  split(subs: SubscriptionHandle[], upcall: Upcall): Cohort {
    const child = new Cohort(
      this.conn,
      upcall,
      this.committedF,
      this.fetchBatch,
      this.factory,
    );
    for (const sub of subs) {
      const m = this.members.get(sub.view);
      if (!m) throw new Error(`split(): '${sub.view}' is not a member`);
      this.members.delete(sub.view);
      child.members.set(sub.view, m);
      m.sub.sink = child.sink;
    }
    void this.pump(); // remaining members' min may advance (laggard moved out)
    void child.pump();
    return child;
  }

  private startMember(
    view: string,
    asOf: bigint,
    snapshot: boolean,
  ): SubscriptionHandle {
    // Register the Member before creating the stream so the shared sink can find
    // it; the stream cannot dispatch synchronously (run() awaits connect first).
    const member: Member = {
      sub: undefined as unknown as SubscriptionHandle,
      frontier: null,
      columns: null,
      buffer: [],
    };
    this.members.set(view, member);
    const sub = this.factory(
      this.conn,
      view,
      asOf,
      snapshot,
      this.sink,
      this.fetchBatch,
    );
    member.sub = sub;
    // A stream that dies (e.g. AS OF below the RETAIN HISTORY window) rejects
    // `failed`, so the consumer fails loudly rather than acting on a partial
    // consistent moment.
    sub.done.catch((e) => this.failReject(e));
    return sub;
  }

  private minFrontier(): bigint | null {
    if (this.members.size === 0) return null;
    let m: bigint | null = null;
    for (const mem of this.members.values()) {
      if (mem.frontier === null) return null; // not all members are live yet
      if (m === null || mem.frontier < m) m = mem.frontier;
    }
    return m;
  }

  /**
   * Serialized commit pump. Releases consistent batches as F = min(frontier)
   * advances. Runs independently of the FETCH loops: a slow upcall grows buffers
   * (the bridge's memory); it never pauses the drain from Materialize. The
   * `pumping` guard serializes commits; because the loop re-reads minFrontier()
   * after each await, a progress that arrives mid-commit is caught on the next
   * iteration.
   */
  private async pump(): Promise<void> {
    if (this.pumping) return;
    this.pumping = true;
    try {
      while (true) {
        const F = this.minFrontier();
        if (F === null || F <= this.committedF) break;
        const batch = this.slice(F);
        await this.upcall(F, batch);
        this.committedF = F;
        this.dropThrough(F);
      }
    } catch (e) {
      this.failReject(e); // the upcall threw — surface it loudly
    } finally {
      this.pumping = false;
    }
  }

  /** Consolidate each member's buffered updates with `ts < F` into net rows. */
  private slice(F: bigint): CohortBatch {
    const batch: CohortBatch = [];
    for (const [view, m] of this.members) {
      const pending = m.buffer.filter((e) => e.ts < F);
      if (pending.length === 0) continue;
      if (m.columns === null) {
        throw new Error(`stream '${view}' produced data before a schema`);
      }
      const byKey = new Map<
        string,
        { values: (string | null)[]; diff: bigint }
      >();
      for (const e of pending) {
        const k = rowKey(e.values);
        const g = byKey.get(k);
        if (g) g.diff += e.diff;
        else byKey.set(k, { values: e.values, diff: e.diff });
      }
      const rows = [...byKey.values()].filter((r) => r.diff !== 0n);
      if (rows.length > 0) batch.push({ view, columns: m.columns, rows });
    }
    return batch;
  }

  /** Drop everything released through F. Safe: recovery is re-SUBSCRIBE. */
  private dropThrough(F: bigint): void {
    for (const m of this.members.values()) {
      m.buffer = m.buffer.filter((e) => e.ts >= F);
    }
  }
}
