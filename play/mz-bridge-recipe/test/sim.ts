import assert from "node:assert";
import { Cohort } from "../src/cohort";
import { resumeAsOf, type CohortBatch } from "../src/types";
import type {
  SubscriptionHandle,
  SubscriptionSink,
  SubscriptionFactory,
} from "../src/mz";

// A fake source we drive by hand, standing in for a live SUBSCRIBE. This lets us
// assert the idioms (consistent moments, consolidation, resume arithmetic,
// split, merge) with no Materialize running — and doubles as executable docs.
class FakeSub implements SubscriptionHandle {
  readonly done = Promise.resolve();
  stopped = false;
  constructor(
    readonly view: string,
    public sink: SubscriptionSink,
    readonly asOf: bigint,
    readonly snapshot: boolean,
  ) {}
  stop(): void {
    this.stopped = true;
  }
  schema(cols: string[]): void {
    this.sink.onSchema(this.view, cols);
  }
  data(ts: bigint, diff: bigint, values: (string | null)[]): void {
    this.sink.onData(this.view, ts, diff, values);
  }
  progress(ts: bigint): void {
    this.sink.onProgress(this.view, ts);
  }
}

function fakeWorld() {
  const subs = new Map<string, FakeSub>();
  const factory: SubscriptionFactory = (_conn, view, asOf, snapshot, sink) => {
    const s = new FakeSub(view, sink, asOf, snapshot);
    subs.set(view, s);
    return s;
  };
  return { subs, factory };
}

function recorder() {
  const moments: { F: bigint; batch: CohortBatch }[] = [];
  const upcall = async (F: bigint, batch: CohortBatch): Promise<void> => {
    moments.push({ F, batch });
  };
  return { moments, upcall };
}

// Let the serialized async pump settle.
const settle = () => new Promise((r) => setImmediate(r));

let passed = 0;
async function test(name: string, fn: () => Promise<void>): Promise<void> {
  await fn();
  passed++;
  console.log(`ok - ${name}`);
}

function byView(batch: CohortBatch): Record<string, CohortBatch[number]["rows"]> {
  return Object.fromEntries(batch.map((v) => [v.view, v.rows]));
}

async function main(): Promise<void> {
  await test("resumeAsOf subtracts one and clamps at zero", async () => {
    assert.equal(resumeAsOf(100n), 99n);
    assert.equal(resumeAsOf(1n), 0n);
    assert.equal(resumeAsOf(0n), 0n);
  });

  await test("resuming picks AS OF from-1, SNAPSHOT false", async () => {
    const { subs, factory } = fakeWorld();
    const { upcall } = recorder();
    await Cohort.resuming("", ["a"], 42n, upcall, { factory });
    const a = subs.get("a")!;
    assert.equal(a.asOf, 41n); // resumeAsOf(42)
    assert.equal(a.snapshot, false);
  });

  await test("consistent moment = min(frontier) across the cohort", async () => {
    const { subs, factory } = fakeWorld();
    const { moments, upcall } = recorder();
    await Cohort.resuming("", ["a", "b"], 10n, upcall, { factory });
    const a = subs.get("a")!;
    const b = subs.get("b")!;
    a.schema(["x"]);
    b.schema(["x"]);
    a.data(10n, 1n, ["a10"]);
    b.data(10n, 1n, ["b10"]);
    a.data(12n, 1n, ["a12"]);
    a.progress(13n); // a -> 13, but b only -> 11, so min = 11
    b.progress(11n);
    await settle();
    assert.equal(moments.length, 1);
    assert.equal(moments[0].F, 11n);
    const v = byView(moments[0].batch);
    // both ts=10 rows released; a12 (>= 11) held back
    assert.equal(v["a"].length, 1);
    assert.equal(v["a"][0].values[0], "a10");
    assert.equal(v["b"][0].values[0], "b10");
  });

  await test("z-set consolidation within a window", async () => {
    const { subs, factory } = fakeWorld();
    const { moments, upcall } = recorder();
    await Cohort.resuming("", ["m"], 0n, upcall, { factory });
    const m = subs.get("m")!;
    m.schema(["name"]);
    m.data(1n, 1n, ["alice"]); // alice: +1 +1 -1 => net +1
    m.data(1n, 1n, ["alice"]);
    m.data(2n, -1n, ["alice"]);
    m.data(1n, 1n, ["bob"]); // bob: +1 -1 => net 0 => dropped
    m.data(2n, -1n, ["bob"]);
    m.progress(5n);
    await settle();
    assert.equal(moments.length, 1);
    const rows = moments[0].batch[0].rows;
    assert.equal(rows.length, 1);
    assert.equal(rows[0].values[0], "alice");
    assert.equal(rows[0].diff, 1n);
  });

  await test("idle advance fires an empty batch (a commit point, no data)", async () => {
    const { subs, factory } = fakeWorld();
    const { moments, upcall } = recorder();
    await Cohort.resuming("", ["a"], 0n, upcall, { factory });
    const a = subs.get("a")!;
    a.schema(["x"]);
    a.progress(5n);
    await settle();
    assert.equal(moments.length, 1);
    assert.equal(moments[0].F, 5n);
    assert.deepEqual(moments[0].batch, []);
  });

  await test("nothing is released until every member is live", async () => {
    const { subs, factory } = fakeWorld();
    const { moments, upcall } = recorder();
    await Cohort.resuming("", ["a", "b"], 0n, upcall, { factory });
    const a = subs.get("a")!;
    const b = subs.get("b")!;
    a.schema(["x"]);
    b.schema(["x"]);
    a.progress(100n); // only a is live; min is undefined
    await settle();
    assert.equal(moments.length, 0);
    b.progress(50n); // now both live, min = 50
    await settle();
    assert.equal(moments.length, 1);
    assert.equal(moments[0].F, 50n);
  });

  await test("split lets a fast subset advance past a laggard", async () => {
    const { subs, factory } = fakeWorld();
    const { moments: m1, upcall: u1 } = recorder();
    const cohort = await Cohort.resuming("", ["fast", "slow"], 0n, u1, {
      factory,
    });
    const fast = subs.get("fast")!;
    const slow = subs.get("slow")!;
    fast.schema(["x"]);
    slow.schema(["x"]);
    fast.data(1n, 1n, ["f1"]);
    fast.progress(10n);
    slow.progress(2n); // min = 2: releases f1 (ts 1) but the cohort is stuck at 2
    await settle();
    assert.equal(m1[m1.length - 1].F, 2n);

    // Peel 'fast' into its own cohort; it should now advance to 10 alone.
    const { moments: m2, upcall: u2 } = recorder();
    const child = cohort.split([cohort.subscription("fast")], u2);
    await settle();
    assert.deepEqual(child.views(), ["fast"]);
    assert.deepEqual(cohort.views(), ["slow"]);
    assert.equal(m2[m2.length - 1].F, 10n);
  });

  await test("drop a laggard so the rest can advance", async () => {
    const { subs, factory } = fakeWorld();
    const { moments, upcall } = recorder();
    const cohort = await Cohort.resuming("", ["a", "slow"], 0n, upcall, {
      factory,
    });
    const a = subs.get("a")!;
    const slow = subs.get("slow")!;
    a.schema(["x"]);
    slow.schema(["x"]);
    a.data(5n, 1n, ["a5"]);
    a.progress(20n);
    slow.progress(3n); // min = 3, stuck
    await settle();
    assert.equal(moments[moments.length - 1].F, 3n);
    cohort.drop(cohort.subscription("slow"));
    await settle();
    assert.deepEqual(cohort.views(), ["a"]);
    assert.equal(moments[moments.length - 1].F, 20n); // a advanced alone
  });

  await test("merge takes the higher commit point; backlog flushes to the new upcall", async () => {
    const { subs, factory } = fakeWorld();
    const { moments: m1, upcall: u1 } = recorder();
    const { upcall: u2 } = recorder();
    const c1 = await Cohort.resuming("", ["a"], 0n, u1, { factory });
    const c2 = await Cohort.resuming("", ["b"], 0n, u2, { factory });
    const a = subs.get("a")!;
    const b = subs.get("b")!;
    a.schema(["x"]);
    b.schema(["x"]);
    a.progress(20n); // c1 advances to 20 alone (idle)
    await settle();

    c1.merge(c2); // committedF = max(20, 0) = 20; union governed by u1
    b.data(5n, 1n, ["b5"]); // b's pre-merge backlog
    a.progress(30n);
    b.progress(25n); // min(30, 25) = 25 > 20 -> release ts < 25 (b5)
    await settle();

    assert.deepEqual(c1.views().sort(), ["a", "b"]);
    assert.deepEqual(c2.views(), []);
    const last = m1[m1.length - 1];
    assert.equal(last.F, 25n);
    const bv = last.batch.find((v) => v.view === "b");
    assert.equal(bv!.rows[0].values[0], "b5");
  });

  await test("released buffers are dropped (re-SUBSCRIBE is the recovery path)", async () => {
    const { subs, factory } = fakeWorld();
    const { moments, upcall } = recorder();
    await Cohort.resuming("", ["a"], 0n, upcall, { factory });
    const a = subs.get("a")!;
    a.schema(["x"]);
    a.data(1n, 1n, ["x1"]);
    a.progress(5n); // release x1
    await settle();
    a.progress(9n); // nothing left below 9 -> idle, not a re-release of x1
    await settle();
    assert.equal(moments.length, 2);
    assert.equal(moments[0].batch[0].rows.length, 1);
    assert.deepEqual(moments[1].batch, []);
  });

  console.log(`\nall ${passed} sim assertions passed`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
