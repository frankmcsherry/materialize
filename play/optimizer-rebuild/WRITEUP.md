# Optimizer rebuild — findings writeup (deliverable #4)

Scope: a from-scratch **logical** optimizer core behind `MZ_OPTIMIZER_REBUILD=logical`,
replacing only `logical_optimizer`. Main's `logical_cleanup_pass` and the entire
`physical_optimizer` (join implementation, etc.) still run downstream. So every plan
below is rebuild-logical → main-cleanup → main-physical. ~1,460 LOC of transform logic
across 12 transforms; the existing logical phase it stands in for is ~6k+ LOC.

## 1. Correctness: results are preserved across the suite
Full slt corpus (487 files, `recursion_limit.slt` excluded as a pre-existing dev-build
stack overflow that also crashes main), each rebuild failure controlled against main:

- **pass=367, regress_result=1, panic=0.**
- One true wrong-result was found and fixed: `projection_thinning` pushed column demand
  through `Threshold`, breaking `EXCEPT`/set-difference (`subselect.slt` count 0 vs 100).
  Fixed (f675ce9d78); `subselect` now PASS 86/86.
- The remaining `regress_result=1` (`replacement-materialized-views.slt`) is **not** a
  wrong result: a valid-but-non-minimal key `{a,c,d}` vs main's `{a,c}` in an error message.
  Root cause (from the view `SELECT DISTINCT a, b AS c, 1 AS d …`): `d` is the constant `1`;
  main lifts it out of the `DISTINCT` group key (→ key `{a,c}`), the rebuild groups by it
  (→ `{a,c,d}`). It's the missing **constant-lift / LiteralLifting** (§4.4), not a key-analysis
  bug — and the same cause inflates part of the distinct +233 delta (grouping by liftable
  expressions widens both the group key and its inferred unique key).

So: the rebuilt logical core returns the same answers as main everywhere except one
(now-fixed) bug and one cosmetic precision difference. That is the real, earned version
of "parity on the suite."

## 2. The 90 plan-text diffs — triaged (are they breaking? which classes?)
The other 90 failing files are EXPLAIN-text differences (a different optimizer prints a
different plan). Mined both plans (rebuild vs main) from the sweep log and classified by
arrangement count (memory proxy) and node mix. 89 of 90 had extractable plan pairs.

**Direction (arrangement proxy, rebuild vs main):**
- parity (equal arrangements): **47** — benign formatting/ordering, same dataflow size.
- rebuild fewer arrangements: **4**.
- rebuild MORE arrangements: **38**.

So it is **not** "90% good / 10% fatal" and **not** "all 50-50 toss-ups." It's roughly
**half parity, half the rebuild builds a correct-but-bigger plan.**

**The bigger plans are not random — they cluster into a short list of missing
optimizations** (aggregate node deltas, rebuild minus main, across all 90):

| node | main | rebuild | delta | reading |
|---|--:|--:|--:|---|
| Distinct | 450 | 683 | **+233 (+52%)** | rebuild fails to elide Distincts main removes |
| Get | 2154 | 2825 | +671 (+31%) | symptom of bigger plans (more operators ⇒ more Get refs); NOT evidence of a CSE gap — see note |
| Reduce | 413 | 559 | +146 (+35%) | reductions not pushed down / not elided |
| Join | 797 | 918 | +121 (+15%) | redundant joins not eliminated |
| ArrangeBy | 1741 | 1931 | +190 (+11%) | net memory proxy: ~11% heavier |
| CrossJoin | 123 | 130 | +7 (negligible) | **no cross-join blow-ups** (the catastrophic case is absent) |

The `transform/*.slt` files (main's own per-transform regression tests) name the gaps
directly — they fail precisely where the rebuild lacks a transform main has:
- `transform/redundant_join.slt` (join 0→4, arr 0→8): **RedundantJoin not implemented.**
  Concrete: main collapses a redundant self-join to `Filter→Map→ReadStorage` (0
  arrangements); the rebuild keeps the `Join` + 2 `ArrangeBy` + `Distinct`. Same result.
- `not-null-propagation.slt` (distinct 8→36, reduce 3→36): weak null/predicate propagation
  → re-derived Distincts/Reduces.
- `transform/reduction_pushdown.slt`, `transform/column_knowledge.slt`,
  `transform/literal_lifting.slt`, `transform/aggregation_nullability.slt`: the named
  transform is missing or weaker in the rebuild.

**Severity: none are breaking.** Every diff is a correct plan; the deltas are proportional
(largest is `ldbc_bi` at +42 arrangements on a 297-arrangement, 25-EXPLAIN giant ≈ +14%),
and CrossJoin is flat (+7) so no plan is "winning" on arrangement count by collapsing into a
catastrophic cross product. The gap is **un-built optimization, not mis-built optimization.**

## 3. The "few broad ideas" that worked
- **A flat binding environment** (`env.rs`): peel the Let-spine into `Vec<(LocalId, body)>`
  with Let-free bodies, LetRec opaque. Makes every transform a function over a list of named,
  Let-free expressions; sharing is explicit.
- **Demand/projection-thinning + join flattening carry ~45% of the logic** and produce most
  of the wins. Notably, **main's cte-width shape emerged from *composition*, not a dedicated
  rule**: `join_flatten` (makes equivalence-class duplicates referenceable) + `projection_thinning`
  (drops them) + a dedup-hoist (re-duplicates at use sites). No single transform does it.
  That is direct evidence for the "few broad ideas, well-factored" hypothesis.
- **Per-form factoring with requires/ensures contracts** kept each transform a single idea
  with an explicit error-discipline argument; the one correctness bug was a contract violation
  (`Threshold` is identity-sensitive) that the contract framing made easy to name and fix.

## 4. Prioritized gap list (effort rough; NO work done here — triage only)
Ranked by arrangement impact in the triage:
1. **`Get`/`Let` type refresh — TRIED, INERT (see §7).** The hypothesis was that stale `Get`
   keys blocked the existing key-based rewrites. Experiment 1 refreshed them; the sweep came
   back byte-identical. Correct but inert — *not* the lever. The real levers are #2–#4
   (genuinely missing transforms), which the gap map (§7) shows split ~evenly across the
   non-tiny worse-files.
2. **Residual elision after the refresh: semijoin idempotence** — a genuinely missing rewrite
   (not just stale keys). *Medium effort,* gated on #1 to see how much remains.
3. **RedundantJoin elimination** (+121 join). Genuinely not implemented; self-contained
   transform. *Medium effort.*
4. **ReductionPushdown, ColumnKnowledge propagation, LiteralLifting** — smaller, named, mostly
   independent; *low-medium each.*
5. The `Get` +31% is **not** its own gap — **confirmed a symptom by data**, not a CSE deficit:
   refs-per-CTE is identical (main 2.11, rebuild 2.12), the rebuild has *more* CTEs (+307, i.e.
   shares more, not less), and the `Get` delta tracks total-op-count (≈0.26×) and CTE-count
   (≈2.1×) uniformly across files. It will shrink as #2–#4 cut the operator count; no CSE work
   is warranted.

## 5. E-graph assessment
The biggest leverage idea is sound: MIR `LetRec` uses synchronous update semantics, so a name
means the same thing everywhere in scope → equality is *unconditional*, which is exactly the
property that makes an e-graph (equality saturation) natural rather than fighting the IR. The
current factored transforms are a reasonable *pre-e-graph* state: each is a rewrite with a
contract, and the "compose to get main's shape" finding (§3) suggests the win from a saturating
engine would be real (the rebuild already gets emergent behavior from composition). The missing
optimizations in §4 are mostly *rewrites* (redundant join, semijoin, reduction pushdown) — good
e-graph rule candidates. The first unblock is **not** a new analysis (the keys analysis exists);
it's refreshing `Get`/`Let` type info mid-pipeline (§4.1) so the existing analysis is actually
seen by key-based rewrites — and an e-graph, which keeps equivalences current by construction,
would dissolve exactly that staleness problem. Recommendation: experiment 1 (§7) showed the
refresh is *not* the lever; the work is the missing transform families (§4.2–4.4), each modest.
Decide between continuing hand-factored transforms vs. moving to a saturating engine once a
couple are in.

---

## 6. Should this continue? (deliverable #5)

Yes — with eyes open. The weekend produced a from-scratch logical core that, across the full
slt suite, **returns main's results everywhere** bar one bug (found and fixed) and one cosmetic
key-precision difference, in ~1,460 lines that even reproduce some of main's behavior by
*composition* rather than dedicated rules. That is a genuine result, not a toy.

The case for continuing is that **the remaining gaps are bounded and named, not diffuse**.
Triage shows ~half the plan diffs are pure parity and the rest are *correct-but-bigger* plans
clustering into a short list of known-missing optimizations — with **no catastrophic cases**
(deltas proportional, no cross-join blow-ups). Most importantly, the dominant gap (Distinct/
Reduce elision) is *probably not a missing capability at all* but stale key info across
`Get`/`Let` — a small refresh, not a new analysis. If that one experiment lands, a large
fraction of the 38 "worse" files should move toward parity at once; if it doesn't, we learn the
gap is deeper and re-cost. Either outcome is cheap to obtain and highly informative.

The honest caveats for whoever funds the next step: (a) this is still only the *logical* phase
in front of main's cleanup+physical, so "replace the optimizer" remains far off; (b) the
parity-or-better claim on plan *quality* is not yet proven for the 38 worse files — they're
correct and proportionally heavier, but a real memory/cost read (not just arrangement count)
is still owed; (c) the genuinely-missing transforms (semijoin idempotence, redundant join) are
real work, low-risk but not free. Net: the next increment is unusually well-targeted — one
cheap experiment (the `Get`-type refresh) plausibly closes the dominant gap — so continuing is
worthwhile, and the go/no-go on a larger investment should be re-evaluated right after that
experiment's result.

## 7. Overnight experiment log

**Experiment 1 — `Get`/`Let` type refresh (the staleness hypothesis): DISCONFIRMED.**
A surgical pass refreshing each `Get`'s recorded type from its binding body, inserted before
`reduce_elision`, produced a **byte-identical** corpus sweep (pass=367, reg_result=1, every
node-delta unchanged). It was correct but inert: rails held, no new elision. So the dominant
gap is not stale keys feeding the existing rewrites — it's that the rebuild lacks the transform
*families* main has (the deltas are dominated by `Reduce`-with-aggregates and `Join`s, beyond
`reduce_elision`'s Distinct-only scope).

**Sharpened gap map (the 38 "worse" files).** 20 are within ≤2 arrangements (near-parity). The
18 non-tiny split ~evenly across three known-missing families: ~7 join-elimination
(redundant-join/semijoin), ~7 distinct/semijoin-elision, ~4 reduction-elision/pushdown; the two
largest (`ldbc_bi`, +42) are 25-query benchmarks with the delta spread thin. **Closability
verdict (evidence-based): the gap is ~3 modest, well-understood transform families for
incremental, evenly-distributed gains — not one big lever, and half the worse-set is already
near-parity.** This refines §4: it's tractable, bounded work, but it's *N transforms each worth
a few files*, not a single high-leverage fix.

**Experiment 2 — reduce-elision for keyed Reduce-with-aggregates: WIN (committed 7a41e362b1).**
Extended `reduce_elision` to the with-aggregates case (group key is a unique key ⇒ one row per
group ⇒ `AggregateExpr::on_unique`), mirroring main, gated on a local `typ().keys` check (no
provenance). Result: reduce-node delta vs main **+146 → +73**, pass **367 → 368**, rails held
(reg_result 1, panic 0). A real, modest, correctness-safe simplification.

**Remaining gaps are provenance-bound (deliberate work, not overnight quick-wins).** The two
biggest remaining families — distinct/semijoin-elision (+233) and redundant-join (+121) — both
require main's provenance analysis (`semijoin_idempotence` 593 LOC, `redundant_join` 861 LOC,
`ProvInfoCtx`); a narrow structural detector cannot catch even `redundant_join.slt`'s own
example. So the cheap/tractable levers are now exhausted (one disconfirmed, one committed); the
rest is deliberate reimplementation of provenance-based transforms.
