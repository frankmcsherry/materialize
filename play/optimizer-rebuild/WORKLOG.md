# Optimizer rebuild — work log

Base: b5df9fd73a409ae3b7aafcc42efc98334d72cf26 (origin/main, pinned 2026-06-12 17:30)
Worktree: .claude/worktrees/optimizer-rebuild, branch optimizer-rebuild
Prompt: play/optimizer-rebuild/LOOP_PROMPT.md

## Journal

### 2026-06-12 17:30 — kickoff
Loop started. Plan: Phase 0 capability map — (1) inventory logical_optimizer's
composition with one-line contribution per registered transform, (2) cross-ref
optimizer-probe corpus as behavior checklist, (3) mine physical_optimizer's
implicit input contracts (most load-bearing), (4) selected ablations for
contract discovery only. Then Phase 1: Let/sharing position, switch, harness.

### 2026-06-12 17:31 — iteration 1
Setup complete: worktree at b5df9fd7, journal, scorecard/, specs/. Captured all
five composition trees verbatim into capability-map.md (logical_optimizer,
normalize, FuseAndCollapse, fuse_and_collapse_fixpoint, cleanup+physical).
First structural observation: FuseAndCollapse is registered in THREE positions
(own fixpoint + inside both logical fixpoints) — the canonical no-trusted-
position smell. Launched background inventory agent for per-transform
idea/mechanism/contracts/quirks annotations. Next iteration: merge agent
output into capability-map, cross-ref probe corpus, mine physical_optimizer's
input contracts, choose Phase-0 ablation set.

### 2026-06-12 17:33 — iteration 2
Per-transform inventory landed (appended to capability-map.md): 25 distinct
transforms; NormalizeLets runs 6+ times, FuseAndCollapse 4, Typecheck 2; four
fixpoints, all limit-100. Notable mined contracts: NormalizeLets must rerun
"before RedundantJoin so key info is correct" and after FoldConstants ("can
break normalized form"); fusion::join re-runs PredicatePushdown+CanonicalizeMfp
ad hoc inside itself (database-issues#7728 admits temporary); FoldConstants <->
LiteralLifting oscillation documented in physical pipeline comments; Demand
"mostly obsoleted by ProjectionPushdown but one unique optimization". Broad
ideas identified: predicate placement, equivalence reasoning, aggregation
placement (ReductionPushdown), distinctness (WillDistinct), semijoin
idempotence, redundancy (RedundantJoin/CSE). Everything else is normalization
or local simplification — consistent with the few-broad-ideas hypothesis.
Next: physical_optimizer input-contract mining (agent), then probe cross-ref.

### 2026-06-12 17:36 — iteration 3
Physical-boundary contracts mined and appended to capability-map.md. The
rebuilt logical pipeline's ensures-set at its output boundary: (a) Let normal
form (root-hoisted, renumbered — multiple PANIC consumers), (b) no Dummies/
unbound locals (typecheck-enforced), (c) canonical MFP order at the boundary,
(d) canonicalized join equivalences w/ strict nullability. Also catalogued the
physical pipeline's own fragile internal orderings (LiteralConstraints/
JoinImplementation/RelationCSE exclusion zone) as hazards NOT to recreate.
Next: probe-corpus cross-ref (launching), then ablation-set decision and the
Let/sharing design position — the latter is the first real design act.

### 2026-06-12 17:38 — resilience
User heading out for the weekend; added hourly resurrection cron (e17249b7,
:23 past each hour) re-entering the loop, so usage-limit blocks only pause
work until the next tick. Journal discipline is the resume mechanism: every
iteration records state here before ending its turn. Probe-corpus cross-ref
agent still in flight; on its arrival Phase 0 mining closes and the Let/
sharing design position (Phase 1) begins.

### 2026-06-12 17:40 — iteration 4: Phase 0 closes, Phase 1 opens
Probe cross-ref banked (with provenance caveat). DECISION: skip the optional
ablation experiments — the mined comments + contracts already surface the
load-bearing dependencies, ablation costs hours per data point, and per the
prompt's epistemics it could not tell us what the optimizer MUST do anyway.
DECISION (design-let-sharing.md): flat binding environment with Let-free
bodies as the working representation; sharing changes are explicit ops;
entry/exit converters own the Let contracts. Rationale and rejected
alternatives in the doc. Next: Phase 1 skeleton — rebuild module, env-var
switch, entry/exit converters + round-trip tests, Typecheck passthrough,
then the evaluation harness.

### 2026-06-12 18:04 — iteration 5: Phase 1 skeleton
rebuild module landed (committed + pushed to fork): BindingEnv (ordered
Let-free bindings + root; LetRec opaque in Phase 1), RebuildLogical transform
(parse -> empty sequence -> emit), MZ_OPTIMIZER_REBUILD=logical switch with
permissive end-Typecheck (strict checks return when rebuild establishes the
properties). Round-trip + rejects-unnormalized unit tests pass. SCAFFOLDING
LOAN journaled: entry uses existing NormalizeLets once as parser. Smoke in
flight: results-only cockroach slts under the switch (first slt-binary build
in this worktree — long). On smoke pass, Phase 1 closes with the harness
script; then transform earning begins (first candidates per capability map:
predicate placement cluster or the fuse/simplify substrate — decide against
the first scorecard).

### 2026-06-12 18:12 — iteration 6: first contracts discovered + first transform earned
Smoke run surfaced two real boundary contracts within minutes (the harness
works): (1) my opaque-LetRec check used visit_pre whose closure cannot stop
descent — Lets at LetRec scope roots are LEGITIMATE normal form; fixed with a
non-descending worklist. (2) NEW CONTRACT not in the capability map: temporal
filter compilation (mz_now() predicates) REJECTS top-level ANDs in Filter
predicates — PANICS at bootstrap on system views. Consumed at MFP lowering;
the existing pipeline satisfies it via canonicalize_predicates as a side
effect. EARNED TRANSFORM #1: rebuild/filter_split.rs — Filter predicate
normal form (no top-level conjunctions), with soundness note (And's
error behavior is grouping-insensitive) and contract doc. Unit tests pass;
smoke re-running. Added contract to track: predicate-conjunct splitting is
load-bearing for temporal filters.

### 2026-06-12 18:16 — iteration 7: next contract bounded, design for transform #2
Smoke now reaches a second LIR contract: ReducePlan::create_from ASSERTS all
aggregates in a Reduce share one reduction class (Accumulable/Hierarchical/
Basic; plan/reduce.rs:346; sibling assert 378 for fused-unnest Basic
singletons). Existing pipeline satisfies via ReduceReduction. DESIGN for
earned transform #2 (reduce_class_split): a mixed Reduce(K, [aggs]) becomes
per-class Reduces over the same input equijoined on K, projected to original
column order; join is 1:1 and total because K is a unique key of each part
and all parts share the same group set; empty-input consistent (all parts
empty). monotonic/expected_group_size copied to parts. v1 CLONES the shared
input (journaled debt: switch to an explicit env binding once BindingEnv
grows an id allocator — this transform is the natural showcase for explicit
sharing). Committed+pushed filter_split + LetRec check fix.

### 2026-06-12 18:44 — iteration 8: PHASE 1 COMPLETE
Smoke PASSES end to end (bootstrap + 322 result queries) with exactly two
earned transforms. Committed + pushed (4fc32e0248). Scorecard harness
(sweep.sh) written: classifies pass / plandiff (with rough arrangement
counts) / resultdiff (stop-the-line) / panic, never rewrites checked-in
slts. First sweep (sweep-001, 12 representative files) running — expect
heavy plandiff (passthrough plans) and possibly fresh panics on harder
shapes; result diffs would be bugs. Phase 2 begins: earn quality transforms
against the scorecard, broad ideas first.

### 2026-06-12 18:45 — iteration 9: sweep-001 verdict + Phase 2 targets
Sweep-001: pass=4 plandiff=8 resultdiff=0 panic=0 over ~1,150 queries.
CORRECTNESS HOLDS with the two-transform pipeline; every divergence is plan
shape (passthrough = unoptimized, expected). Classifier debt: the
arrangement counter samples only the first diff hunk — refine before using
it for plan-quality judgments. PHASE 2 ORDER DECIDED: (#3) linear-operator
normalization — Map/Filter/Project fusion + compaction per form; it is
substrate (predicate placement operates on fused Filters, and unfused
lowering noise drowns the scorecard); then (#4) predicate placement (the
first broad idea: saturate-equivalences, place-each-conjunct-lowest, with
the no-new-errors discipline and conjoin-don't-replace available); then
(#5) join graph flattening. Each per the from-scratch discipline: idea in
own words in this journal before reading any existing counterpart code.

### 2026-06-12 18:51 — iteration 10 (cron tick): transform #3 earned
linear_fuse: canonical M-F-P per node via the shared MapFilterProject IR
utility; non-errors extraction keeps rule 1; ordered BEFORE filter_split
(fusion can form ANDs — a requires/ensures chain doing its job). Unit tests
pass (6 total in rebuild). Committed, pushed; sweep-002 running on the
plandiff set to measure movement vs sweep-001. Pipeline is now:
linear_fuse -> filter_split -> reduce_class_split. Next per plan: #4
predicate placement (first broad idea) — write the own-words design at next
wake before any code.

### 2026-06-12 18:54 — iteration 11: sweep-002 clean; predicate placement designed
sweep-002: resultdiff=0 panic=0 with linear_fuse live (plandiffs persist as
expected pre-quality-transforms). design-predicate-placement.md written: one
top-down pass, per-operator descent table, subset-or-infallible error rule
unifying rule 1 across operators. Implementation next iteration; ordering
will be linear_fuse -> filter_split -> predicate_placement -> linear_fuse?
No: deposit emits Filters; sequence becomes [linear_fuse, filter_split,
predicate_placement, reduce_class_split] with predicate_placement preserving
both upstream normal forms by construction (deposits are split conjuncts on
single Filter nodes).

## Iteration 12 (2026-06-12 ~19:30)

Implemented `predicate_placement` exactly per design-predicate-placement.md.
Mechanics: Filters dissolve into a pending conjunct set on the way down;
per-operator routing (Project remap, Map if no map-col refs, Union copy to
all branches, Join home-input with local remap, Reduce via group-key
substitution, TopK key-column subset, Negate/Threshold free); deposits
recreate single split-conjunct Filters where conjuncts stop. The one error
rule from the design ("subset-or-infallible") is the only gate.

Notes earned during implementation:
- Reduce key substitution happens AFTER the infallibility check on the
  conjunct itself; a fallible group_key expr is fine because Reduce already
  evaluates it on every input row — no new errors (recorded in module doc).
- `visit_mut_children` (VisitChildren trait) handles the opaque catch-all
  (ArrangeBy/LetRec/leaves): children still get placement with empty pending,
  deposits stay above. ArrangeBy passthrough is deliberate v1 conservatism.
- Lesson re-learned: scripted import cleanup removed a *used* top-level
  import because the compiler warning pointed at the test-module duplicate.
  Check the warning's line number before deleting.

Wired sequence: [linear_fuse, filter_split, predicate_placement,
reduce_class_split]. 9/9 unit tests (3 new: join home-input descent,
union branch copy, fallible-stays-above-join).

Sweep-003 (same 12 files as sweep-001): pass=4 plandiff=8 resultdiff=0
panic=0. Semantic safety holds. Plandiff movement is small (tpch_select
21->20 failing EXPLAINs; join_fusion arrangements 30/16 -> 28/14) — as
expected, since main's PredicatePushdown already places these and the
remaining diffs are join planning, not predicates. The transform's value
shows up once join graph flattening (#5) consumes filters adjacent to
joins.

Commit 9108bb0f57 pushed to me/optimizer-rebuild.

Next: design + implement join graph flattening (#5): adjacent Join/Filter/
Map/Project regions flattened into a single join graph with equivalence
classes; this is where arrangement counts move. Then sweep-004.

## Iteration 13 (2026-06-12 ~19:45)

Spot-check of predicate_placement through the live pipeline (tiny /tmp slt,
never a checked-in file): `SELECT * FROM t1, t2 WHERE t1.a = t2.c AND
t2.d > 7` shows `d > 7` on t2's join input and both filters lifted into the
storage-read MFPs. Matches main's shape. (Caveat recorded: later phases —
logical_cleanup/physical — still run main's transforms, so equality→
equivalence conversion in that plan is not yet the rebuild's own work.)

Wrote design-join-flatten.md (transform #5): join regions (maximal
Join/Filter/Project trees) rebuilt as Project(Filter(flat Join)); patterns:
Join-of-Join splice with offset bookkeeping, equality-conjunct absorption
into equivalence classes WITH synthesized IS NOT NULL guards (datum-equality
vs SQL-equality null semantics — the guard is elidable when the side is
non-nullable), Project-between-joins remap; boundaries: Map (v1), Negate/
Threshold/Union (outer-join shapes, never crossed). Reuses
canonicalize_equivalences from mz-expr rather than reimplementing its
termination argument. New pipeline order puts join_flatten BEFORE
predicate_placement so guards/residue get routed by placement afterward.

Honesty note in the design: flattening can evaluate a fallible constraint
on rows the nested form had eliminated. This is parity with main
(JoinFusion flattens unconditionally; Join constraint order is
unspecified), so rule 1 holds relative to Join's contract, not the nested
rendering. Stated rather than hidden.

Next iteration: implement join_flatten + unit tests, reorder sequence,
sweep-004. Expect first real arrangement-count movement on joins.slt /
tpch_select.slt / join_fusion.slt.

## Iteration 14 (2026-06-12 ~20:15)

Implemented join_flatten (commit 50003e6371, pushed). Two findings earned
from sweep-004/005 plan diffs, both now encoded:

1. **Both-sides guards.** Sweep-004 regressed joins.slt (7->9 failing
   EXPLAINs): one-sided IS NOT NULL guards left the other side's input
   unfiltered, so RelationCSE downstream could not share the filtered
   arrangement (saw 2 arrangements where main has 1 on the IN-chain
   query). Guarding both sides is row-neutral given the class, and each
   guard sinks to a different input. Sweep-005: joins.slt 7->6 vs the
   pre-flatten baseline, tpch 21->20, join_fusion arrangements 28/14.
   Lesson for the writeup: the optimizer's value often lives in what
   LATER passes can see — a "semantically redundant" predicate is the
   difference between shared and duplicated arrangements.

2. **Datum-equality is load-bearing.** Main only synthesizes bare
   IS NOT NULL when consuming SQL equality filters; when localizing
   pre-existing equivalence members it pushes (e1 = e2) OR (both NULL).
   Confirmed in predicate_pushdown.rs:744-752. So equivalence classes
   cannot be mined for non-null facts in general — recorded in the
   capability map's contracts section candidate list.

Remaining joins.slt diffs classified (6):
- 2 diffs: unit-constant cross joins (outer-join lowering artifacts);
  implemented is_join_unit collapse in emit() this iteration (15/15
  tests), sweep-006 in flight.
- 3 diffs: cte projection thinning (main stores 3-col cte, re-projects
  duplicates at use sites; we arrange 4 cols). Next big transform:
  column demand / projection thinning across bindings (#6).
- 1 diff: subquery semijoin shape (main gets a 3-input delta join where
  we nest 2-input joins under a Distinct). Deeper; queued to
  JUDGMENT_QUEUE.md side — needs thought about Distinct-permeable
  flattening or a different decorrelation normal form.

## Iteration 15 (2026-06-12 ~20:30)

Two rounds of scorecard-driven refinement on join_flatten's constant
handling, both taught by reading the rebuild's own LOCALLY OPTIMIZED
output (EXPLAIN LOCALLY OPTIMIZED under MZ_OPTIMIZER_REBUILD=logical is
the right probe — the final OPTIMIZED plan mixes in main's later phases):

1. Binding-aware units: the unit constants in real plans are CSE'd into
   Let bindings, so join inputs are `Get l2`, not literal Constants.
   apply() now tracks unit bindings in dependency order.
2. Generalized to ANY single-row multiplicity-one constant input
   (commit c93122ac3c): values substitute as literals into equivalences/
   residue; visible columns re-emit as Map expressions. The
   aggregate-default idiom (sum over cross join -> Union with Map(null))
   now matches main's logical shape exactly.

Sweep-008: resultdiff=0 panic=0; failing-EXPLAIN counts flat but
first-diff arrangement samples down everywhere (joins 43/23, tpch 84/43,
subquery 36/21) — the remaining diffs are tighter.

Gap inventory after this iteration (next capabilities, in value order):
- (#6) Projection thinning / column demand across bindings: 3 joins.slt
  diffs + likely several subquery/tpch ones (main stores narrower ctes,
  re-projects at use sites). Design next iteration.
- (#7) Reduce simplification: Distinct over single-row constants should
  fold; Distinct over <=1-row inputs (Reduce with empty key) should
  elide to Project — needs a small cardinality/keys analysis. Visible in
  the fuel and outer-join-lowering diffs.
- (queued) Subquery semijoin shape (joins.slt:153): main flattens to a
  3-input delta join where we nest under Distinct. Revisit after #6/#7.

## Iteration 16 (2026-06-12 ~20:40)

Wrote design-projection-thinning.md (transform #6, column demand).
Consumers-before-producers over the flat env (root, then bindings last to
first); thin(expr, demand) rewrites in place and returns the old->new
column mapping; per-operator rules with the two genuine deletions (Map
scalars and Reduce aggregates nobody observes — both only remove error
possibilities, never rows) called out for the rule-1 argument. Bindings
narrow to the union of use-site demands; use sites re-project, which is
exactly the cte-width shape main produces on joins.slt. Pipeline re-runs
linear_fuse after thinning instead of teaching this pass M-F-P repair.

Implementation next iteration (largest transform so far — likely 2
iterations: core + Reduce/FlatMap cases), then sweep-009.

## Iterations 17-18 (2026-06-12 ~21:15)

Implemented projection_thinning (transform #6) and two follow-on
refinements, commit a018fc874e pushed. Pipeline is now:
[linear_fuse, filter_split, join_flatten, predicate_placement,
projection_thinning, linear_fuse, reduce_class_split].

Design notes that earned their keep:
- The marker discipline (Project-over-Get with ORIGINAL columns, fixed up
  after all layouts finalize) cleanly solves the
  consumers-rewritten-before-producers-finalize circularity.
- Equivalence-representative substitution in join_flatten + demand
  thinning + binding dedup hoist COMPOSE to produce main's cte shape:
  flatten makes duplicates referencable, thinning makes them droppable,
  the hoist moves the re-duplication to use sites. No single transform
  does it; the writeup should present this as evidence for the "few broad
  ideas, well-factored" hypothesis.
- Bare full-demand Gets (identity-wrap elision) need wrapping at fixup if
  their binding deduped — caught in design, not by a failure.

Sweep-010: joins.slt failing EXPLAINs 6 -> 3 (the three cte-width diffs
resolved); resultdiff=0 panic=0 throughout.

Remaining gap inventory:
- joins.slt 3: subquery semijoin shape (1), fuel/Distinct-fold remnants (2
  — need Reduce-over-constant folding + <=1-row Reduce elision, #7).
- subquery.slt 5, tpch 20, window_funcs 11: unexamined since thinning;
  re-classify next iteration.
- predicate_reduction.slt 3: likely actual predicate simplification
  (AND/OR absorption) — not yet attempted in rebuild.

## Iteration 19 (2026-06-12 ~21:30)

Classified all 5 remaining subquery.slt diffs:
- Diffs 1-3 + part of 4: missing unique-key reasoning. We keep Distinct
  over collections already keyed by the group columns (Distinct over
  Distinct, over keyed joins, over keyed Gets). Main's ReduceElision uses
  key knowledge. Get typs in our env remain sound-conservative (semantic
  preservation keeps lowering-assigned keys true; thinning's narrow_typ
  drops keys that lose columns), so typ().keys is usable as the v1
  analysis.
- Diff 4 also needs Union cancellation: Union(A, Negate(A), unit) -> unit
  once Distinct elision makes the A's syntactically equal. Requires Union
  flattening (nested Unions) first — not yet done anywhere in rebuild.
- Diff 0: redundant join (semijoin of l0 against Distinct(project of l0)
  on the same key is identity). Largest; after #7/#8.

Plan: transform #7 reduce_elision (Distinct case: group_key all columns,
input key subset of group cols -> Project) + #8 union_flatten+cancel in
one iteration; pipeline inserts [reduce_elision, union_cancel] after
predicate_placement, before projection_thinning. Then sweep-011 and
re-classify tpch/window_funcs.

## Iteration 20 (2026-06-12 ~21:45)

Implemented reduce_elision (#7) and union_cancel (#8); commit pushed.
Pipeline (9 stages): [linear_fuse, filter_split, join_flatten,
predicate_placement, reduce_elision, union_cancel, projection_thinning,
linear_fuse, reduce_class_split].

Earned detail: a cancelled-to-empty Union is itself the union identity in
an enclosing Union — the empty-constant branch drop is what lets the
default-row idiom collapse fully (found by unit test, not sweep).

Sweep-011: joins.slt 3->2, subquery 5->4, tpch 20->16, window_funcs flat
at 11; resultdiff=0 panic=0. joins.slt trajectory across the day:
7 -> 6 -> 3 -> 2 failing EXPLAINs.

Next: classify tpch_select (16) and window_funcs (11) diffs; expected
families: redundant join (#9, known from subquery diff 0), literal
lifting/CSE differences, and whatever tpch's aggregations reveal.

## Iteration 21 (2026-06-12 ~22:00)

Implemented reduce_inline (Map/Project below Reduce fold into group key
and aggregate exprs; Filter stops the fold). Chosen by frequency: the
"-group_by_expr" signature appeared in 13/16 tpch diffs. Commit pushed.

Sweep-012: outer_join_lowering.slt PASSES IN FULL (4 -> 0). window_funcs
11 -> 5, tpch 16 -> 14, pass=5 plandiff=7 resultdiff=0 panic=0.

Scoreboard across the day (failing EXPLAINs):
  joins 7->2, subquery 5->4, outer_join_lowering 4->0, window_funcs
  11->5, tpch 21->14, join_fusion 4 (flat), aggregates 3 (unexamined),
  predicate_reduction 3 (unexamined).

Remaining known families: window-func devolution on keyed partitions
(family A, ~5 window_funcs diffs), redundant join (#9, subquery diff 0 +
several tpch), predicate simplification (predicate_reduction.slt),
aggregates.slt unexamined. Next: classify aggregates + join_fusion +
predicate_reduction (the three small files) in one pass, then decide
between redundant-join (broad) vs window devolution (deep).

## Iteration 22 (2026-06-12 ~22:15) — final iteration before model shutoff

Classified the three small files in one pass:
- aggregates.slt (3): two are aggregate-arg Maps that reduce_inline already
  folds elsewhere but the projection above the Reduce kept; one is a
  duplicated sum(x) aggregate.
- predicate_reduction.slt (3): conjunct-level simplification — dedup,
  OR-absorption, implied-non-null.
- join_fusion.slt (4): delta-join ordering and a sargability detail
  (#0>0 vs IS NOT NULL guard placement); NOT cheap — left for later.

Implemented two transforms:
- aggregate_dedup (committed f5f6d31ab9): Reduce computes each distinct
  aggregate once; duplicates become a Project of the first. aggregates.slt
  3 -> 2.
- predicate_simplify (committed d3f6b4c9ce, post-shutoff by Opus): the three
  conjunct rules above. predicate_reduction.slt 3 -> 1.

Sweep-014 (final): pass=5 plandiff=7 resultdiff=0 panic=0. Full-day
scoreboard of failing EXPLAINs: joins 7->2, subquery 5->4,
outer_join_lowering 4->0, window_funcs 11->5, tpch 21->14,
aggregates 3->2, predicate_reduction 3->1, join_fusion 4 (untouched).

** MODEL SHUTOFF **: Fable was turned off Friday night mid-iteration 22.
The loop stopped cleanly: sweep-014 had completed but its results were
unread, and predicate_simplify was written + warning-fixed + sweep-
validated but uncommitted. No half-applied edits. Session-only
resurrection cron e17249b7 had no live runtime to fire against, so no
weekend token spend. On resume (Opus, 2026-06-15): confirmed 27/27 unit
tests green, committed predicate_simplify (d3f6b4c9ce), deleted the cron.

## Monday-deliverable status at shutoff
1. Capability map + contracts: DONE (capability-map.md).
2. rebuild/ with switch + 11 transforms + unit tests + contracts: DONE;
   EXCEPT the boundary-spec slts under specs/ were never created.
3. Scorecard: DONE (sweeps 001-014).
4. Writeup (few broad ideas / long tail / e-graph assessment / gap list):
   NOT WRITTEN.
5. Honest "should this continue" paragraph: NOT WRITTEN.
The two synthesis deliverables (4, 5) and the specs/ slts (2) are the
wrap-up artifacts the weekend never reached.

## Iteration 23 (2026-06-15) — corpus sweep, triage, first real correctness fix

Resumed on Opus after Fable shutoff. Ran the FULL slt corpus (not the 12
curated files) under MZ_OPTIMIZER_REBUILD=logical, main-controlled.

Measurement traps hit and fixed (control-first earned its keep):
- macOS has no `timeout`; v1 corpus sweep was 100% artifact (all 488 exit-127,
  split only by whether file contains "EXPLAIN"). Fixed: perl SIGALRM wrapper.
- v2 (per-invocation, 120s cap, main-controlled): pass=366 reg_explain=90
  reg_result=2 bothfail=28 panic=0 timeout=2.
- macOS app firewall re-prompts per launch of ad-hoc-signed clusterd (managed
  machine, can't change firewall); 488 invocations = flood. Fixed: batched
  sweep (all files in ONE invocation = 1 clusterd launch = ~1 prompt;
  batched-sweep.sh). One process for all files aborts on the first fatal
  crash -> excluded recursion_limit.slt (pre-existing dev-build tokio-worker
  stack overflow, crashes main too; CI uses --optimized).

Triage of the 4 non-plan discrepancies:
- subselect.slt:671 (correlated lateral EXCEPT, full rows) = REAL wrong result,
  count 0 vs 100. ROOT CAUSE: projection_thinning pushed reduced column demand
  through Threshold; Threshold implements set-difference by clamping
  multiplicities and compares whole rows, so thinning its inputs below the
  distinguishing columns cancelled rows wrongly. FIX (f675ce9d78): Threshold
  demands all input columns, narrows above. Negate left pass-through (its only
  identity-sensitive consumer, Threshold, now demands all).
- replacement-materialized-views.slt = NOT a wrong result; non-minimal key
  {a,c,d} vs main's {a,c} in an error message = key-minimization PRECISION gap
  (weak analysis), untouched by the fix.
- materialized_views.slt, privilege_grants.slt = NOT hangs; 120s-cap artifacts.
  privilege_grants PASSES (rebuild 133s vs main 252s); materialized_views has 1
  timing-dependent EXPLAIN test (rebuild finishes too fast to hit an expected
  statement timeout). Both benign.

Post-fix full sweep (487 files): pass=367 reg_explain=90 reg_result=1
bothfail=29 panic=0. reg_result 2->1 (subselect cleared; remaining = the key
precision gap). reg_explain unchanged at 90 => NONE were masking the Threshold
bug; all 90 are genuine plan-text diffs. subselect.slt PASS 86/86.

Net: one true corpus-wide correctness bug, found and fixed; results otherwise
preserved across the suite; the long tail is plan-text differences + one
weak-analysis precision gap. Scorecards: corpus_prefix.md (pre-fix),
corpus.md (post-fix).

## Iteration 24 (2026-06-15) — overnight loop start; experiment 1 (Get-type refresh)

Green-lit to continue overnight closing the triaged plan-quality gaps
(OVERNIGHT_PLAN.md). User guidance: (a) the refresh logic properly lives in
normalize_lets.rs and is subtle — reuse/import rather than reinvent, but I'm
starting with the minimal surgical version for clean attribution;
(b) semijoin-idempotence (#2) and redundant-join (#3) are one family —
"remove joins/semijoins that, for different reasons, do nothing".

Experiment 1: src/transform/src/rebuild/refresh.rs — forward pass refreshing
each local Get's recorded type from its binding body's current typ() (deps in
order; LetRec opaque). Strictly more precise, cannot change results. Wired
before reduce_elision. 28/28 unit tests. Tests the stale-Get-keys hypothesis
(code-confirmed: Gets recorded at parse, never refreshed mid-pipeline).

Baseline corpus_exp0.md (post-Threshold): pass=367 reg_explain=90 reg_result=1.
Measurement sweep launched (batched, recursion_limit excluded). Rails for the
commit decision: reg_result must stay <=1, panic=0; revert if correctness
regresses; keep only if it reduces the Distinct/arrangement deltas.
Next on sweep completion: compare vs corpus_exp0.md (node-delta analysis),
commit-or-revert + journal, then the no-op-join/semijoin family (#2/#3).

## Iteration 25 (2026-06-15) — experiment 1 result: staleness DISCONFIRMED

Refresh sweep: byte-identical to corpus_exp0.md. pass=367 reg_explain=90
reg_result=1 panic=0; node deltas UNCHANGED (arr +190, join +121, distinct
+233, reduce +146, get +671). The Get-type refresh was inert: rails held (it
only sharpens types) but it enabled no new elision. Conclusion: the dominant
gap is NOT stale Get keys feeding reduce_elision; it's that reduce_elision is
far narrower than main's family (deltas dominated by Reduce-with-aggregates and
Joins, which neither key-refresh nor a Distinct-only elision can address).
refresh.rs left uncommitted and disconfirmed; revert at cleanup unless a later
transform needs cross-Get keys (experiment showed none does so far).

Sharpened gap map (38 "worse" files): 20 tiny (≤2 arr, near-parity, skip);
18 non-tiny split ~evenly: 7 join-dominant (redundant-join/semijoin), 7
distinct-dominant (distinct/semijoin elision), 4 reduce-dominant (reduction
elision/pushdown). Biggest (ldbc_bi +42) are giant benchmarks, delta spread
thin. => closability = ~3 known modest transform families, incremental/even
gains, no single lever.

Next: implement redundant-join elimination (the most-confirmed: redundant_join.slt
main collapses to Filter->Map->Read, rebuild keeps Join+2 ArrangeBy+Distinct),
carefully, rail-protected (reg_result must stay <=1), in a fresh iteration.

## Iteration 26 (2026-06-15) — experiment 2: reduce-elision extension

Checked the three families' feasibility against main's transforms:
- redundant_join (861 LOC) and semijoin_idempotence (593 LOC) both rely on
  PROVENANCE analysis (ProvInfoCtx), not structural matching — confirmed a
  narrow structural detector can't even catch redundant_join.slt's own example
  (a Map breaks structural equality). NOT safe overnight quick-wins; deferred
  as deliberate production work.
- ReduceElision is only 121 LOC and the hard part (aggregate value on a
  single-row group) is a reusable primitive: AggregateExpr::on_unique. TRACTABLE.

Implemented: extended rebuild reduce_elision to the keyed-Reduce-WITH-aggregates
case, mirroring main — input.map(group_key ++ aggregates.on_unique(input_type))
.project(...) gated on input keys ⊇ group key (local typ().keys check, no
provenance). Kept the working Distinct path untouched. 29/29 unit tests
(new: elides_keyed_reduce_with_aggregate). Reverted the inert refresh first so
the swept binary == committable code.

Baseline corpus_exp0.md: pass=367 reg_explain=90 reg_result=1; reduce delta
+146. Measurement sweep launched (bb9glgx6p). Commit if reduce/arr deltas shrink
AND rails hold (reg_result<=1, panic=0); revert otherwise. Next on completion:
compare, commit-or-revert + journal.

## Iteration 27 (2026-06-15) — experiment 2 WIN committed; loop winding down

Experiment 2 (reduce-elision for keyed Reduce-with-aggregates) is a clean
rail-safe win: reduce delta +146 -> +73, pass 367 -> 368, reg_explain 90 -> 89,
reg_result still 1, panic 0. Committed 7a41e362b1, pushed.

Remaining gap families both need provenance analysis (assessed against main):
distinct/semijoin-elision (+233; semijoin_idempotence 593 LOC, ProvInfoCtx) and
redundant-join (+121; redundant_join 861 LOC, ProvInfoCtx). Neither is a safe
unattended overnight quick-win (a narrow structural detector can't even catch
redundant_join.slt's own example). residual reduce (+73) and reduction_pushdown
(492 LOC) similar. So the cheap/tractable levers are exhausted; the rest is
deliberate production work.

Loop conclusion: stopping cleanly. Overnight delivered: Threshold correctness
fix (f675ce9d78), experiment 1 (staleness) cleanly disconfirmed + reverted, and
experiment 2 (reduce-elision extension) committed (7a41e362b1). Scorecards:
corpus_prefix.md (pre-Threshold), corpus_exp0.md (post-Threshold), corpus.md
(post-exp2). WRITEUP.md §6/§7 carry the should-continue verdict.

## Iteration 28 (2026-06-16) — Step 1: literal_lift (constant-lift from Reduce group keys)

Implemented rebuild/literal_lift.rs: lifts literal-Ok group-key entries out of
Reduce (drop from group key, Map back, Project to restore order). Narrows the
group key and its inferred unique key; targets the weak-key (replacement-mv
{a,c,d} -> {a,c}) and part of the distinct +233. Local, no provenance; only
literal-Ok lifted (errors stay) so error behaviour unchanged. Wired AFTER
reduce_inline (so Map-fed constants are materialised into the group key first).
30/30 rebuild unit tests (new: lifts_literal_group_key_and_narrows_key,
leaves_literal_free_reduce_alone).

Baseline corpus_exp2.md (post reduce-elision, tip 7a41e362b1): pass=368
reg_explain=89 reg_result=1 (replacement-mv), distinct delta +233. Measurement
sweep launched (b0prkn62d). Expect reg_result 1->0 (weak key closed) + some
distinct chip. Commit if it helps + rails hold (reg_result<=prev, panic=0).
Then Step 2: provenance-based elimination USING the existing Provenance analysis
(reuse, not bespoke) as a legibility/hygiene test.
