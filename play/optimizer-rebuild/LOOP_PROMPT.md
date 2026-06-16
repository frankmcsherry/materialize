# Optimizer rebuild — weekend loop prompt

You are building a **replacement for an optimizer entry point** of Materialize's
MIR optimizer, from scratch. The current optimizer works but is a tangle:
phase-ordering chaos, transforms registered multiple times because no single
position is trusted, fixpoints with `limit: 100` standing in for confluence
arguments, and implicit normal-form contracts written down nowhere. Rather than
poke at it locally (which spirals), remove something large and rebuild it
aimed at the same intended properties, without inheriting the quirks.

## Target and scope

- Replace the body of **`Optimizer::logical_optimizer`** (src/transform/src/lib.rs)
  first. If that goes well, `logical_cleanup_pass` and `physical_optimizer` are
  next, in that order. Whole-optimizer replacement is in scope if the structure
  you build naturally extends; do not force it.
- The signature stays the same: a transform of `MirRelationExpr`, with side
  information available through `TransformCtx` and the `Analysis` framework
  (`Derived` / `DerivedBuilder` — see `analysis.rs`; this framework is the
  modern substrate and ages well, lean on it).
- New code lives in `src/transform/src/rebuild/` (module per transform), with a
  selector so both pipelines coexist: environment variable
  `MZ_OPTIMIZER_REBUILD=logical` switches the entry point to yours. Never modify
  the existing pipeline's behavior when the switch is off.
- MIR-logical only. No changes to join implementation selection or physical
  planning until the logical rebuild has a clean scorecard. The rebuilt
  logical pipeline feeds the EXISTING physical optimizer: its final
  transforms must establish whatever normal forms the physical pipeline
  implicitly requires (discover and document these in Phase 0 — they are the
  most load-bearing contracts of all).
- Improving shared `Analysis` implementations in place is allowed and
  encouraged when they are weak or inconsistent — both pipelines benefit, and
  such improvements are independently mergeable. Keep them in separate
  commits from the rebuild itself.
- Pin the base: branch from origin/main at the start of the weekend and do
  not chase main until the Monday writeup.

## Semantic rules (non-negotiable)

1. **No new errors.** The rebuilt pipeline must not introduce erroring
   computations the input plan could not produce. This includes errors via
   *synthesized arithmetic*: if a rewrite computes a derived quantity (a
   cardinality, a moved bound), the synthesized expression must be infallible
   on every input where the original was — compute exactly at rewrite time and
   decline otherwise (worked example: the all-literal generate_series fold and
   its decline-on-overflow; see memory and PR #36776).
2. **Filters commute.** WHERE/HAVING conjuncts may move and reorder freely.
   This is licensed by existing semantics, not a new permission: `And::eval`
   returns false when any conjunct is false even if a sibling errored
   (src/expr/src/scalar/func/variadic.rs), so error visibility is already
   order-independent. The "conjoin, don't replace" pattern (derive an implied
   infallible predicate, keep the original) buys placement freedom with exact
   error parity when you need it.
3. **`If` is the only user-facing evaluation guard.** Never elide or hoist
   evaluation across an `If` test unless you can prove the guarded expression
   cannot error on the rows that reach it.
4. **LetRec uses synchronous update semantics.** Unlike WITH MUTUALLY
   RECURSIVE's surface presentation (bindings update in turn), MIR `LetRec`
   updates all bindings atomically: a name denotes the same collection at
   every use site, independent of position in scope. Equality of expressions
   is therefore unconditional with respect to position — exploit this for CSE
   and equality reasoning, and do not import sequential-semantics caution that
   does not apply.
5. Plans must typecheck between transforms (keep `Typecheck` in the loop in
   strict mode while developing).

## Architecture requirements

- **Factored, not monolithic.** The ideal shape is many small transforms, each
  embodying exactly one idea: a normalization per form, intro/outro rules per
  form, strength reduction per form — rather than conflations of concerns
  where it was expedient. The output is judged on **legibility and
  maintainability**: a reviewer should be able to read any one transform,
  state its idea in a sentence, and believe its correctness argument. (If it
  cannot plausibly be merged it was entertainment; but do not optimize
  exclusively for mergeability either.)
- **Confluence over independence.** Every transform declares explicit
  contracts: `requires` (normal forms it assumes) and `ensures` (normal forms
  it establishes/preserves), as doc comments AND debug assertions. Pipeline
  order must be *derivable* from the contracts. No fixpoint without a written
  termination-and-convergence argument; `limit: 100` is the smell we are
  removing, not a tool.
- **Analyses for side information.** Pure, memoized, composable analyses
  (arity, types, unique keys, non-negativity, equivalences, cardinality
  bounds) feed transforms; transforms do not recompute global facts ad hoc.
  Corollary: `typ()` is called ad hoc everywhere and must stay cheap —
  expensive inference belongs in the memoized Analysis layer, never inside
  `typ()`. (Exhibit: PR #28867's generalized Union key detection inside
  `typ()` turned NormalizeLets into four ~100-second calls on a 25-way
  union chain; the identical logic as a memoized Analysis would be linear.)
- **From-scratch discipline.** When building a transform: first write down,
  in your own words in the journal, the *idea* of any existing counterpart,
  plus its known quirks (read its comments, its bug history via git log, the
  optimizer-probe findings that touch it). Then implement from your written
  idea — not from the code. Learn from the existing ones; do not copy their
  problems.
- **e-graphs / equality saturation:** consider explicitly and write an
  assessment (fit, cost, what the Analysis framework would become). Rule 4
  (referentially transparent names) removes the main conditional-equality
  obstacle, which is suggestive. Adopt if you judge a minimal version
  tractable within the weekend; otherwise design the transform set so a later
  e-graph core could subsume it, and say how.

## Method

**Phase 0 — capability map (bounded: one day at most).** Mine, do not build:
read the entry point's composition once; inventory what each registered
transform contributes; cross-reference the optimizer-probe corpus
(~/OrbStack/mz-claude-0/home/frankmcsherry/projects/materialize/play/optimizer-probe/index.md)
as the behavior checklist. Ablation experiments on the existing pipeline are
allowed here *for learning only* — disable things, observe slt fallout, record
the implicit contracts you discover. **Epistemics of ablation:** it reveals
what the optimizer *happens to do*, never what it *must* do. The language is
declarative and leaves great flexibility; that the current pipeline does not
do something is not evidence that it should not be done, and several existing
analyses are weak or inconsistent — improving on them is good, not scope
creep. Treat ablation output as a floor of capabilities and a map of hazards,
not as a specification. The replacement itself starts EMPTY and earns every
transform; do not arrive at it by thinning the existing pipeline.

**Phase 1 — skeleton + harness.** The switch, an entry point that is just
Typecheck + NormalizeLets-equivalent-of-your-design (you must take a position
on the Let/sharing story first; it is the substrate everything else stands
on), and the evaluation harness below, running end to end.

**Phase 2..n — earn transforms.** Repeat: run the corpus; pick the largest
plan-quality gap vs main; write down the idea; implement the transform with
contracts and tests; re-run; commit with a journal entry. Prefer the broad
ideas (predicate placement via equivalence saturation, redundancy elimination,
reduction pushdown/elision, distinctness reasoning) over long tails of local
simplifications; note the long tail for later instead of chasing it.

## Evaluation protocol (the continuation criterion)

The SLT corpus is the oracle. With `MZ_OPTIMIZER_REBUILD=logical`, run the
fast-slt file set; every divergence from checked-in expectations is a signal
to classify, never to silence:

- **Plan-quality comparison.** The real metric is predicted memory use;
  arrangement count is its everyday proxy. Score plans as: (1) estimated
  retained memory — sum over arrangements of (key arity + value arity),
  weighted by cardinality bounds where the Cardinality analysis offers one;
  (2) arrangement count as the tie-breaker and sanity proxy; (3) operator
  count. **Guardrail:** these measures can be gamed — a cross join of all
  inputs minimizes arrangements and storage while being catastrophic. Any
  plan that introduces a cross join (or drops a join key) absent from main's
  plan is plan-worse regardless of the scores. When the comparison is
  ambiguous under these rules, do not auto-judge: append the case to
  play/optimizer-rebuild/scorecard/JUDGMENT_QUEUE.md with both plans and your
  analysis, and move on. If main's plan is clearly better, the rebuild is
  wrong or missing a transform: fix it. If the rebuild's plan is equal or
  better, record the win in the scorecard.
- **Never rewrite checked-in slt files.** Capture failures, classify from the
  diffs, and keep results in play/optimizer-rebuild/scorecard/ (one file per
  sweep, plus a running summary table: files passed / plan-better /
  plan-worse / errors / not-yet-implemented).
- **Correctness:** result rows must match everywhere (a result diff is a stop-
  the-line bug in the rebuild); error-pinning tests must keep erroring.
- **Cost canaries:** recursion_limit.slt and the wide/deep-plan files must
  complete; judge optimizer cost by the internal optimizer_metrics
  per-transform traces, not wall clocks (dev-box wall time has ±30s noise;
  also note recursion_limit.slt stack-overflows DEV builds even on main —
  use it under optimized builds or EXPLAIN-only forms).

## Working agreements

- Worktree: .claude/worktrees/optimizer-rebuild, branch `optimizer-rebuild`,
  pushed to fork remote `me` for backup only — never to shared branches, no
  PRs, no comments on PRs.
- Journal: play/optimizer-rebuild/WORKLOG.md — every experiment gets an entry
  (intent, what happened, decision). Decisions of consequence (the Let story,
  the e-graph verdict, contract definitions) get their own files. Commit per
  experiment; messages explain *why*.
- Time-box: any single investigation showing no progress for ~90 minutes gets
  journaled and parked. The "note it, don't force it" rule from the PR-refresh
  sweep applies to everything.
- Long sweeps run backgrounded; use the wakeup loop with generous heartbeats;
  builds are ~5 minutes and full slt sweeps are hours — schedule accordingly.
- Record durable findings in memory (project type) as you go, so any session
  can resume.

## Monday deliverables

1. The capability map and discovered-contract notes (Phase 0 output).
2. `src/transform/src/rebuild/` with the switch, N transforms with contracts,
   each with datadriven tests and a boundary-spec slt under
   play/optimizer-rebuild/specs/ (slt-format, run manually — not in test/).
3. The scorecard: corpus sweep summary vs main, by the criteria above.
4. A writeup: the "few broad ideas" found, the long tail catalogued, the
   e-graph assessment, and a prioritized gap list with effort estimates.
5. An honest paragraph on whether this should continue, written for a reader
   deciding whether to fund the next weekend.
