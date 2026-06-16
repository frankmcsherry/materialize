# Capability map — Phase 0

## Composition tree (as registered, base b5df9fd7)

### logical_optimizer
```
    pub fn logical_optimizer(ctx: &mut TransformCtx) -> Self {
        let transforms: Vec<Box<dyn Transform>> = transforms![
            // 0. `Transform`s that don't actually change the plan.
            Box::new(Typecheck::new(ctx.typechecking_context()).strict_join_equivalences()),
            Box::new(CollectNotices),
            // 1. Structure-agnostic cleanup
            Box::new(normalize()),
            Box::new(NonNullRequirements::default()),
            // 2. Collapse constants, joins, unions, and lets as much as possible.
            // TODO: lift filters/maps to maximize ability to collapse
            // things down?
            Box::new(fuse_and_collapse_fixpoint()),
            // 3. Needs to happen before LiteralLifting, EquivalencePropagation
            // make (literal) filters look more complicated than what the NonNegative Analysis can
            // recognize.
            Box::new(ThresholdElision),
            // 4. Move predicate information up and down the tree.
            //    This also fixes the shape of joins in the plan.
            Box::new(Fixpoint {
                name: "fixpoint_logical_01",
                limit: 100,
                transforms: vec![
                    // Predicate pushdown sets the equivalence classes of joins.
                    Box::new(PredicatePushdown::default()),
                    Box::new(EquivalencePropagation::default()),
                    // Lifts the information `col1 = col2`
                    Box::new(Demand::default()),
                    Box::new(FuseAndCollapse::default()),
                ],
            }),
            // 5. Reduce/Join simplifications.
            Box::new(Fixpoint {
                name: "fixpoint_logical_02",
                limit: 100,
                transforms: vec![
                    Box::new(SemijoinIdempotence::default()),
                    // Pushes aggregations down
                    Box::new(ReductionPushdown),
                    // Replaces reduces with maps when the group keys are
                    // unique with maps
                    Box::new(ReduceElision),
                    // Rips complex reduces apart.
                    Box::new(ReduceReduction),
                    // Converts `Cross Join {Constant(Literal) + Input}` to
                    // `Map {Cross Join (Input, Constant()), Literal}`.
                    // Join fusion will clean this up to `Map{Input, Literal}`
                    Box::new(LiteralLifting::default()),
                    // Identifies common relation subexpressions.
                    Box::new(cse::relation_cse::RelationCSE::new(false)),
                    Box::new(FuseAndCollapse::default()),
                ],
            }),
            Box::new(
                Typecheck::new(ctx.typechecking_context())
                    .disallow_new_globals()
                    .strict_join_equivalences()
            ),
        ];
        Self {
            name: "logical",
            transforms,
        }
    }
```
### normalize()
```
pub fn normalize() -> Fixpoint {
    Fixpoint {
        name: "normalize",
        limit: 100,
        transforms: vec![Box::new(NormalizeLets::new(false)), Box::new(NormalizeOps)],
    }
}
```
### FuseAndCollapse::default()
```
impl Default for FuseAndCollapse {
    fn default() -> Self {
        Self {
            // TODO: The relative orders of the transforms have not been
            // determined except where there are comments.
            // TODO (database-issues#2036): All the transforms here except for `ProjectionLifting`
            //  and `RedundantJoin` can be implemented as free functions.
            transforms: vec![
                Box::new(canonicalization::ProjectionExtraction),
                Box::new(movement::ProjectionLifting::default()),
                Box::new(fusion::Fusion),
                Box::new(canonicalization::FlatMapElimination),
                Box::new(fusion::join::Join),
                Box::new(NormalizeLets::new(false)),
                Box::new(fusion::reduce::Reduce),
                Box::new(WillDistinct),
                Box::new(compound::UnionNegateFusion),
                // This goes after union fusion so we can cancel out
                // more branches at a time.
                Box::new(UnionBranchCancellation),
                // This should run before redundant join to ensure that key info
                // is correct.
                Box::new(NormalizeLets::new(false)),
                // Removes redundant inputs from joins.
                // Note that this eliminates one redundant input per join,
                // so it is necessary to run this section in a loop.
                Box::new(RedundantJoin::default()),
                // As a final logical action, convert any constant expression to a constant.
                // Some optimizations fight against this, and we want to be sure to end as a
                // `MirRelationExpr::Constant` if that is the case, so that subsequent use can
                // clearly see this.
                Box::new(fold_constants_fixpoint(true)),
            ],
        }
    }
}
```
### fuse_and_collapse_fixpoint()
```
pub fn fuse_and_collapse_fixpoint() -> Fixpoint {
    Fixpoint {
        name: "fuse_and_collapse_fixpoint",
        limit: 100,
        transforms: FuseAndCollapse::default().transforms,
    }
}
```
### logical_cleanup_pass + physical_optimizer (downstream contracts)
```
    pub fn logical_cleanup_pass(ctx: &mut TransformCtx, allow_new_globals: bool) -> Self {
        let mut repr_typechecker =
            Typecheck::new(ctx.typechecking_context()).strict_join_equivalences();
        if !allow_new_globals {
            repr_typechecker = repr_typechecker.disallow_new_globals();
        }

        let transforms: Vec<Box<dyn Transform>> = transforms![
            Box::new(repr_typechecker),
            // Delete unnecessary maps.
            Box::new(fusion::Fusion),
            Box::new(Fixpoint {
                name: "fixpoint_logical_cleanup_pass_01",
                limit: 100,
                transforms: vec![
                    Box::new(CanonicalizeMfp),
                    // Remove threshold operators which have no effect.
                    Box::new(ThresholdElision),
                    // Projection pushdown may unblock fusing joins and unions.
                    Box::new(fusion::join::Join),
                    // Predicate pushdown required to tidy after join fusion.
                    Box::new(PredicatePushdown::default()),
                    Box::new(RedundantJoin::default()),
                    // Redundant join produces projects that need to be fused.
                    Box::new(fusion::Fusion),
                    Box::new(compound::UnionNegateFusion),
                    // This goes after union fusion so we can cancel out
                    // more branches at a time.
                    Box::new(UnionBranchCancellation),
                    // The last RelationCSE before JoinImplementation should be with
                    // inline_mfp = true.
                    Box::new(cse::relation_cse::RelationCSE::new(true)),
                    Box::new(fold_constants_fixpoint(true)),
                ],
            }),
            Box::new(
                Typecheck::new(ctx.typechecking_context())
                    .disallow_new_globals()
                    .strict_join_equivalences()
            ),
        ];
        Self {
            name: "logical_cleanup",
            transforms,
        }
    }
    pub fn physical_optimizer(ctx: &mut TransformCtx) -> Self {
        // Implementation transformations
        let transforms: Vec<Box<dyn Transform>> = transforms![
            Box::new(
                Typecheck::new(ctx.typechecking_context())
                    .disallow_new_globals()
                    .strict_join_equivalences(),
            ),
            // Considerations for the relationship between JoinImplementation and other transforms:
            // - there should be a run of LiteralConstraints before JoinImplementation lifts away
            //   the Filters from the Gets;
            // - there should be no RelationCSE between this LiteralConstraints and
            //   JoinImplementation, because that could move an IndexedFilter behind a Get.
            // - The last RelationCSE before JoinImplementation should be with inline_mfp = true.
            // - Currently, JoinImplementation can't be before LiteralLifting because the latter
            //   sometimes creates `Unimplemented` joins (despite LiteralLifting already having been
            //   run in the logical optimizer).
            // - Not running EquivalencePropagation in the same fixpoint loop with JoinImplementation
            //   is slightly hurting our plans. However, I'd say we should fix these problems by
            //   making EquivalencePropagation (and/or JoinImplementation) smarter (database-issues#5289), rather than
            //   having them in the same fixpoint loop. If they would be in the same fixpoint loop,
            //   then we either run the risk of EquivalencePropagation invalidating a join plan (database-issues#5260),
            //   or we would have to run JoinImplementation an unbounded number of times, which is
            //   also not good database-issues#4639.
            //   (The same is true for FoldConstants, Demand, and LiteralLifting to a lesser
            //   extent.)
            //
            // Also note that FoldConstants and LiteralLifting are not confluent. They can
            // oscillate between e.g.:
            //         Constant
            //           - (4)
            // and
            //         Map (4)
            //           Constant
            //             - ()
            Box::new(Fixpoint {
                name: "fixpoint_physical_01",
                limit: 100,
                transforms: transforms![
                    Box::new(EquivalencePropagation::default()),
                    Box::new(fold_constants_fixpoint(true)),
                    Box::new(coalesce_case::CoalesceCase::default());
                        if ctx.features.enable_coalesce_case_transform,
                    Box::new(Demand::default()),
                    // Demand might have introduced dummies, so let's also do a ProjectionPushdown.
                    Box::new(ProjectionPushdown::default()),
                    Box::new(LiteralLifting::default()),
                ],
            }),
            Box::new(LiteralConstraints),
            Box::new(Fixpoint {
                name: "fixpoint_join_impl",
                limit: 100,
                transforms: vec![Box::new(JoinImplementation::default())],
            }),
            Box::new(CanonicalizeMfp),
            // Identifies common relation subexpressions.
            Box::new(cse::relation_cse::RelationCSE::new(false)),
            // `RelationCSE` can create new points of interest for `ProjectionPushdown`: If an MFP
            // is cut in half by `RelationCSE`, then we'd like to push projections behind the new
            // Get as much as possible. This is because a fork in the plan involves copying the
            // data. (But we need `ProjectionPushdown` to skip joins, because it can't deal with
            // filled in JoinImplementations.)
            Box::new(ProjectionPushdown::skip_joins());
                if ctx.features.enable_projection_pushdown_after_relation_cse,
            // Plans look nicer if we tidy MFPs again after ProjectionPushdown.
            Box::new(CanonicalizeMfp);
                if ctx.features.enable_projection_pushdown_after_relation_cse,
            // Rewrite If-chains matching a single expression against literals
            // into a CaseLiteral lookup for O(log n) evaluation.
            Box::new(case_literal::CaseLiteralTransform);
                if ctx.features.enable_case_literal_transform,
            // Do a last run of constant folding. Importantly, this also runs `NormalizeLets`!
            // We need `NormalizeLets` at the end of the MIR pipeline for various reasons:
            // - The rendering expects some invariants about Let/LetRecs.
            // - `CollectIndexRequests` needs a normalized plan.
            //   https://github.com/MaterializeInc/database-issues/issues/6371
            Box::new(fold_constants_fixpoint(true)),
            Box::new(
                Typecheck::new(ctx.typechecking_context())
                    .disallow_new_globals()
                    .disallow_dummy()
                    .strict_join_equivalences(),
            ),
        ];
        Self {
            name: "physical",
            transforms,
        }
    }
```

## Per-transform inventory (agent-mined, iteration 2)

### Bookkeeping
- **Typecheck** (x2: start permissive, end disallow-new-globals + strict join equivalences). Validation only.
- **CollectNotices**: scans for user-mistake patterns (`= NULL`); records notices; no plan change.

### Normalization
- **NormalizeLets** (runs 6+ times): hoists Let/LetRec to scope root, canonical order, renumbered ids; `inline_mfp` flag normally false except before JoinImplementation. CONTRACTS MINED: must rerun "before RedundantJoin to ensure key info is correct"; must rerun after FoldConstants which "can break normalized form by removing all references to a Let". This is the substrate transform.
- **NormalizeOps**: bottom-up FlatMapElimination+Fusion+join-fusion+ProjectionExtraction; partial collapse only.
- **CanonicalizeMfp** (in cleanup/physical phases): MFP canonical M-F-P order. Position-sensitive (see #28867 exhibit).

### Local simplification
- **ProjectionExtraction**, **ProjectionLifting** (TODO admits relative order undetermined), **Fusion** (per-operator-kind fusing), **FlatMapElimination** (literal-arg table funcs -> Map/empty), **fusion::reduce::Reduce** (nested Reduce fusing, conservative), **UnionNegateFusion** (flatten unions, distribute Negate), **UnionBranchCancellation** (Union(X, Negate(X)) cancel; safe via same-Let-binding reasoning), **FoldConstants** (size-limited; Let-propagation deferred to NormalizeLets, TODO database-issues#5346), **ReduceScalars** (skips ArrangeBy exprs), **ThresholdElision** (NonNegative analysis), **ReduceElision** (UniqueKeys analysis), **ReduceReduction** (split mixed aggregate classes).
- **fusion::join::Join**: fuses to multiway joins; QUIRK: re-runs PredicatePushdown + CanonicalizeMfp ad hoc inside itself when it fires (database-issues#7728: "temporary").

### Movement
- **LiteralLifting**: hoists literals out of Maps through operators. QUIRK: documented oscillation with FoldConstants (Constant <-> Map(Constant)) in physical pipeline.
- **Demand**: dummy-column substitution for joins; "mostly obsoleted by ProjectionPushdown but still does one unique optimization".
- **NonNullRequirements**: pushes non-null constraints toward sources, prunes contradicted branches.

### Broad ideas (the short list)
1. **Predicate placement**: PredicatePushdown (also sets join equivalence classes — dual role!).
2. **Equivalence reasoning**: EquivalencePropagation (Analysis-based; falls back to ColumnKnowledge comparison path).
3. **Aggregation placement**: ReductionPushdown (push Reduce through Join under coverage conditions).
4. **Distinctness reasoning**: WillDistinct (flag-gated, NonNegative-based).
5. **Semijoin idempotence / redundancy**: SemijoinIdempotence (requires renumbered bindings), RedundantJoin (one elimination per pass — hence fixpoint).
6. **Sharing**: RelationCSE (ANF then NormalizeLets).

### Structural statistics
- 25 distinct transforms; 4 fixpoints (normalize, fuse_and_collapse, logical_01, logical_02), all limit 100.
- Repeat registrations: NormalizeLets 6+, FuseAndCollapse 4 (as fixpoint + in both logical fixpoints), Typecheck 2.
- Analyses in use: NonNegative, UniqueKeys, Equivalences, SubtreeSize, ReprRelationType.

## Physical-boundary contracts (agent-mined, iteration 3) — what the rebuilt logical output MUST establish

Highest-impact (panic-if-violated):
1. join.implementation filled before LIR lowering (lowering.rs:727) — but this is PHYSICAL's job; the logical boundary contract is rather: joins in canonicalized-equivalences form with strict nullability (Typecheck::strict_join_equivalences at physical entry).
2. Let/LetRec normal form: all Lets at scope root, ordered/renumbered ids — Get binding lookups panic otherwise; LetRec bindings must expose raw (unarranged) collections (lowering.rs:382 unwrap).
3. No Dummy datums or unbound locals escaping the logical phase (Typecheck disallow_dummy / assert on unbound Gets); Demand introduces Dummies → ProjectionPushdown must remove them.
4. MFPs in canonical Map-Filter-Project sequence at lowering boundaries; "This operator should have been extracted" panics in lowering.rs:195+.
5. Physical-internal ordering hazards a rebuild must not recreate at its boundary: LiteralConstraints-before-JoinImplementation; NO RelationCSE between them (IndexedFilter visibility); CanonicalizeMfp immediately after JoinImplementation; final NormalizeLets+FoldConstants after everything ("rendering expects invariants").

Full agent table preserved in WORKLOG references; severities: PANIC (binding lookups, unimplemented join, dummy escape, LetRec raw-arrangement), SOFT_PANIC+fallback (delta-join arrangements absent), WRONG_PLAN (MFP canonicality, equivalence canonicalization, arrangement/projection interaction).

## Probe cross-reference (iteration 4; caveat: agent leaned partly on session memory rather than pure index mining — consult the probe index directly when building each transform)

Gap clusters worth systematic treatment (not pattern-by-pattern): null-guarded constant-result folds; monotonicity metadata completeness; predicate implication via inverse/preimage reasoning (difference=0, LIKE-exact, injective cast inversion, cast-implied ranges); temporal constant movement; any/all-to-IN.

Architecture hints adopted into planning:
1. **Inverse/preimage registry**: generalize unary inverse() (left-inverse gated on preserves_uniqueness, audited 2026-06-12) to binary-with-argument-index; drives implication rewrites systematically.
2. **Interval/range lattice as an Analysis**: bounds on scalars; gates monotonicity reasoning, temporal constant moves, cast-implied ranges.
3. **Cardinality as a first-class hardened Analysis** (table funcs, cross products, constants) — count-aware reduction and collapse rewrites query it instead of synthesizing bounds ad hoc.
4. **Contracts document per sequence** with termination measures — the replacement for limit:100.

## Contracts discovered empirically (Phase 1 smoke)
- **Filter predicates must be conjunction-free** for temporal filter (mz_now)
  compilation at MFP lowering — PANIC at bootstrap otherwise. The existing
  pipeline establishes this implicitly via canonicalize_predicates. (Found
  iteration 6; rebuild/filter_split.rs owns it now.)
- **Lets at LetRec scope roots are legitimate Let normal form** — NormalizeLets
  hoists per-scope, not globally. Checkers must scope accordingly.
