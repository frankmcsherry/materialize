# Design: projection_thinning (rebuild transform #6)

Goal: every operator computes, stores, and transmits only the columns its
consumers demand. The scorecard motivation is cte width: main stores
narrow bindings and re-projects (even re-duplicates) columns at use sites,
while the rebuild currently arranges every column a binding happens to
produce (three joins.slt diffs show a 4-column arrangement where main has
3; tpch and subquery have analogous, wider cases).

## Shape

One pass per environment, processing consumers before producers. In the
flat [`BindingEnv`], binding `i` can only be used by bindings `j > i` and
the root, so the order is: root first (all of its output columns demanded),
then bindings from last to first, each rewritten against the union of
demands its use sites recorded.

`thin(expr, demand: &BTreeSet<usize>) -> Vec<usize>`: rewrites `expr` in
place so it produces exactly the demanded columns (ascending old-column
order), and returns the mapping from old columns to the new layout. The
caller patches its own column references through the returned mapping.
Top-level calls demand all columns, so plan signatures never change;
narrowing happens strictly inside.

## Per-operator demand rules

- **Project**: demand maps through `outputs`; the node's own projection
  composes with whatever the input returns. Duplicate references collapse
  (demand is a set) — this is what lets a use-site Project re-duplicate a
  column the binding stores once.
- **Map**: scalars partition into demanded and not; demanded scalars add
  their support to the input demand (transitively through scalars that
  reference other map columns). Undemanded scalars drop — including
  fallible ones: an expression no consumer observes only ever *removed*
  rows... it never removed rows; dropping it can only remove errors, which
  rule 1 permits. State this explicitly in the module doc.
- **Filter**: predicates' support adds to demand; predicates never drop.
- **Join**: demand splits per input by column ranges; every equivalence
  member's support adds to its input's demand. (Equivalence classes are
  never dropped here — that is join planning's concern, not demand's.)
- **Reduce**: group key support is always demanded of the input; aggregates
  partition into demanded and not, and undemanded aggregates DROP (the
  group structure persists — a Reduce emits one row per group regardless
  of how many aggregates it carries). This is where window-function and
  wide-aggregate plans narrow substantially.
- **TopK**: group, order, and limit-expression columns are demanded plus
  whatever the consumer demands; TopK emits its input's full row, so the
  input demand is the union, and a Project above restores the narrow set.
- **Union**: the same demand flows to every branch.
- **Negate/Threshold**: pass through.
- **FlatMap**: the function's arguments add to input demand; the function's
  output columns are fixed (drop none), but undemanded *input* columns
  still thin, with the function's column references patched.
- **Get (local)**: record demand against the binding (union across use
  sites); the use site rewrites to the binding's eventual layout, known
  because the binding is processed after all its consumers. Global `Get`s
  thin via a Project directly above.
- **Constant**: rows narrow to demanded columns.
- **ArrangeBy/LetRec and anything else**: conservative — demand everything
  below, thin nothing through the node. (ArrangeBy keys are arrangement
  contracts; LetRec is Phase-1 opaque.)

## Contract

Requires: BindingEnv (Let-free bodies; LetRec opaque). Ensures: no operator
produces a column no consumer can observe; binding bodies produce exactly
the union of their use sites' demands; signatures of root and bindings'
visible columns unchanged at use sites (compensating Projects). The M-F-P
normal form may be locally disturbed by inserted Projects — the pipeline
re-runs `linear_fuse` afterward rather than complicating this pass.

Pipeline: `linear_fuse → filter_split → join_flatten → predicate_placement
→ projection_thinning → linear_fuse → reduce_class_split`.

## Error discipline

The only deletions are (a) Map scalars and Reduce aggregates no consumer
observes, and (b) columns of Constants. (a) can only remove error
possibilities, never rows (Map/Reduce row counts don't depend on dropped
expressions); (b) is value-level. Nothing reorders or synthesizes
evaluation, so rule 1 holds without caveats.

## Tests

- Wide Map under narrow Project: undemanded (fallible) scalar drops.
- Binding used twice with different demands: body narrows to the union;
  each use site re-projects.
- Use-site duplicate columns (the joins.slt shape): binding stores one
  copy; the Project above the Get re-duplicates.
- Reduce with two aggregates, one demanded: aggregate drops, key stays.
- Demand through Join equivalences: member support stays alive.
