# Design: join_flatten (rebuild transform #5)

Goal: a single flat n-ary `Join` with merged, canonicalized equivalence
classes as the logical normal form for each join region. This is the
transform where arrangement counts move: nested binary joins each arrange
their intermediate, while a flat join lets the physical `JoinImplementation`
choose orders and share arrangements.

## Patterns consumed (one bottom-up pass per body)

A *join region* is a maximal tree of `Join`, `Filter`, and `Project` nodes.
The pass rebuilds each region into:

```
Project (restore outer column order)
  Filter (residue: non-equality conjuncts + synthesized IS NOT NULL guards)
    Join (flat inputs, merged equivalences)
```

Empty Filter/identity Project layers are elided.

### 1. Join-of-Join (associativity)

`Join(..., Join(inner_inputs, inner_eqs), ...)` splices `inner_inputs` in
place of the nested input. Column bookkeeping: inner equivalence members are
offset by the splice position; outer members at or beyond the spliced input
shift by `inner_arity - 1` input slots. The flat constraint set is the union
of inner and outer classes.

*Rows*: cross-product associativity plus the fact that the constraint set is
a conjunction over the flat product — grouping-insensitive, the same And
argument as filter_split.

*Errors*: constraint expressions can be fallible (`1/#x = #y`). Nesting
determines which candidate rows a constraint might be evaluated on, and the
flat form can evaluate one on rows the nested form had already eliminated.
MIR's Join contract leaves constraint evaluation order unspecified and
main's JoinFusion flattens unconditionally, so this is parity-with-main, not
a new hazard — but it IS a hazard, and the honest statement is: rule 1 holds
relative to main's semantics for Join, not relative to the nested rendering.
Documented here and in the module doc rather than hidden.

### 2. Filter over a join (equality absorption)

`Filter(Join)` inside a region: each conjunct of shape `e1 = e2` (top-level
`BinaryFunc::Eq`) is absorbed into the equivalence classes; everything else
stays in the residue Filter above.

**Null semantics.** A filter `e1 = e2` drops rows where either side is NULL
(the comparison returns NULL, filter keeps only TRUE). Join equivalences use
datum equality, where NULL matches NULL. Absorbing the conjunct therefore
requires synthesizing `e1 IS NOT NULL` (one side suffices: under datum
equality, if the sides are equal and one is non-null, both are) into the
residue Filter — exactly the guards main's PredicatePushdown emits. When the
type of `e1` is non-nullable, the guard is elided immediately.

Equality of fallible members: absorbing changes evaluation context the same
way pattern 1 does; same parity-with-main statement.

### 3. Project between joins

`Join(..., Project(inner, outputs), ...)` splices `inner` and rewrites
references through `outputs`. Dropped columns reappear in the flat join's
intermediate arity; the region's outer Project restores the original
signature, so downstream sees no change. Duplicated columns in `outputs`
become two references to one inner column — handled by the same remap.

### 4. Boundaries (not crossed)

- `Map` between joins: stop the region; v1 conservatism (substituting map
  scalars into equivalences trades clarity for little gain — revisit only if
  scorecards demand).
- `Negate`/`Threshold`/`Union`: outer-join lowering shapes; crossing them
  changes multiplicity semantics. Never crossed.
- `Get`/`Constant`/`Reduce`/etc.: region leaves (flat join inputs).
- Single-input joins (`Join([x], eqs)`) are unwrapped to `Filter(x)` with the
  equivalence constraints as equality conjuncts where possible.

## Canonicalization

After building each flat join, call `mz_expr::canonicalize::
canonicalize_equivalences` with the per-input column types. It dedups
classes, merges overlapping ones, and rewrites members toward minimal
complexity with a termination argument already proven in mz-expr — reuse,
don't reimplement. Cost note (the #28867 lesson): gather input types once
per flatten via one `typ()` call per flat input, never inside a loop over
classes.

## Ordering in the pipeline

```
linear_fuse → filter_split → join_flatten → predicate_placement → reduce_class_split
```

join_flatten runs BEFORE predicate_placement so that (a) its synthesized
IS NOT NULL guards and non-equality residue get routed into join inputs by
placement, and (b) placement's join rule sees the final flat input layout.
filter_split before it guarantees conjunction-free Filters, so equality
detection is per-conjunct with no And-walking.

## Ensures

- No nested Join directly under Join/Filter/Project within a region.
- Equivalence classes canonicalized; each class ≥ 2 members.
- Filters remain conjunction-free (residues are emitted split).
- Region output columns identical to input (outer Project restores order).

## Tests

- Nested binary joins flatten to one 3-ary join with merged classes.
- Equality filter absorbed; nullable side gets IS NOT NULL residue;
  non-nullable side gets no residue.
- Project-between-joins remap (drop + permute).
- Map boundary: region stops, no flatten across.
- Fallible non-equality conjunct stays in residue.
