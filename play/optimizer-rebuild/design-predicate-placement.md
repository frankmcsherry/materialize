# Design: predicate placement (transform #4, first broad idea)

## Idea (own words)
Each Filter conjunct should evaluate as close to the data as its column
dependencies and the error rules allow. Working form: binding env, bodies in
M-F-P normal form with conjunction-free Filters (ensured by linear_fuse +
filter_split). One top-down pass per body carries a set of pending conjuncts;
at each operator a conjunct either descends (possibly rewritten), splits
(Union), or is deposited in a Filter above the operator. Structural recursion,
no fixpoint; termination is by descent.

## Per-operator descent rules
- Filter: absorb (conjuncts commute — rule 2 license).
- Project: remap column refs through the projection; always descends.
- Map: descends if it references no map-produced columns; v1 does NOT
  substitute map expressions into conjuncts (growth control; revisit).
- Union: copy into every branch. Each branch's rows are a subset of what the
  conjunct would have seen above, so this is safe even for fallible
  conjuncts.
- Join: a conjunct whose support lies in one input descends to that input —
  but the input's row set is not a subset of the join output's, so only
  INFALLIBLE conjuncts descend here (rule 1). Equality conjuncts joining two
  inputs' columns are left in place in v1 (migration into join equivalences
  belongs to join planning). Multi-input conjuncts stay above.
- Reduce: descends iff support ⊆ group key columns (group membership is
  determined by the key, so filtering inputs by a key predicate filters
  exactly the dropped groups' rows); infallible-only (pre-aggregation rows
  are a superset of the groups' representative rows? No — every input row
  belongs to some output group, but groups dropped by the predicate
  contribute rows that would never have met the conjunct above; those rows
  DO evaluate it below. Superset ⇒ infallible-only).
- TopK: descends iff support ⊆ group columns; superset reasoning as Reduce ⇒
  infallible-only. (Within-group predicates must NOT descend past the limit.)
- Negate, Threshold: descend freely (pointwise in rows).
- FlatMap: descends iff support ⊆ input columns; the input rows are exactly
  the rows extended, and every input row reaches the conjunct above iff the
  func emits rows — emitting zero rows for an input row means the conjunct
  above never sees it ⇒ superset below ⇒ infallible-only.
- Get/Constant/ArrangeBy/LetRec(opaque): deposit.

## Error discipline summary (rule 1)
A conjunct may descend through an operator iff the row set it evaluates on
below is a subset of (or equal to) the set above, OR the conjunct is
infallible. Row-preserving: Map/Project/Filter/Negate/Threshold/Union-branch.
Superset (infallible-only): Join-input, Reduce, TopK, FlatMap.
The conjoin-don't-replace pattern is available later for fallible
sargable derivations; v1 keeps fallible conjuncts conservative.

## Ensures
Every conjunct sits at the lowest position permitted by the rules; Filters
remain conjunction-free (deposit re-splits); M-F-P normal form is preserved
by depositing as Filter nodes and re-running linear_fuse afterwards in the
sequence (ordering contract recorded in rebuild.rs).
