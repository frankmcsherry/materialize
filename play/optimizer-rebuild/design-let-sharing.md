# Design position: the Let/sharing story

## Decision
The rebuild's working representation is a **flat binding environment**: at
entry, parse all Let/LetRec structure once into (bindings: ordered map
LocalId -> body, root), where every body is **Let-free**. Transforms operate
per-binding over Let-free trees. Sharing changes are **explicit operations**
(extract-binding, inline-binding, merge-equal-bindings), never a side effect
of normalization. At exit, emit nested Lets in exactly the physical
pipeline's contract form (root-hoisted, ordered, renumbered) — once.

## Why
- The mined pipeline runs NormalizeLets 6+ times because Let normal form is
  an *implicit* contract every transform can break and none owns. Making the
  normal form true *by construction* deletes the transform class entirely.
- Rule 4 (LetRec synchronous semantics: names denote the same collection at
  every use site) makes per-binding processing sound and equality of
  expressions position-independent — the property that makes a flat
  environment (and later, e-graph-style reasoning) work.
- Per-binding Let-free bodies give every transform a simpler input language
  (no Let arms at all) and give analyses a natural memoization boundary
  (per-binding results, invalidated on binding change) — structurally
  preventing the #28867 class of typ()-recomputation blowups.
- CSE becomes hash-consing over binding bodies plus merge-equal-bindings;
  RelationCSE's ANF-then-renormalize dance disappears.

## Alternatives considered
(a) Persistent Let-normal-form over the ordinary tree: still leaves every
    transform able to break it; contract enforcement by discipline, the
    thing we are escaping. Rejected.
(b) Re-normalize at phase boundaries (status quo shape): accepted cost today,
    known failure mode. Rejected for the rebuild.
(c) Aggressive early inlining, CSE late: duplicates work in analyses and
    risks exponential tree growth on diamond sharing. Rejected, though
    inline-binding remains available as an explicit local operation.

## Consequences / obligations
- Entry/exit converters are correctness-critical and need their own tests
  (round-trip identity on a corpus of plans, including LetRec scopes).
- LetRec: bindings carry their recursive scope structure in the environment
  (scope tree over binding ids); per-binding processing within a recursive
  scope must respect rule 4's atomic-update meaning when reasoning about
  cross-binding facts (document per analysis).
- The boundary emission owns the physical contracts: root-hoisted, ordered,
  renumbered ids, no unbound locals, no dummies.
