# Overnight plan — close the triaged plan-quality gaps

Goal: move the 38 "rebuild builds a bigger plan" files toward parity, **without
regressing correctness**. Triage (WRITEUP.md) says the gaps are bounded and named.

## Hard rails (never violate)
- `regress_result` must stay ≤ 1 and `panic` = 0. If any change increases
  wrong-results, **revert it** — correctness is not tradeable for plan size.
- Each change behind `MZ_OPTIMIZER_REBUILD=logical`, on branch `optimizer-rebuild`,
  committed + journaled (WORKLOG.md) with before/after numbers.
- One change at a time; measure before/after; don't force an unclear change —
  note it and move on.

## Measurement (minimize macOS firewall prompts)
- Use `batched-sweep.sh` (ALL files in one invocation; recursion_limit excluded).
  ~2 prompts per full sweep, not ~680. Tests complete regardless of the prompts
  (clusterd uses loopback, which the firewall doesn't gate) — expect a stack of
  harmless, dismissable dialogs by morning.
- Compare `corpus.md` vs `corpus_prefix.md`; re-run the node-delta analysis
  (Distinct/Reduce/Join/ArrangeBy, rebuild vs main) to see the 38 move.

## Experiment sequence (stop early if a rail trips)
1. **`Get`/`Let` type refresh** (the dominant hypothesis; cheap, high-leverage).
   Add a pass that refreshes each `Get`'s recorded type from its binding body's
   current `typ()` (or re-run `NormalizeLets`), placed before the key-using
   transforms (esp. `reduce_elision`) and after the key-establishing ones
   (`join_flatten`, etc.). Re-sweep. Expect: Distinct/Reduce/arrangement deltas
   shrink; `replacement-mv` precision gap may also close (same stale-key root).
   **This is the experiment that decides the rest** — if it lands big, the gap was
   staleness; if not, the gap is deeper (missing rewrites) and we re-cost.
2. **Semijoin idempotence** — only the residual after (1). A genuine missing
   rewrite (A⋈A on a key = A). Behind the flag; measure.
3. **RedundantJoin elimination** — genuinely unimplemented (confirmed on
   redundant_join.slt). Self-contained; measure.
4. Re-run the full triage analysis; update WRITEUP.md §2/§4 with the new numbers
   and a revised should-continue note.

## Per-iteration rhythm
edit → `cargo test -p mz-transform --lib rebuild` → `batched-sweep.sh` →
compare deltas + confirm rails → commit + journal → next. Surface findings after
each experiment (don't silently chain).
