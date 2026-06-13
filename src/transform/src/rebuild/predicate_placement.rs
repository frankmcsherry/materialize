// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Predicate placement: each conjunct evaluates as low as the rules allow.
//!
//! See play/optimizer-rebuild/design-predicate-placement.md for the design.
//! One top-down pass per body: `Filter`s dissolve into a pending set of
//! conjuncts carried downward; at each operator a conjunct descends
//! (possibly rewritten), splits across `Union` branches, or is deposited in
//! a `Filter` above the operator where it stopped.
//!
//! **Error discipline (rule 1).** A conjunct may descend through an operator
//! iff the row set it would evaluate on below is a subset of the row set
//! above — true of Map/Project/Filter/Negate/Threshold and of each `Union`
//! branch — or the conjunct is infallible (`!could_error`), which licenses
//! descent through superset boundaries: a `Join` input, a `Reduce` or
//! `TopK` group key, a `FlatMap` input.
//!
//! **Contract.** Requires: a [`BindingEnv`] with conjunction-free `Filter`s
//! (filter_split). Ensures: conjunction-free `Filter`s at the lowest
//! permitted positions; M-F-P content otherwise untouched.

use mz_expr::visit::{Visit, VisitChildren};
use mz_expr::{Columns, Eval, MirRelationExpr, MirScalarExpr};

use crate::rebuild::env::BindingEnv;

/// Applies the transform to every binding body and the root.
pub fn apply(env: &mut BindingEnv) {
    for (_id, body) in env.bindings.iter_mut() {
        place(body, Vec::new());
    }
    place(&mut env.root, Vec::new());
}

/// Pushes `pending` (conjuncts over `expr`'s output columns) into `expr` as
/// far as the rules allow, depositing whatever stops here.
fn place(expr: &mut MirRelationExpr, mut pending: Vec<MirScalarExpr>) {
    match expr {
        MirRelationExpr::Filter { input, predicates } => {
            // Dissolve: predicates join the pending set; the Filter node
            // disappears (deposits will recreate what stops).
            pending.append(predicates);
            let mut inner = std::mem::replace(
                input,
                Box::new(MirRelationExpr::constant(
                    vec![],
                    mz_repr::ReprRelationType::new(vec![]),
                )),
            );
            place(&mut inner, pending);
            *expr = *inner;
        }
        MirRelationExpr::Project { input, outputs } => {
            // Remap conjunct columns through the projection; all descend.
            for p in pending.iter_mut() {
                remap_columns(p, |c| outputs[c]);
            }
            place(input, pending);
        }
        MirRelationExpr::Map { input, scalars: _ } => {
            let input_arity = input.arity();
            let (down, stay): (Vec<_>, Vec<_>) = pending
                .drain(..)
                .partition(|p| p.support().iter().all(|c| *c < input_arity));
            place(input, down);
            deposit(expr, stay);
        }
        MirRelationExpr::Union { base, inputs } => {
            // Copy every conjunct into every branch (each branch's rows are
            // a subset of the rows above: safe even for fallible conjuncts).
            place(base, pending.clone());
            for input in inputs.iter_mut() {
                place(input, pending.clone());
            }
        }
        MirRelationExpr::Join {
            inputs,
            equivalences: _,
            implementation: _,
        } => {
            // Column ranges per input.
            let arities: Vec<usize> = inputs.iter().map(|i| i.arity()).collect();
            let mut starts = Vec::with_capacity(arities.len());
            let mut acc = 0;
            for a in &arities {
                starts.push(acc);
                acc += a;
            }
            let mut per_input: Vec<Vec<MirScalarExpr>> =
                inputs.iter().map(|_| Vec::new()).collect();
            let mut stay = Vec::new();
            for mut p in pending.drain(..) {
                let support = p.support();
                let home = (0..inputs.len()).find(|i| {
                    support
                        .iter()
                        .all(|c| *c >= starts[*i] && *c < starts[*i] + arities[*i])
                });
                match home {
                    // Superset boundary: infallible conjuncts only.
                    Some(i) if !p.could_error() => {
                        remap_columns(&mut p, |c| c - starts[i]);
                        per_input[i].push(p);
                    }
                    _ => stay.push(p),
                }
            }
            for (input, downs) in inputs.iter_mut().zip(per_input) {
                place(input, downs);
            }
            deposit(expr, stay);
        }
        MirRelationExpr::Reduce {
            input, group_key, ..
        } => {
            let key_arity = group_key.len();
            let (down, stay): (Vec<_>, Vec<_>) = pending
                .drain(..)
                .partition(|p| !p.could_error() && p.support().iter().all(|c| *c < key_arity));
            // Substitute key expressions for key-column references.
            let mut down_sub = Vec::with_capacity(down.len());
            for mut p in down {
                substitute_columns(&mut p, group_key);
                down_sub.push(p);
            }
            place(input, down_sub);
            deposit(expr, stay);
        }
        MirRelationExpr::TopK {
            input, group_key, ..
        } => {
            let (down, stay): (Vec<_>, Vec<_>) = pending.drain(..).partition(|p| {
                !p.could_error() && p.support().iter().all(|c| group_key.contains(c))
            });
            place(input, down);
            deposit(expr, stay);
        }
        MirRelationExpr::Negate { input } | MirRelationExpr::Threshold { input } => {
            place(input, pending);
        }
        MirRelationExpr::FlatMap { input, .. } => {
            let input_arity = input.arity();
            let (down, stay): (Vec<_>, Vec<_>) = pending
                .drain(..)
                .partition(|p| !p.could_error() && p.support().iter().all(|c| *c < input_arity));
            place(input, down);
            deposit(expr, stay);
        }
        // Leaves and opaque nodes: deposit everything; recurse nowhere.
        other => {
            // Still visit children of operators not handled above (e.g.
            // ArrangeBy, LetRec) with empty pending, so nested Filters are
            // also placed.
            other.visit_mut_children(|child| place(child, Vec::new()));
            let stay = std::mem::take(&mut pending);
            deposit(other, stay);
        }
    }
}

/// Wraps `expr` in a `Filter` holding `conjuncts`, if any.
fn deposit(expr: &mut MirRelationExpr, conjuncts: Vec<MirScalarExpr>) {
    if !conjuncts.is_empty() {
        let inner = std::mem::replace(
            expr,
            MirRelationExpr::constant(vec![], mz_repr::ReprRelationType::new(vec![])),
        );
        *expr = inner.filter(conjuncts);
    }
}

/// Rewrites every column reference in `e` via `f`.
pub(crate) fn remap_columns<F: Fn(usize) -> usize>(e: &mut MirScalarExpr, f: F) {
    e.visit_mut_post(&mut |node| {
        if let MirScalarExpr::Column(c, _) = node {
            *c = f(*c);
        }
    });
}

/// Replaces column `i` references in `e` with `exprs[i]`.
fn substitute_columns(e: &mut MirScalarExpr, exprs: &[MirScalarExpr]) {
    e.visit_mut_post(&mut |node| {
        if let MirScalarExpr::Column(c, _) = node {
            *node = exprs[*c].clone();
        }
    });
}

#[cfg(test)]
mod tests {
    use mz_expr::func;
    use mz_repr::{Datum, ReprRelationType, ReprScalarType};

    use super::*;
    use crate::rebuild::env::BindingEnv;

    fn table2() -> MirRelationExpr {
        MirRelationExpr::constant(
            vec![],
            ReprRelationType::new(vec![ReprScalarType::Int64.nullable(false); 2]),
        )
    }

    fn gt0(col: usize) -> MirScalarExpr {
        MirScalarExpr::column(col).call_binary(
            MirScalarExpr::literal_ok(Datum::Int64(0), ReprScalarType::Int64),
            func::Gt,
        )
    }

    #[mz_ore::test]
    fn pushes_through_join_to_home_input() {
        let join = MirRelationExpr::join(vec![table2(), table2()], vec![vec![(0, 0), (1, 0)]]);
        let mut env = BindingEnv {
            bindings: vec![],
            root: join.filter(vec![gt0(3)]), // col 3 lives in input 1
        };
        apply(&mut env);
        // The conjunct now sits on input 1, as column 1 there.
        let MirRelationExpr::Join { inputs, .. } = &env.root else {
            panic!("expected bare Join at root, got {:?}", env.root)
        };
        assert!(
            matches!(&inputs[1], MirRelationExpr::Filter { predicates, .. }
            if predicates == &vec![gt0(1)])
        );
    }

    #[mz_ore::test]
    fn copies_into_union_branches() {
        let union = table2().union(table2());
        let mut env = BindingEnv {
            bindings: vec![],
            root: union.filter(vec![gt0(0)]),
        };
        apply(&mut env);
        let MirRelationExpr::Union { base, inputs } = &env.root else {
            panic!("expected Union at root")
        };
        assert!(matches!(&**base, MirRelationExpr::Filter { .. }));
        assert!(matches!(&inputs[0], MirRelationExpr::Filter { .. }));
    }

    #[mz_ore::test]
    fn fallible_stays_above_join() {
        // 1/col stays above the join (could error), even though its support
        // is a single input.
        let div = MirScalarExpr::literal_ok(Datum::Int64(1), ReprScalarType::Int64)
            .call_binary(MirScalarExpr::column(0), func::DivInt64)
            .call_binary(
                MirScalarExpr::literal_ok(Datum::Int64(0), ReprScalarType::Int64),
                func::Gt,
            );
        let join = MirRelationExpr::join(vec![table2(), table2()], vec![vec![(0, 0), (1, 0)]]);
        let mut env = BindingEnv {
            bindings: vec![],
            root: join.filter(vec![div.clone()]),
        };
        apply(&mut env);
        assert!(
            matches!(&env.root, MirRelationExpr::Filter { predicates, .. }
            if predicates == &vec![div])
        );
    }
}
