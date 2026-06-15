// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Union flattening and Negate-pair cancellation.
//!
//! Nested `Union`s splice into one (columns align, no remapping). Then a
//! branch `Negate(X)` cancels against a syntactically equal positive branch
//! `X`: as multisets, X - X = 0, exact for any X. Cancellation typically
//! becomes available only after reduce_elision normalizes both sides of an
//! outer-join default-row idiom to the same expression.
//!
//! Cancelled-to-nothing Unions become empty Constants of the same type
//! (one typ() call at that rare site).
//!
//! **Contract.** Requires: a [`BindingEnv`]. Local rewrites only; safe
//! inside LetRec values.

use mz_expr::MirRelationExpr;
use mz_expr::visit::VisitChildren;

use crate::rebuild::env::BindingEnv;

/// Applies the transform to every binding body and the root.
pub fn apply(env: &mut BindingEnv) {
    for (_id, body) in env.bindings.iter_mut() {
        apply_expr(body);
    }
    apply_expr(&mut env.root);
}

fn apply_expr(expr: &mut MirRelationExpr) {
    expr.visit_mut_children(apply_expr);
    if !matches!(expr, MirRelationExpr::Union { .. }) {
        return;
    }
    let typ_backup = expr.typ();
    let MirRelationExpr::Union { base, inputs } = expr else {
        unreachable!("matched above")
    };
    // Flatten nested Unions (children already processed, so one level).
    let mut branches = Vec::with_capacity(1 + inputs.len());
    let splice = |branch: MirRelationExpr, branches: &mut Vec<MirRelationExpr>| match branch {
        MirRelationExpr::Union { base, inputs } => {
            branches.push(*base);
            branches.extend(inputs);
        }
        other => branches.push(other),
    };
    let base = std::mem::replace(
        &mut **base,
        MirRelationExpr::constant(vec![], mz_repr::ReprRelationType::new(vec![])),
    );
    splice(base, &mut branches);
    for input in inputs.drain(..) {
        splice(input, &mut branches);
    }
    // Cancel Negate(X) against an equal positive X.
    let mut removed = vec![false; branches.len()];
    for i in 0..branches.len() {
        if removed[i] {
            continue;
        }
        let MirRelationExpr::Negate { input: negated } = &branches[i] else {
            continue;
        };
        if let Some(j) =
            (0..branches.len()).find(|j| !removed[*j] && *j != i && branches[*j] == **negated)
        {
            removed[i] = true;
            removed[j] = true;
        }
    }
    let mut branches: Vec<MirRelationExpr> = branches
        .into_iter()
        .zip(removed)
        .filter_map(|(b, r)| (!r).then_some(b))
        // Empty constants are the union identity.
        .filter(
            |b| !matches!(b, MirRelationExpr::Constant { rows: Ok(rows), .. } if rows.is_empty()),
        )
        .collect();
    *expr = match branches.len() {
        0 => MirRelationExpr::Constant {
            rows: Ok(vec![]),
            typ: typ_backup,
        },
        1 => branches.pop().expect("len checked"),
        _ => {
            let base = branches.remove(0);
            MirRelationExpr::Union {
                base: Box::new(base),
                inputs: branches,
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use mz_repr::{ReprRelationType, ReprScalarType};

    use super::*;
    use crate::rebuild::env::BindingEnv;

    fn t() -> MirRelationExpr {
        MirRelationExpr::local_get(
            mz_expr::LocalId::new(3),
            ReprRelationType::new(vec![ReprScalarType::Int64.nullable(true)]),
        )
    }

    fn unit() -> MirRelationExpr {
        MirRelationExpr::constant(vec![vec![]], ReprRelationType::new(vec![]))
    }

    #[mz_ore::test]
    fn cancels_negate_pair() {
        // Union(X, Negate(X), unit-typed constant) -> the constant.
        let x = t().project(vec![]);
        let union = x
            .clone()
            .union(MirRelationExpr::Negate { input: Box::new(x) });
        let union = MirRelationExpr::Union {
            base: Box::new(union),
            inputs: vec![unit()],
        };
        let mut env = BindingEnv {
            bindings: vec![],
            root: union,
        };
        apply(&mut env);
        assert!(
            matches!(&env.root, MirRelationExpr::Constant { rows: Ok(rows), .. } if rows.len() == 1),
            "expected unit constant, got {:?}",
            env.root
        );
    }

    #[mz_ore::test]
    fn flattens_nested_unions() {
        let union = t().union(t()).union(t());
        let mut env = BindingEnv {
            bindings: vec![],
            root: union,
        };
        apply(&mut env);
        let MirRelationExpr::Union { inputs, .. } = &env.root else {
            panic!("expected Union, got {:?}", env.root)
        };
        assert_eq!(inputs.len(), 2, "three branches in one flat Union");
    }
}
