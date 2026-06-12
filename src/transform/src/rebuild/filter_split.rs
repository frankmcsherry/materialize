// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Filter predicate normal form: no top-level conjunctions.
//!
//! **Idea.** A `Filter` retains rows on which every predicate evaluates to
//! true, so a conjunction among its predicates is redundant structure:
//! `Filter [a AND b]` and `Filter [a, b]` are the same operator. We keep the
//! split form, in which each predicate is a conjunction-free expression.
//!
//! **Why it matters.** Downstream consumers reason per-conjunct: temporal
//! filter compilation requires each `mz_now()` comparison to be its own
//! predicate (a top-level `AND` is rejected at MFP lowering — discovered
//! empirically: bootstrap panics on system views without this form), and
//! predicate placement moves conjuncts independently.
//!
//! **Soundness.** `And`'s evaluation is order- and grouping-insensitive,
//! including its error behavior (it returns false if any conjunct is false,
//! even if another errored), so splitting cannot change results or error
//! visibility.
//!
//! **Contract.** Requires: a [`BindingEnv`]. Ensures: no `Filter` predicate
//! anywhere in the environment has a top-level `And`; existing predicate
//! order is preserved with conjuncts expanded in place, depth-first.

use mz_expr::visit::Visit;
use mz_expr::{MirRelationExpr, MirScalarExpr, VariadicFunc};

use crate::rebuild::env::BindingEnv;

/// Applies the transform to every binding body and the root.
pub fn apply(env: &mut BindingEnv) {
    for (_id, body) in env.bindings.iter_mut() {
        apply_expr(body);
    }
    apply_expr(&mut env.root);
}

fn apply_expr(expr: &mut MirRelationExpr) {
    expr.visit_mut_post(&mut |e| {
        if let MirRelationExpr::Filter { predicates, .. } = e {
            if predicates.iter().any(is_and) {
                let mut split = Vec::with_capacity(predicates.len());
                for p in predicates.drain(..) {
                    push_conjuncts(p, &mut split);
                }
                *predicates = split;
            }
        }
    });
}

fn is_and(e: &MirScalarExpr) -> bool {
    matches!(
        e,
        MirScalarExpr::CallVariadic {
            func: VariadicFunc::And(_),
            ..
        }
    )
}

/// Appends `e`'s conjuncts to `out`, flattening nested `And`s, preserving
/// left-to-right order.
fn push_conjuncts(e: MirScalarExpr, out: &mut Vec<MirScalarExpr>) {
    match e {
        MirScalarExpr::CallVariadic {
            func: VariadicFunc::And(_),
            exprs,
        } => {
            for inner in exprs {
                push_conjuncts(inner, out);
            }
        }
        other => out.push(other),
    }
}

#[cfg(test)]
mod tests {
    use mz_expr::{MirRelationExpr, MirScalarExpr};
    use mz_repr::{Datum, ReprRelationType, ReprScalarType};

    use super::*;

    #[mz_ore::test]
    fn splits_nested_ands() {
        let t = ReprRelationType::new(vec![ReprScalarType::Bool.nullable(false); 3]);
        let input = MirRelationExpr::constant(vec![], t);
        let c = MirScalarExpr::column;
        let and = |exprs: Vec<MirScalarExpr>| {
            MirScalarExpr::call_variadic(mz_expr::func::variadic::And, exprs)
        };
        let pred = and(vec![c(0), and(vec![c(1), c(2)])]);
        let mut env = BindingEnv {
            bindings: vec![],
            root: input.clone().filter(vec![pred]),
        };
        apply(&mut env);
        assert_eq!(env.root, input.filter(vec![c(0), c(1), c(2)]));
    }

    #[mz_ore::test]
    fn non_and_untouched() {
        let t = ReprRelationType::new(vec![ReprScalarType::Bool.nullable(false)]);
        let input = MirRelationExpr::constant(vec![], t);
        let lit_true = MirScalarExpr::literal_ok(Datum::True, ReprScalarType::Bool);
        let mut env = BindingEnv {
            bindings: vec![],
            root: input.clone().filter(vec![lit_true.clone()]),
        };
        apply(&mut env);
        assert_eq!(env.root, input.filter(vec![lit_true]));
    }
}
