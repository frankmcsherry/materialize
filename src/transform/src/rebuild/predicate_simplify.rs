// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Conjunct-level predicate simplification on settled `Filter`s.
//!
//! Three rules, each justified by filter semantics (a row passes iff every
//! conjunct evaluates to TRUE; NULL and FALSE both drop the row):
//!
//! 1. **Dedup**: repeated conjuncts evaluate identically (MIR determinism).
//! 2. **Absorption**: a disjunction with another conjunct among its
//!    disjuncts is implied by it — `p AND (p OR q)` keeps only `p`. (The
//!    disjunction cannot turn a passing row into a dropped one: when `p` is
//!    TRUE the Or is TRUE regardless of `q`, by Or's error-absorbing eval.)
//! 3. **Implied non-null**: `e IS NOT NULL` is redundant beside a conjunct
//!    that propagates NULL from `e` to its own result — if `e` were NULL,
//!    that conjunct would evaluate to NULL and drop the row anyway.
//!
//! Runs after predicate_placement, when every conjunct has reached its
//! final Filter; all three rules are per-Filter local.
//!
//! **Contract.** Requires: conjunction-free Filters ([`BindingEnv`], after
//! filter_split). Ensures: no Filter carries a conjunct implied by its
//! neighbors under rules 1-3. Emptied Filters elide.

use mz_expr::visit::VisitChildren;
use mz_expr::{MirRelationExpr, MirScalarExpr, UnaryFunc, VariadicFunc};

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
    let MirRelationExpr::Filter { input, predicates } = expr else {
        return;
    };
    // Rule 1: dedup (order-preserving).
    let mut seen: Vec<MirScalarExpr> = Vec::with_capacity(predicates.len());
    predicates.retain(|p| {
        if seen.contains(p) {
            false
        } else {
            seen.push(p.clone());
            true
        }
    });
    // Rule 2: absorption.
    let all = predicates.clone();
    predicates.retain(|p| {
        let MirScalarExpr::CallVariadic {
            func: VariadicFunc::Or(_),
            exprs,
        } = p
        else {
            return true;
        };
        !exprs.iter().any(|d| all.iter().any(|q| q != p && q == d))
    });
    // Rule 3: implied non-null.
    let all = predicates.clone();
    predicates.retain(|p| {
        let MirScalarExpr::CallUnary { func, expr: inner } = p else {
            return true;
        };
        if !matches!(func, UnaryFunc::Not(_)) {
            return true;
        }
        let MirScalarExpr::CallUnary {
            func: inner_func,
            expr: e,
        } = &**inner
        else {
            return true;
        };
        if !matches!(inner_func, UnaryFunc::IsNull(_)) {
            return true;
        }
        !all.iter().any(|q| q != p && null_rejecting(q, e))
    });
    if predicates.is_empty() {
        let inner = std::mem::replace(
            &mut **input,
            MirRelationExpr::constant(vec![], mz_repr::ReprRelationType::new(vec![])),
        );
        *expr = inner;
    }
}

/// True iff a NULL value of `e` forces `q` to evaluate to NULL: `e` occurs
/// in `q` along a path of null-propagating functions. (`If` and friends are
/// lazy, so they end such paths.)
fn null_rejecting(q: &MirScalarExpr, e: &MirScalarExpr) -> bool {
    if q == e {
        return true;
    }
    match q {
        MirScalarExpr::CallUnary { func, expr } => {
            func.propagates_nulls() && null_rejecting(expr, e)
        }
        MirScalarExpr::CallBinary { func, expr1, expr2 } => {
            func.propagates_nulls() && (null_rejecting(expr1, e) || null_rejecting(expr2, e))
        }
        MirScalarExpr::CallVariadic { func, exprs } => {
            func.propagates_nulls() && exprs.iter().any(|x| null_rejecting(x, e))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use mz_expr::func;
    use mz_repr::{Datum, ReprRelationType, ReprScalarType};

    use super::*;
    use crate::rebuild::env::BindingEnv;

    fn t() -> MirRelationExpr {
        MirRelationExpr::local_get(
            mz_expr::LocalId::new(3),
            ReprRelationType::new(vec![ReprScalarType::Int64.nullable(true); 2]),
        )
    }

    fn gt0(col: usize) -> MirScalarExpr {
        MirScalarExpr::column(col).call_binary(
            MirScalarExpr::literal_ok(Datum::Int64(0), ReprScalarType::Int64),
            func::Gt,
        )
    }

    fn simplify(root: MirRelationExpr) -> MirRelationExpr {
        let mut env = BindingEnv {
            bindings: vec![],
            root,
        };
        apply(&mut env);
        env.root
    }

    #[mz_ore::test]
    fn absorbs_or_with_present_conjunct() {
        let p = gt0(0);
        let p_or_q = p.clone().or(gt0(1));
        let result = simplify(t().filter(vec![p.clone(), p_or_q]));
        assert!(matches!(&result, MirRelationExpr::Filter { predicates, .. }
            if predicates == &vec![p]));
    }

    #[mz_ore::test]
    fn drops_implied_non_null() {
        let guard = MirScalarExpr::column(0).call_is_null().not();
        let result = simplify(t().filter(vec![guard, gt0(0)]));
        assert!(matches!(&result, MirRelationExpr::Filter { predicates, .. }
            if predicates == &vec![gt0(0)]));
    }

    #[mz_ore::test]
    fn keeps_unrelated_non_null() {
        let guard = MirScalarExpr::column(1).call_is_null().not();
        let result = simplify(t().filter(vec![guard.clone(), gt0(0)]));
        assert!(matches!(&result, MirRelationExpr::Filter { predicates, .. }
            if predicates.len() == 2));
    }
}
