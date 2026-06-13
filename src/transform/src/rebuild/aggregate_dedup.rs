// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Aggregate deduplication: a `Reduce` computes each distinct aggregate
//! once; duplicates become projections of the first occurrence.
//!
//! MIR is deterministic, so two syntactically equal `AggregateExpr`s in
//! the same `Reduce` produce equal values (the same argument multiset per
//! group). Arrangements then store one copy.
//!
//! **Contract.** Requires: a [`BindingEnv`]. Local rewrites; safe inside
//! LetRec values.

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
    let MirRelationExpr::Reduce {
        group_key,
        aggregates,
        ..
    } = expr
    else {
        return;
    };
    let klen = group_key.len();
    // Keep each aggregate's first occurrence; map every position to it.
    let mut kept = Vec::with_capacity(aggregates.len());
    let mut firsts = Vec::with_capacity(aggregates.len());
    for agg in aggregates.drain(..) {
        match kept.iter().position(|k| *k == agg) {
            Some(first) => firsts.push(first),
            None => {
                firsts.push(kept.len());
                kept.push(agg);
            }
        }
    }
    if kept.len() == firsts.len() {
        *aggregates = kept;
        return;
    }
    let outputs: Vec<usize> = (0..klen).chain(firsts.iter().map(|f| klen + f)).collect();
    *aggregates = kept;
    let inner = std::mem::replace(
        expr,
        MirRelationExpr::constant(vec![], mz_repr::ReprRelationType::new(vec![])),
    );
    *expr = inner.project(outputs);
}

#[cfg(test)]
mod tests {
    use mz_expr::{AggregateExpr, AggregateFunc, MirScalarExpr};
    use mz_repr::{ReprRelationType, ReprScalarType};

    use super::*;
    use crate::rebuild::env::BindingEnv;

    #[mz_ore::test]
    fn dedups_repeated_aggregate() {
        let table = MirRelationExpr::local_get(
            mz_expr::LocalId::new(3),
            ReprRelationType::new(vec![ReprScalarType::Int64.nullable(true); 2]),
        );
        let sum = |col: usize| AggregateExpr {
            func: AggregateFunc::SumInt64,
            expr: MirScalarExpr::column(col),
            distinct: false,
        };
        let reduce = MirRelationExpr::Reduce {
            input: Box::new(table),
            group_key: vec![MirScalarExpr::column(0)],
            aggregates: vec![sum(1), sum(0), sum(1)],
            monotonic: false,
            expected_group_size: None,
        };
        let mut env = BindingEnv {
            bindings: vec![],
            root: reduce,
        };
        apply(&mut env);
        let MirRelationExpr::Project { input, outputs } = &env.root else {
            panic!("expected Project, got {:?}", env.root)
        };
        assert_eq!(outputs, &vec![0, 1, 2, 1]);
        assert!(
            matches!(&**input, MirRelationExpr::Reduce { aggregates, .. }
            if aggregates.len() == 2)
        );
    }
}
