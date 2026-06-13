// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Reduce input inlining: `Map` and `Project` directly below a `Reduce`
//! fold into the group key and aggregate expressions.
//!
//! Group keys and aggregate arguments are arbitrary scalar expressions, so
//! a `Map` evaluated only to feed a `Reduce` is an extra dataflow operator
//! (and an extra row-width) for nothing: substitute its scalars into the
//! consuming expressions and drop the node. A `Project` below a `Reduce`
//! is a column remap of the same expressions.
//!
//! *Errors*: a substituted scalar is still evaluated once per input row,
//! in the same operator position (the Reduce's key/aggregate evaluation),
//! on exactly the rows the Map saw. A scalar feeding several expressions
//! is duplicated, which duplicates work but not error behavior (parity
//! with main's fusion).
//!
//! `Filter` below the Reduce stops the fold: predicates change the row set
//! and stay put.
//!
//! **Contract.** Requires: a [`BindingEnv`]. Local rewrites; safe inside
//! LetRec values.

use mz_expr::visit::VisitChildren;
use mz_expr::{MirRelationExpr, MirScalarExpr};

use crate::rebuild::env::BindingEnv;
use crate::rebuild::predicate_placement::remap_columns;

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
        input,
        group_key,
        aggregates,
        ..
    } = expr
    else {
        return;
    };
    loop {
        match &mut **input {
            MirRelationExpr::Project {
                input: inner,
                outputs,
            } => {
                for k in group_key.iter_mut() {
                    remap_columns(k, |c| outputs[c]);
                }
                for agg in aggregates.iter_mut() {
                    remap_columns(&mut agg.expr, |c| outputs[c]);
                }
                let inner = std::mem::replace(
                    inner,
                    Box::new(MirRelationExpr::constant(
                        vec![],
                        mz_repr::ReprRelationType::new(vec![]),
                    )),
                );
                **input = *inner;
            }
            MirRelationExpr::Map {
                input: inner,
                scalars,
            } => {
                let a_in = inner.arity();
                // Resolve map columns within the scalars themselves first,
                // so substitution into the Reduce is single-step.
                let mut resolved: Vec<MirScalarExpr> = Vec::with_capacity(scalars.len());
                for scalar in scalars.iter() {
                    let mut scalar = scalar.clone();
                    substitute_above(&mut scalar, a_in, &resolved);
                    resolved.push(scalar);
                }
                for k in group_key.iter_mut() {
                    substitute_above(k, a_in, &resolved);
                }
                for agg in aggregates.iter_mut() {
                    substitute_above(&mut agg.expr, a_in, &resolved);
                }
                let inner = std::mem::replace(
                    inner,
                    Box::new(MirRelationExpr::constant(
                        vec![],
                        mz_repr::ReprRelationType::new(vec![]),
                    )),
                );
                **input = *inner;
            }
            _ => break,
        }
    }
}

/// Replaces references to columns at or above `arity` with `defs[c - arity]`.
fn substitute_above(e: &mut MirScalarExpr, arity: usize, defs: &[MirScalarExpr]) {
    use mz_expr::visit::Visit;
    e.visit_mut_post(&mut |node| {
        if let MirScalarExpr::Column(c, _) = node {
            if *c >= arity {
                *node = defs[*c - arity].clone();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use mz_expr::{AggregateExpr, AggregateFunc, func};
    use mz_repr::{ReprRelationType, ReprScalarType};

    use super::*;
    use crate::rebuild::env::BindingEnv;

    #[mz_ore::test]
    fn inlines_map_into_group_key() {
        let table = MirRelationExpr::local_get(
            mz_expr::LocalId::new(3),
            ReprRelationType::new(vec![ReprScalarType::Int64.nullable(true); 2]),
        );
        let two_x = MirScalarExpr::literal_ok(mz_repr::Datum::Int64(2), ReprScalarType::Int64)
            .call_binary(MirScalarExpr::column(0), func::MulInt64);
        let reduce = MirRelationExpr::Reduce {
            input: Box::new(table.map(vec![two_x.clone()])),
            group_key: vec![MirScalarExpr::column(2)],
            aggregates: vec![AggregateExpr {
                func: AggregateFunc::SumInt64,
                expr: MirScalarExpr::column(1),
                distinct: false,
            }],
            monotonic: false,
            expected_group_size: None,
        };
        let mut env = BindingEnv {
            bindings: vec![],
            root: reduce,
        };
        apply(&mut env);
        let MirRelationExpr::Reduce {
            input, group_key, ..
        } = &env.root
        else {
            panic!("expected Reduce, got {:?}", env.root)
        };
        assert!(
            matches!(&**input, MirRelationExpr::Get { .. }),
            "Map folded away"
        );
        assert_eq!(group_key, &vec![two_x]);
    }
}
