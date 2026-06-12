// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Reduce normal form: one reduction class per `Reduce`.
//!
//! **Idea.** LIR plans each `Reduce` as one of three implementation classes
//! (accumulable, hierarchical, basic) and asserts the aggregates of a
//! `Reduce` all belong to one class (`ReducePlan::create_from`). A `Reduce`
//! mixing classes is equivalent to one `Reduce` per class over the same
//! input and group key, equijoined on the key and projected back to the
//! original column order.
//!
//! **Soundness.** Each per-class `Reduce` emits exactly one row per group of
//! the shared input, with the group key as a unique key, and all of them see
//! the identical group set — so the key-equijoin is one-to-one and total,
//! and the projection restores the original arity and order. On empty input
//! every part is empty, so the join is empty, matching the original.
//!
//! **Contract.** Requires: a [`BindingEnv`]. Ensures: every `Reduce` in the
//! environment has aggregates of a single reduction class (the LIR lowering
//! contract). Debt (journaled): the shared input is cloned per class; once
//! `BindingEnv` grows an id allocator this becomes an explicit shared
//! binding.

use mz_compute_types::plan::reduce::reduction_type;
use mz_expr::MirRelationExpr;
use mz_expr::visit::Visit;

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
        let MirRelationExpr::Reduce {
            input,
            group_key,
            aggregates,
            monotonic,
            expected_group_size,
        } = e
        else {
            return;
        };
        // Partition aggregate positions by reduction class, preserving
        // per-class order.
        let mut classes: Vec<(_, Vec<usize>)> = Vec::new();
        for (index, aggr) in aggregates.iter().enumerate() {
            let class = reduction_type(&aggr.func);
            match classes.iter_mut().find(|(c, _)| *c == class) {
                Some((_, indexes)) => indexes.push(index),
                None => classes.push((class, vec![index])),
            }
        }
        if classes.len() <= 1 {
            return;
        }

        let key_arity = group_key.len();
        // One Reduce per class, each over (a clone of) the shared input.
        let parts: Vec<MirRelationExpr> = classes
            .iter()
            .map(|(_, indexes)| MirRelationExpr::Reduce {
                input: input.clone(),
                group_key: group_key.clone(),
                aggregates: indexes.iter().map(|i| aggregates[*i].clone()).collect(),
                monotonic: *monotonic,
                expected_group_size: *expected_group_size,
            })
            .collect();

        // Equijoin all parts on the group key columns.
        let equivalences: Vec<Vec<(usize, usize)>> = (0..key_arity)
            .map(|col| (0..parts.len()).map(|part| (part, col)).collect())
            .collect();

        // Column offsets of each part within the join product.
        let mut offsets = Vec::with_capacity(parts.len());
        let mut offset = 0;
        for (part, (_, indexes)) in parts.iter().zip(&classes) {
            let _ = part;
            offsets.push(offset);
            offset += key_arity + indexes.len();
        }

        // Restore the original column order: keys from part 0, then each
        // aggregate from its class's part at its within-class position.
        let mut projection: Vec<usize> = (0..key_arity).collect();
        projection.extend((0..aggregates.len()).map(|original| {
            let (part, within) = classes
                .iter()
                .enumerate()
                .find_map(|(p, (_, indexes))| {
                    indexes.iter().position(|i| *i == original).map(|w| (p, w))
                })
                .expect("every aggregate belongs to exactly one class");
            offsets[part] + key_arity + within
        }));

        *e = MirRelationExpr::join(parts, equivalences).project(projection);
    });
}

#[cfg(test)]
mod tests {
    use mz_compute_types::plan::reduce::reduction_type;
    use mz_expr::{AggregateExpr, AggregateFunc, MirRelationExpr, MirScalarExpr};
    use mz_repr::{ReprRelationType, ReprScalarType};

    use super::*;

    #[mz_ore::test]
    fn splits_mixed_reduce() {
        let t = ReprRelationType::new(vec![ReprScalarType::Int64.nullable(false); 2]);
        let input = MirRelationExpr::constant(vec![], t);
        let agg = |func| AggregateExpr {
            func,
            expr: MirScalarExpr::column(1),
            distinct: false,
        };
        let mixed = input.reduce(
            vec![0],
            vec![
                agg(AggregateFunc::SumInt64), // accumulable
                agg(AggregateFunc::MaxInt64), // hierarchical
                agg(AggregateFunc::Count),    // accumulable
            ],
            None,
        );
        let mut env = BindingEnv {
            bindings: vec![],
            root: mixed,
        };
        apply(&mut env);

        // Every Reduce now has a single class, and the root restores arity 4
        // (key + three aggregates).
        assert_eq!(env.root.arity(), 4);
        env.root.visit_pre(|e| {
            if let MirRelationExpr::Reduce { aggregates, .. } = e {
                let mut classes: Vec<_> =
                    aggregates.iter().map(|a| reduction_type(&a.func)).collect();
                classes.dedup();
                assert_eq!(classes.len(), 1, "Reduce still mixes classes");
            }
        });
    }
}
