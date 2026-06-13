// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Reduce elision: a `Distinct` whose input is already unique on the group
//! columns is a projection.
//!
//! A `Reduce` with no aggregates (Distinct) groups by columns G. If the
//! input has a unique key K with K a subset of G, every group holds exactly
//! one input row, so the Distinct emits exactly the G-projection of its
//! input: replace it with `Project`. Key knowledge comes from `typ()`:
//! lowering-assigned `Get` types remain true under semantics-preserving
//! rewrites, and projection_thinning narrows them soundly, so `typ().keys`
//! is conservative (it may miss keys, never invent them) — elision can only
//! under-fire, which is safe.
//!
//! These shapes arise from subquery decorrelation (Distinct over Distinct,
//! Distinct over keyed semijoins) and outer-join lowering.
//!
//! **Contract.** Requires: a [`BindingEnv`]. Ensures: no Distinct whose
//! input `typ()` proves unique on the group columns. Local rewrites only;
//! safe inside LetRec values (synchronous semantics make local equivalence
//! position-independent).

use mz_expr::visit::VisitChildren;
use mz_expr::{MirRelationExpr, MirScalarExpr};

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
        input,
        group_key,
        aggregates,
        ..
    } = expr
    else {
        return;
    };
    if !aggregates.is_empty() {
        return;
    }
    let group_cols: Option<Vec<usize>> = group_key
        .iter()
        .map(|k| match k {
            MirScalarExpr::Column(c, _) => Some(*c),
            _ => None,
        })
        .collect();
    let Some(group_cols) = group_cols else { return };
    let keyed = input
        .typ()
        .keys
        .iter()
        .any(|key| key.iter().all(|k| group_cols.contains(k)));
    if !keyed {
        return;
    }
    let input = std::mem::replace(
        &mut **input,
        MirRelationExpr::constant(vec![], mz_repr::ReprRelationType::new(vec![])),
    );
    let identity =
        group_cols.len() == input.arity() && group_cols.iter().enumerate().all(|(i, c)| i == *c);
    *expr = if identity {
        input
    } else {
        input.project(group_cols)
    };
}

#[cfg(test)]
mod tests {
    use mz_repr::{ReprRelationType, ReprScalarType};

    use super::*;
    use crate::rebuild::env::BindingEnv;

    fn keyed_table() -> MirRelationExpr {
        MirRelationExpr::local_get(
            mz_expr::LocalId::new(3),
            ReprRelationType::new(vec![ReprScalarType::Int64.nullable(true); 2]).with_key(vec![0]),
        )
    }

    #[mz_ore::test]
    fn elides_distinct_on_key() {
        let distinct = keyed_table().distinct_by(vec![0, 1]);
        let mut env = BindingEnv {
            bindings: vec![],
            root: distinct,
        };
        apply(&mut env);
        assert!(
            matches!(&env.root, MirRelationExpr::Get { .. }),
            "expected bare Get, got {:?}",
            env.root
        );
    }

    #[mz_ore::test]
    fn keeps_distinct_on_non_key() {
        let distinct = keyed_table().distinct_by(vec![1]);
        let mut env = BindingEnv {
            bindings: vec![],
            root: distinct,
        };
        apply(&mut env);
        assert!(matches!(&env.root, MirRelationExpr::Reduce { .. }));
    }
}
