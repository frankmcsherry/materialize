// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Linear-operator normal form: one canonical Map-Filter-Project per node.
//!
//! **Idea.** `Map`, `Filter`, and `Project` are linear operators: any chain
//! of them is equivalent to a single composite, and the composite has one
//! canonical emission (Map, then Filter, then Project). We collapse every
//! maximal linear chain to that form, so all other transforms see at most
//! one of each linear operator above any non-linear node.
//!
//! **Mechanism.** The composite algebra is the shared IR utility
//! [`MapFilterProject`] (infrastructure, like analyses): extract the chain
//! above each node, `optimize()` the composite (common subexpression and
//! identity cleanup within the linear fragment), and re-emit canonically.
//!
//! **Error discipline.** Extraction uses the non-errors variant: expressions
//! that may error are not relocated relative to the predicates that guard
//! the rows reaching them, so no new errors are introduced (rule 1), while
//! infallible fragments reorder freely under the filter-commutation license
//! (rule 2).
//!
//! **Contract.** Requires: a [`BindingEnv`]. Ensures: in every binding body
//! and the root, no linear operator has a linear operator as its input;
//! linear chains appear as canonical M-F-P with optimized scalar content.

use mz_expr::visit::Visit;
use mz_expr::{MapFilterProject, MirRelationExpr};

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
        // Only bother where a linear chain of length >= 2 begins. The
        // post-order visit means inputs are already in normal form, so a
        // linear node whose input is linear is exactly a chain to collapse.
        if is_linear(e) && is_linear(linear_input(e)) {
            let mut mfp = MapFilterProject::extract_non_errors_from_expr_mut(e);
            mfp.optimize();
            if !mfp.is_identity() {
                let (map, filter, project) = mfp.as_map_filter_project();
                let mut out = std::mem::replace(
                    e,
                    MirRelationExpr::constant(vec![], mz_repr::ReprRelationType::new(vec![])),
                );
                if !map.is_empty() {
                    out = out.map(map);
                }
                if !filter.is_empty() {
                    out = out.filter(filter);
                }
                out = out.project(project);
                *e = out;
            }
        }
    });
}

fn is_linear(e: &MirRelationExpr) -> bool {
    matches!(
        e,
        MirRelationExpr::Map { .. }
            | MirRelationExpr::Filter { .. }
            | MirRelationExpr::Project { .. }
    )
}

fn linear_input(e: &MirRelationExpr) -> &MirRelationExpr {
    match e {
        MirRelationExpr::Map { input, .. }
        | MirRelationExpr::Filter { input, .. }
        | MirRelationExpr::Project { input, .. } => input,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use mz_expr::{MirRelationExpr, MirScalarExpr};
    use mz_repr::{Datum, ReprRelationType, ReprScalarType};

    use super::*;

    #[mz_ore::test]
    fn collapses_chains() {
        let t = ReprRelationType::new(vec![ReprScalarType::Int64.nullable(false); 2]);
        let base = MirRelationExpr::constant(vec![], t);
        let lit = MirScalarExpr::literal_ok(Datum::Int64(7), ReprScalarType::Int64);
        let chain = base
            .clone()
            .map(vec![lit.clone()])
            .project(vec![0, 2])
            .map(vec![lit.clone()])
            .project(vec![1, 2]);
        let mut env = BindingEnv {
            bindings: vec![],
            root: chain,
        };
        apply(&mut env);

        // At most one of each linear operator in a row.
        let mut depth_linear: usize = 0;
        let mut max_chain = 0;
        fn walk(e: &MirRelationExpr, chain: usize, max: &mut usize) {
            let chain = if super::is_linear(e) { chain + 1 } else { 0 };
            *max = (*max).max(chain);
            for c in e.children() {
                walk(c, chain, max);
            }
        }
        walk(&env.root, depth_linear, &mut max_chain);
        depth_linear = 0;
        let _ = depth_linear;
        assert!(max_chain <= 3, "linear chain longer than canonical M-F-P");
        // And semantics-relevant arity preserved.
        assert_eq!(env.root.arity(), 2);
    }
}
