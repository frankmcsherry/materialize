// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! The flat binding environment the rebuild operates over.
//!
//! All `Let` structure lives in an ordered binding list; binding bodies and
//! the root are `Let`-free, so per-binding transforms see a simpler language
//! and sharing changes are explicit operations rather than side effects of
//! normalization. `LetRec` scopes are held opaque in Phase 1: a `LetRec`
//! subtree is treated as a leaf of the body that contains it (sound — we
//! simply do not yet optimize inside recursive scopes; the design doc records
//! the obligation to flatten them, justified by their synchronous-update
//! semantics making names position-independent).

use mz_expr::{LocalId, MirRelationExpr};

use crate::TransformError;

/// An optimization unit: ordered bindings (each `Let`-free outside opaque
/// `LetRec` scopes) and a `Let`-free root, equivalent to the nested-`Let`
/// expression it was parsed from.
#[derive(Debug)]
pub struct BindingEnv {
    /// Bindings in dependency order: later bindings and the root may
    /// reference earlier bindings; never vice versa.
    pub bindings: Vec<(LocalId, MirRelationExpr)>,
    /// The result expression.
    pub root: MirRelationExpr,
}

impl BindingEnv {
    /// Parses an expression already in `Let` normal form (all `Let`s in a
    /// root-spine, as `NormalizeLets` establishes) into the environment.
    ///
    /// Returns an error if a `Let` survives anywhere below the spine, which
    /// would mean the input was not in normal form.
    pub fn from_normalized(mut expr: MirRelationExpr) -> Result<Self, TransformError> {
        let mut bindings = Vec::new();
        loop {
            match expr {
                MirRelationExpr::Let { id, value, body } => {
                    bindings.push((id, *value));
                    expr = *body;
                }
                other => {
                    expr = other;
                    break;
                }
            }
        }
        let env = BindingEnv {
            bindings,
            root: expr,
        };
        env.assert_let_free()?;
        Ok(env)
    }

    /// Emits the environment as a nested-`Let` expression in the contract
    /// form downstream consumers require: the binding spine at the root, in
    /// dependency order, with the ids the environment carries.
    pub fn into_expr(self) -> MirRelationExpr {
        let mut expr = self.root;
        for (id, value) in self.bindings.into_iter().rev() {
            expr = MirRelationExpr::Let {
                id,
                value: Box::new(value),
                body: Box::new(expr),
            };
        }
        expr
    }

    /// Verifies no `Let` occurs within any binding body or the root,
    /// excluding the interiors of (opaque) `LetRec` scopes, where `Let`s at
    /// scope roots are legitimate normal form.
    fn assert_let_free(&self) -> Result<(), TransformError> {
        let mut todo: Vec<&MirRelationExpr> = self
            .bindings
            .iter()
            .map(|(_id, body)| body)
            .chain(std::iter::once(&self.root))
            .collect();
        while let Some(e) = todo.pop() {
            match e {
                // Opaque in Phase 1: do not descend.
                MirRelationExpr::LetRec { .. } => {}
                MirRelationExpr::Let { .. } => {
                    return Err(TransformError::Internal(
                        "BindingEnv: input was not in Let normal form".to_string(),
                    ));
                }
                other => todo.extend(other.children()),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use mz_repr::{Datum, ReprRelationType, ReprScalarType};

    use super::*;

    fn leaf(v: i64) -> MirRelationExpr {
        MirRelationExpr::constant(
            vec![vec![Datum::Int64(v)]],
            ReprRelationType::new(vec![ReprScalarType::Int64.nullable(false)]),
        )
    }

    /// Parsing a normalized Let spine and emitting it reproduces the
    /// expression exactly.
    #[mz_ore::test]
    fn round_trip_identity() {
        let l0 = LocalId::new(0);
        let l1 = LocalId::new(1);
        let get = |id, of: &MirRelationExpr| MirRelationExpr::local_get(id, of.typ());
        let v0 = leaf(1);
        let v1 = get(l0, &v0).union(leaf(2));
        let body = get(l0, &v0).union(get(l1, &v1));
        let expr = MirRelationExpr::Let {
            id: l0,
            value: Box::new(v0.clone()),
            body: Box::new(MirRelationExpr::Let {
                id: l1,
                value: Box::new(v1.clone()),
                body: Box::new(body.clone()),
            }),
        };
        let env = BindingEnv::from_normalized(expr.clone()).unwrap();
        assert_eq!(env.bindings.len(), 2);
        assert_eq!(env.into_expr(), expr);
    }

    /// A Let below the spine (not in normal form) is rejected.
    #[mz_ore::test]
    fn rejects_unnormalized() {
        let l0 = LocalId::new(0);
        let inner = MirRelationExpr::Let {
            id: l0,
            value: Box::new(leaf(1)),
            body: Box::new(leaf(2)),
        };
        let expr = inner.union(leaf(3));
        assert!(BindingEnv::from_normalized(expr).is_err());
    }
}
