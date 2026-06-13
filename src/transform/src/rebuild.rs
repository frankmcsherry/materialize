// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! A from-scratch rebuild of the logical optimizer (experimental).
//!
//! Selected by `MZ_OPTIMIZER_REBUILD=logical`. The rebuild operates over a
//! flat binding environment ([`env::BindingEnv`]): all `Let` structure is
//! parsed into an ordered binding list whose bodies are `Let`-free, transforms
//! run per binding over that simpler language, sharing changes are explicit
//! operations, and the exit emits exactly the nested-`Let` contract form the
//! physical pipeline requires. See play/optimizer-rebuild/ for the design
//! journal, capability map, and scorecard.

pub mod env;
pub mod filter_split;
pub mod join_flatten;
pub mod linear_fuse;
pub mod predicate_placement;
pub mod projection_thinning;
pub mod reduce_class_split;

use mz_expr::MirRelationExpr;

use crate::{TransformCtx, TransformError};

/// The rebuilt logical pipeline: parse to the binding environment, apply the
/// per-binding transform sequence (currently empty — transforms are earned;
/// see the loop prompt), and emit the contract form.
#[derive(Debug)]
pub struct RebuildLogical;

impl crate::Transform for RebuildLogical {
    fn name(&self) -> &'static str {
        "RebuildLogical"
    }

    #[mz_ore::instrument(
        target = "optimizer",
        level = "debug",
        fields(path.segment = "rebuild_logical")
    )]
    fn actually_perform_transform(
        &self,
        relation: &mut MirRelationExpr,
        ctx: &mut TransformCtx,
    ) -> Result<(), TransformError> {
        // Phase 1 scaffolding loan (to repay): use the existing NormalizeLets
        // once as the *parser* into binding form. The rebuild will own its
        // hoisting before this module is proposed for merging.
        crate::normalize_lets::NormalizeLets::new(false)
            .actually_perform_transform(relation, ctx)?;

        let env = env::BindingEnv::from_normalized(std::mem::replace(
            relation,
            MirRelationExpr::constant(vec![], mz_repr::ReprRelationType::new(vec![])),
        ))?;

        // The transform sequence; each entry earned against the scorecard.
        let mut env = env;
        linear_fuse::apply(&mut env);
        filter_split::apply(&mut env);
        join_flatten::apply(&mut env);
        predicate_placement::apply(&mut env);
        projection_thinning::apply(&mut env);
        linear_fuse::apply(&mut env);
        reduce_class_split::apply(&mut env);

        *relation = env.into_expr();
        mz_repr::explain::trace_plan(&*relation);
        Ok(())
    }
}
