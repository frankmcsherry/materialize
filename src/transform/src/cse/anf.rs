// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Identifies common relation subexpressions and places them behind `Let`
//! bindings.
//!
//! All structurally equivalent expressions, defined recursively as having
//! structurally equivalent inputs, and identical parameters, will be placed
//! behind `Let` bindings. The resulting expressions likely have an excess of
//! `Let` expressions, and therefore this transform is usually followed by a
//! `NormalizeLets` application.

use std::collections::BTreeMap;

use mz_expr::visit::VisitChildren;
use mz_expr::{AccessStrategy, Id, LocalId, MirRelationExpr, RECURSION_LIMIT};
use mz_ore::id_gen::IdGen;
use mz_ore::stack::{CheckedRecursion, RecursionGuard};

/// Transform an MirRelationExpr into an administrative normal form (ANF).
#[derive(Default, Debug)]
pub struct ANF;

use crate::TransformCtx;

impl crate::Transform for ANF {
    #[mz_ore::instrument(
        target = "optimizer",
        level = "debug",
        fields(path.segment = "anf")
    )]
    fn transform(
        &self,
        relation: &mut MirRelationExpr,
        _ctx: &mut TransformCtx,
    ) -> Result<(), crate::TransformError> {
        let result = self.transform_without_trace(relation);
        mz_repr::explain::trace_plan(&*relation);
        result
    }
}

impl ANF {
    /// Performs the `NormalizeLets` transformation without tracing the result.
    pub fn transform_without_trace(
        &self,
        relation: &mut MirRelationExpr,
    ) -> Result<(), crate::TransformError> {
        let mut bindings = Bindings::default();
        bindings.intern_expression(&mut IdGen::default(), relation)?;
        bindings.populate_expression(relation);
        Ok(())
    }
}

/// Maintains `Let` bindings in a compact, explicit representation.
///
/// The `bindings` map contains neither `Let` bindings nor two structurally
/// equivalent expressions.
///
/// The bindings can be interpreted as an ordered sequence of let bindings,
/// ordered by their identifier, that should be applied in order before the
/// use of the expression from which they have been extracted.
#[derive(Clone, Debug)]
struct Bindings {
    /// A list of let-bound expressions and their order / identifier.
    bindings: BTreeMap<MirRelationExpr, u64>,
    /// Mapping from conventional local `Get` identifiers to new ones.
    rebindings: BTreeMap<LocalId, LocalId>,
    // A guard for tracking the maximum depth of recursive tree traversal.
    recursion_guard: RecursionGuard,
}

impl CheckedRecursion for Bindings {
    fn recursion_guard(&self) -> &RecursionGuard {
        &self.recursion_guard
    }
}

impl Default for Bindings {
    fn default() -> Bindings {
        Bindings {
            bindings: BTreeMap::new(),
            rebindings: BTreeMap::new(),
            recursion_guard: RecursionGuard::with_limit(RECURSION_LIMIT),
        }
    }
}

impl Bindings {
    fn new(rebindings: BTreeMap<LocalId, LocalId>) -> Bindings {
        Bindings {
            rebindings,
            ..Bindings::default()
        }
    }
}

impl Bindings {
    /// Replace `relation` with an equivalent `Get` expression referencing a location in `bindings`.
    ///
    /// The algorithm performs a post-order traversal of the expression tree, binding each distinct
    /// expression to a new local identifier. It maintains the invariant that `bindings` contains no
    /// `Let` expressions, nor any two structurally equivalent expressions.
    ///
    /// Once each sub-expression is replaced by a canonical `Get` expression, each expression is also
    /// in a canonical representation, which is used to check for prior instances and drives re-use.
    fn intern_expression(
        &mut self,
        id_gen: &mut IdGen,
        relation: &mut MirRelationExpr,
    ) -> Result<(), crate::TransformError> {
        self.checked_recur_mut(|this| {
            match relation {
                MirRelationExpr::LetRec {
                    ids,
                    values,
                    body,
                    limits,
                } => {
                    // Introduce a new copy of `self`, which will be specific to this scope.
                    // This makes expressions used in the outer scope available for re-use
                    // in this new recursive scope. By retaining `self`, we'll be able to see
                    // the new bindings, and install only them when we reform the expression.
                    let mut scoped_anf = this.clone();

                    // Used to distinguish new bindings from old bindings.
                    let id_boundary = id_gen.allocate_id();

                    // Each identifier in `ids` will be given *two* new identifiers,
                    // one "old" and one "new". This is important to ensure that use
                    // of the "old" collection does not result in hits for uses of the
                    // "new" collection. We will need to unify these identifiers before
                    // returning, but this should just be a matter of rewritting one
                    // with the other (their distinction comes only from the moment they
                    // are referenced).

                    // The plan is to walk the bindings `values` in turn, producing a sequence
                    // of ANF bindings that represent the same computation. As we reach each
                    // `value`, we'll mint a new identifier and commit that term, but also
                    // 1. Update the rebindings map to the new identifier, and
                    // 2. Replace references to the old identifier with the new identifier.
                    // Once finished, we'll lay out the new sequence of `ids` and `values`.

                    // For each bound identifier from `ids`, a temporary identifier for the "old" version.
                    let prevs = ids
                        .iter()
                        .map(|_id| LocalId::new(id_gen.allocate_id()))
                        .collect::<Vec<_>>();
                    let mut nexts = Vec::new();

                    // Install the "old" rebindings to start.
                    // As we discover uses of `id`, we'll replace them with `old`.
                    scoped_anf
                        .rebindings
                        .extend(ids.iter().zip(prevs.iter()).map(|(x, y)| (*x, *y)));

                    // Intern the sequence of values and then body.
                    // Care is taken as each value is bound to alter the rebinding
                    // of the identifier, so that uses of the "old" value do not
                    // match uses of the "new" value.
                    for (index, value) in values.iter_mut().enumerate() {
                        scoped_anf.intern_expression(id_gen, value)?;
                        // Now replace the rebinding of `id` from `old` to `new`.
                        // We will ultimately use `new_id` for `value`, and should
                        // imagine a `let new_id = value` that will appear at this
                        // point. We cannot use `scoped_anf` for this, because `value`
                        // may already be bound to something else, and we cannot set
                        // all references to `value` to now be `new_id`. We can set
                        // *subsequent* references to `value` to be `new_id`, but we
                        // oughtn't uninstall `value` if it already exists.
                        let new_id = id_gen.allocate_id();
                        nexts.push(new_id);
                        scoped_anf
                            .rebindings
                            .insert(ids[index].clone(), LocalId::new(new_id));
                    }

                    // We handle `body` separately, as it is an error to rely on arrangements from within the WMR.
                    // Ideally we wouldn't need that complexity here, but this is called on arrangement-laden MRE
                    // after join planning where we need to have locked in arrangements. Revisit if we correct that.
                    let mut body_anf = Bindings::new(this.rebindings.clone());
                    for id in ids.iter() {
                        body_anf
                            .rebindings
                            .insert(*id, scoped_anf.rebindings[id].clone());
                    }
                    body_anf.intern_expression(id_gen, body)?;
                    body_anf.populate_expression(body);

                    // We now want to rebuild the let bindings that will make `body`
                    // the correct answer. We have these in `scoped_anf`, but we must
                    // 1. rewrite occurrences of `Get(old)` into `Get(new)`,
                    // 2. insert `let new = value` for each existing binding.
                    // 3. update `ids` and `values` to reflect all of this.

                    // Map from "old" identifiers to "new" identifiers.
                    // If we have a hit in this map, we should perform the replacement.
                    let mut remap = BTreeMap::new();
                    for (p, n) in prevs
                        .iter()
                        .zip(nexts.iter())
                        .map(|(p, n)| (*p, LocalId::new(*n)))
                    {
                        remap.insert(p, n);
                    }

                    // Convert the bindings in to a sequence, by the local identifier.
                    let mut bindings = scoped_anf
                        .bindings
                        .into_iter()
                        .filter(|(_e, i)| i > &id_boundary)
                        .map(|(e, i)| (i, e))
                        .collect::<Vec<_>>();
                    // Add bindings corresponding to `(ids, values)`
                    bindings.extend(nexts.into_iter().zip(values.drain(..)));
                    bindings.sort();
                    for (_id, expr) in bindings.iter_mut() {
                        let mut todo = vec![&mut *expr];
                        while let Some(e) = todo.pop() {
                            if let MirRelationExpr::Get {
                                id: Id::Local(i), ..
                            } = e
                            {
                                if let Some(next) = remap.get(i) {
                                    i.clone_from(next);
                                }
                            }
                            todo.extend(e.children_mut());
                        }
                    }

                    // New ids and new values can be extracted from the bindings.
                    let (new_ids, new_values): (Vec<_>, Vec<_>) = bindings.into_iter().unzip();
                    use itertools::Itertools;
                    // New limits will all be `None`, except for any pre-existing limits.
                    let mut new_limits: BTreeMap<LocalId, _> = BTreeMap::default();
                    for (id, limit) in ids.iter().zip_eq(limits.iter()) {
                        new_limits.insert(scoped_anf.rebindings[id], limit.clone());
                    }
                    for id in new_ids.iter() {
                        if !new_limits.contains_key(&LocalId::new(*id)) {
                            new_limits.insert(LocalId::new(*id), None);
                        }
                    }
                    let new_limits = new_limits.into_values().collect::<Vec<_>>();

                    *ids = new_ids.into_iter().map(LocalId::new).collect();
                    *values = new_values;
                    *limits = new_limits;
                }
                MirRelationExpr::Let { id, value, body } => {
                    this.intern_expression(id_gen, value)?;
                    let new_id = if let MirRelationExpr::Get {
                        id: Id::Local(x), ..
                    } = **value
                    {
                        x
                    } else {
                        panic!("Invariant violated")
                    };
                    this.rebindings.insert(*id, new_id);
                    this.intern_expression(id_gen, body)?;
                    let body = body.take_dangerous();
                    this.rebindings.remove(id);
                    *relation = body;
                }
                MirRelationExpr::Get { id, .. } => {
                    if let Id::Local(id) = id {
                        if let Some(rebound) = this.rebindings.get(id) {
                            *id = *rebound;
                        } else {
                            Err(crate::TransformError::Internal(format!(
                                "Identifier missing: {:?}",
                                id
                            )))?;
                        }
                    }
                }

                _ => {
                    // All other expressions just need to apply the logic recursively.
                    relation.try_visit_mut_children(|expr| this.intern_expression(id_gen, expr))?;
                }
            };

            // This should be fast, as it depends directly on only `Get` expressions.
            let typ = relation.typ();
            // We want to maintain the invariant that `relation` ends up as a local `Get`.
            if let MirRelationExpr::Get {
                id: Id::Local(_), ..
            } = relation
            {
                // Do nothing, as the expression is already a local `Get` expression.
            } else {
                // Either find an instance of `relation` or insert this one.
                let id = this
                    .bindings
                    .entry(relation.take_dangerous())
                    .or_insert_with(|| id_gen.allocate_id());
                *relation = MirRelationExpr::Get {
                    id: Id::Local(LocalId::new(*id)),
                    typ,
                    access_strategy: AccessStrategy::UnknownOrLocal,
                }
            }

            Ok(())
        })
    }

    /// Populates `expression` with necessary `Let` bindings.
    ///
    /// This population may result in substantially more `Let` bindings that one
    /// might expect. It is very appropriate to run the `NormalizeLets` transformation
    /// afterwards to remove `Let` bindings that it deems unhelpful.
    fn populate_expression(self, expression: &mut MirRelationExpr) {
        // Convert the bindings in to a sequence, by the local identifier.
        let mut bindings = self.bindings.into_iter().collect::<Vec<_>>();
        bindings.sort_by_key(|(_, i)| *i);

        for (value, index) in bindings.into_iter().rev() {
            let new_expression = MirRelationExpr::Let {
                id: LocalId::new(index),
                value: Box::new(value),
                body: Box::new(expression.take_dangerous()),
            };
            *expression = new_expression;
        }
    }
}
