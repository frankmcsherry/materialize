// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Projection thinning: operators produce only columns consumers demand.
//!
//! See play/optimizer-rebuild/design-projection-thinning.md for the design.
//! Consumers are processed before producers: the root first (all columns
//! demanded), then bindings from last to first, each thinned against the
//! union of the demands its use sites recorded. [`thin`] rewrites an
//! expression in place to produce a superset of the demanded columns (in
//! ascending original order) and returns that list; callers remap their
//! references through it. Exact narrowing (a `Project`) is forced only
//! where layouts must agree: binding bodies and `Union` branches.
//!
//! Use sites of local bindings become `Project`-over-`Get` markers carrying
//! *original* binding columns; once every consumer has been processed and a
//! binding's layout is final, a fixup pass rewrites the markers (and the
//! `Get` types) into the new layout.
//!
//! **Error discipline.** The only deletions are `Map` scalars and `Reduce`
//! aggregates no consumer observes, and columns of `Constant`s. Dropping an
//! unobserved expression never changes row counts (`Map` and `Reduce`
//! multiplicities do not depend on it) and can only remove the possibility
//! of an error, which rule 1 permits.
//!
//! **Contract.** Requires: a [`BindingEnv`] (LetRec opaque: anything used
//! inside one is demanded in full, which keeps those bindings — and the
//! `Get`s inside the LetRec — at their original layout). Ensures: binding
//! bodies produce exactly the union of use-site demands; root and use-site
//! visible columns unchanged. May disturb the M-F-P normal form with
//! inserted `Project`s; the pipeline re-runs `linear_fuse` afterwards.

use std::collections::{BTreeMap, BTreeSet};

use mz_expr::visit::VisitChildren;
use mz_expr::{Columns, Id, LocalId, MirRelationExpr};
use mz_repr::ReprRelationType;

use crate::rebuild::env::BindingEnv;
use crate::rebuild::predicate_placement::remap_columns;

/// Applies the transform to the environment.
pub fn apply(env: &mut BindingEnv) {
    let mut demands: BTreeMap<LocalId, BTreeSet<usize>> = BTreeMap::new();
    // Root: all columns demanded.
    let root_arity = env.root.arity();
    let full: BTreeSet<usize> = (0..root_arity).collect();
    thin(&mut env.root, &full, &mut demands);
    // Bindings, consumers first. A binding's demand is final here because
    // its consumers (later bindings and the root) are all processed.
    let mut layouts: BTreeMap<LocalId, Layout> = BTreeMap::new();
    for (id, body) in env.bindings.iter_mut().rev() {
        let layout: Vec<usize> = demands.remove(id).unwrap_or_default().into_iter().collect();
        let demand: BTreeSet<usize> = layout.iter().cloned().collect();
        let produced = thin(body, &demand, &mut demands);
        if produced != layout {
            let positions: Vec<usize> = layout.iter().map(|l| position(&produced, *l)).collect();
            let inner = std::mem::replace(
                body,
                MirRelationExpr::constant(vec![], ReprRelationType::new(vec![])),
            );
            *body = inner.project(positions);
        }
        // If the body ends in a Project with duplicate outputs, the binding
        // would store the same column twice. Hoist the duplication to the
        // use sites: store the deduplicated body, and have each use site
        // re-duplicate (`dup` maps demanded positions to stored positions).
        let mut dup: Vec<usize> = (0..layout.len()).collect();
        if let MirRelationExpr::Project { input, outputs } = body {
            let mut stored: Vec<usize> = Vec::new();
            dup = outputs
                .iter()
                .map(|o| match stored.iter().position(|s| s == o) {
                    Some(pos) => pos,
                    None => {
                        stored.push(*o);
                        stored.len() - 1
                    }
                })
                .collect();
            if stored.len() < outputs.len() {
                let inner = std::mem::replace(
                    input,
                    Box::new(MirRelationExpr::constant(
                        vec![],
                        ReprRelationType::new(vec![]),
                    )),
                );
                *body = inner.project(stored);
            } else {
                dup = (0..layout.len()).collect();
            }
        }
        // Original column represented at each stored position.
        let stored_len = dup.iter().max().map_or(0, |m| m + 1);
        let mut stored_orig = vec![usize::MAX; stored_len];
        for (i, d) in dup.iter().enumerate() {
            if stored_orig[*d] == usize::MAX {
                stored_orig[*d] = layout[i];
            }
        }
        layouts.insert(
            *id,
            Layout {
                layout,
                dup,
                stored_orig,
            },
        );
    }
    // Fix up the use-site markers now that every layout is final.
    for (_id, body) in env.bindings.iter_mut() {
        fixup_markers(body, &layouts);
    }
    fixup_markers(&mut env.root, &layouts);
}

/// A thinned binding's final shape, for rewriting its use sites.
struct Layout {
    /// Original columns the binding's consumers demanded (ascending).
    layout: Vec<usize>,
    /// For each demanded position, the stored position holding its value.
    dup: Vec<usize>,
    /// For each stored position, the original column it represents.
    stored_orig: Vec<usize>,
}

/// Index of `col` in the produced-columns list `m`.
fn position(m: &[usize], col: usize) -> usize {
    m.binary_search(&col).expect("demanded column was produced")
}

/// Rewrites `expr` to produce a superset of `demand` (ascending original
/// columns) and returns the produced list. Records local `Get` demands.
fn thin(
    expr: &mut MirRelationExpr,
    demand: &BTreeSet<usize>,
    demands: &mut BTreeMap<LocalId, BTreeSet<usize>>,
) -> Vec<usize> {
    let owned = std::mem::replace(
        expr,
        MirRelationExpr::constant(vec![], ReprRelationType::new(vec![])),
    );
    let (rebuilt, produced) = thin_owned(owned, demand, demands);
    *expr = rebuilt;
    produced
}

fn thin_owned(
    expr: MirRelationExpr,
    demand: &BTreeSet<usize>,
    demands: &mut BTreeMap<LocalId, BTreeSet<usize>>,
) -> (MirRelationExpr, Vec<usize>) {
    match expr {
        MirRelationExpr::Constant { rows, typ } => {
            let produced: Vec<usize> = demand.iter().cloned().collect();
            let new_typ = narrow_typ(&typ, &produced);
            let new_rows = rows.map(|rows| {
                rows.into_iter()
                    .map(|(row, diff)| {
                        let datums: Vec<_> = row.iter().collect();
                        (
                            mz_repr::Row::pack(produced.iter().map(|c| datums[*c])),
                            diff,
                        )
                    })
                    .collect()
            });
            (
                MirRelationExpr::Constant {
                    rows: new_rows,
                    typ: new_typ,
                },
                produced,
            )
        }
        MirRelationExpr::Get {
            id: Id::Local(id),
            typ,
            access_strategy,
        } => {
            demands.entry(id).or_default().extend(demand.iter());
            let produced: Vec<usize> = demand.iter().cloned().collect();
            let get = MirRelationExpr::Get {
                id: Id::Local(id),
                typ,
                access_strategy,
            };
            if produced.len() == get.arity() {
                // Full demand: the binding's layout will be full too (it is
                // a union including this site), so no marker is needed.
                (get, produced)
            } else {
                // Marker: outputs are ORIGINAL binding columns until fixup.
                (get.project(produced.clone()), produced)
            }
        }
        get @ MirRelationExpr::Get { .. } => {
            let produced: Vec<usize> = demand.iter().cloned().collect();
            if produced.len() == get.arity() {
                (get, produced)
            } else {
                (get.project(produced.clone()), produced)
            }
        }
        MirRelationExpr::Project { mut input, outputs } => {
            let child_demand: BTreeSet<usize> = demand.iter().map(|d| outputs[*d]).collect();
            let m = thin(&mut input, &child_demand, demands);
            let produced: Vec<usize> = demand.iter().cloned().collect();
            let new_outputs: Vec<usize> =
                produced.iter().map(|d| position(&m, outputs[*d])).collect();
            (input.project(new_outputs), produced)
        }
        MirRelationExpr::Map { mut input, scalars } => {
            let a_in = input.arity();
            let mut needed = vec![false; scalars.len()];
            let mut input_demand: BTreeSet<usize> = BTreeSet::new();
            for d in demand {
                if *d >= a_in {
                    needed[*d - a_in] = true;
                } else {
                    input_demand.insert(*d);
                }
            }
            for i in (0..scalars.len()).rev() {
                if needed[i] {
                    for c in scalars[i].support() {
                        if c < a_in {
                            input_demand.insert(c);
                        } else {
                            needed[c - a_in] = true;
                        }
                    }
                }
            }
            let m_in = thin(&mut input, &input_demand, demands);
            let mut old_to_new: BTreeMap<usize, usize> =
                m_in.iter().enumerate().map(|(n, o)| (*o, n)).collect();
            let mut kept = Vec::new();
            let mut produced = m_in.clone();
            for (i, mut scalar) in scalars.into_iter().enumerate() {
                if needed[i] {
                    remap_columns(&mut scalar, |c| old_to_new[&c]);
                    old_to_new.insert(a_in + i, m_in.len() + kept.len());
                    kept.push(scalar);
                    produced.push(a_in + i);
                }
            }
            let rebuilt = if kept.is_empty() {
                *input
            } else {
                input.map(kept)
            };
            (rebuilt, produced)
        }
        MirRelationExpr::Filter {
            mut input,
            mut predicates,
        } => {
            let mut child_demand = demand.clone();
            for p in predicates.iter() {
                child_demand.extend(p.support());
            }
            let m = thin(&mut input, &child_demand, demands);
            for p in predicates.iter_mut() {
                remap_columns(p, |c| position(&m, c));
            }
            (input.filter(predicates), m)
        }
        MirRelationExpr::Join {
            mut inputs,
            mut equivalences,
            implementation,
        } => {
            let arities: Vec<usize> = inputs.iter().map(|i| i.arity()).collect();
            let mut starts = Vec::with_capacity(arities.len());
            let mut acc = 0;
            for a in &arities {
                starts.push(acc);
                acc += a;
            }
            let mut input_demands: Vec<BTreeSet<usize>> =
                inputs.iter().map(|_| BTreeSet::new()).collect();
            let locate = |c: usize| {
                let i = starts
                    .iter()
                    .rposition(|s| *s <= c)
                    .expect("column in range");
                (i, c - starts[i])
            };
            for d in demand {
                let (i, local) = locate(*d);
                input_demands[i].insert(local);
            }
            for class in equivalences.iter() {
                for member in class.iter() {
                    for c in member.support() {
                        let (i, local) = locate(c);
                        input_demands[i].insert(local);
                    }
                }
            }
            let mut produced = Vec::new();
            let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
            let mut new_offset = 0;
            for (i, input) in inputs.iter_mut().enumerate() {
                let m = thin(input, &input_demands[i], demands);
                for (n, local) in m.iter().enumerate() {
                    produced.push(starts[i] + local);
                    old_to_new.insert(starts[i] + local, new_offset + n);
                }
                new_offset += m.len();
            }
            for class in equivalences.iter_mut() {
                for member in class.iter_mut() {
                    remap_columns(member, |c| old_to_new[&c]);
                }
            }
            (
                MirRelationExpr::Join {
                    inputs,
                    equivalences,
                    implementation,
                },
                produced,
            )
        }
        MirRelationExpr::FlatMap {
            mut input,
            func,
            mut exprs,
        } => {
            let a_in = input.arity();
            let func_arity = func.output_type().column_types.len();
            let mut child_demand: BTreeSet<usize> =
                demand.iter().filter(|d| **d < a_in).cloned().collect();
            for e in exprs.iter() {
                child_demand.extend(e.support());
            }
            let m = thin(&mut input, &child_demand, demands);
            for e in exprs.iter_mut() {
                remap_columns(e, |c| position(&m, c));
            }
            let mut produced = m;
            produced.extend(a_in..a_in + func_arity);
            (MirRelationExpr::FlatMap { input, func, exprs }, produced)
        }
        MirRelationExpr::Reduce {
            mut input,
            group_key,
            aggregates,
            monotonic,
            expected_group_size,
        } => {
            let klen = group_key.len();
            let mut child_demand: BTreeSet<usize> = BTreeSet::new();
            for k in group_key.iter() {
                child_demand.extend(k.support());
            }
            let keep: Vec<bool> = (0..aggregates.len())
                .map(|j| demand.contains(&(klen + j)))
                .collect();
            for (j, agg) in aggregates.iter().enumerate() {
                if keep[j] {
                    child_demand.extend(agg.expr.support());
                }
            }
            let m = thin(&mut input, &child_demand, demands);
            let mut group_key = group_key;
            for k in group_key.iter_mut() {
                remap_columns(k, |c| position(&m, c));
            }
            let mut produced: Vec<usize> = (0..klen).collect();
            let mut kept_aggs = Vec::new();
            for (j, mut agg) in aggregates.into_iter().enumerate() {
                if keep[j] {
                    remap_columns(&mut agg.expr, |c| position(&m, c));
                    kept_aggs.push(agg);
                    produced.push(klen + j);
                }
            }
            (
                MirRelationExpr::Reduce {
                    input,
                    group_key,
                    aggregates: kept_aggs,
                    monotonic,
                    expected_group_size,
                },
                produced,
            )
        }
        MirRelationExpr::TopK {
            mut input,
            group_key,
            order_key,
            limit,
            offset,
            monotonic,
            expected_group_size,
        } => {
            let mut child_demand = demand.clone();
            child_demand.extend(group_key.iter().cloned());
            child_demand.extend(order_key.iter().map(|o| o.column));
            if let Some(l) = &limit {
                child_demand.extend(l.support());
            }
            let m = thin(&mut input, &child_demand, demands);
            let group_key = group_key.iter().map(|c| position(&m, *c)).collect();
            let mut order_key = order_key;
            for o in order_key.iter_mut() {
                o.column = position(&m, o.column);
            }
            let mut limit = limit;
            if let Some(l) = limit.as_mut() {
                remap_columns(l, |c| position(&m, c));
            }
            (
                MirRelationExpr::TopK {
                    input,
                    group_key,
                    order_key,
                    limit,
                    offset,
                    monotonic,
                    expected_group_size,
                },
                m,
            )
        }
        MirRelationExpr::Negate { mut input } => {
            let m = thin(&mut input, demand, demands);
            (MirRelationExpr::Negate { input }, m)
        }
        MirRelationExpr::Threshold { mut input } => {
            let m = thin(&mut input, demand, demands);
            (MirRelationExpr::Threshold { input }, m)
        }
        MirRelationExpr::Union {
            mut base,
            mut inputs,
        } => {
            let produced: Vec<usize> = demand.iter().cloned().collect();
            let narrow =
                |branch: &mut MirRelationExpr, demands: &mut BTreeMap<LocalId, BTreeSet<usize>>| {
                    let m = thin(branch, demand, demands);
                    if m != produced {
                        let positions: Vec<usize> =
                            produced.iter().map(|d| position(&m, *d)).collect();
                        let inner = std::mem::replace(
                            branch,
                            MirRelationExpr::constant(vec![], ReprRelationType::new(vec![])),
                        );
                        *branch = inner.project(positions);
                    }
                };
            narrow(&mut base, demands);
            for input in inputs.iter_mut() {
                narrow(input, demands);
            }
            (MirRelationExpr::Union { base, inputs }, produced)
        }
        MirRelationExpr::ArrangeBy { mut input, keys } => {
            // Arrangement keys are contracts; demand everything below.
            let full: BTreeSet<usize> = (0..input.arity()).collect();
            let m = thin(&mut input, &full, demands);
            (MirRelationExpr::ArrangeBy { input, keys }, m)
        }
        // LetRec (Phase-1 opaque) and anything unexpected: record full
        // demand for every local Get inside, thin nothing.
        opaque => {
            record_full_demands(&opaque, demands);
            let arity = opaque.arity();
            (opaque, (0..arity).collect())
        }
    }
}

/// Records full-arity demand for every local `Get` in `expr`'s subtree.
fn record_full_demands(expr: &MirRelationExpr, demands: &mut BTreeMap<LocalId, BTreeSet<usize>>) {
    expr.visit_pre(|e| {
        if let MirRelationExpr::Get {
            id: Id::Local(id),
            typ,
            ..
        } = e
        {
            demands.entry(*id).or_default().extend(0..typ.arity());
        }
    });
}

/// Rewrites `Project`-over-local-`Get` markers (outputs in original binding
/// columns) into the bindings' final layouts, narrowing the `Get` types.
fn fixup_markers(expr: &mut MirRelationExpr, layouts: &BTreeMap<LocalId, Layout>) {
    match expr {
        // A Project directly above a local Get is a marker: outputs are
        // original binding columns.
        MirRelationExpr::Project { input, outputs }
            if matches!(
                &**input,
                MirRelationExpr::Get {
                    id: Id::Local(_),
                    ..
                }
            ) =>
        {
            let MirRelationExpr::Get {
                id: Id::Local(id),
                typ,
                ..
            } = &mut **input
            else {
                unreachable!("matched above")
            };
            if let Some(l) = layouts.get(id) {
                for o in outputs.iter_mut() {
                    *o = l.dup[position(&l.layout, *o)];
                }
                *typ = narrow_typ(typ, &l.stored_orig);
            }
        }
        // A bare local Get is a full-demand site; if the binding stores a
        // deduplicated layout, re-duplicate here.
        MirRelationExpr::Get {
            id: Id::Local(id),
            typ,
            ..
        } => {
            if let Some(l) = layouts.get(id) {
                if l.dup.iter().enumerate().any(|(i, d)| i != *d) {
                    *typ = narrow_typ(typ, &l.stored_orig);
                    let inner = std::mem::replace(
                        expr,
                        MirRelationExpr::constant(vec![], ReprRelationType::new(vec![])),
                    );
                    *expr = inner.project(l.dup.clone());
                }
            }
        }
        other => other.visit_mut_children(|child| fixup_markers(child, layouts)),
    }
}

/// Restricts a relation type to the given original columns, keeping keys
/// that survive in full.
fn narrow_typ(typ: &ReprRelationType, produced: &[usize]) -> ReprRelationType {
    let column_types = produced
        .iter()
        .map(|c| typ.column_types[*c].clone())
        .collect();
    let mut new_typ = ReprRelationType::new(column_types);
    let kept: BTreeSet<usize> = produced.iter().cloned().collect();
    for key in typ.keys.iter() {
        if key.iter().all(|k| kept.contains(k)) {
            new_typ = new_typ.with_key(key.iter().map(|k| position(produced, *k)).collect());
        }
    }
    new_typ
}

#[cfg(test)]
mod tests {
    use mz_expr::{AggregateExpr, AggregateFunc, MirScalarExpr, func};
    use mz_repr::{Datum, ReprScalarType};

    use super::*;
    use crate::rebuild::env::BindingEnv;

    fn table3() -> MirRelationExpr {
        MirRelationExpr::local_get(
            LocalId::new(7),
            ReprRelationType::new(vec![ReprScalarType::Int64.nullable(true); 3]),
        )
    }

    #[mz_ore::test]
    fn drops_undemanded_fallible_scalar() {
        // Map [#0+#1, 1/#0]; only the first map column is demanded.
        let fallible = MirScalarExpr::literal_ok(Datum::Int64(1), ReprScalarType::Int64)
            .call_binary(MirScalarExpr::column(0), func::DivInt64);
        let mapped = table3().map(vec![
            MirScalarExpr::column(0).call_binary(MirScalarExpr::column(1), func::AddInt64),
            fallible,
        ]);
        let mut env = BindingEnv {
            bindings: vec![],
            root: mapped.project(vec![3]),
        };
        apply(&mut env);
        let MirRelationExpr::Project { input, .. } = &env.root else {
            panic!("expected Project at root, got {:?}", env.root)
        };
        let MirRelationExpr::Map { scalars, .. } = &**input else {
            panic!("expected Map below, got {:?}", input)
        };
        assert_eq!(scalars.len(), 1, "fallible unused scalar should drop");
    }

    #[mz_ore::test]
    fn narrows_binding_to_use_site_union() {
        // Binding produces 3 columns; uses demand {0} and {2}.
        let id = LocalId::new(0);
        let value = table3();
        let get = MirRelationExpr::local_get(
            id,
            ReprRelationType::new(vec![ReprScalarType::Int64.nullable(true); 3]),
        );
        let root = get.clone().project(vec![0]).union(get.project(vec![2]));
        let mut env = BindingEnv {
            bindings: vec![(id, value)],
            root,
        };
        apply(&mut env);
        // The binding body narrows to columns {0, 2}.
        assert_eq!(env.bindings[0].1.arity(), 2);
        // Use sites remap through the layout [0, 2].
        let MirRelationExpr::Union { base, inputs } = &env.root else {
            panic!("expected Union at root")
        };
        for (branch, expect) in [(&**base, 0), (&inputs[0], 1)] {
            let MirRelationExpr::Project { outputs, .. } = branch else {
                panic!("expected Project marker, got {:?}", branch)
            };
            assert_eq!(outputs, &vec![expect]);
        }
    }

    #[mz_ore::test]
    fn drops_undemanded_aggregate() {
        let agg = |col: usize, distinct| AggregateExpr {
            func: AggregateFunc::SumInt64,
            expr: MirScalarExpr::column(col),
            distinct,
        };
        let reduce = table3().reduce(vec![0], vec![agg(1, false), agg(2, false)], None);
        let mut env = BindingEnv {
            bindings: vec![],
            root: reduce.project(vec![0, 2]),
        };
        apply(&mut env);
        let MirRelationExpr::Project { input, .. } = &env.root else {
            panic!("expected Project at root")
        };
        let MirRelationExpr::Reduce { aggregates, .. } = &**input else {
            panic!("expected Reduce below, got {:?}", input)
        };
        assert_eq!(aggregates.len(), 1);
        assert_eq!(aggregates[0].expr, MirScalarExpr::column(1));
    }
}
