// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Join flattening: each join region becomes one flat n-ary `Join`.
//!
//! See play/optimizer-rebuild/design-join-flatten.md for the design.
//! A *join region* is a maximal tree of `Join`, `Filter`, and `Project`
//! nodes. Each region is rebuilt as
//! `Project(Filter(Join(flat inputs, merged equivalences)))`, with empty
//! Filter and identity Project layers elided.
//!
//! Equality conjuncts whose support spans multiple flat inputs are absorbed
//! into the equivalence classes, with `IS NOT NULL` guards synthesized for
//! both sides (filter equality is SQL equality, which drops NULLs; join
//! equivalences use datum equality, where NULL matches NULL). Guards over
//! non-nullable columns are reduced away at emission.
//!
//! **Error discipline.** A `Filter` below a join joins the region only if
//! all its conjuncts are infallible: lifting an infallible conjunct into
//! the residue is harmless (predicate_placement re-sinks it), and lifting a
//! fallible one could only *remove* errors (the flat form evaluates it on
//! join results, each containing a row the nested form also evaluated) —
//! but it would also strand the conjunct above the join, so we leave such
//! Filters as region leaves instead. Fallible *equivalence members* are
//! parity-with-main: MIR leaves Join constraint evaluation order
//! unspecified and main's JoinFusion flattens unconditionally.
//!
//! **Contract.** Requires: a [`BindingEnv`] with conjunction-free `Filter`s
//! (filter_split). Ensures: no Join directly under Join/Filter/Project;
//! equivalences canonicalized; Filters remain conjunction-free; each
//! region's visible columns are unchanged.

use mz_expr::visit::{Visit, VisitChildren};
use mz_expr::{BinaryFunc, Columns, Eval, MirRelationExpr, MirScalarExpr};
use mz_repr::ReprColumnType;

use crate::rebuild::env::BindingEnv;
use crate::rebuild::predicate_placement::remap_columns;

/// Applies the transform to every binding body and the root.
pub fn apply(env: &mut BindingEnv) {
    // Bindings whose (already processed) bodies are the join identity, so
    // `Get`s of them collapse too. Bindings are in dependency order, so a
    // binding's unit-ness is settled before any use site is processed.
    let mut units = std::collections::BTreeSet::new();
    for (id, body) in env.bindings.iter_mut() {
        apply_expr(body, &units);
        if matches!(single_row_literals(body, &units), Some(lits) if lits.is_empty()) {
            units.insert(*id);
        }
    }
    apply_expr(&mut env.root, &units);
}

/// Flattens every join region in `expr`, recursively.
fn apply_expr(expr: &mut MirRelationExpr, units: &std::collections::BTreeSet<mz_expr::LocalId>) {
    if starts_region(expr) {
        let taken = std::mem::replace(
            expr,
            MirRelationExpr::constant(vec![], mz_repr::ReprRelationType::new(vec![])),
        );
        let mut region = flatten_region(taken, true);
        // Regions stopped at boundaries (Map, fallible Filter, ...) may hold
        // further joins inside their leaves.
        for input in region.inputs.iter_mut() {
            apply_expr(input, units);
        }
        *expr = emit(region, units);
    } else {
        expr.visit_mut_children(|child| apply_expr(child, units));
    }
}

/// True iff `expr` is a Filter/Project chain ending at a `Join`.
fn starts_region(expr: &MirRelationExpr) -> bool {
    let mut cur = expr;
    loop {
        match cur {
            MirRelationExpr::Join { .. } => return true,
            MirRelationExpr::Filter { input, .. } | MirRelationExpr::Project { input, .. } => {
                cur = input
            }
            _ => return false,
        }
    }
}

/// A join region mid-flattening. `equivalences`, `residue`, and `outputs`
/// are all in *flat* coordinates: the concatenation of `inputs`' columns.
struct Region {
    inputs: Vec<MirRelationExpr>,
    equivalences: Vec<Vec<MirScalarExpr>>,
    residue: Vec<MirScalarExpr>,
    /// Maps the region's visible columns to flat columns.
    outputs: Vec<usize>,
    /// Total flat arity (sum of input arities).
    flat_arity: usize,
}

/// Decomposes a Filter/Project/Join tree into a [`Region`]. `above_top_join`
/// is true while we have not yet descended through a `Join`: Filter layers
/// up there sit at residue height already, so fallibility is no obstacle.
fn flatten_region(expr: MirRelationExpr, above_top_join: bool) -> Region {
    match expr {
        MirRelationExpr::Filter { input, predicates }
            if above_top_join || predicates.iter().all(|p| !p.could_error()) =>
        {
            let mut region = flatten_region(*input, above_top_join);
            for mut p in predicates {
                remap_columns(&mut p, |c| region.outputs[c]);
                match as_cross_input_equality(&p, &region) {
                    Some((e1, e2)) => {
                        // Guards on BOTH sides: one would suffice for row
                        // semantics (the class then implies the other), but
                        // each guard sinks to a different input, and the
                        // early filters are what downstream sharing sees.
                        region.residue.push(e1.clone().call_is_null().not());
                        region.residue.push(e2.clone().call_is_null().not());
                        region.equivalences.push(vec![e1, e2]);
                    }
                    None => region.residue.push(p),
                }
            }
            region
        }
        MirRelationExpr::Project { input, outputs } => {
            let region = flatten_region(*input, above_top_join);
            Region {
                outputs: outputs.iter().map(|c| region.outputs[*c]).collect(),
                ..region
            }
        }
        MirRelationExpr::Join {
            inputs,
            equivalences,
            ..
        } => {
            let regions: Vec<Region> = inputs
                .into_iter()
                .map(|input| flatten_region(input, false))
                .collect();
            let mut flat = Region {
                inputs: Vec::new(),
                equivalences: Vec::new(),
                residue: Vec::new(),
                outputs: Vec::new(),
                flat_arity: 0,
            };
            for mut r in regions {
                let offset = flat.flat_arity;
                flat.outputs.extend(r.outputs.iter().map(|c| offset + c));
                for mut class in r.equivalences {
                    for member in class.iter_mut() {
                        remap_columns(member, |c| offset + c);
                    }
                    flat.equivalences.push(class);
                }
                for mut p in r.residue {
                    remap_columns(&mut p, |c| offset + c);
                    flat.residue.push(p);
                }
                flat.inputs.append(&mut r.inputs);
                flat.flat_arity += r.flat_arity;
            }
            // The outer equivalences reference the concatenated *visible*
            // columns of the child regions, which `flat.outputs` maps.
            for mut class in equivalences {
                for member in class.iter_mut() {
                    remap_columns(member, |c| flat.outputs[c]);
                }
                flat.equivalences.push(class);
            }
            flat
        }
        leaf => {
            let arity = leaf.arity();
            Region {
                inputs: vec![leaf],
                equivalences: Vec::new(),
                residue: Vec::new(),
                outputs: (0..arity).collect(),
                flat_arity: arity,
            }
        }
    }
}

/// If `p` is `e1 = e2` with support spanning more than one flat input,
/// returns the sides. Single-input equalities stay as residue: placement
/// will sink them below the join, which beats hoisting them into the
/// equivalence classes.
fn as_cross_input_equality(
    p: &MirScalarExpr,
    region: &Region,
) -> Option<(MirScalarExpr, MirScalarExpr)> {
    let MirScalarExpr::CallBinary {
        func: BinaryFunc::Eq(_),
        expr1,
        expr2,
    } = p
    else {
        return None;
    };
    let mut boundaries = Vec::with_capacity(region.inputs.len());
    let mut acc = 0;
    for input in region.inputs.iter() {
        acc += input.arity();
        boundaries.push(acc);
    }
    let home = |c: &usize| boundaries.iter().position(|b| c < b);
    let support = p.support();
    let mut homes = support.iter().map(home);
    let first = homes.next()?;
    if homes.all(|h| h == first) {
        None
    } else {
        Some(((**expr1).clone(), (**expr2).clone()))
    }
}

/// Emits `Project(Map(Filter(Join(...))))` for a finished region,
/// canonicalizing equivalences and reducing residue conjuncts against the
/// flat types (which removes guards over non-nullable columns). Trivial
/// layers elide.
///
/// Single-row constant inputs (multiplicity one) are removed from the join:
/// they multiply rows by one and contribute only fixed values, which we
/// substitute as literals into the equivalences and residue, and re-emit as
/// `Map` expressions for any visible columns. These arise from outer-join
/// and aggregate-default lowerings and otherwise cost arrangements.
fn emit(region: Region, units: &std::collections::BTreeSet<mz_expr::LocalId>) -> MirRelationExpr {
    let Region {
        inputs,
        mut equivalences,
        mut residue,
        outputs,
        flat_arity: _,
    } = region;
    // Partition inputs: single-row constants become per-column literal
    // substitutions; the rest are kept, their columns compacted.
    let mut subst: std::collections::BTreeMap<usize, MirScalarExpr> = Default::default();
    let mut old_to_new: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut kept = Vec::new();
    let mut offset = 0;
    for input in inputs {
        let arity = input.arity();
        match single_row_literals(&input, units) {
            Some(literals) => {
                for (j, lit) in literals.into_iter().enumerate() {
                    subst.insert(offset + j, lit);
                }
            }
            None => {
                for j in 0..arity {
                    old_to_new.insert(offset + j, old_to_new.len());
                }
                kept.push(input);
            }
        }
        offset += arity;
    }
    let kept_arity = old_to_new.len();
    if !subst.is_empty() {
        let rewrite = |e: &mut MirScalarExpr| {
            e.visit_mut_post(&mut |node| {
                if let MirScalarExpr::Column(c, _) = node {
                    if let Some(lit) = subst.get(c) {
                        *node = lit.clone();
                    } else {
                        let n = old_to_new[&*c];
                        *c = n;
                    }
                }
            });
        };
        for class in equivalences.iter_mut() {
            for member in class.iter_mut() {
                rewrite(member);
            }
        }
        for p in residue.iter_mut() {
            rewrite(p);
        }
    }
    if kept.is_empty() {
        kept.push(MirRelationExpr::constant(
            vec![vec![]],
            mz_repr::ReprRelationType::new(vec![]),
        ));
    }
    // Visible columns of removed constants become appended Map expressions.
    let mut appended = Vec::new();
    let mut appended_at: std::collections::BTreeMap<usize, usize> = Default::default();
    let outputs: Vec<usize> = outputs
        .iter()
        .map(|o| match old_to_new.get(o) {
            Some(n) => *n,
            None => *appended_at.entry(*o).or_insert_with(|| {
                appended.push(subst[o].clone());
                kept_arity + appended.len() - 1
            }),
        })
        .collect();

    if kept.len() == 1 && equivalences.is_empty() {
        let mut result = kept.pop().expect("len checked");
        if !residue.is_empty() {
            // Residue conjuncts have not been reduced against types here;
            // keep them as written (placement will route them).
            result = result.filter(residue);
        }
        if !appended.is_empty() {
            result = result.map(appended);
        }
        let emitted_arity = kept_arity + appended_at.len();
        if outputs.len() != emitted_arity || outputs.iter().enumerate().any(|(i, c)| i != *c) {
            result = result.project(outputs);
        }
        return result;
    }
    let inputs = kept;
    let flat_arity = kept_arity;
    // One typ() per input, gathered once (cost discipline: see the
    // capability map's notes on typ() in transforms).
    let input_types: Vec<Vec<ReprColumnType>> = inputs
        .iter()
        .map(|input| input.typ().column_types)
        .collect();
    mz_expr::canonicalize::canonicalize_equivalences(&mut equivalences, input_types.iter());
    // Columns in one equivalence class are datum-equal on every surviving
    // row, so the region's visible columns can all reference the class
    // representative (least column); thinning then drops the duplicates
    // from the join, and the region's Project re-duplicates above it —
    // arrangements store one copy.
    let mut representative: std::collections::BTreeMap<usize, usize> = Default::default();
    for class in equivalences.iter() {
        let mut cols: Vec<usize> = class
            .iter()
            .filter_map(|m| match m {
                MirScalarExpr::Column(c, _) => Some(*c),
                _ => None,
            })
            .collect();
        cols.sort();
        for c in cols.iter().skip(1) {
            representative.insert(*c, cols[0]);
        }
    }
    let outputs: Vec<usize> = outputs
        .iter()
        .map(|o| representative.get(o).copied().unwrap_or(*o))
        .collect();
    let flat_types: Vec<ReprColumnType> = input_types.into_iter().flatten().collect();
    for p in residue.iter_mut() {
        p.reduce(&flat_types);
    }
    residue.retain(|p| !p.is_literal_true());
    residue.sort();
    residue.dedup();

    let mut result = MirRelationExpr::join_scalars(inputs, equivalences);
    if !residue.is_empty() {
        result = result.filter(residue);
    }
    if !appended.is_empty() {
        result = result.map(appended);
    }
    let emitted_arity = flat_arity + appended_at.len();
    if outputs.len() != emitted_arity || outputs.iter().enumerate().any(|(i, c)| i != *c) {
        result = result.project(outputs);
    }
    result
}

/// If `input` is a constant with exactly one row at multiplicity one (or a
/// `Get` of a binding known to hold the zero-column one), returns its
/// columns as literal expressions. Such an input multiplies the join by one
/// and contributes only fixed values.
fn single_row_literals(
    input: &MirRelationExpr,
    units: &std::collections::BTreeSet<mz_expr::LocalId>,
) -> Option<Vec<MirScalarExpr>> {
    match input {
        MirRelationExpr::Constant {
            rows: Ok(rows),
            typ,
        } if rows.len() == 1 && rows[0].1 == mz_repr::Diff::ONE => Some(
            rows[0]
                .0
                .iter()
                .zip(typ.column_types.iter())
                .map(|(datum, ty)| MirScalarExpr::literal_ok(datum, ty.scalar_type.clone()))
                .collect(),
        ),
        MirRelationExpr::Get {
            id: mz_expr::Id::Local(id),
            ..
        } if units.contains(id) => Some(Vec::new()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use mz_expr::func;
    use mz_repr::{Datum, ReprRelationType, ReprScalarType};

    use super::*;
    use crate::rebuild::env::BindingEnv;

    fn table2(nullable: bool) -> MirRelationExpr {
        // Get rather than Constant: typ() on an empty Constant tightens
        // nullability to false, which would elide the guards under test.
        MirRelationExpr::local_get(
            mz_expr::LocalId::new(0),
            ReprRelationType::new(vec![ReprScalarType::Int64.nullable(nullable); 2]),
        )
    }

    fn eq(a: usize, b: usize) -> MirScalarExpr {
        MirScalarExpr::column(a).call_binary(MirScalarExpr::column(b), func::Eq)
    }

    /// Strips re-duplicating Project layers introduced by representative
    /// substitution; tests assert the structure beneath.
    fn through_projects(mut e: &MirRelationExpr) -> &MirRelationExpr {
        while let MirRelationExpr::Project { input, .. } = e {
            e = input;
        }
        e
    }

    fn flatten(root: MirRelationExpr) -> MirRelationExpr {
        let mut env = BindingEnv {
            bindings: vec![],
            root,
        };
        apply(&mut env);
        env.root
    }

    #[mz_ore::test]
    fn flattens_nested_joins() {
        let inner = MirRelationExpr::join(
            vec![table2(false), table2(false)],
            vec![vec![(0, 0), (1, 0)]],
        );
        let outer = MirRelationExpr::join(vec![inner, table2(false)], vec![vec![(0, 1), (1, 1)]]);
        let result = flatten(outer);
        let MirRelationExpr::Join {
            inputs,
            equivalences,
            ..
        } = through_projects(&result)
        else {
            panic!("expected flat Join at root, got {:?}", result)
        };
        assert_eq!(inputs.len(), 3);
        assert_eq!(equivalences.len(), 2);
        assert!(
            inputs
                .iter()
                .all(|i| !matches!(i, MirRelationExpr::Join { .. }))
        );
    }

    #[mz_ore::test]
    fn absorbs_equality_with_guard() {
        // Nullable columns: the guard must survive.
        let cross = MirRelationExpr::join(vec![table2(true), table2(true)], vec![]);
        let result = flatten(cross.filter(vec![eq(0, 2)]));
        let MirRelationExpr::Filter { input, predicates } = through_projects(&result) else {
            panic!("expected guard Filter at root, got {:?}", result)
        };
        assert_eq!(
            predicates,
            &vec![
                MirScalarExpr::column(0).call_is_null().not(),
                MirScalarExpr::column(2).call_is_null().not(),
            ]
        );
        assert!(
            matches!(&**input, MirRelationExpr::Join { equivalences, .. }
            if equivalences == &vec![vec![MirScalarExpr::column(0), MirScalarExpr::column(2)]])
        );
    }

    #[mz_ore::test]
    fn elides_guard_for_non_nullable() {
        let cross = MirRelationExpr::join(vec![table2(false), table2(false)], vec![]);
        let result = flatten(cross.filter(vec![eq(0, 2)]));
        assert!(
            matches!(through_projects(&result), MirRelationExpr::Join { equivalences, .. }
            if equivalences.len() == 1)
        );
    }

    #[mz_ore::test]
    fn single_input_equality_stays_residue() {
        let cross = MirRelationExpr::join(vec![table2(false), table2(false)], vec![]);
        let result = flatten(cross.filter(vec![eq(0, 1)]));
        let MirRelationExpr::Filter { input, predicates } = &result else {
            panic!("expected residue Filter at root, got {:?}", result)
        };
        assert_eq!(predicates, &vec![eq(0, 1)]);
        assert!(
            matches!(&**input, MirRelationExpr::Join { equivalences, .. }
            if equivalences.is_empty())
        );
    }

    #[mz_ore::test]
    fn fallible_filter_below_join_stops_region() {
        let div = MirScalarExpr::literal_ok(Datum::Int64(1), ReprScalarType::Int64)
            .call_binary(MirScalarExpr::column(0), func::DivInt64)
            .call_binary(
                MirScalarExpr::literal_ok(Datum::Int64(0), ReprScalarType::Int64),
                func::Gt,
            );
        let inner = MirRelationExpr::join(
            vec![table2(false), table2(false)],
            vec![vec![(0, 0), (1, 0)]],
        )
        .filter(vec![div]);
        let outer = MirRelationExpr::join(vec![inner, table2(false)], vec![vec![(0, 1), (1, 1)]]);
        let result = flatten(outer);
        let MirRelationExpr::Join { inputs, .. } = through_projects(&result) else {
            panic!("expected Join at root, got {:?}", result)
        };
        // The fallible Filter is a leaf; its inner join still flattened
        // (trivially) beneath it.
        assert_eq!(inputs.len(), 2);
        assert!(matches!(
            through_projects(&inputs[0]),
            MirRelationExpr::Filter { .. }
        ));
    }

    #[mz_ore::test]
    fn collapses_unit_inputs() {
        let unit = MirRelationExpr::constant(vec![vec![]], ReprRelationType::new(vec![]));
        let result = flatten(MirRelationExpr::join(vec![unit, table2(false)], vec![]));
        assert!(
            matches!(&result, MirRelationExpr::Get { .. }),
            "expected bare Get, got {:?}",
            result
        );
    }
}
