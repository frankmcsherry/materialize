// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Renders a plan into a timely/differential dataflow computation.
//!
//! ## Error handling
//!
//! Timely and differential have no idioms for computations that can error. The
//! philosophy is, reasonably, to define the semantics of the computation such
//! that errors are unnecessary: e.g., by using wrap-around semantics for
//! integer overflow.
//!
//! Unfortunately, SQL semantics are not nearly so elegant, and require errors
//! in myriad cases. The classic example is a division by zero, but invalid
//! input for casts, overflowing integer operations, and dozens of other
//! functions need the ability to produce errors ar runtime.
//!
//! At the moment, only *scalar* expression evaluation can fail, so only
//! operators that evaluate scalar expressions can fail. At the time of writing,
//! that includes map, filter, reduce, and join operators. Constants are a bit
//! of a special case: they can be either a constant vector of rows *or* a
//! constant, singular error.
//!
//! The approach taken is to build two parallel trees of computation: one for
//! the rows that have been successfully evaluated (the "oks tree"), and one for
//! the errors that have been generated (the "errs tree"). For example:
//!
//! ```text
//!    oks1  errs1       oks2  errs2
//!      |     |           |     |
//!      |     |           |     |
//!   project  |           |     |
//!      |     |           |     |
//!      |     |           |     |
//!     map    |           |     |
//!      |\    |           |     |
//!      | \   |           |     |
//!      |  \  |           |     |
//!      |   \ |           |     |
//!      |    \|           |     |
//!   project  +           +     +
//!      |     |          /     /
//!      |     |         /     /
//!    join ------------+     /
//!      |     |             /
//!      |     | +----------+
//!      |     |/
//!     oks   errs
//! ```
//!
//! The project operation cannot fail, so errors from errs1 are propagated
//! directly. Map operators are fallible and so can inject additional errors
//! into the stream. Join operators combine the errors from each of their
//! inputs.
//!
//! The semantics of the error stream are minimal. From the perspective of SQL,
//! a dataflow is considered to be in an error state if there is at least one
//! element in the final errs collection. The error value returned to the user
//! is selected arbitrarily; SQL only makes provisions to return one error to
//! the user at a time. There are plans to make the err collection accessible to
//! end users, so they can see all errors at once.
//!
//! To make errors transient, simply ensure that the operator can retract any
//! produced errors when corrected data arrives. To make errors permanent, write
//! the operator such that it never retracts the errors it produced. Future work
//! will likely want to introduce some sort of sort order for errors, so that
//! permanent errors are returned to the user ahead of transient errors—probably
//! by introducing a new error type a la:
//!
//! ```no_run
//! # struct EvalError;
//! # struct SourceError;
//! enum DataflowError {
//!     Transient(EvalError),
//!     Permanent(SourceError),
//! }
//! ```
//!
//! If the error stream is empty, the oks stream must be correct. If the error
//! stream is non-empty, then there are no semantics for the oks stream. This is
//! sufficient to support SQL in its current form, but is likely to be
//! unsatisfactory long term. We suspect that we can continue to imbue the oks
//! stream with semantics if we are very careful in describing what data should
//! and should not be produced upon encountering an error. Roughly speaking, the
//! oks stream could represent the correct result of the computation where all
//! rows that caused an error have been pruned from the stream. There are
//! strange and confusing questions here around foreign keys, though: what if
//! the optimizer proves that a particular key must exist in a collection, but
//! the key gets pruned away because its row participated in a scalar expression
//! evaluation that errored?
//!
//! In the meantime, it is probably wise for operators to keep the oks stream
//! roughly "as correct as possible" even when errors are present in the errs
//! stream. This reduces the amount of recomputation that must be performed
//! if/when the errors are retracted.

use std::collections::BTreeMap;
use std::rc::Rc;

use differential_dataflow::AsCollection;
use timely::dataflow::operators::to_stream::ToStream;
use timely::dataflow::operators::InspectCore;
use timely::dataflow::scopes::Child;
use timely::dataflow::{Scope, Stream};
use timely::order::Product;
use timely::progress::Timestamp;
use timely::PartialOrder;

use mz_compute_client::plan::Plan;
use mz_compute_client::types::dataflows::IndexDesc;
use mz_expr::Id;
use mz_repr::{Diff, GlobalId, Row};
use mz_storage_client::types::errors::DataflowError;

use crate::compute_state::ComputeState;
use crate::logging::compute::ComputeEvent;
use crate::logging::compute::Logger;
pub use context::CollectionBundle;
use context::{ArrangementFlavor, Context};

pub mod context;
mod flat_map;
mod iteration;
mod join;
mod reduce;
pub mod sinks;
mod threshold;
mod top_k;

pub use iteration::build_compute_dataflow;

// This helper function adds an operator to track source instantiation frontier advancements
// in a dataflow. The tracking supports instrospection sources populated by compute logging.
fn intercept_source_instantiation_frontiers<G>(
    source_instantiation: &Stream<G, (Row, mz_repr::Timestamp, Diff)>,
    logger: Logger,
    source_id: GlobalId,
    dataflow_ids: Vec<GlobalId>,
) -> Stream<G, (Row, mz_repr::Timestamp, Diff)>
where
    G: Scope<Timestamp = mz_repr::Timestamp>,
{
    let mut previous_time = None;
    source_instantiation.inspect_container(move |event| {
        if let Err(frontier) = event {
            if let Some(previous) = previous_time {
                for dataflow_id in dataflow_ids.iter() {
                    logger.log(ComputeEvent::SourceFrontier(
                        *dataflow_id,
                        source_id,
                        previous,
                        -1,
                    ));
                }
            }
            if let Some(time) = frontier.get(0) {
                for dataflow_id in dataflow_ids.iter() {
                    logger.log(ComputeEvent::SourceFrontier(
                        *dataflow_id,
                        source_id,
                        *time,
                        1,
                    ));
                }
                previous_time = Some(*time);
            } else {
                previous_time = None;
            }
        }
    })
}

// This implementation block allows child timestamps to vary from parent timestamps,
// but requires the parent timestamp to be `repr::Timestamp`.
impl<'g, G, T> Context<Child<'g, G, T>, Row>
where
    G: Scope<Timestamp = mz_repr::Timestamp>,
    T: Refines<G::Timestamp> + RenderTimestamp,
{
    pub(crate) fn import_index(
        &mut self,
        compute_state: &mut ComputeState,
        tokens: &mut BTreeMap<GlobalId, Rc<dyn std::any::Any>>,
        scope: &mut G,
        region: &mut Child<'g, G, T>,
        idx_id: GlobalId,
        idx: &IndexDesc,
    ) {
        if let Some(traces) = compute_state.traces.get_mut(&idx_id) {
            assert!(
                PartialOrder::less_equal(&traces.compaction_frontier(), &self.as_of_frontier),
                "Index {idx_id} has been allowed to compact beyond the dataflow as_of"
            );

            let token = traces.to_drop().clone();
            let (ok_arranged, ok_button) = traces.oks_mut().import_frontier_core(
                scope,
                &format!("Index({}, {:?})", idx.on_id, idx.key),
                self.as_of_frontier.clone(),
                self.until.clone(),
            );
            let (err_arranged, err_button) = traces.errs_mut().import_frontier_core(
                scope,
                &format!("ErrIndex({}, {:?})", idx.on_id, idx.key),
                self.as_of_frontier.clone(),
                self.until.clone(),
            );
            let ok_arranged = ok_arranged.enter(region);
            let err_arranged = err_arranged.enter(region);
            self.update_id(
                Id::Global(idx.on_id),
                CollectionBundle::from_expressions(
                    idx.key.clone(),
                    ArrangementFlavor::Trace(idx_id, ok_arranged, err_arranged),
                ),
            );
            tokens.insert(
                idx_id,
                Rc::new((ok_button.press_on_drop(), err_button.press_on_drop(), token)),
            );
        } else {
            panic!(
                "import of index {} failed while building dataflow {}",
                idx_id, self.dataflow_id
            );
        }
    }
}

impl<G> Context<G, Row>
where
    G: Scope,
    G::Timestamp: RenderTimestamp,
{
    /// Renders a plan to a differential dataflow, producing the collection of results.
    ///
    /// The return type reflects the uncertainty about the data representation, perhaps
    /// as a stream of data, perhaps as an arrangement, perhaps as a stream of batches.
    pub fn render_plan(
        &mut self,
        plan: Plan,
        scope: &mut G,
        worker_index: usize,
    ) -> CollectionBundle<G, Row> {
        match plan {
            Plan::Constant { rows } => {
                // Produce both rows and errs to avoid conditional dataflow construction.
                let (rows, errs) = match rows {
                    Ok(rows) => (rows, Vec::new()),
                    Err(e) => (Vec::new(), vec![e]),
                };

                // We should advance times in constant collections to start from `as_of`.
                let as_of_frontier = self.as_of_frontier.clone();
                let until = self.until.clone();
                let ok_collection = rows
                    .into_iter()
                    .filter_map(move |(row, mut time, diff)| {
                        time.advance_by(as_of_frontier.borrow());
                        if !until.less_equal(&time) {
                            Some((
                                row,
                                <G::Timestamp as Refines<mz_repr::Timestamp>>::to_inner(time),
                                diff,
                            ))
                        } else {
                            None
                        }
                    })
                    .to_stream(scope)
                    .as_collection();

                let mut error_time: mz_repr::Timestamp = Timestamp::minimum();
                error_time.advance_by(self.as_of_frontier.borrow());
                let err_collection = errs
                    .into_iter()
                    .map(move |e| {
                        (
                            DataflowError::from(e),
                            <G::Timestamp as Refines<mz_repr::Timestamp>>::to_inner(error_time),
                            1,
                        )
                    })
                    .to_stream(scope)
                    .as_collection();

                CollectionBundle::from_collections(ok_collection, err_collection)
            }
            Plan::Get { id, keys, plan } => {
                // Recover the collection from `self` and then apply `mfp` to it.
                // If `mfp` happens to be trivial, we can just return the collection.
                let mut collection = self
                    .lookup_id(id)
                    .unwrap_or_else(|| panic!("Get({:?}) not found at render time", id));
                match plan {
                    mz_compute_client::plan::GetPlan::PassArrangements => {
                        // Assert that each of `keys` are present in `collection`.
                        assert!(keys
                            .arranged
                            .iter()
                            .all(|(key, _, _)| collection.arranged.contains_key(key)));
                        assert!(keys.raw <= collection.collection.is_some());
                        // Retain only those keys we want to import.
                        collection.arranged.retain(|key, _value| {
                            keys.arranged.iter().any(|(key2, _, _)| key2 == key)
                        });
                        collection
                    }
                    mz_compute_client::plan::GetPlan::Arrangement(key, row, mfp) => {
                        let (oks, errs) = collection.as_collection_core(
                            mfp,
                            Some((key, row)),
                            self.until.clone(),
                        );
                        CollectionBundle::from_collections(oks, errs)
                    }
                    mz_compute_client::plan::GetPlan::Collection(mfp) => {
                        let (oks, errs) =
                            collection.as_collection_core(mfp, None, self.until.clone());
                        CollectionBundle::from_collections(oks, errs)
                    }
                }
            }
            Plan::Let { id, value, body } => {
                // Render `value` and bind it to `id`. Complain if this shadows an id.
                let value = self.render_plan(*value, scope, worker_index);
                let prebound = self.insert_id(Id::Local(id), value);
                assert!(prebound.is_none());

                let body = self.render_plan(*body, scope, worker_index);
                self.remove_id(Id::Local(id));
                body
            }
            Plan::Mfp {
                input,
                mfp,
                input_key_val,
            } => {
                let input = self.render_plan(*input, scope, worker_index);
                // If `mfp` is non-trivial, we should apply it and produce a collection.
                if mfp.is_identity() {
                    input
                } else {
                    let (oks, errs) =
                        input.as_collection_core(mfp, input_key_val, self.until.clone());
                    CollectionBundle::from_collections(oks, errs)
                }
            }
            Plan::FlatMap {
                input,
                func,
                exprs,
                mfp,
                input_key,
            } => {
                let input = self.render_plan(*input, scope, worker_index);
                self.render_flat_map(input, func, exprs, mfp, input_key)
            }
            Plan::Join { inputs, plan } => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| self.render_plan(input, scope, worker_index))
                    .collect();
                match plan {
                    mz_compute_client::plan::join::JoinPlan::Linear(linear_plan) => {
                        self.render_join(inputs, linear_plan, scope)
                    }
                    mz_compute_client::plan::join::JoinPlan::Delta(delta_plan) => {
                        self.render_delta_join(inputs, delta_plan, scope)
                    }
                }
            }
            Plan::Reduce {
                input,
                key_val_plan,
                plan,
                input_key,
            } => {
                let input = self.render_plan(*input, scope, worker_index);
                self.render_reduce(input, key_val_plan, plan, input_key)
            }
            Plan::TopK { input, top_k_plan } => {
                let input = self.render_plan(*input, scope, worker_index);
                self.render_topk(input, top_k_plan)
            }
            Plan::Negate { input } => {
                let input = self.render_plan(*input, scope, worker_index);
                let (oks, errs) = input.as_specific_collection(None);
                CollectionBundle::from_collections(oks.negate(), errs)
            }
            Plan::Threshold {
                input,
                threshold_plan,
            } => {
                let input = self.render_plan(*input, scope, worker_index);
                self.render_threshold(input, threshold_plan)
            }
            Plan::Union { inputs } => {
                let mut oks = Vec::new();
                let mut errs = Vec::new();
                for input in inputs.into_iter() {
                    let (os, es) = self
                        .render_plan(input, scope, worker_index)
                        .as_specific_collection(None);
                    oks.push(os);
                    errs.push(es);
                }
                let oks = differential_dataflow::collection::concatenate(scope, oks);
                let errs = differential_dataflow::collection::concatenate(scope, errs);
                CollectionBundle::from_collections(oks, errs)
            }
            Plan::ArrangeBy {
                input,
                forms: keys,
                input_key,
                input_mfp,
            } => {
                let input = self.render_plan(*input, scope, worker_index);
                input.ensure_collections(keys, input_key, input_mfp, self.until.clone())
            }
        }
    }
}

use differential_dataflow::lattice::Lattice;
use mz_storage_client::source::persist_source::NO_FLOW_CONTROL;
use timely::progress::timestamp::Refines;

/// A timestamp type that can be used for operations within MZ's dataflow layer.
pub trait RenderTimestamp: Timestamp + Lattice + Refines<mz_repr::Timestamp> {
    /// The system timestamp component of the timestamp.
    ///
    /// This is useful for manipulating the system time, as when delaying
    /// updates for subsequent cancellation, as with monotonic reduction.
    fn system_time(&mut self) -> &mut mz_repr::Timestamp;
    /// Effects a system delay in terms of the timestamp summary.
    fn system_delay(delay: mz_repr::Timestamp) -> <Self as Timestamp>::Summary;
    /// The event timestamp component of the timestamp.
    fn event_time(&mut self) -> &mut mz_repr::Timestamp;
    /// Effects an event delay in terms of the timestamp summary.
    fn event_delay(delay: mz_repr::Timestamp) -> <Self as Timestamp>::Summary;
    /// Steps the timestamp back so that logical compaction to the output will
    /// not conflate `self` with any historical times.
    fn step_back(&self) -> Self;
}

impl RenderTimestamp for mz_repr::Timestamp {
    fn system_time(&mut self) -> &mut mz_repr::Timestamp {
        self
    }
    fn system_delay(delay: mz_repr::Timestamp) -> <Self as Timestamp>::Summary {
        delay
    }
    fn event_time(&mut self) -> &mut mz_repr::Timestamp {
        self
    }
    fn event_delay(delay: mz_repr::Timestamp) -> <Self as Timestamp>::Summary {
        delay
    }
    fn step_back(&self) -> Self {
        self.saturating_sub(1)
    }
}

impl<T: Timestamp + Lattice> RenderTimestamp for Product<mz_repr::Timestamp, T> {
    fn system_time(&mut self) -> &mut mz_repr::Timestamp {
        &mut self.outer
    }
    fn system_delay(delay: mz_repr::Timestamp) -> <Self as Timestamp>::Summary {
        Product::new(delay, Default::default())
    }
    fn event_time(&mut self) -> &mut mz_repr::Timestamp {
        &mut self.outer
    }
    fn event_delay(delay: mz_repr::Timestamp) -> <Self as Timestamp>::Summary {
        Product::new(delay, Default::default())
    }
    fn step_back(&self) -> Self {
        Product::new(self.outer.saturating_sub(1), self.inner.clone())
    }
}
