// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use differential_dataflow::AsCollection;
use timely::communication::Allocate;
use timely::dataflow::scopes::Child;
use timely::dataflow::Scope;
use timely::order::Product;
use timely::worker::Worker as TimelyWorker;

use mz_compute_client::plan::Plan;
use mz_compute_client::types::dataflows::{DataflowDescription, IndexDesc};
use mz_expr::Id;
use mz_ore::collections::CollectionExt as IteratorExt;
use mz_repr::{GlobalId, Row};
use mz_storage_client::controller::CollectionMetadata;
use mz_storage_client::source::persist_source;

pub use super::context::CollectionBundle;
use super::context::{ArrangementFlavor, Context};
use crate::arrangement::manager::TraceBundle;
use crate::compute_state::ComputeState;
use crate::render::RenderTimestamp;

/// Assemble the "compute"  side of a dataflow, i.e. all but the sources.
///
/// This method imports sources from provided assets, and then builds the remaining
/// dataflow using "compute-local" assets like shared arrangements, and producing
/// both arrangements and sinks.
pub fn build_compute_dataflow<A: Allocate>(
    timely_worker: &mut TimelyWorker<A>,
    compute_state: &mut ComputeState,
    dataflow: DataflowDescription<Plan, CollectionMetadata>,
) {
    let worker_logging = timely_worker.log_register().get("timely");
    let name = format!("Dataflow: {}", &dataflow.debug_name);

    timely_worker.dataflow_core(&name, worker_logging, Box::new(()), |_, scope| {
        // The scope.clone() occurs to allow import in the region.
        // We build a region here to establish a pattern of a scope inside the dataflow,
        // so that other similar uses (e.g. with iterative scopes) do not require weird
        // alternate type signatures.
        scope.clone().iterative::<usize, _, _>(|region| {
            let mut context = crate::render::context::Context::for_dataflow(
                &dataflow,
                scope.addr().into_element(),
            );
            let mut tokens = BTreeMap::new();

            // Import declared sources into the rendering context.
            for (source_id, (source, _monotonic)) in dataflow.source_imports.iter() {
                let mut mfp = source.arguments.operators.clone().map(|ops| {
                    mz_expr::MfpPlan::create_from(ops)
                        .expect("Linear operators should always be valid")
                });

                // Note: For correctness, we require that sources only emit times advanced by
                // `dataflow.as_of`. `persist_source` is documented to provide this guarantee.
                let (mut ok_stream, err_stream, token) = persist_source::persist_source(
                    scope,
                    *source_id,
                    Arc::clone(&compute_state.persist_clients),
                    source.storage_metadata.clone(),
                    dataflow.as_of.clone(),
                    dataflow.until.clone(),
                    mfp.as_mut(),
                    // TODO: provide a more meaningful flow control input
                    &timely::dataflow::operators::generic::operator::empty(scope),
                    super::NO_FLOW_CONTROL,
                    // Copy the logic in DeltaJoin/Get/Join to start.
                    |_timer, count| count > 1_000_000,
                );

                // If `mfp` is non-identity, we need to apply what remains.
                // For the moment, assert that it is either trivial or `None`.
                assert!(mfp.map(|x| x.is_identity()).unwrap_or(true));

                // If logging is enabled, intercept frontier advancements coming from persist to track materialization lags.
                // Note that we do this here instead of in the server.rs worker loop since we want to catch the wall-clock
                // time of the frontier advancement for each dataflow as early as possible.
                if let Some(logger) = compute_state.compute_logger.clone() {
                    let export_ids = dataflow.export_ids().collect();
                    ok_stream = crate::render::intercept_source_instantiation_frontiers(
                        &ok_stream, logger, *source_id, export_ids,
                    );
                }

                // TODO(petrosagg): this is just wrapping an Arc<T> into an Rc<Arc<T>> to make the
                // type checker happy. We should decide what we want our tokens to look like
                let token: Rc<dyn Any> = Rc::new(token);

                let (oks, errs) = (
                    ok_stream.as_collection().enter(region),
                    err_stream.as_collection().enter(region),
                );

                // Associate collection bundle with the source identifier.
                context.insert_id(
                    mz_expr::Id::Global(*source_id),
                    crate::render::CollectionBundle::from_collections(oks, errs),
                );
                // Associate returned tokens with the source identifier.
                tokens.insert(*source_id, token);
            }

            // Import declared indexes into the rendering context.
            for (idx_id, idx) in &dataflow.index_imports {
                context.import_index(compute_state, &mut tokens, scope, region, *idx_id, &idx.0);
            }

            // We first determine indexes and sinks to export, then build the declared object, and
            // finally export indexes and sinks. The reason for this is that we want to avoid
            // cloning the dataflow plan for `build_object`, which can be expensive.

            // Determine indexes to export
            let indexes = dataflow
                .index_exports
                .iter()
                .map(|(idx_id, (idx, _typ))| (*idx_id, dataflow.depends_on(idx.on_id), idx.clone()))
                .collect::<Vec<_>>();

            // Determine sinks to export
            let sinks = dataflow
                .sink_exports
                .iter()
                .map(|(sink_id, sink)| (*sink_id, dataflow.depends_on(sink.from), sink.clone()))
                .collect::<Vec<_>>();

            // Build declared objects.
            // It is important that we only use the `Variable` until the object is bound.
            // At that point, all subsequent uses should have access to the object itself.
            let mut variables = BTreeMap::new();
            for object in dataflow.objects_to_build.iter() {
                use differential_dataflow::operators::iterate::Variable;
                use mz_expr::CollectionPlan;

                let mut depends_on = BTreeSet::new();
                object.plan.depends_on_into(&mut depends_on);
                let recursive = depends_on.into_iter().any(|id| id >= object.id);
                if !recursive {
                    continue;
                }

                let oks_v = Variable::new(region, Product::new(Default::default(), 1));
                let err_v = Variable::new(region, Product::new(Default::default(), 1));

                context.insert_id(
                    Id::Global(object.id),
                    CollectionBundle::from_collections(oks_v.clone(), err_v.clone()),
                );
                variables.insert(object.id, (oks_v, err_v));
            }
            for object in dataflow.objects_to_build {
                let id = object.id;
                let bundle = context.render_plan(object.plan, region, region.index());
                // We need to ensure that the raw collection exists, but do not have enough information
                // here to cause that to happen.
                if let Some((oks_v, err_v)) = variables.remove(&id) {
                    let (oks, err) = bundle.collection.as_ref().expect("Collection must exist");
                    oks_v.set(oks);
                    err_v.set(err);
                }
                context.insert_id(Id::Global(object.id), bundle);
            }

            // Export declared indexes.
            for (idx_id, imports, idx) in indexes {
                context.export_index_iterative(compute_state, &mut tokens, imports, idx_id, &idx);
            }

            // Export declared sinks.
            for (sink_id, imports, sink) in sinks {
                context.export_sink(compute_state, &mut tokens, imports, sink_id, &sink);
            }
        });
    })
}

// This implementation block requires the scopes have the same timestamp as the trace manager.
// That makes some sense, because we are hoping to deposit an arrangement in the trace manager.
impl<'g, G, T> Context<Child<'g, G, T>, Row>
where
    G: Scope<Timestamp = mz_repr::Timestamp>,
    T: RenderTimestamp,
{
    pub(crate) fn export_index_iterative(
        &mut self,
        compute_state: &mut ComputeState,
        tokens: &mut BTreeMap<GlobalId, Rc<dyn std::any::Any>>,
        import_ids: BTreeSet<GlobalId>,
        idx_id: GlobalId,
        idx: &IndexDesc,
    ) {
        // put together tokens that belong to the export
        let mut needed_tokens = Vec::new();
        for import_id in import_ids {
            if let Some(token) = tokens.get(&import_id) {
                needed_tokens.push(Rc::clone(token));
            }
        }
        let bundle = self.lookup_id(Id::Global(idx_id)).unwrap_or_else(|| {
            panic!(
                "Arrangement alarmingly absent! id: {:?}",
                Id::Global(idx_id)
            )
        });
        match bundle.arrangement(&idx.key) {
            Some(ArrangementFlavor::Local(oks, errs)) => {
                use differential_dataflow::operators::arrange::Arrange;
                let oks = oks
                    .as_collection(|k, v| (k.clone(), v.clone()))
                    .leave()
                    .arrange();
                let errs = errs
                    .as_collection(|k, v| (k.clone(), v.clone()))
                    .leave()
                    .arrange();
                compute_state.traces.set(
                    idx_id,
                    TraceBundle::new(oks.trace, errs.trace).with_drop(needed_tokens),
                );
            }
            Some(ArrangementFlavor::Trace(gid, _, _)) => {
                // Duplicate of existing arrangement with id `gid`, so
                // just create another handle to that arrangement.
                let trace = compute_state.traces.get(&gid).unwrap().clone();
                compute_state.traces.set(idx_id, trace);
            }
            None => {
                println!("collection available: {:?}", bundle.collection.is_none());
                println!(
                    "keys available: {:?}",
                    bundle.arranged.keys().collect::<Vec<_>>()
                );
                panic!(
                    "Arrangement alarmingly absent! id: {:?}, keys: {:?}",
                    Id::Global(idx_id),
                    &idx.key
                );
            }
        };
    }
}
