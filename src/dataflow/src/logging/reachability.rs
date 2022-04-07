// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Logging dataflows for events generated by timely dataflow.

use std::{collections::HashMap, time::Duration};

use differential_dataflow::operators::arrange::arrangement::Arrange;
use mz_expr::{permutation_for_arrangement, MirScalarExpr};
use timely::communication::Allocate;
use timely::dataflow::channels::pact::Exchange;
use timely::dataflow::operators::capture::EventLink;
use timely::logging::WorkerIdentifier;

use mz_dataflow_types::logging::LoggingConfig;
use mz_dataflow_types::KeysValsHandle;
use mz_dataflow_types::RowSpine;
use mz_ore::iter::IteratorExt;
use mz_repr::{Datum, Diff, Row, RowArena, Timestamp};

use super::{LogVariant, TimelyLog};
use crate::common::activator::RcActivator;
use crate::common::replay::MzReplay;
use crate::logging::ConsolidateBuffer;

/// Constructs the logging dataflow for reachability logs.
///
/// Params
/// * `worker`: The Timely worker hosting the log analysis dataflow.
/// * `config`: Logging configuration
/// * `linked`: The source to read log events from.
/// * `activator`: A handle to acknowledge activations.
///
/// Returns a map from log variant to a tuple of a trace handle and a permutation to reconstruct
/// the original rows.
pub fn construct<A: Allocate>(
    worker: &mut timely::worker::Worker<A>,
    config: &LoggingConfig,
    linked: std::rc::Rc<
        EventLink<
            Timestamp,
            (
                Duration,
                WorkerIdentifier,
                (
                    Vec<usize>,
                    Vec<(usize, usize, bool, Option<Timestamp>, Diff)>,
                ),
            ),
        >,
    >,
    activator: RcActivator,
) -> std::collections::HashMap<LogVariant, KeysValsHandle> {
    let granularity_ms = std::cmp::max(1, config.granularity_ns / 1_000_000) as Timestamp;

    // A dataflow for multiple log-derived arrangements.
    let traces = worker.dataflow_named("Dataflow: timely reachability logging", move |scope| {
        use differential_dataflow::collection::AsCollection;

        let logs = Some(linked).mz_replay(
            scope,
            "reachability logs",
            Duration::from_nanos(config.granularity_ns as u64),
            activator,
        );

        use timely::dataflow::operators::generic::builder_rc::OperatorBuilder;

        let construct_reachability = |key: Vec<_>, value: Vec<_>| {
            let mut flatten = OperatorBuilder::new(
                "Timely Reachability Logging Flatten ".to_string(),
                scope.clone(),
            );

            use timely::dataflow::channels::pact::Pipeline;
            let mut input = flatten.new_input(&logs, Pipeline);

            let (mut updates_out, updates) = flatten.new_output();

            let mut buffer = Vec::new();
            flatten.build(move |_capability| {
                move |_frontiers| {
                    let updates = updates_out.activate();
                    let mut updates_session = ConsolidateBuffer::new(updates, 0);

                    input.for_each(|cap, data| {
                        data.swap(&mut buffer);

                        for (time, worker, (addr, massaged)) in buffer.drain(..) {
                            let time_ms = (((time.as_millis() as Timestamp / granularity_ms) + 1)
                                * granularity_ms)
                                as Timestamp;
                            for (source, port, update_type, ts, diff) in massaged {
                                updates_session.give(
                                    &cap,
                                    (
                                        (update_type, addr.clone(), source, port, worker, ts),
                                        time_ms,
                                        diff,
                                    ),
                                );
                            }
                        }
                    });
                }
            });

            let mut row_buf = Row::default();
            updates
                .as_collection()
                .arrange_core::<_, RowSpine<_, _, _, _>>(
                    Exchange::new(|(((_, _, _, _, w, _), ()), _, _)| *w as u64),
                    "PreArrange Timely reachability",
                )
                .as_collection(move |(update_type, addr, source, port, worker, ts), _| {
                    let row_arena = RowArena::default();
                    let update_type = if *update_type { "source" } else { "target" };
                    row_buf.packer().push_list(
                        addr.iter()
                            .chain_one(&source)
                            .map(|id| Datum::Int64(*id as i64)),
                    );
                    let datums = &[
                        row_arena.push_unary_row(row_buf.clone()),
                        Datum::Int64(*port as i64),
                        Datum::Int64(*worker as i64),
                        Datum::String(&update_type),
                        Datum::from(ts.and_then(|ts| i64::try_from(ts).ok())),
                    ];
                    row_buf.packer().extend(key.iter().map(|k| datums[*k]));
                    let key_row = row_buf.clone();
                    row_buf.packer().extend(value.iter().map(|k| datums[*k]));
                    let value_row = row_buf.clone();
                    (key_row, value_row)
                })
        };

        // Restrict results by those logs that are meant to be active.
        let logs = vec![(LogVariant::Timely(TimelyLog::Reachability))];

        let mut result = std::collections::HashMap::new();
        for variant in logs {
            if config.active_logs.contains_key(&variant) {
                let key = variant.index_by();
                let (_, value) = permutation_for_arrangement::<HashMap<_, _>>(
                    &key.iter()
                        .cloned()
                        .map(MirScalarExpr::Column)
                        .collect::<Vec<_>>(),
                    variant.desc().arity(),
                );
                let trace = construct_reachability(key.clone(), value)
                    .arrange_named::<RowSpine<_, _, _, _>>(&format!("Arrange {:?}", variant))
                    .trace;
                result.insert(variant, trace);
            }
        }
        result
    });

    traces
}
