// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Provides parsing and convenience functions for working with Kafka from the `sql` package.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::bail;
use mz_kafka_util::client::DEFAULT_FETCH_METADATA_TIMEOUT;
use mz_ore::task;
use mz_sql_parser::ast::display::AstDisplay;
use mz_sql_parser::ast::{AstInfo, KafkaConfigOption, KafkaConfigOptionName};
use mz_storage_types::connections::StringOrSecret;
use mz_storage_types::sinks::KafkaSinkCompressionType;
use rdkafka::consumer::{BaseConsumer, Consumer, ConsumerContext};
use rdkafka::{Offset, TopicPartitionList};
use tokio::time::Duration;

use crate::ast::Value;
use crate::names::Aug;
use crate::normalize::generate_extracted_config;
use crate::plan::with_options::{ImpliedValue, TryFromValue};
use crate::plan::PlanError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum KafkaOptionCheckContext {
    Source,
    Sink,
}

/// Verifies that [`KafkaConfigOption`]s are only used in the appropriate contexts
pub fn validate_options_for_context<T: AstInfo>(
    options: &[KafkaConfigOption<T>],
    context: KafkaOptionCheckContext,
) -> Result<(), anyhow::Error> {
    use KafkaConfigOptionName::*;
    use KafkaOptionCheckContext::*;

    for KafkaConfigOption { name, .. } in options {
        let limited_to_context = match name {
            CompressionType => Some(Sink),
            GroupIdPrefix => None,
            Topic => None,
            TopicMetadataRefreshIntervalMs => None,
            StartTimestamp => Some(Source),
            StartOffset => Some(Source),
            PartitionCount => Some(Sink),
            ReplicationFactor => Some(Sink),
            RetentionBytes => Some(Sink),
            RetentionMs => Some(Sink),
        };
        if limited_to_context.is_some() && limited_to_context != Some(context) {
            bail!(
                "cannot set {} for {}",
                name.to_ast_string(),
                match context {
                    Source => "SOURCE",
                    Sink => "SINK",
                }
            );
        }
    }

    Ok(())
}

generate_extracted_config!(
    KafkaConfigOption,
    (
        CompressionType,
        KafkaSinkCompressionType,
        Default(KafkaSinkCompressionType::None)
    ),
    (GroupIdPrefix, String),
    (Topic, String),
    (TopicMetadataRefreshIntervalMs, i32),
    (StartTimestamp, i64),
    (StartOffset, Vec<i64>),
    (PartitionCount, i32, Default(-1)),
    (ReplicationFactor, i32, Default(-1)),
    (RetentionBytes, i64),
    (RetentionMs, i64)
);

impl TryFromValue<Value> for KafkaSinkCompressionType {
    fn try_from_value(v: Value) -> Result<Self, PlanError> {
        match v {
            Value::String(v) => match v.to_lowercase().as_str() {
                "none" => Ok(KafkaSinkCompressionType::None),
                "gzip" => Ok(KafkaSinkCompressionType::Gzip),
                "snappy" => Ok(KafkaSinkCompressionType::Snappy),
                "lz4" => Ok(KafkaSinkCompressionType::Lz4),
                "zstd" => Ok(KafkaSinkCompressionType::Zstd),
                // The caller will add context, resulting in an error like
                // "invalid COMPRESSION TYPE: <bad-compression-type>".
                _ => sql_bail!("{}", v),
            },
            _ => sql_bail!("compression type must be a string"),
        }
    }

    fn name() -> String {
        "Kafka sink compression type".to_string()
    }
}

impl ImpliedValue for KafkaSinkCompressionType {
    fn implied_value() -> Result<Self, PlanError> {
        sql_bail!("must provide a compression type value")
    }
}

/// The config options we expect to pass along when connecting to librdkafka.
///
/// Note that these are meant to be disjoint from the options we permit being
/// set on Kafka connections (CREATE CONNECTION), i.e. these are meant to be
/// per-client options (CREATE SOURCE, CREATE SINK).
#[derive(Debug)]
pub struct LibRdKafkaConfig(pub BTreeMap<String, StringOrSecret>);

impl TryFrom<&KafkaConfigOptionExtracted> for LibRdKafkaConfig {
    type Error = PlanError;
    // We are in a macro, so allow calling a closure immediately.
    #[allow(clippy::redundant_closure_call)]
    fn try_from(
        KafkaConfigOptionExtracted {
            topic_metadata_refresh_interval_ms,
            ..
        }: &KafkaConfigOptionExtracted,
    ) -> Result<LibRdKafkaConfig, Self::Error> {
        let mut o = BTreeMap::new();

        macro_rules! fill_options {
            // Values that are not option can just be wrapped in some before being passed to the macro
            ($v:expr, $s:expr) => {
                if let Some(v) = $v {
                    o.insert($s.to_string(), StringOrSecret::String(v.to_string()));
                }
            };
            ($v:expr, $s:expr, $check:expr, $err:expr) => {
                if let Some(v) = $v {
                    if !$check(v) {
                        sql_bail!($err);
                    }
                    o.insert($s.to_string(), StringOrSecret::String(v.to_string()));
                }
            };
        }

        fill_options!(
            topic_metadata_refresh_interval_ms,
            "topic.metadata.refresh.interval.ms",
            |i: &i32| { 0 <= *i && *i <= 3_600_000 },
            "TOPIC METADATA REFRESH INTERVAL MS must be within [0, 3,600,000]"
        );

        Ok(LibRdKafkaConfig(o))
    }
}

/// An enum that represents start offsets for a kafka consumer.
#[derive(Debug)]
pub enum KafkaStartOffsetType {
    /// Fully specified, either by the user or generated.
    StartOffset(Vec<i64>),
    /// Specified by the user.
    StartTimestamp(i64),
}

impl TryFrom<&KafkaConfigOptionExtracted> for Option<KafkaStartOffsetType> {
    type Error = PlanError;
    fn try_from(
        KafkaConfigOptionExtracted {
            start_offset,
            start_timestamp,
            ..
        }: &KafkaConfigOptionExtracted,
    ) -> Result<Option<KafkaStartOffsetType>, Self::Error> {
        Ok(match (start_offset, start_timestamp) {
            (Some(_), Some(_)) => {
                sql_bail!("cannot specify START TIMESTAMP and START OFFSET at same time")
            }
            (Some(so), _) => Some(KafkaStartOffsetType::StartOffset(so.clone())),
            (_, Some(sto)) => Some(KafkaStartOffsetType::StartTimestamp(*sto)),
            _ => None,
        })
    }
}

/// Returns start offsets for the partitions of `topic` and the provided
/// `START TIMESTAMP` option.
///
/// For each partition, the returned offset is the earliest offset whose
/// timestamp is greater than or equal to the given timestamp for the
/// partition. If no such message exists (or the Kafka broker is before
/// 0.10.0), the current end offset is returned for the partition.
///
/// The provided `START TIMESTAMP` option must be a non-zero number:
/// * Non-Negative numbers will used as is (e.g. `1622659034343`)
/// * Negative numbers will be translated to a timestamp in millis
///   before now (e.g. `-10` means 10 millis ago)
///
/// If `START TIMESTAMP` has not been configured, an empty Option is
/// returned.
pub async fn lookup_start_offsets<C>(
    consumer: Arc<BaseConsumer<C>>,
    topic: &str,
    offsets: KafkaStartOffsetType,
    now: u64,
) -> Result<Option<Vec<i64>>, PlanError>
where
    C: ConsumerContext + 'static,
{
    let time_offset = match offsets {
        KafkaStartOffsetType::StartTimestamp(time) => time,
        _ => return Ok(None),
    };

    let time_offset = if time_offset < 0 {
        let now: i64 = now.try_into()?;
        let ts = now - time_offset.abs();

        if ts <= 0 {
            sql_bail!("Relative START TIMESTAMP must be smaller than current system timestamp")
        }
        ts
    } else {
        time_offset
    };

    // Lookup offsets
    // TODO(guswynn): see if we can add broker to this name
    task::spawn_blocking(|| format!("kafka_lookup_start_offsets:{topic}"), {
        let topic = topic.to_string();
        move || {
            // There cannot be more than i32 partitions
            let num_partitions = mz_kafka_util::client::get_partitions(
                consumer.as_ref().client(),
                &topic,
                DEFAULT_FETCH_METADATA_TIMEOUT,
            )
            .map_err(|e| sql_err!("{}", e))?
            .len();

            let num_partitions_i32 = i32::try_from(num_partitions)
                .map_err(|_| sql_err!("kafka topic had more than {} partitions", i32::MAX))?;

            let mut tpl = TopicPartitionList::with_capacity(1);
            tpl.add_partition_range(&topic, 0, num_partitions_i32 - 1);
            tpl.set_all_offsets(Offset::Offset(time_offset))
                .map_err(|e| sql_err!("{}", e))?;

            let offsets_for_times = consumer
                .offsets_for_times(tpl, Duration::from_secs(10))
                .map_err(|e| sql_err!("{}", e))?;

            // Translate to `start_offsets`
            let start_offsets = offsets_for_times
                .elements()
                .iter()
                .map(|elem| match elem.offset() {
                    Offset::Offset(offset) => Ok(offset),
                    Offset::End => fetch_end_offset(&consumer, &topic, elem.partition()),
                    _ => sql_bail!(
                        "Unexpected offset {:?} for partition {}",
                        elem.offset(),
                        elem.partition()
                    ),
                })
                .collect::<Result<Vec<_>, _>>()?;

            if start_offsets.len() != num_partitions {
                sql_bail!(
                    "Expected offsets for {} partitions, but received {}",
                    num_partitions,
                    start_offsets.len(),
                );
            }

            Ok(Some(start_offsets))
        }
    })
    .await
    .map_err(|e| sql_err!("{}", e))?
}

// Kafka supports bulk lookup of watermarks, but it is not exposed in rdkafka.
// If that ever changes, we will want to first collect all pids that have no
// offset for a given timestamp and then do a single request (instead of doing
// a request for each partition individually).
fn fetch_end_offset<C>(consumer: &BaseConsumer<C>, topic: &str, pid: i32) -> Result<i64, PlanError>
where
    C: ConsumerContext,
{
    let (_low, high) = consumer
        .fetch_watermarks(topic, pid, Duration::from_secs(10))
        .map_err(|e| sql_err!("{}", e))?;
    Ok(high)
}
