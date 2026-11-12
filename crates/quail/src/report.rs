//! A region's run, reported from what each worker ran.
//!
//! Each worker answers a region's end with its counters as JSON and
//! its nodes' output tables ([`crate::workers::Ran`]). [`region_report`]
//! merges the workers: one table per output port, one set of metrics
//! per node, and the run's totals, throughput, and GPU cost.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::{RecordBatch, UInt32Array};
use arrow::compute::{cast, concat_batches, take_record_batch};
use arrow::row::{RowConverter, SortField};
use datafusion::common::Result;
use quail_planner::physical::{PhysicalGraph, PhysicalNode, ValueType, topological_nodes};
use quail_planner::stages::compute_nodes;
use serde_json::{Map, Value, json};

use crate::shared::Table;
use crate::workers::{Ran, failure};

/// A node's measured metrics, in the order a report lists them; every
/// one adds across workers but the peak, which is the largest.
const METRICS: [&str; 14] = [
    "wall_s",
    "input_rows",
    "output_rows",
    "evaluated_documents",
    "evaluated_document_pairs",
    "fresh_tokens",
    "cached_tokens",
    "kv_hits",
    "kv_misses",
    "kv_removals",
    "kv_recomputations",
    "peak_gpu_bytes",
    "gpu_s",
    "chunks",
];

/// The fields a report shows of each executed join and barrier.
const JOIN_FIELDS: [(&str, &[&str]); 2] = [
    (
        "quail.ai_join",
        &[
            "written_position",
            "exec_index",
            "anchor",
            "partners",
            "semantics",
            "selectivity",
            "expected_rows",
            "anchor_frame_tokens",
            "pair_tail_tokens",
            "anchor_resident",
            "tuple_tokens",
            "pairs_from",
            "keep_kv",
            "attention",
            "share_prefixes",
        ],
    ),
    ("quail.barrier", &["next_anchor", "aliases"]),
];

/// The sum of two counters: an integer when both are.
fn added(left: &Value, right: &Value) -> Value {
    match (left.as_i64(), right.as_i64()) {
        (Some(left), Some(right)) => json!(left + right),
        _ => json!(left.as_f64().unwrap_or(0.0) + right.as_f64().unwrap_or(0.0)),
    }
}

fn is_count(value: &Value) -> bool {
    value.is_number()
}

/// One node's metrics from two workers: counters add, the peak is the
/// larger, and an extension keeps the later worker's descriptive values.
fn merged_metrics(left: &Map<String, Value>, right: &Map<String, Value>) -> Map<String, Value> {
    let zero = json!(0);
    let mut merged = Map::new();
    for name in METRICS {
        let (one, two) = (
            left.get(name).unwrap_or(&zero),
            right.get(name).unwrap_or(&zero),
        );
        let value = if name == "peak_gpu_bytes" {
            json!(one.as_i64().unwrap_or(0).max(two.as_i64().unwrap_or(0)))
        } else {
            added(one, two)
        };
        merged.insert(name.into(), value);
    }
    let empty = Map::new();
    let extension = |metrics: &Map<String, Value>| {
        (metrics.get("extension").and_then(Value::as_object).cloned()).unwrap_or_default()
    };
    let (mut kept, later) = (extension(left), extension(right));
    let earlier = left
        .get("extension")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    for (name, value) in later {
        let value = match earlier.get(&name) {
            Some(before) if is_count(&value) => added(before, &value),
            _ => value,
        };
        kept.insert(name, value);
    }
    merged.insert("extension".into(), Value::Object(kept));
    merged
}

fn no_metrics() -> Map<String, Value> {
    let mut metrics: Map<String, Value> = METRICS
        .iter()
        .map(|name| {
            let zero = if matches!(*name, "wall_s" | "gpu_s") {
                json!(0.0)
            } else {
                json!(0)
            };
            ((*name).to_string(), zero)
        })
        .collect();
    metrics.insert("extension".into(), json!({}));
    metrics
}

/// A node's metrics as a report lists them: the standard fields, then
/// each extension metric that is a number.
fn scalar_metrics(metrics: &Map<String, Value>) -> Value {
    let mut scalars = Map::new();
    for name in METRICS {
        scalars.insert(name.into(), metrics.get(name).cloned().unwrap_or(json!(0)));
    }
    if let Some(extension) = metrics.get("extension").and_then(Value::as_object) {
        for (name, value) in extension {
            if is_count(value) {
                scalars.insert(name.clone(), value.clone());
            }
        }
    }
    Value::Object(scalars)
}

/// The node and port a worker's output table is tagged with.
fn tag(table: &Table) -> Option<(&String, &String)> {
    let metadata = table.0.metadata();
    Some((metadata.get("quail.node")?, metadata.get("quail.port")?))
}

/// One port's rows from every worker as one table: each worker's rows
/// in turn, and for a rows port each set of ids once.
///
/// A worker with no rows of a port may type its columns apart, so its
/// empty table is left out.
fn merged_table(tables: &[&Table], rows_port: bool) -> Result<Table> {
    let filled: Vec<&&Table> = (tables.iter())
        .filter(|(_, batches)| batches.iter().any(|batch| batch.num_rows() > 0))
        .collect();
    let Some(first) = filled.first().copied().or(tables.first()) else {
        return Err(failure("a port has no table"));
    };
    let schema = Arc::clone(&first.0);
    let mut batches = Vec::new();
    for (found, rows) in filled.iter().map(|table| (&table.0, &table.1)) {
        for batch in rows {
            if *found == schema {
                batches.push(batch.clone());
                continue;
            }
            // the same columns under another type
            let columns = (schema.fields().iter())
                .map(|field| {
                    let column = batch.column_by_name(field.name()).ok_or_else(|| {
                        failure(format!("a worker's rows lack the column {}", field.name()))
                    })?;
                    Ok(cast(column, field.data_type())?)
                })
                .collect::<Result<Vec<_>>>()?;
            batches.push(RecordBatch::try_new(Arc::clone(&schema), columns)?);
        }
    }
    let merged = concat_batches(&schema, &batches)?;
    let ids: Vec<usize> = (schema.fields().iter().enumerate())
        .filter(|(_, field)| field.metadata().contains_key("quail.alias"))
        .map(|(index, _)| index)
        .collect();
    if !rows_port || ids.is_empty() || merged.num_rows() == 0 {
        return Ok((schema, vec![merged]));
    }
    let columns: Vec<_> = (ids.iter().map(|index| Arc::clone(merged.column(*index)))).collect();
    let fields = (columns.iter())
        .map(|column| SortField::new(column.data_type().clone()))
        .collect();
    let rows = RowConverter::new(fields)?.convert_columns(&columns)?;
    let mut seen: HashMap<&[u8], ()> = HashMap::new();
    let first: Vec<u32> = (rows.iter().enumerate())
        .filter(|(_, row)| seen.insert(row.data(), ()).is_none())
        .map(|(position, _)| position as u32)
        .collect();
    if first.len() == merged.num_rows() {
        return Ok((schema, vec![merged]));
    }
    let kept = take_record_batch(&merged, &UInt32Array::from(first))?;
    Ok((schema, vec![kept]))
}

/// The workers' memory reports as one: capacity and bytes add, the
/// peak is the largest, and the trace sums each worker's latest sample.
fn merged_memory(workers: &[Value]) -> Map<String, Value> {
    let rows = |worker: &Value, name: &str| -> Vec<Value> {
        worker[name].as_array().cloned().unwrap_or_default()
    };
    let mut records: Vec<(f64, usize, Value)> = Vec::new();
    let mut events: Vec<(f64, usize, Value)> = Vec::new();
    for (worker, report) in workers.iter().enumerate() {
        for row in rows(report, "memory_trace") {
            records.push((row[0].as_f64().unwrap_or(0.0), worker, row));
        }
        for event in rows(report, "memory_events") {
            let mut detail = event[2].as_object().cloned().unwrap_or_default();
            detail.insert("worker".into(), json!(worker));
            let stamp = event[0].as_f64().unwrap_or(0.0);
            events.push((stamp, worker, json!([event[0], event[1], detail])));
        }
    }
    let by_time = |one: &(f64, usize, Value), two: &(f64, usize, Value)| {
        one.0.total_cmp(&two.0).then(one.1.cmp(&two.1))
    };
    records.sort_by(by_time);
    events.sort_by(by_time);
    let mut latest: BTreeMap<usize, Value> = BTreeMap::new();
    let mut trace = Vec::new();
    for (_, worker, row) in records {
        latest.insert(worker, row.clone());
        let summed = |index: usize| {
            (latest.values()).fold(json!(0), |sum, sample| added(&sum, &sample[index]))
        };
        let mut sample = vec![row[0].clone()];
        sample.extend((1..7).map(summed));
        sample.extend([row[7].clone(), row[8].clone(), summed(9)]);
        trace.push(Value::Array(sample));
    }
    let started = (workers.iter())
        .filter_map(|worker| worker.get("memory_trace_started")?.as_f64())
        .fold(None, |least: Option<f64>, found| {
            Some(least.map_or(found, |least| least.min(found)))
        });
    let peak = (workers.iter())
        .filter_map(|worker| worker.get("peak_gib")?.as_f64())
        .fold(0.0, f64::max);
    let mut merged = Map::new();
    merged.insert("memory_workers".into(), json!(workers));
    merged.insert("memory_trace".into(), Value::Array(trace));
    merged.insert("memory_trace_started".into(), json!(started.unwrap_or(0.0)));
    let events = events.into_iter().map(|(_, _, event)| event).collect();
    merged.insert("memory_events".into(), Value::Array(events));
    merged.insert("peak_gib".into(), json!(peak));
    for field in ["memory_capacity", "memory_bytes"] {
        let mut totals = Map::new();
        for worker in workers {
            for (key, value) in worker[field].as_object().into_iter().flatten() {
                let total = totals
                    .get(key)
                    .map_or(value.clone(), |sum| added(sum, value));
                totals.insert(key.clone(), total);
            }
        }
        merged.insert(field.into(), Value::Object(totals));
    }
    merged
}

/// A counter from every worker as one: numbers add, and counters by
/// alias add alias by alias.
fn summed(values: &[&Value]) -> Value {
    if values.first().is_some_and(|value| value.is_object()) {
        let mut totals = Map::new();
        for (key, value) in
            (values.iter()).flat_map(|value| value.as_object().into_iter().flatten())
        {
            let total = totals
                .get(key)
                .map_or(value.clone(), |sum| added(sum, value));
            totals.insert(key.clone(), total);
        }
        return Value::Object(totals);
    }
    values
        .iter()
        .fold(json!(0), |sum, value| added(&sum, value))
}

/// The report of a region's run on its workers, and the run's output
/// tables, one per port, tagged with node and port.
///
/// `graph` is the region's graph; `ran` what each worker ran; `wall`
/// the seconds the region took. `usd_per_hour` prices one GPU.
///
/// # Errors
///
/// A worker's results do not read, or a port's rows do not merge.
pub fn region_report(
    graph: &PhysicalGraph,
    ran: &[Ran],
    wall: f64,
    usd_per_hour: Option<f64>,
    gpus: i64,
    gpu_timing: bool,
) -> Result<(Value, Vec<Table>)> {
    let mut outputs = Vec::new();
    let mut nodes: Vec<(&PhysicalNode, Map<String, Value>)> = Vec::new();
    for index in compute_nodes(graph) {
        let node = &graph.nodes[index];
        let measured: Vec<&Map<String, Value>> = (ran.iter())
            .filter_map(|(found, _)| found["metrics"][node.id()].as_object())
            .collect();
        let Some((first, rest)) = measured.split_first() else {
            continue;
        };
        let metrics = (rest.iter()).fold((*first).clone(), |sum, next| merged_metrics(&sum, next));
        for port in node.output_ports() {
            let tables: Vec<&Table> = (ran.iter())
                .flat_map(|(_, tables)| tables)
                .filter(|table| tag(table) == Some((&node.id().to_string(), &port.name)))
                .collect();
            if tables.is_empty() {
                continue;
            }
            let (schema, batches) = merged_table(&tables, port.value_type == ValueType::Rows)?;
            // the first table's tag names the port
            outputs.push((Arc::clone(&tables[0].0), batches, schema));
        }
        nodes.push((node, metrics));
    }
    let outputs: Vec<Table> = outputs
        .into_iter()
        .map(|(tagged, batches, schema)| {
            let schema = Arc::new(
                schema
                    .as_ref()
                    .clone()
                    .with_metadata(tagged.metadata().clone()),
            );
            let batches = batches
                .into_iter()
                .map(|batch| batch.with_schema(Arc::clone(&schema)))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok((schema, batches))
        })
        .collect::<Result<_>>()?;
    let total = (nodes.iter()).fold(no_metrics(), |sum, (_, metrics)| {
        merged_metrics(&sum, metrics)
    });
    let mut report = Map::new();
    report.insert("wall_s".into(), json!((wall * 100.0).round() / 100.0));
    for name in [
        "fresh_tokens",
        "cached_tokens",
        "evaluated_documents",
        "evaluated_document_pairs",
    ] {
        report.insert(name.into(), total[name].clone());
    }
    report.insert(
        "usd_per_query".into(),
        json!(usd_per_hour.map(|price| wall / 3600.0 * gpus as f64 * price)),
    );
    let inferences: Vec<Value> = (nodes.iter())
        .filter(|(node, _)| node.is_inference())
        .map(|(_, metrics)| metrics["extension"].clone())
        .collect();
    report.insert("backend_metrics".into(), json!({"inferences": inferences}));
    let measured: Map<String, Value> = (nodes.iter())
        .map(|(node, metrics)| (node.id().to_string(), scalar_metrics(metrics)))
        .collect();
    report.insert("node_metrics".into(), Value::Object(measured));
    let mut joins = Vec::new();
    for index in topological_nodes(&graph.nodes) {
        let node = &graph.nodes[index];
        let Some((_, fields)) = (JOIN_FIELDS.iter()).find(|(kind, _)| *kind == node.type_name())
        else {
            continue;
        };
        let written = serde_json::to_value(node).map_err(failure)?;
        let mut shown = Map::new();
        shown.insert("type".into(), json!(node.type_name()));
        shown.insert("id".into(), json!(node.id()));
        for field in *fields {
            shown.insert((*field).into(), written["attributes"][*field].clone());
        }
        joins.push(Value::Object(shown));
    }
    report.insert("executed_join_plan".into(), Value::Array(joins));
    let seconds = wall.max(1e-9);
    let pairs = total["evaluated_document_pairs"].as_f64().unwrap_or(0.0);
    if pairs != 0.0 {
        report.insert("document_pairs_per_second".into(), json!(pairs / seconds));
    } else {
        let documents: i64 = (graph.nodes.iter())
            .filter_map(|node| match node {
                PhysicalNode::Scan(scan) => Some(scan.attributes.num_documents),
                _ => None,
            })
            .sum();
        report.insert(
            "documents_per_second".into(),
            json!(documents as f64 / seconds),
        );
    }
    if gpu_timing {
        let gpu = total["gpu_s"].as_f64().unwrap_or(0.0);
        report.insert("gpu_s".into(), json!((gpu * 1000.0).round() / 1000.0));
        report.insert("chunks".into(), total["chunks"].clone());
    }
    let counters: Vec<&Value> = ran.iter().map(|(found, _)| &found["kv_manager"]).collect();
    let mut kv_manager = Map::new();
    for name in counters
        .first()
        .and_then(|first| first.as_object())
        .into_iter()
        .flat_map(Map::keys)
    {
        let values: Vec<&Value> = counters.iter().map(|worker| &worker[name]).collect();
        kv_manager.insert(name.clone(), summed(&values));
    }
    report.insert("kv_manager".into(), Value::Object(kv_manager));
    let memory: Vec<Value> = ran
        .iter()
        .map(|(found, _)| found["memory"].clone())
        .collect();
    report.extend(merged_memory(&memory));
    Ok((Value::Object(report), outputs))
}
