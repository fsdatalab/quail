//! Run a planned region on the session's GPU workers.
//!
//! [`run_region`] sends the region and its tables to each worker, runs
//! its stages, and builds the region's rows as the workers finish them
//! ([`FinishedRows`]). It returns what each worker ran, for the run's
//! report.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, Int32Array, RecordBatch};
use arrow::compute::concat;
use datafusion::common::{DataFusionError, Result};
use quail_datafusion::derive::expressions::PhysicalValue;
use quail_planner::error::CompileError;
use quail_planner::expressions::column_references;
use quail_planner::operators::Operators;
use quail_planner::physical::{PhysicalGraph, PhysicalNode};
use quail_planner::results::source_columns;
use quail_planner::stages::{Stage, compute_nodes};
use serde_json::{Value, json};

use crate::finished::FinishedRows;
use crate::request::{Scanned, request_tables, stage_groupings};
use crate::results::{AnswerPort, AnswerRequest, OutputPort};
use crate::shared::Table;
use crate::workers::{Ask, Ran, Region, Workers, failure};

fn compile(error: CompileError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

/// A planned region, as its run reads it.
pub struct Planned<'a> {
    /// The plan's envelope: its graph, model, device, and settings.
    pub envelope: &'a Value,
    /// The operators the plan lowers, and their JSON.
    pub operators: &'a Operators,
    pub operators_json: &'a str,
    /// What each scan reads.
    pub scans: &'a [Scanned],
    /// By alias, each document's row in the registered table, for a
    /// scan that reads some of a table's rows.
    pub source_rows: &'a BTreeMap<String, Vec<i32>>,
}

impl Planned<'_> {
    fn graph(&self) -> Result<PhysicalGraph> {
        serde_json::from_value(self.envelope["graph"].clone()).map_err(failure)
    }

    /// The region's compute nodes as a graph: the nodes the workers run
    /// and the scans they read.
    fn compute(&self) -> Result<PhysicalGraph> {
        let graph = self.graph()?;
        let nodes: Vec<PhysicalNode> = compute_nodes(&graph)
            .into_iter()
            .map(|index| graph.nodes[index].clone())
            .collect();
        let last = (nodes.last()).ok_or_else(|| failure("the plan has no compute operation"))?;
        let port = (last.output_ports().into_iter().next())
            .ok_or_else(|| failure("the plan's last node has no output"))?;
        let root = quail_planner::physical::port(last.id(), &port.name);
        Ok(PhysicalGraph { nodes, root })
    }

    /// What the region's rows are built from: its operators, every port
    /// its run returns, and each scan's number of documents.
    ///
    /// # Errors
    ///
    /// The plan's graph does not decode.
    pub fn answer_request(&self) -> Result<AnswerRequest> {
        let ports = (self.compute()?.nodes.iter())
            .flat_map(|node| {
                node.output_ports().into_iter().map(|port| OutputPort {
                    node_type: node.type_name().to_string(),
                    node: node.id().to_string(),
                    port: port.name,
                    value_type: serde_json::to_value(port.value_type)
                        .ok()
                        .and_then(|value| value.as_str().map(str::to_string))
                        .unwrap_or_default(),
                    answer: port.answer.map(|answer| AnswerPort {
                        kind: answer.kind,
                        name: answer.name,
                        columns: answer.columns,
                    }),
                })
            })
            .collect();
        let num_documents = (self.scans.iter())
            .map(|(alias, (_, batches), _)| {
                let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
                (alias.clone(), rows as i64)
            })
            .collect();
        Ok(AnswerRequest {
            operators: self.operators_json.to_string(),
            ports,
            num_documents,
            source_rows: self.source_rows.clone(),
        })
    }

    /// Each column of the region's rows that a source expression
    /// computes, by name: one value per document of the scan it reads.
    ///
    /// # Errors
    ///
    /// A source reads a column its scan did not load.
    pub fn sources(&self) -> Result<HashMap<String, ArrayRef>> {
        let mut found = HashMap::new();
        for (name, expression) in source_columns(self.operators) {
            let references = column_references(&expression);
            let alias = (references.first())
                .map(|reference| reference.alias.clone())
                .ok_or_else(|| failure(format!("the column {name:?} reads no table")))?;
            let (schema, batches) = (self.scans.iter())
                .find(|(known, _, _)| *known == alias)
                .map(|(_, _, columns)| columns)
                .ok_or_else(|| failure(format!("no rows for the scan of {alias}")))?;
            if !(references.iter()).all(|reference| schema.index_of(&reference.column).is_ok()) {
                return Err(failure(format!(
                    "the column {name:?} was not loaded by the scan of {alias:?}"
                )));
            }
            let built = PhysicalValue::new(&expression).map_err(compile)?;
            let chunks = (batches.iter())
                .map(|batch| built.evaluate(batch, false))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(compile)?;
            let chunks: Vec<&dyn Array> = chunks.iter().map(AsRef::as_ref).collect();
            let values = match chunks.is_empty() {
                true => built
                    .evaluate(&RecordBatch::new_empty(schema.clone()), false)
                    .map_err(compile)?,
                false => concat(&chunks)?,
            };
            found.insert(name, values);
        }
        Ok(found)
    }

    /// The rows of the region, to build as its operators finish them.
    ///
    /// # Errors
    ///
    /// The graph does not decode, or a column has no source.
    pub fn finished_rows(&self) -> Result<FinishedRows> {
        FinishedRows::new(&self.answer_request()?, &self.sources()?).map_err(failure)
    }
}

/// What a region's run tells its workers besides the plan.
pub struct Listening {
    /// Each scan alias's table name, for the tables the region's hint
    /// holds rows of; None when the region has no hint.
    pub hinted: Option<BTreeMap<String, String>>,
    /// Whether someone takes progress counts, and answers as the model
    /// gives them.
    pub progress: bool,
    pub answers: bool,
}

/// Builds the region rows a table of finished rows completes.
type Reporter = Arc<dyn Fn(&Table) -> Result<()> + Send + Sync>;

/// The node and port a finished table is tagged with.
fn tag(table: &Table) -> Result<(String, String)> {
    let metadata = table.0.metadata();
    match (metadata.get("quail.node"), metadata.get("quail.port")) {
        (Some(node), Some(port)) => Ok((node.clone(), port.clone())),
        _ => Err(failure(
            "finished rows need quail.node and quail.port metadata",
        )),
    }
}

/// Run `planned` on `workers`, building the region's rows in
/// `finished` and handing each batch of them to `on_rows` as it is
/// built; returns what each worker ran.
///
/// A worker's question that is not the hint's goes to `ask`.
///
/// # Errors
///
/// The plan does not decode, a worker fails, or the rows do not build.
pub fn run_region(
    workers: &mut Workers,
    planned: &Planned<'_>,
    listening: &Listening,
    finished: &Arc<Mutex<FinishedRows>>,
    on_rows: Arc<dyn Fn(RecordBatch) -> Result<()> + Send + Sync>,
    ask: Ask,
) -> Result<Vec<Ran>> {
    let envelope = planned.envelope;
    let compute = planned.compute()?;
    let settings = envelope["settings"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let cut: Vec<Stage> =
        serde_json::from_value(settings.get("stages").cloned().unwrap_or(json!([])))
            .map_err(failure)?;
    let wanted: Vec<(String, String)> = {
        let rows = finished.lock().expect("the rows lock");
        (compute.nodes.iter())
            .flat_map(|node| {
                let ports = node.output_ports().into_iter();
                ports.map(|port| (node.id().to_string(), port.name))
            })
            .filter(|(node, port)| rows.wants(node, port))
            .collect()
    };
    let mut query = settings.clone();
    query.insert("physical_plan".into(), envelope.clone());
    query.insert("model".into(), envelope["model"].clone());
    query.insert("gpus".into(), envelope["gpus"].clone());
    let region = json!({
        "query": query, "wanted": wanted, "hinted": listening.hinted,
        "listening": [listening.progress, listening.answers],
    });
    let mut tables = request_tables(envelope, planned.operators, planned.scans).map_err(compile)?;
    // a worker may name a document by its row in the registered table
    for (alias, positions) in planned.source_rows {
        let positions: ArrayRef = Arc::new(Int32Array::from(positions.clone()));
        let batch = RecordBatch::try_from_iter([("row", positions)])?;
        tables.push((
            format!("source_rows:{alias}"),
            (batch.schema(), vec![batch]),
        ));
    }
    let deduplicated: Vec<String> = match settings.get("deduplicate") {
        Some(Value::Object(found)) => found.keys().cloned().collect(),
        _ => Vec::new(),
    };
    let groupings = stage_groupings(&compute, &deduplicated, &tables).map_err(compile)?;
    // a table of finished rows builds the region rows it completes
    let report: Reporter = {
        let (finished, on_rows) = (Arc::clone(finished), Arc::clone(&on_rows));
        Arc::new(move |table: &Table| {
            let (node, port) = tag(table)?;
            let empty = [RecordBatch::new_empty(table.0.clone())];
            // an empty table still gives its columns' types
            let batches = match table.1.is_empty() {
                true => &empty[..],
                false => &table.1[..],
            };
            for batch in batches {
                let built = (finished.lock().expect("the rows lock"))
                    .report(&node, &port, batch)
                    .map_err(failure)?;
                if let Some(rows) = built {
                    on_rows(rows)?;
                }
            }
            Ok(())
        })
    };
    let calls = Region {
        report: {
            let report = Arc::clone(&report);
            Box::new(move |table| report(&table))
        },
        ask,
    };
    let ran = workers.run_region(&region, &tables, |connections| {
        crate::stages::run_stages(&compute, cut, groupings, connections, calls)
    })?;
    // each returned output builds the rows its operator did not report;
    // a worker with no rows of a port may type its columns apart
    for (node, port) in &wanted {
        let named: Vec<&Table> = (ran.iter())
            .flat_map(|(_, tables)| tables)
            .filter(|table| tag(table).is_ok_and(|(n, p)| n == *node && p == *port))
            .collect();
        let rows = |table: &&Table| table.1.iter().any(|batch| batch.num_rows() > 0);
        let filled: Vec<&Table> = named.iter().copied().filter(rows).collect();
        for table in if filled.is_empty() {
            named.iter().take(1).copied().collect()
        } else {
            filled
        } {
            report(table)?;
        }
    }
    let built = (finished.lock().expect("the rows lock"))
        .finish()
        .map_err(failure)?;
    if let Some(rows) = built {
        on_rows(rows)?;
    }
    Ok(ran)
}
