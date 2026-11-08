//! The model region runner that calls Quail's Python executor, and the
//! tables whose rows a Python provider returns.

use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, RecordBatch, new_null_array};
use datafusion::arrow::compute::cast;
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{Constraints, DataFusionError, Result};
use datafusion::datasource::TableType;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::collections::HashMap;

use quail_datafusion::{
    IdentifiedTable, InputBatches, InputRows, InputStream, ModelRegion, RegionBatches,
    RegionColumns, RegionHint, RegionRunner, SharedInput, derive, quail_column_names,
};
use quail_planner::operators::{Operators, Scan};
use quail_planner::prompts::Turn;
use tokio::sync::mpsc::UnboundedSender;

use crate::input::{Documents, HintRows, InputHint, InputReader, hinted_rows};
use crate::planning::{
    PlanOutcome, Planner, Settings, TableRows, extracted_aliases, planned_json, source_column,
};
use crate::pyarrow::{from_pyarrow, stream_from_pyarrow, to_pyarrow};
use crate::python_failure;

/// Runs each model region by calling the engine's
/// `execute(planned, tables, sink)` or `run_region(payload, tables,
/// fetch)`. `execute` hands the sink each batch of finished rows as the
/// region runs and returns None, or returns the rows; `run_region`
/// returns them, as a pyarrow Table or RecordBatchReader.
#[derive(Debug)]
pub(crate) struct PythonRunner {
    pub(crate) engine: Py<PyAny>,
    /// The model's chat-turn text and prompt layout, which each region's
    /// prompts are bound in.
    pub(crate) prompting: (Turn, String),
    /// While a query runs: the thread that called DataFusion, and where
    /// a worker sends it a region to run.
    pub(crate) caller: Mutex<Option<(ThreadId, UnboundedSender<RegionCall>)>>,
    /// The planner and configuration Rust plans each region with, then
    /// calls the engine's `execute`; None calls `run_region`.
    pub(crate) planning: Option<(Arc<Planner>, Settings)>,
}

/// A region a DataFusion worker hands to the calling thread to run.
pub(crate) struct RegionCall {
    pub(crate) region: ModelRegion,
    pub(crate) inputs: Vec<InputStream>,
    pub(crate) fetch: Option<usize>,
    pub(crate) hint: RegionHint,
    pub(crate) rows: RowSender,
}

/// One message of a region's rows: a batch, a failure, or the end.
#[derive(Debug)]
pub(crate) enum RegionRows {
    Batch(Result<RecordBatch>),
    Done,
}

/// Where a running region sends its rows, in order.
pub(crate) type RowSender = std::sync::mpsc::Sender<RegionRows>;

/// The rows a region sends, read as DataFusion's iterator of batches.
///
/// A sender dropped before the end, such as when the query stopped,
/// reads as a failure.
struct ReceivedRows {
    receiver: std::sync::mpsc::Receiver<RegionRows>,
    ended: bool,
}

impl Iterator for ReceivedRows {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ended {
            return None;
        }
        match self.receiver.recv() {
            Ok(RegionRows::Batch(batch)) => Some(batch),
            Ok(RegionRows::Done) => {
                self.ended = true;
                None
            }
            Err(_) => {
                self.ended = true;
                Some(Err(DataFusionError::Execution(
                    "the region did not finish".into(),
                )))
            }
        }
    }
}

/// The callable a region's executor hands each batch of finished rows,
/// as a pyarrow Table or RecordBatchReader, while the region runs.
#[pyclass(name = "RegionSink", module = "_quail_native", frozen)]
pub(crate) struct RegionSink {
    rows: RowSender,
    names: Vec<String>,
    schema: SchemaRef,
}

#[pymethods]
impl RegionSink {
    /// Send a table's rows on to DataFusion, in the region's schema.
    fn __call__(&self, table: &Bound<'_, PyAny>) -> PyResult<()> {
        let failure = |error: DataFusionError| PyRuntimeError::new_err(error.to_string());
        for batch in stream_from_pyarrow(table)? {
            let batch = batch.map_err(|error| failure(error.into()))?;
            let batch = conform(batch, &self.names, &self.schema).map_err(failure)?;
            // the receiver is gone once DataFusion stops reading
            let _ = self.rows.send(RegionRows::Batch(Ok(batch)));
        }
        Ok(())
    }
}

impl PythonRunner {
    /// One region's operators and DataFusion plan, as [`region_json`]
    /// writes them.
    fn request(&self, region: &ModelRegion) -> Result<String> {
        let (turn, layout) = &self.prompting;
        let (operators, plan) = derive::derive_region(region, turn, layout)
            .map_err(|error| DataFusionError::Plan(format!("CompileError: {}", error.0)))?;
        region_json(&operators, &plan)
    }

    /// Plan a region in Rust, or None when Python plans it: a region that
    /// asks no model. `payload` is the region as [`region_json`] writes it;
    /// `tables` holds the rows of each input, by name. The plan carries
    /// the region's DataFusion plan on to Python.
    fn plan_here(
        &self,
        region: &ModelRegion,
        tables: HashMap<String, TableRows>,
        payload: &str,
        fetch: Option<usize>,
    ) -> Result<Option<String>> {
        let Some((planner, settings)) = &self.planning else {
            return Ok(None);
        };
        let (mut operators, logical) = read_region_json(payload)?;
        if let Some(fetch) = fetch {
            operators = operators.limited(fetch as i64);
        }
        if !asks_model(&operators) {
            return Ok(None);
        }
        let mut all = named_tables(region)?;
        all.extend(tables);
        let compile = |error: quail_planner::CompileError| {
            DataFusionError::Plan(format!("CompileError: {}", error.0))
        };
        let mut planned = match planner.plan(&operators, &all, settings) {
            Ok(PlanOutcome::Planned(planned)) => planned_json(&planned).map_err(compile)?,
            Ok(PlanOutcome::Refused(refused)) => crate::planning::refused_json(&refused),
            Ok(PlanOutcome::Fixed(plan)) => plan,
            Err(error) => return Err(compile(error)),
        };
        planned["logical"] = logical.into();
        Ok(Some(planned.to_string()))
    }

    /// Plan a region without running it, over the rows of its inputs that
    /// call no model; an input that calls a model plans as empty.
    ///
    /// Returns the region as [`region_json`] writes it, and the planned
    /// region as JSON or None when Python plans it.
    pub(crate) fn plan_only(
        &self,
        region: &ModelRegion,
        inputs: &[InputRows],
    ) -> Result<(String, Option<String>)> {
        let payload = self.request(region)?;
        let mut sorted = Vec::new();
        for ((name, batches), input) in inputs.iter().zip(&region.inputs) {
            sorted.push((name.clone(), in_source_order(input, batches)?));
        }
        let planned = self.plan_here(region, input_tables(region, &sorted), &payload, None)?;
        Ok((payload, planned))
    }

    /// Run a call's region on this thread and send its rows, then the end.
    pub(crate) fn serve(&self, call: RegionCall) {
        let found = self.run_here(
            &call.region,
            call.inputs,
            call.fetch,
            &call.hint,
            &call.rows,
        );
        if let Err(error) = found {
            let _ = call.rows.send(RegionRows::Batch(Err(error)));
        }
        let _ = call.rows.send(RegionRows::Done);
    }

    /// Run the region on this thread, sending its rows to `rows` as the
    /// executor finishes them.
    ///
    /// The region runs once, as one plan. It waits for each input to end
    /// and plans on its exact rows, except the one input it reads while
    /// DataFusion produces it (see [`streamed_input`]), which it plans
    /// from the rows of the table its documents come from. With a hint,
    /// each row an input or a table the region reads by name holds is
    /// dropped where a test rejects it: an input's rows before planning,
    /// a streamed input's as they arrive, and the executor reads the
    /// hint to drop more as it tightens.
    fn run_here(
        &self,
        region: &ModelRegion,
        inputs: Vec<InputStream>,
        fetch: Option<usize>,
        hint: &RegionHint,
        rows: &RowSender,
    ) -> Result<()> {
        let payload = self.request(region)?;
        let Some((planner, settings)) = &self.planning else {
            let mut read = Vec::new();
            for (name, batches) in inputs {
                read.push((name, read_to_end(batches)?));
            }
            return self.call_engine(region, Call::Python(&payload, &read, fetch), rows);
        };
        let (operators, _) = read_region_json(&payload)?;
        let columns = RegionColumns::of(region);
        // a fetch or a stop key keeps the first rows: dropping one changes
        // which rows those are
        let hinted = fetch.is_none() && operators.stop_keys.is_empty() && !hint.is_empty();
        let hint_tables = match hinted {
            true => hint.tables(&columns),
            false => Vec::new(),
        };
        let hint_rows = HintRows::default();
        // several GPUs' workers take every document at the start
        let streamed = match fetch {
            None if settings.gpus == 1 && asks_model(&operators) => {
                streamed_input(region, &operators)
            }
            _ => None,
        };
        let mut read = Vec::new();
        let mut reader = None;
        for (index, (name, batches)) in inputs.into_iter().enumerate() {
            if let Some((_, scan)) = streamed.as_ref().filter(|(found, _)| *found == index) {
                let hint = hint_tables
                    .contains(&name)
                    .then(|| (hint.clone(), columns.clone(), Arc::clone(&hint_rows)));
                let documents = Documents {
                    planner: Arc::clone(planner),
                    column: scan.column.clone(),
                    with_offsets: extracted_aliases(std::iter::once(&operators))
                        .contains(&scan.alias),
                };
                let handed = handed_columns(region, index, scan)?;
                reader = Some((
                    index,
                    scan.clone(),
                    InputReader::new(name, batches, documents, handed, hint),
                ));
                continue;
            }
            let mut batches = in_source_order(&region.inputs[index], &read_to_end(batches)?)?;
            if hint_tables.contains(&name) {
                let schema = Arc::new(region.inputs[index].schema().as_arrow().clone());
                let all = concat_batches(&schema, &batches)?;
                let kept = hinted_rows(hint, &columns, &name, &all)?;
                hint_rows
                    .lock()
                    .expect("the hint rows lock")
                    .insert(name.clone(), vec![kept.clone()]);
                batches = vec![kept];
            }
            read.push((name, batches));
        }
        for table in &hint_tables {
            let known = hint_rows
                .lock()
                .expect("the hint rows lock")
                .contains_key(table);
            if !known && let Some(found) = named_rows(region, table)? {
                hint_rows
                    .lock()
                    .expect("the hint rows lock")
                    .insert(table.clone(), vec![found]);
            }
        }
        let mut tables = input_tables(region, &read);
        if let Some((index, scan, reader)) = &reader {
            tables.insert(reader.table_name(), table_statistics(region, *index, scan)?);
        }
        let planned = self.plan_here(region, tables, &payload, fetch)?;
        let hint =
            (!hint_tables.is_empty()).then(|| InputHint::new(hint.clone(), columns, hint_rows));
        let call = match &planned {
            Some(planned) => Call::Planned {
                planned,
                inputs: &read,
                hint,
                reader: reader.map(|(_, _, reader)| Box::new(reader)),
            },
            None => Call::Python(&payload, &read, fetch),
        };
        self.call_engine(region, call, rows)
    }

    /// Hand the region to the engine's `execute`, or to `run_region`
    /// when Python plans it, and send its rows to `rows`.
    fn call_engine(&self, region: &ModelRegion, call: Call<'_>, rows: &RowSender) -> Result<()> {
        let schema = Arc::clone(region.plan.schema());
        let names = quail_column_names(&schema);
        let sink = RegionSink {
            rows: rows.clone(),
            names: names.clone(),
            schema: Arc::clone(schema.inner()),
        };
        let reader = Python::attach(|py| {
            let answer = match call {
                Call::Planned {
                    planned,
                    inputs,
                    hint,
                    reader,
                } => {
                    let tables = pyarrow_tables(py, region, inputs)?;
                    let sink = Bound::new(py, sink)?;
                    let hint = hint.map(|hint| Bound::new(py, hint)).transpose()?;
                    let reader = reader.map(|reader| Bound::new(py, *reader)).transpose()?;
                    self.engine.call_method1(
                        py,
                        "execute",
                        (planned, tables, sink, hint, reader),
                    )?
                }
                Call::Python(payload, inputs, fetch) => {
                    let tables = pyarrow_tables(py, region, inputs)?;
                    self.engine
                        .call_method1(py, "run_region", (payload, tables, fetch))?
                }
            };
            // an engine that sent every row to the sink returns None
            let answer = answer.bind(py);
            if answer.is_none() {
                return Ok(None);
            }
            stream_from_pyarrow(answer).map(Some)
        })
        .map_err(python_failure)?;
        for batch in reader.into_iter().flatten() {
            let batch = conform(batch?, &names, schema.inner())?;
            if rows.send(RegionRows::Batch(Ok(batch))).is_err() {
                break;
            }
        }
        Ok(())
    }
}

/// Each input as a pyarrow Table, by the table name the region reads.
fn pyarrow_tables<'py>(
    py: Python<'py>,
    region: &ModelRegion,
    inputs: &[InputRows],
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let mut tables = Vec::new();
    for (name, batches) in inputs {
        let input = region
            .input_names
            .iter()
            .position(|found| found == name)
            .expect("an input of the region");
        let schema = Arc::new(region.inputs[input].schema().as_arrow().clone());
        tables.push((name.clone(), to_pyarrow(py, schema, batches.clone())?));
    }
    tables.into_py_dict(py)
}

/// How a region goes to the engine.
enum Call<'a> {
    /// Planned in Rust: the plan as JSON, the inputs read to their end,
    /// the hint, and the input read as it arrives.
    Planned {
        planned: &'a str,
        inputs: &'a [InputRows],
        hint: Option<InputHint>,
        reader: Option<Box<InputReader>>,
    },
    /// Planned by Python: the region as JSON, its inputs, and its fetch.
    Python(&'a str, &'a [InputRows], Option<usize>),
}

/// The columns of a streamed input the executor reads with each row: the
/// scan's value columns and the column its documents come from.
fn handed_columns(region: &ModelRegion, index: usize, scan: &Scan) -> Result<SchemaRef> {
    let schema = region.inputs[index].schema().as_arrow().clone();
    let mut fields = Vec::new();
    for name in scan
        .columns
        .iter()
        .cloned()
        .chain(source_column(&scan.column))
    {
        let field = schema.field_with_name(&name)?.clone();
        if !fields.contains(&field) {
            fields.push(field);
        }
    }
    Ok(Arc::new(datafusion::arrow::datatypes::Schema::new(fields)))
}

/// Read an input's batches to its end.
fn read_to_end(batches: InputBatches) -> Result<Vec<RecordBatch>> {
    batches.into_iter().collect()
}

/// Whether a region's operators ask the model; Python plans a region
/// that does not.
fn asks_model(operators: &Operators) -> bool {
    !(operators.filters.is_empty()
        && operators.joins.is_empty()
        && operators.classifies.is_empty()
        && operators.extracts.is_empty()
        && operators.sae_features.is_empty()
        && operators.maps.is_empty()
        && operators.projections.is_empty())
}

/// The input the region reads while DataFusion produces it, by position,
/// with the scan whose documents it holds.
///
/// It is the first input that holds one scan's documents, which the
/// scan does not test itself, and that DataFusion computes with more than
/// a table's filters and projections, such as a join, from tables alone.
/// The region waits for every other input to end: a scan's rows arrive
/// at once, and another region's rows are complete before this region
/// starts, as both run on the calling thread one after the other.
fn streamed_input(region: &ModelRegion, operators: &Operators) -> Option<(usize, Scan)> {
    use datafusion::common::tree_node::TreeNode;
    use datafusion::logical_expr::LogicalPlan;
    let scanned = |input: &LogicalPlan| {
        !input
            .exists(|node| {
                Ok(match node {
                    LogicalPlan::TableScan(_)
                    | LogicalPlan::Filter(_)
                    | LogicalPlan::Projection(_)
                    | LogicalPlan::SubqueryAlias(_) => false,
                    LogicalPlan::Extension(extension) => extension
                        .node
                        .as_any()
                        .downcast_ref::<SharedInput>()
                        .is_none(),
                    _ => true,
                })
            })
            .unwrap_or(true)
    };
    let calls_model = |input: &LogicalPlan| {
        input
            .exists(|node| {
                Ok(matches!(node, LogicalPlan::Extension(extension)
                    if extension.node.as_any().downcast_ref::<ModelRegion>().is_some()))
            })
            .unwrap_or(true)
    };
    region
        .input_names
        .iter()
        .zip(&region.inputs)
        .enumerate()
        .find_map(|(index, (name, input))| {
            let mut scans = operators.scans.iter().filter(|scan| &scan.provider == name);
            let scan = scans.next()?;
            let streamed = scans.next().is_none()
                && scan.predicates.is_empty()
                && !scanned(input)
                && !calls_model(input);
            streamed.then(|| (index, scan.clone()))
        })
}

/// The rows a streamed input is planned over, in the input's schema: the
/// rows of a registered table the input reads that holds the input's
/// document column, each input column taken from the table's column of
/// the same name, or no rows when no such table is found.
fn table_statistics(region: &ModelRegion, index: usize, scan: &Scan) -> Result<TableRows> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source_as_provider;
    use datafusion::logical_expr::LogicalPlan;
    let input = &region.inputs[index];
    // a column the table lacks is planned as nulls
    let schema = Arc::new(datafusion::arrow::datatypes::Schema::new(
        input
            .schema()
            .as_arrow()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone().with_nullable(true))
            .collect::<Vec<_>>(),
    ));
    let source = source_column(&scan.column);
    let mut found: Option<(String, Arc<dyn TableProvider>)> = None;
    input.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && let Ok(provider) = source_as_provider(&scan.source)
            && let Some(table) = provider.downcast_ref::<IdentifiedTable>()
            && let Some(python) = table.inner().downcast_ref::<PythonTable>()
            && source
                .as_ref()
                .is_some_and(|name| python.schema.index_of(name).is_ok())
        {
            found = Some((table.identity().to_string(), Arc::clone(&provider)));
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    let Some((identity, provider)) = found else {
        return Ok(TableRows {
            identity: None,
            schema,
            batches: Vec::new(),
        });
    };
    let python = provider
        .downcast_ref::<IdentifiedTable>()
        .and_then(|table| table.inner().downcast_ref::<PythonTable>())
        .expect("a Python table");
    let mut batches = Vec::new();
    for batch in python.rows()? {
        let columns = schema
            .fields()
            .iter()
            .map(|field| {
                let column = batch
                    .column_by_name(field.name())
                    .and_then(|column| cast(column, field.data_type()).ok());
                column.unwrap_or_else(|| new_null_array(field.data_type(), batch.num_rows()))
            })
            .collect();
        batches.push(RecordBatch::try_new(Arc::clone(&schema), columns)?);
    }
    // the token store is the table's, kept by its identity
    Ok(TableRows {
        identity: Some(identity),
        schema,
        batches,
    })
}

/// The rows of each input read to its end, by name, for planning.
fn input_tables(region: &ModelRegion, inputs: &[InputRows]) -> HashMap<String, TableRows> {
    let mut tables = HashMap::new();
    for (name, batches) in inputs {
        let Some(input) = region.input_names.iter().position(|found| found == name) else {
            continue;
        };
        tables.insert(
            name.clone(),
            TableRows {
                identity: whole_table_identity(&region.inputs[input]),
                schema: Arc::new(region.inputs[input].schema().as_arrow().clone()),
                batches: batches.clone(),
            },
        );
    }
    tables
}

impl RegionRunner for PythonRunner {
    /// Run the region on the thread that called DataFusion, so the GPU
    /// process's per-thread state is set up once, not on each worker.
    fn run(
        &self,
        region: &ModelRegion,
        inputs: Vec<InputStream>,
        fetch: Option<usize>,
        hint: &RegionHint,
    ) -> Result<RegionBatches> {
        let caller = self.caller.lock().expect("the caller lock").clone();
        let (rows, receiver) = std::sync::mpsc::channel();
        let call = RegionCall {
            region: region.clone(),
            inputs,
            fetch,
            hint: hint.clone(),
            rows,
        };
        match caller {
            Some((thread, sender)) if thread != std::thread::current().id() => sender
                .send(call)
                .map_err(|_| DataFusionError::Execution("the query ended".into()))?,
            _ => self.serve(call),
        }
        Ok(Box::new(ReceivedRows {
            receiver,
            ended: false,
        }))
    }
}

/// Each identified table a region's plan reads by name, by table name.
fn identified_scans(region: &ModelRegion) -> Result<Vec<(String, Arc<dyn TableProvider>)>> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source_as_provider;
    use datafusion::logical_expr::LogicalPlan;
    let mut found: Vec<(String, Arc<dyn TableProvider>)> = Vec::new();
    region.plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && let Ok(provider) = source_as_provider(&scan.source)
            && provider.downcast_ref::<IdentifiedTable>().is_some()
        {
            found.push((scan.table_name.table().to_string(), provider));
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

/// Every row of a table the region reads by name, as one batch, when
/// its rows come from Python.
fn named_rows(region: &ModelRegion, table: &str) -> Result<Option<RecordBatch>> {
    let found = named_tables(region)?;
    let Some(rows) = found.get(table) else {
        return Ok(None);
    };
    Ok(Some(concat_batches(&rows.schema, &rows.batches)?))
}

/// A region as JSON: `operators`, its operators as JSON text, and
/// `logical`, DataFusion's plan of the region as `display_indent` writes
/// it.
///
/// # Errors
///
/// The operators do not write.
pub fn region_json(
    operators: &Operators,
    plan: &datafusion::logical_expr::LogicalPlan,
) -> Result<String> {
    let operators = crate::planning::operators_json(operators)
        .map_err(|error| DataFusionError::Plan(format!("CompileError: {}", error.0)))?;
    Ok(serde_json::json!({
        "operators": operators,
        "logical": plan.display_indent().to_string(),
    })
    .to_string())
}

/// A region's operators and DataFusion plan text from [`region_json`].
fn read_region_json(payload: &str) -> Result<(Operators, String)> {
    let external = |error: serde_json::Error| DataFusionError::External(Box::new(error));
    let found: serde_json::Value = serde_json::from_str(payload).map_err(external)?;
    let operators =
        serde_json::from_str(found["operators"].as_str().unwrap_or_default()).map_err(external)?;
    let logical = found["logical"].as_str().unwrap_or_default().to_string();
    Ok((operators, logical))
}

/// An input's rows in the order of the registered table it holds rows of,
/// found by the table's id column; the rows unchanged for any other input.
///
/// DataFusion scans and filters in parallel, so the rows reach the region
/// in no fixed order.
fn in_source_order(
    input: &datafusion::logical_expr::LogicalPlan,
    batches: &[RecordBatch],
) -> Result<Vec<RecordBatch>> {
    use datafusion::arrow::compute::{concat_batches, sort_to_indices, take_record_batch};
    use datafusion::arrow::datatypes::DataType;
    use datafusion::common::Constraint;
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source_as_provider;
    use datafusion::logical_expr::LogicalPlan;
    let mut source = None;
    input.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && let Ok(provider) = source_as_provider(&scan.source)
            && provider.downcast_ref::<IdentifiedTable>().is_some()
        {
            source = Some(provider);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    let Some(provider) = source else {
        return Ok(batches.to_vec());
    };
    let table = provider
        .downcast_ref::<IdentifiedTable>()
        .expect("an identified table");
    let Some(python) = table.inner().downcast_ref::<PythonTable>() else {
        return Ok(batches.to_vec());
    };
    let key = python
        .constraints
        .iter()
        .find_map(|constraint| match constraint {
            Constraint::PrimaryKey(keys) if keys.len() == 1 => Some(keys[0]),
            _ => None,
        });
    let Some(key) = key.map(|index| python.schema.field(index).name().clone()) else {
        return Ok(batches.to_vec());
    };
    let Some(first) = batches.first() else {
        return Ok(Vec::new());
    };
    let rows = concat_batches(&first.schema(), batches)?;
    let Ok(column) = rows.schema().index_of(&key) else {
        return Ok(batches.to_vec());
    };
    let texts = |values: &ArrayRef| -> Result<Vec<Option<String>>> {
        let values = cast(values, &DataType::Utf8)?;
        let values = values
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StringArray>()
            .expect("a string array");
        Ok(values
            .iter()
            .map(|value| value.map(str::to_string))
            .collect())
    };
    let mut positions: HashMap<String, u64> = HashMap::new();
    let mut next = 0u64;
    for batch in python.column_rows(&key)? {
        for value in texts(batch.column(0))? {
            if let Some(value) = value {
                positions.entry(value).or_insert(next);
            }
            next += 1;
        }
    }
    let found: datafusion::arrow::array::UInt64Array = texts(rows.column(column))?
        .into_iter()
        .map(|value| value.and_then(|value| positions.get(&value).copied()))
        .collect();
    let order = sort_to_indices(&found, None, None)?;
    Ok(vec![take_record_batch(&rows, &order)?])
}

/// The rows and identity of each identified table a region reads by
/// name, by table name.
fn named_tables(region: &ModelRegion) -> Result<HashMap<String, TableRows>> {
    let found = identified_scans(region)?;
    let mut tables = HashMap::new();
    for (name, provider) in found {
        let table = provider
            .downcast_ref::<IdentifiedTable>()
            .expect("an identified table");
        let Some(python) = table.inner().downcast_ref::<PythonTable>() else {
            continue;
        };
        tables.insert(
            name,
            TableRows {
                identity: Some(table.identity().to_string()),
                schema: Arc::clone(&python.schema),
                batches: python.rows()?,
            },
        );
    }
    Ok(tables)
}

/// The content identity of an input that is a whole table with one,
/// read without filters.
fn whole_table_identity(input: &datafusion::logical_expr::LogicalPlan) -> Option<String> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source_as_provider;
    use datafusion::logical_expr::LogicalPlan;
    let mut identity = None;
    let mut whole = true;
    let _ = input.apply(|node| {
        match node {
            LogicalPlan::TableScan(scan) => {
                whole &= scan.filters.is_empty() && scan.fetch.is_none();
                identity = source_as_provider(&scan.source).ok().and_then(|provider| {
                    provider
                        .downcast_ref::<IdentifiedTable>()
                        .map(|table| table.identity().to_string())
                });
            }
            LogicalPlan::SubqueryAlias(_) | LogicalPlan::Projection(_) => {}
            LogicalPlan::Extension(extension)
                if extension
                    .node
                    .as_any()
                    .downcast_ref::<SharedInput>()
                    .is_some() => {}
            _ => whole = false,
        }
        Ok(TreeNodeRecursion::Continue)
    });
    if whole { identity } else { None }
}

/// Give a batch of Quail's rows the region's schema: each region
/// column is the Quail column of the same name, cast to its type.
///
/// A column Quail names differently, such as `r.details.body` for
/// DataFusion's `r.details[body]`, is the one in the same position when
/// both list the same number of columns, as both follow the SELECT list.
fn conform(batch: RecordBatch, names: &[String], schema: &SchemaRef) -> Result<RecordBatch> {
    let positional = batch.num_columns() == names.len();
    let columns = names
        .iter()
        .zip(schema.fields())
        .enumerate()
        .map(|(index, (name, field))| {
            let found = batch
                .column_by_name(name)
                .or_else(|| positional.then(|| batch.column(index)));
            let column = found.ok_or_else(|| {
                let returned: Vec<_> = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                DataFusionError::Execution(format!(
                    "Quail returned columns {returned:?}, without the region's column {name}"
                ))
            })?;
            Ok(cast(column, field.data_type())?)
        })
        .collect::<Result<Vec<ArrayRef>>>()?;
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

/// A table whose rows a Python function returns on each scan.
///
/// Model regions read it through the engine, which keeps its token
/// counts, so DataFusion scans it only for operators outside every
/// region.
#[derive(Debug)]
pub(crate) struct PythonTable {
    pub(crate) schema: SchemaRef,
    pub(crate) scan: Py<PyAny>,
    /// The id column as a primary key, which DataFusion's rules read.
    pub(crate) constraints: Constraints,
}

impl PythonTable {
    /// Every row of the table.
    pub(crate) fn rows(&self) -> Result<Vec<RecordBatch>> {
        let columns: Vec<String> = self
            .schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        self.scan_columns(columns)
    }

    /// One column of the table's rows.
    fn column_rows(&self, name: &str) -> Result<Vec<RecordBatch>> {
        self.scan_columns(vec![name.to_string()])
    }

    fn scan_columns(&self, columns: Vec<String>) -> Result<Vec<RecordBatch>> {
        Python::attach(|py| {
            let answer = self.scan.call1(py, (columns,))?;
            from_pyarrow(answer.bind(py))
        })
        .map_err(python_failure)
    }
}

#[async_trait]
impl TableProvider for PythonTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn constraints(&self) -> Option<&Constraints> {
        Some(&self.constraints)
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => Arc::clone(&self.schema),
        };
        let columns: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        let batches = Python::attach(|py| {
            let answer = self.scan.call1(py, (columns,))?;
            from_pyarrow(answer.bind(py))
        })
        .map_err(python_failure)?;
        // one partition per thread DataFusion runs, each a slice of the
        // rows, so the operators above read the table in parallel
        let num_partitions = state.config().target_partitions().max(1);
        let mut partitions: Vec<Vec<RecordBatch>> = vec![Vec::new(); num_partitions];
        for batch in batches {
            let size = batch.num_rows().div_ceil(num_partitions).max(1);
            let mut offset = 0;
            let mut index = 0;
            while offset < batch.num_rows() {
                let length = size.min(batch.num_rows() - offset);
                partitions[index % num_partitions].push(batch.slice(offset, length));
                offset += length;
                index += 1;
            }
        }
        Ok(MemorySourceConfig::try_new_exec(&partitions, schema, None)?)
    }
}
