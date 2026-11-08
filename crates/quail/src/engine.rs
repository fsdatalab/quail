//! The model region runner that calls Quail's Python executor, and the
//! tables whose rows a Python provider returns.

use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, BooleanArray, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::compute::{concat_batches, filter_record_batch};
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
    RegionColumns, RegionHint, RegionRunner, SharedInput, collect_inputs, derive,
    quail_column_names,
};
use quail_planner::operators::Operators;
use quail_planner::prompts::Turn;
use tokio::sync::mpsc::UnboundedSender;

use crate::planning::{PlanOutcome, Planner, Settings, TableRows, planned_json};
use crate::pyarrow::{from_pyarrow, stream_from_pyarrow, to_pyarrow};

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

/// The rows of a region input the executor runs at once, when the
/// region asks each row on its own: a larger input runs in windows of
/// this many rows, each planned and run as it arrives.
pub const INPUT_WINDOW_ROWS: usize = 65_536;

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

/// A region's hint over the rows of its one input that the executor
/// reads: whether each row may still pass the tests above the region, as
/// they tighten while the region runs.
#[pyclass(name = "InputHint", module = "_quail_native", frozen)]
pub(crate) struct InputHint {
    hint: RegionHint,
    columns: RegionColumns,
    rows: RecordBatch,
    /// The table name the region reads the input by.
    #[pyo3(get)]
    table: String,
}

#[pymethods]
impl InputHint {
    /// A number that changes whenever a test tightens.
    fn generation(&self) -> u64 {
        self.hint.generation()
    }

    /// Whether each input row, in the order the executor reads them, may
    /// still pass every test.
    fn keep(&self) -> PyResult<Vec<bool>> {
        self.hint
            .keep(&self.columns, &self.table, &self.rows)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
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
    /// the plan carries its DataFusion plan on to Python.
    fn plan_here(
        &self,
        region: &ModelRegion,
        inputs: &[InputRows],
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
        // Python plans a region that asks no model
        let asks_model = !(operators.filters.is_empty()
            && operators.joins.is_empty()
            && operators.classifies.is_empty()
            && operators.extracts.is_empty()
            && operators.sae_features.is_empty()
            && operators.maps.is_empty()
            && operators.projections.is_empty());
        if !asks_model {
            return Ok(None);
        }
        let mut tables = named_tables(region)?;
        for ((name, batches), input) in inputs.iter().zip(&region.inputs) {
            tables.insert(
                name.clone(),
                TableRows {
                    identity: whole_table_identity(input),
                    schema: Arc::new(input.schema().as_arrow().clone()),
                    batches: batches.clone(),
                },
            );
        }
        let compile = |error: quail_planner::CompileError| {
            DataFusionError::Plan(format!("CompileError: {}", error.0))
        };
        let mut planned = match planner.plan(&operators, &tables, settings) {
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
        let sorted = inputs
            .iter()
            .zip(&region.inputs)
            .map(|((name, batches), input)| Ok((name.clone(), in_source_order(input, batches)?)))
            .collect::<Result<Vec<InputRows>>>()?;
        let planned = self.plan_here(region, &sorted, &payload, None)?;
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
    /// A region that asks each row of its one input on its own runs the
    /// input in windows of rows as they arrive, each planned on its own
    /// rows; any other region, and an input that ends within one window,
    /// runs once over every row. Such a region also drops each input row
    /// a test of `hint` rejects, by the input columns it passes on, before
    /// planning, and hands the executor the hint to drop rows by as it
    /// tightens.
    fn run_here(
        &self,
        region: &ModelRegion,
        inputs: Vec<InputStream>,
        fetch: Option<usize>,
        hint: &RegionHint,
        rows: &RowSender,
    ) -> Result<()> {
        let payload = self.request(region)?;
        let columns = RegionColumns::of(region);
        let source = match self.planning.is_some() && fetch.is_none() && !hint.is_empty() {
            true => hinted_table(region, &payload)?,
            false => None,
        };
        let hint = source.as_deref().map(|table| (hint, &columns, table));
        if !(self.planning.is_some() && fetch.is_none() && row_local(region, &payload)?) {
            let inputs = collect_inputs(inputs)?;
            return self.run_window(region, &payload, &inputs, fetch, hint, true, rows);
        }
        let size = self.input_window_rows()?;
        let mut inputs = inputs;
        let (name, mut batches) = inputs.remove(0);
        let mut first = true;
        let mut rest = None;
        loop {
            let (window, ended) = read_window(&mut batches, &mut rest, size)?;
            let window = [(name.clone(), window)];
            self.run_window(region, &payload, &window, fetch, hint, first, rows)?;
            first = false;
            if ended {
                return Ok(());
            }
        }
    }

    /// The rows of an input window, from the engine's
    /// `input_window_rows` when it has one.
    fn input_window_rows(&self) -> Result<usize> {
        Python::attach(
            |py| match self.engine.bind(py).getattr("input_window_rows") {
                Ok(found) => found.extract::<usize>(),
                Err(_) => Ok(INPUT_WINDOW_ROWS),
            },
        )
        .map_err(|error| DataFusionError::External(Box::new(error)))
    }

    /// Plan and run the region over these rows of its inputs, sending
    /// its rows to `rows`. With a hint over the one table the region
    /// reads, an input's rows the hint rejects are dropped first, and
    /// the executor reads the hint as it tightens. A window after the
    /// `first` that keeps no row does not run.
    #[allow(clippy::too_many_arguments)]
    fn run_window(
        &self,
        region: &ModelRegion,
        payload: &str,
        inputs: &[InputRows],
        fetch: Option<usize>,
        hint: Option<(&RegionHint, &RegionColumns, &str)>,
        first: bool,
        rows: &RowSender,
    ) -> Result<()> {
        let mut sorted;
        let mut held = None;
        let (planned, inputs) = if self.planning.is_some() {
            sorted = inputs
                .iter()
                .zip(&region.inputs)
                .map(|((name, batches), input)| {
                    Ok((name.clone(), in_source_order(input, batches)?))
                })
                .collect::<Result<Vec<InputRows>>>()?;
            if let Some((hint, columns, table)) = hint {
                let found = sorted.iter().position(|(name, _)| name == table);
                let found_rows = match found {
                    Some(input) => {
                        let schema = Arc::new(region.inputs[input].schema().as_arrow().clone());
                        let window = concat_batches(&schema, &sorted[input].1)?;
                        let keep = BooleanArray::from(hint.keep(columns, table, &window)?);
                        let window = filter_record_batch(&window, &keep)?;
                        if window.num_rows() == 0 && !first {
                            return Ok(());
                        }
                        sorted[input].1 = vec![window.clone()];
                        Some(window)
                    }
                    None => named_rows(region, table)?,
                };
                held = found_rows.map(|rows| InputHint {
                    hint: hint.clone(),
                    columns: columns.clone(),
                    rows,
                    table: table.to_string(),
                });
            }
            (
                self.plan_here(region, &sorted, payload, fetch)?,
                sorted.as_slice(),
            )
        } else {
            (None, inputs)
        };
        let schema = Arc::clone(region.plan.schema());
        let names = quail_column_names(&schema);
        let sink = RegionSink {
            rows: rows.clone(),
            names: names.clone(),
            schema: Arc::clone(schema.inner()),
        };
        let reader = Python::attach(|py| {
            // each input as a pyarrow Table, by the table name the region reads
            let mut tables = Vec::new();
            for ((name, batches), input) in inputs.iter().zip(&region.inputs) {
                let schema = Arc::new(input.schema().as_arrow().clone());
                tables.push((name.clone(), to_pyarrow(py, schema, batches.clone())?));
            }
            let tables = tables.into_py_dict(py)?;
            let answer = match &planned {
                Some(found) => {
                    let sink = Bound::new(py, sink)?;
                    let hint = held.map(|hint| Bound::new(py, hint)).transpose()?;
                    self.engine
                        .call_method1(py, "execute", (found, tables, sink, hint))?
                }
                None => self
                    .engine
                    .call_method1(py, "run_region", (payload, tables, fetch))?,
            };
            // an engine that sent every row to the sink returns None
            let answer = answer.bind(py);
            if answer.is_none() {
                return Ok(None);
            }
            stream_from_pyarrow(answer).map(Some)
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
        for batch in reader.into_iter().flatten() {
            let batch = conform(batch?, &names, schema.inner())?;
            if rows.send(RegionRows::Batch(Ok(batch))).is_err() {
                break;
            }
        }
        Ok(())
    }
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

/// Read batches until they hold `size` rows or the input ends; a batch
/// past `size` rows is cut, and its rest starts the next window.
///
/// Returns the batches and whether the input ended.
fn read_window(
    batches: &mut InputBatches,
    rest: &mut Option<RecordBatch>,
    size: usize,
) -> Result<(Vec<RecordBatch>, bool)> {
    let mut window = Vec::new();
    let mut held = 0;
    while held < size {
        let batch = match rest.take() {
            Some(batch) => batch,
            None => match batches.next() {
                Some(batch) => batch?,
                None => return Ok((window, true)),
            },
        };
        let taken = batch.num_rows().min(size - held);
        if taken < batch.num_rows() {
            *rest = Some(batch.slice(taken, batch.num_rows() - taken));
        }
        held += taken;
        window.push(batch.slice(0, taken));
    }
    Ok((window, false))
}

/// Whether each of a region's rows comes from one row of its one input
/// alone: no join, no table read by name, no fetch, and no stop key, so
/// the region can run its input's rows in windows.
fn row_local(region: &ModelRegion, payload: &str) -> Result<bool> {
    let (operators, _) = read_region_json(payload)?;
    Ok(region.inputs.len() == 1
        && identified_scans(region)?.is_empty()
        && operators.joins.is_empty()
        && operators.fetch.is_none()
        && operators.stop_keys.is_empty())
}

/// The one table a region reads, by input name or table name, when a
/// hint over it can drop the region's rows: each row comes from one row
/// of that table, with no join, fetch, or stop key.
fn hinted_table(region: &ModelRegion, payload: &str) -> Result<Option<String>> {
    let (operators, _) = read_region_json(payload)?;
    if !(operators.joins.is_empty() && operators.fetch.is_none() && operators.stop_keys.is_empty())
    {
        return Ok(None);
    }
    let mut tables: Vec<String> = region.input_names.clone();
    for (name, _) in identified_scans(region)? {
        if !tables.contains(&name) {
            tables.push(name);
        }
    }
    Ok(match tables.as_slice() {
        [table] => Some(table.clone()),
        _ => None,
    })
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
        .map_err(|error| DataFusionError::External(Box::new(error)))
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
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
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
