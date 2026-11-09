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
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::physical_plan::ExecutionPlan;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::collections::HashMap;

use quail_datafusion::{
    IdentifiedTable, InputRows, InputStream, ModelRegion, RegionBatches, RegionColumns, RegionHint,
    RegionRunner, SharedInput, collect_inputs, derive, quail_column_names,
};
use quail_planner::operators::Operators;
use quail_planner::prompts::Turn;
use tokio::sync::mpsc::UnboundedSender;

use crate::planning::{PlannedRegion, Planner, ScanInputs, Settings, TableRows, outcome_json};
use crate::pyarrow::{from_pyarrow, stream_from_pyarrow, to_pyarrow};
use crate::{EngineConfig, python_failure, start_python};

/// Runs each model region by calling the engine's
/// `execute(planned, tables, sink, hint)` or `run_region(payload,
/// tables, fetch)`; `planned` is the planned region's JSON and its scans,
/// as [`scans_to_pyarrow`] hands them over. `execute` hands the sink each batch of finished rows as the
/// region runs and returns None, or returns the rows; `run_region`
/// returns them, as a pyarrow Table or RecordBatchReader.
///
/// It is the [`RegionRunner`] a [`crate::Session`] runs model regions
/// on. To add Quail to a DataFusion session of your own, register
/// `QuailExtension::new(Arc::new(Executor::open(&config)?))`.
#[derive(Debug)]
pub struct Executor {
    pub(crate) engine: Py<PyAny>,
    /// The optimizer rules each model region runs, by name, in order.
    pub(crate) rules: Vec<String>,
    /// Whether this executor opened its engine, and closes it.
    owns_engine: bool,
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

impl Executor {
    /// Open Quail's executor for `config`, starting Python if this process
    /// has not, and loading it from the `quail` package.
    ///
    /// The interpreter finds `quail` on its own path, or in the virtual
    /// environment `VIRTUAL_ENV` names.
    ///
    /// # Errors
    ///
    /// Python cannot import `quail`, or Quail refuses the configuration.
    pub fn open(config: &EngineConfig) -> Result<Self> {
        start_python()?;
        let engine = Python::attach(|py| {
            let engine = py
                .import("quail.api.engine")?
                .getattr("Engine")?
                .call_method1(
                    "open",
                    (
                        config.model.as_str(),
                        config.device.as_str(),
                        config.gpus,
                        config.backend.as_str(),
                    ),
                )?;
            Ok::<_, PyErr>(engine.unbind())
        })
        .map_err(python_failure)?;
        let mut executor = Self::with_engine(engine)?;
        executor.owns_engine = true;
        Ok(executor)
    }

    /// An executor that runs model regions on `engine`, a Python object
    /// as [`crate::Session::with_engine`] describes, planning each region
    /// in Rust when the engine has a `tokenizer_file` attribute.
    ///
    /// # Errors
    ///
    /// An attribute is missing or malformed.
    pub fn with_engine(engine: Py<PyAny>) -> Result<Self> {
        let tokenizer_file = Python::attach(|py| match engine.bind(py).getattr("tokenizer_file") {
            Ok(found) => found.extract::<Option<String>>(),
            Err(_) => Ok(None),
        })
        .map_err(python_failure)?;
        let planner = tokenizer_file
            .map(|path| Planner::new(std::path::Path::new(&path)))
            .transpose()
            .map_err(|error| DataFusionError::Plan(format!("CompileError: {}", error.message())))?
            .map(Arc::new);
        Self::with_planner(engine, planner)
    }

    /// An executor as [`Executor::with_engine`] describes, whose regions
    /// `planner` plans in Rust when the engine has a `settings`
    /// attribute; without a planner every region goes to `run_region`.
    ///
    /// # Errors
    ///
    /// An attribute is missing or malformed.
    pub fn with_planner(engine: Py<PyAny>, planner: Option<Arc<Planner>>) -> Result<Self> {
        let (prompting, rules, settings) = Python::attach(|py| {
            let bound = engine.bind(py);
            let turn: (String, String) = bound.getattr("turn")?.extract()?;
            let layout: String = bound.getattr("layout")?.extract()?;
            let rules: Option<Vec<String>> = bound.getattr("rules")?.extract()?;
            let settings: Option<String> = match bound.getattr("settings") {
                Ok(found) => found.extract()?,
                Err(_) => None,
            };
            Ok::<_, PyErr>(((turn, layout), rules, settings))
        })
        .map_err(python_failure)?;
        let planning = match (planner, settings) {
            (Some(planner), Some(settings)) => Some((
                planner,
                serde_json::from_str::<Settings>(&settings)
                    .map_err(|error| DataFusionError::External(Box::new(error)))?,
            )),
            _ => None,
        };
        let rules = rules.unwrap_or_else(|| {
            quail_datafusion::LOGICAL_RULES
                .iter()
                .map(ToString::to_string)
                .collect()
        });
        Ok(Self {
            engine,
            rules,
            owns_engine: false,
            prompting,
            caller: Mutex::new(None),
            planning,
        })
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        if self.owns_engine {
            Python::attach(|py| {
                // a close that fails leaves nothing this executor can release
                let _ = self.engine.call_method0(py, "close");
            });
        }
    }
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

/// A region's hint over the rows of the tables the executor reads:
/// whether each row may still pass the tests above the region, as they
/// tighten while the region runs.
#[pyclass(name = "InputHint", module = "_quail_native", frozen)]
pub(crate) struct InputHint {
    hint: RegionHint,
    columns: RegionColumns,
    /// Each table's rows, by table name, in the order the executor reads
    /// them.
    rows: HashMap<String, RecordBatch>,
}

#[pymethods]
impl InputHint {
    /// A number that changes whenever a test tightens.
    fn generation(&self) -> u64 {
        self.hint.generation()
    }

    /// The tables the hint holds rows of.
    fn tables(&self) -> Vec<String> {
        self.rows.keys().cloned().collect()
    }

    /// Whether each row of `table`, in the order the executor reads them,
    /// may still pass every test.
    fn keep(&self, table: &str) -> PyResult<Vec<bool>> {
        let Some(rows) = self.rows.get(table) else {
            return Ok(Vec::new());
        };
        self.hint
            .keep(&self.columns, table, rows)
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

impl Executor {
    /// One region's operators and DataFusion plan.
    fn request(&self, region: &ModelRegion) -> Result<(Operators, LogicalPlan)> {
        let (turn, layout) = &self.prompting;
        derive::derive_region(region, turn, layout)
            .map_err(|error| DataFusionError::Plan(format!("CompileError: {}", error.message())))
    }

    /// Plan a region in Rust, or None when Python plans it: a region that
    /// asks no model. `derived` is the region's operators and DataFusion
    /// plan; the plan carries the DataFusion plan's text on to Python,
    /// beside what its scans read.
    fn plan_here(
        &self,
        region: &ModelRegion,
        inputs: &[InputRows],
        derived: &(Operators, LogicalPlan),
        fetch: Option<usize>,
    ) -> Result<Option<PlannedRegion>> {
        let Some((planner, settings)) = &self.planning else {
            return Ok(None);
        };
        let mut operators = derived.0.clone();
        let logical = derived.1.display_indent().to_string();
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
        // planning reads, and tokenizes, each input's documents first
        let started = std::time::Instant::now();
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
        let compile = |error: quail_planner::error::CompileError| {
            DataFusionError::Plan(format!("CompileError: {}", error.message()))
        };
        let (outcome, scans) = planner
            .plan(&operators, &tables, settings)
            .map_err(compile)?;
        let mut planned = outcome_json(&outcome).map_err(compile)?;
        planned["logical"] = logical.into();
        planned["planning_s"] = started.elapsed().as_secs_f64().into();
        Ok(Some((planned.to_string(), scans)))
    }

    /// Plan a region without running it, over the rows of its inputs that
    /// call no model; an input that calls a model plans as empty.
    ///
    /// Returns the region as [`region_json`] writes it, and the planned
    /// region or None when Python plans it.
    pub(crate) fn plan_only(
        &self,
        region: &ModelRegion,
        inputs: &[InputRows],
    ) -> Result<(String, Option<PlannedRegion>)> {
        let derived = self.request(region)?;
        let sorted = inputs
            .iter()
            .zip(&region.inputs)
            .map(|((name, batches), input)| Ok((name.clone(), in_source_order(input, batches)?)))
            .collect::<Result<Vec<InputRows>>>()?;
        let planned = self.plan_here(region, &sorted, &derived, None)?;
        Ok((region_json(&derived.0, &derived.1)?, planned))
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
    /// The region runs once, as one plan over every row of its inputs.
    /// With a hint (see [`input_hint`]), it first drops each input row a
    /// test rejects, and hands the executor the hint to drop rows by as
    /// the tests tighten.
    fn run_here(
        &self,
        region: &ModelRegion,
        inputs: Vec<InputStream>,
        fetch: Option<usize>,
        hint: &RegionHint,
        rows: &RowSender,
    ) -> Result<()> {
        let derived = self.request(region)?;
        let inputs = collect_inputs(inputs)?;
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
            held = input_hint(region, &derived.0, fetch, hint, &mut sorted)?;
            (
                self.plan_here(region, &sorted, &derived, fetch)?,
                sorted.as_slice(),
            )
        } else {
            (None, inputs.as_slice())
        };
        let schema = Arc::clone(region.plan.schema());
        let names = quail_column_names(&region.plan);
        let sink = RegionSink {
            rows: rows.clone(),
            names: names.clone(),
            schema: Arc::clone(schema.inner()),
        };
        // Python reads a region it plans as JSON
        let payload = match &planned {
            None => Some(region_json(&derived.0, &derived.1)?),
            Some(_) => None,
        };
        let reader = Python::attach(|py| {
            // each input as a pyarrow Table, by the table name the region reads
            let mut tables = Vec::new();
            for ((name, batches), input) in inputs.iter().zip(&region.inputs) {
                let schema = Arc::new(input.schema().as_arrow().clone());
                tables.push((name.clone(), to_pyarrow(py, schema, batches.clone())?));
            }
            let tables = tables.into_py_dict(py)?;
            let answer = match planned {
                Some((found, scans)) => {
                    let found = (found, scans_to_pyarrow(py, &scans)?);
                    let sink = Bound::new(py, sink)?;
                    let hint = held.map(|hint| Bound::new(py, hint)).transpose()?;
                    self.engine
                        .call_method1(py, "execute", (found, tables, sink, hint))?
                }
                None => self.engine.call_method1(
                    py,
                    "run_region",
                    (payload.as_deref(), tables, fetch),
                )?,
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

/// What each scan reads, by alias, as Python's `adopt_plan` takes it:
/// (the token store, the value columns), each a pyarrow Table over the
/// same buffers.
///
/// # Errors
///
/// pyarrow is not importable.
pub fn scans_to_pyarrow<'py>(
    py: Python<'py>,
    scans: &[ScanInputs],
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let found = pyo3::types::PyDict::new(py);
    for scan in scans {
        let tokens = to_pyarrow(
            py,
            Arc::clone(scan.tokens.schema()),
            scan.tokens.record_batches().to_vec(),
        )?;
        let (schema, batches) = &scan.columns;
        let columns = to_pyarrow(py, Arc::clone(schema), batches.clone())?;
        found.set_item(&scan.alias, (tokens, columns))?;
    }
    Ok(found)
}

impl RegionRunner for Executor {
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

/// The hint over the rows of each table a test of `hint` reads, or None
/// when the region takes none. Each row of `inputs` a test rejects is
/// dropped first.
///
/// A fetch or a stop key keeps the first rows: dropping one changes which
/// rows those are, so such a region takes no hint.
fn input_hint(
    region: &ModelRegion,
    operators: &Operators,
    fetch: Option<usize>,
    hint: &RegionHint,
    inputs: &mut [InputRows],
) -> Result<Option<InputHint>> {
    if hint.is_empty()
        || fetch.is_some()
        || operators.fetch.is_some()
        || !operators.stop_keys.is_empty()
    {
        return Ok(None);
    }
    let columns = RegionColumns::of(region);
    let mut rows = HashMap::new();
    for table in hint.tables(&columns) {
        let found = match inputs.iter().position(|(name, _)| *name == table) {
            Some(input) => {
                let schema = Arc::new(region.inputs[input].schema().as_arrow().clone());
                let all = concat_batches(&schema, &inputs[input].1)?;
                let keep = BooleanArray::from(hint.keep(&columns, &table, &all)?);
                let kept = filter_record_batch(&all, &keep)?;
                inputs[input].1 = vec![kept.clone()];
                Some(kept)
            }
            None => named_rows(region, &table)?,
        };
        if let Some(found) = found {
            rows.insert(table, found);
        }
    }
    Ok((!rows.is_empty()).then(|| InputHint {
        hint: hint.clone(),
        columns,
        rows,
    }))
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
        .map_err(|error| DataFusionError::Plan(format!("CompileError: {}", error.message())))?;
    // calls a typo apart; calls that differ only in spelling are refused
    let warnings: Vec<String> = quail_datafusion::derive::similar::similar_calls(plan)
        .iter()
        .filter(|found| found.likeness == quail_datafusion::derive::similar::Likeness::Typo)
        .map(quail_datafusion::derive::similar::Similar::message)
        .collect();
    Ok(serde_json::json!({
        "operators": operators,
        "logical": plan.display_indent().to_string(),
        "warnings": warnings,
    })
    .to_string())
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
        let column = batch.columns().first().ok_or_else(|| {
            DataFusionError::Execution(format!("the Python provider returned no column {key}"))
        })?;
        for value in texts(column)? {
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

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    #[test]
    fn executor_rows_missing_a_required_column_return_an_error() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "answer",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
        let error = conform(batch, &["answer".into()], &schema).unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        assert!(
            error
                .to_string()
                .contains("without the region's column answer")
        );
    }
}
