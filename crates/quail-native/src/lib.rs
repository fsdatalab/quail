//! The `_quail_native` Python module: Quail's Rust crates for Python.
//!
//! Tables and results cross between Python and Rust as Arrow IPC
//! stream bytes, so the module does not depend on pyarrow's version.

mod cost;

use std::io::Cursor;

use async_trait::async_trait;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ipc::reader::StreamReader;
use datafusion::arrow::ipc::writer::StreamWriter;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{
    DataFusionError, Result, tree_node::TreeNode, tree_node::TreeNodeRecursion,
};
use datafusion::datasource::TableType;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use quail_datafusion::{
    ModelRegion, RegionRunner, bind, quail_column_names, quail_schema, region_sql, session_context,
    sql,
};
use quail_planner::prompts::Turn;
use tokio::runtime::Runtime;

/// Runs each model region by calling a Python function with the
/// region's SQL; the function returns the rows as IPC stream bytes.
#[derive(Debug)]
struct PythonRunner {
    run_region: Py<PyAny>,
    /// The session model's chat-turn text and prompt layout, to bind
    /// regions in Rust; None sends each region's SQL instead.
    prompting: Option<(Turn, String)>,
}

impl PythonRunner {
    /// The request for one region: ("plan", its logical plan as JSON),
    /// or ("sql", its SQL) for a region the Rust binder does not read.
    fn request(&self, region: &ModelRegion) -> Result<(&'static str, String)> {
        let Some((turn, layout)) = &self.prompting else {
            return Ok(("sql", region_sql(region)?));
        };
        match bind::bind_region(&region.plan, turn, layout) {
            Ok(plan) => Ok((
                "plan",
                quail_planner::write_logical_plan(&plan)
                    .map_err(|error| DataFusionError::External(Box::new(error)))?,
            )),
            Err(error) if error.0.contains(bind::UNSUPPORTED) => Ok(("sql", region_sql(region)?)),
            Err(error) => Err(DataFusionError::Plan(format!("CompileError: {}", error.0))),
        }
    }
}

impl RegionRunner for PythonRunner {
    fn run(&self, region: &ModelRegion) -> Result<Vec<RecordBatch>> {
        let (kind, payload) = self.request(region)?;
        let bytes: Vec<u8> = Python::attach(|py| {
            self.run_region
                .call1(py, (kind, payload))
                .and_then(|answer| answer.extract::<Vec<u8>>(py))
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
        let schema = region.plan.schema();
        let names = quail_column_names(schema);
        read_ipc(&bytes)?
            .into_iter()
            .map(|batch| conform(batch, &names, schema.inner()))
            .collect()
    }
}

/// Give a batch of Quail's rows the region's schema: each region
/// column is the Quail column of the same name, cast to its type.
fn conform(batch: RecordBatch, names: &[String], schema: &SchemaRef) -> Result<RecordBatch> {
    let columns = names
        .iter()
        .zip(schema.fields())
        .map(|(name, field)| {
            let column = batch.column_by_name(name).ok_or_else(|| {
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
/// Model regions read their tables through Quail, so DataFusion scans
/// a table only for operators outside every region.
#[derive(Debug)]
struct PythonTable {
    schema: SchemaRef,
    scan: Py<PyAny>,
}

#[async_trait]
impl TableProvider for PythonTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => Arc::clone(&self.schema),
        };
        let columns: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        let bytes: Vec<u8> = Python::attach(|py| {
            self.scan
                .call1(py, (columns,))
                .and_then(|answer| answer.extract::<Vec<u8>>(py))
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
        let batches = read_ipc(&bytes)?;
        Ok(MemorySourceConfig::try_new_exec(&[batches], schema, None)?)
    }
}

fn read_ipc(bytes: &[u8]) -> Result<Vec<RecordBatch>> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None)?;
    Ok(reader.collect::<std::result::Result<Vec<_>, _>>()?)
}

fn write_ipc(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut writer = StreamWriter::try_new(&mut bytes, schema)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    drop(writer);
    Ok(bytes)
}

fn python_error(error: DataFusionError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A DataFusion session context with Quail registered.
#[pyclass(name = "NativeContext", module = "_quail_native")]
struct NativeContext {
    context: SessionContext,
    runtime: Runtime,
}

#[pymethods]
impl NativeContext {
    /// Create a context whose model regions run through `run_region`.
    ///
    /// `run_region(kind, payload)` takes ("plan", a region's Quail
    /// logical plan as JSON), bound in Rust with the model's chat-turn
    /// text `turn` and prompt `layout`, or ("sql", the region's SQL)
    /// for a region the Rust binder does not read, or for every region
    /// when no turn and layout are given.
    #[new]
    #[pyo3(signature = (run_region, turn=None, layout=None))]
    fn new(
        run_region: Py<PyAny>,
        turn: Option<(String, String)>,
        layout: Option<String>,
    ) -> PyResult<Self> {
        let runtime = Runtime::new()?;
        let prompting = turn.zip(layout);
        let context = session_context(Arc::new(PythonRunner {
            run_region,
            prompting,
        }));
        Ok(Self { context, runtime })
    }

    /// Register a table by its schema, given as the IPC stream bytes of
    /// an empty table, and a function `scan(columns)` that returns
    /// those columns' rows as IPC stream bytes.
    fn register(&self, name: &str, schema: &[u8], scan: Py<PyAny>) -> PyResult<()> {
        let reader = StreamReader::try_new(Cursor::new(schema), None)
            .map_err(|error| python_error(error.into()))?;
        let table = PythonTable {
            schema: reader.schema(),
            scan,
        };
        self.context
            .register_table(name, Arc::new(table))
            .map_err(python_error)?;
        Ok(())
    }

    /// Run a query and return its rows as IPC stream bytes.
    fn sql<'py>(&self, py: Python<'py>, query: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.runtime.block_on(async {
                    let frame = sql(&self.context, query).await?;
                    let schema = quail_schema(frame.schema());
                    let batches = frame
                        .collect()
                        .await?
                        .into_iter()
                        .map(|batch| {
                            RecordBatch::try_new(Arc::clone(&schema), batch.columns().to_vec())
                        })
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    write_ipc(&schema, &batches)
                })
            })
            .map_err(python_error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Return the SQL each model region of a query sends to Quail.
    fn region_sql(&self, py: Python<'_>, query: &str) -> PyResult<Vec<String>> {
        py.detach(|| {
            self.runtime.block_on(async {
                let plan = sql(&self.context, query).await?.into_optimized_plan()?;
                let mut texts = Vec::new();
                plan.apply(|node| {
                    if let LogicalPlan::Extension(extension) = node
                        && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
                    {
                        texts.push(region_sql(region)?);
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
                Ok(texts)
            })
        })
        .map_err(python_error)
    }

    /// Bind each model region of a query into Quail's logical plan.
    ///
    /// Returns one (plan JSON, None) or (None, CompileError message)
    /// per region, and whether one region is the whole query.
    /// `turn` and `layout` are the model's chat-turn text and prompt
    /// layout.
    #[allow(clippy::type_complexity)]
    fn logical_plans(
        &self,
        py: Python<'_>,
        query: &str,
        turn: (String, String),
        layout: &str,
    ) -> PyResult<(Vec<(Option<String>, Option<String>)>, bool)> {
        py.detach(|| {
            self.runtime.block_on(async {
                let plan = sql(&self.context, query).await?.into_optimized_plan()?;
                let mut bound = Vec::new();
                plan.apply(|node| {
                    if let LogicalPlan::Extension(extension) = node
                        && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
                    {
                        bound.push(
                            match quail_datafusion::bind::bind_region(&region.plan, &turn, layout) {
                                Ok(found) => (
                                    Some(quail_planner::write_logical_plan(&found).map_err(
                                        |error| {
                                            datafusion::error::DataFusionError::External(Box::new(
                                                error,
                                            ))
                                        },
                                    )?),
                                    None,
                                ),
                                Err(error) => (None, Some(error.0)),
                            },
                        );
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
                let whole = matches!(&plan, LogicalPlan::Extension(_));
                Ok((bound, whole))
            })
        })
        .map_err(python_error)
    }

    /// Return DataFusion's optimized logical plan for a query.
    fn explain(&self, py: Python<'_>, query: &str) -> PyResult<String> {
        py.detach(|| {
            self.runtime.block_on(async {
                let plan = sql(&self.context, query).await?.into_optimized_plan()?;
                Ok(plan.display_indent().to_string())
            })
        })
        .map_err(python_error)
    }
}

/// Read a physical plan envelope into the Rust plan types and write
/// it back as JSON.
///
/// Raises a ValueError when the plan does not fit the Rust types.
#[pyfunction]
fn read_and_write_plan(text: &str) -> PyResult<String> {
    let plan = quail_physical::read_plan(text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    quail_physical::write_plan(&plan)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Read a logical plan written by the Python codec into the Rust types
/// and write it back.
///
/// Raises a ValueError when the plan does not fit the Rust types.
#[pyfunction]
fn read_and_write_logical_plan(text: &str) -> PyResult<String> {
    let plan = quail_planner::read_logical_plan(text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    quail_planner::write_logical_plan(&plan)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Check a logical plan as `LogicalPlan.validate` does: None, or the
/// message of the CompileError it raises.
#[pyfunction]
fn validate_logical_plan(text: &str) -> PyResult<Option<String>> {
    let plan = quail_planner::read_logical_plan(text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(quail_planner::nodes::validate_plan(&plan)
        .err()
        .map(|error| error.0))
}

/// Place a bound SELECT and build its logical plan, as the front end's
/// `place_relation` and `LogicalPlanBuilder.project` do.
///
/// Returns (the plan as JSON, None), or (None, the CompileError's
/// message). Without project arguments it places the relation only and
/// returns (None, None) when that succeeds.
#[pyfunction]
#[pyo3(signature = (place, project=None))]
fn build_select(place: &str, project: Option<&str>) -> PyResult<(Option<String>, Option<String>)> {
    let invalid =
        |error: serde_json::Error| pyo3::exceptions::PyValueError::new_err(error.to_string());
    let place: quail_planner::builder::PlaceRelationArguments =
        serde_json::from_str(place).map_err(invalid)?;
    let placed = match quail_planner::builder::place_relation(place) {
        Ok(placed) => placed,
        Err(error) => return Ok((None, Some(error.0))),
    };
    let Some(project) = project else {
        return Ok((None, None));
    };
    let project: quail_planner::builder::ProjectArguments =
        serde_json::from_str(project).map_err(invalid)?;
    match placed.builder.project(&project, &placed.rows) {
        Ok(plan) => Ok((
            Some(quail_planner::write_logical_plan(&plan).map_err(invalid)?),
            None,
        )),
        Err(error) => Ok((None, Some(error.0))),
    }
}

/// Bind one AI function call's prompt, as `PromptBinder.bind` does.
///
/// Returns (the Prompt as JSON, None), or (None, the CompileError's message).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bind_prompt(
    kind: &str,
    template: &str,
    args: &str,
    labels: Vec<String>,
    descriptions: Vec<String>,
    task_description: &str,
    turn: (String, String),
    layout: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    let invalid =
        |error: serde_json::Error| pyo3::exceptions::PyValueError::new_err(error.to_string());
    let args: Vec<quail_physical::ValueExpression> = serde_json::from_str(args).map_err(invalid)?;
    match quail_planner::prompts::bind(
        kind,
        template,
        &args,
        &labels,
        &descriptions,
        task_description,
        &turn,
        layout,
    ) {
        Ok(prompt) => Ok((Some(serde_json::to_string(&prompt).map_err(invalid)?), None)),
        Err(error) => Ok((None, Some(error.0))),
    }
}

/// Apply the built-in logical rules, as `logical_alternatives` does.
///
/// Returns ([(plan JSON, rule names), ...], None), or ([], the
/// CompileError's message). `id_columns` maps each provider to its id
/// column, or is None without a catalog.
#[pyfunction]
#[pyo3(signature = (plan, rules, id_columns=None))]
#[allow(clippy::type_complexity)]
fn logical_alternatives(
    plan: &str,
    rules: Vec<String>,
    id_columns: Option<std::collections::BTreeMap<String, String>>,
) -> PyResult<(Vec<(String, Vec<String>)>, Option<String>)> {
    let invalid =
        |error: serde_json::Error| pyo3::exceptions::PyValueError::new_err(error.to_string());
    let plan = quail_planner::read_logical_plan(plan).map_err(invalid)?;
    let context = quail_planner::rules::RuleContext { id_columns };
    match quail_planner::rules::logical_alternatives(plan, &rules, &context) {
        Ok(found) => Ok((
            found
                .into_iter()
                .map(|alternative| {
                    Ok((
                        quail_planner::write_logical_plan(&alternative.plan).map_err(invalid)?,
                        alternative.rules,
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?,
            None,
        )),
        Err(error) => Ok((Vec::new(), Some(error.0))),
    }
}

#[pymodule]
fn _quail_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeContext>()?;
    module.add_class::<cost::CostModel>()?;
    module.add_function(wrap_pyfunction!(read_and_write_logical_plan, module)?)?;
    module.add_function(wrap_pyfunction!(validate_logical_plan, module)?)?;
    module.add_function(wrap_pyfunction!(build_select, module)?)?;
    module.add_function(wrap_pyfunction!(bind_prompt, module)?)?;
    module.add_function(wrap_pyfunction!(logical_alternatives, module)?)?;
    module.add_function(wrap_pyfunction!(read_and_write_plan, module)?)
}
