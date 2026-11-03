//! The `_quail_datafusion` Python module.
//!
//! Tables and results cross between Python and Rust as Arrow IPC
//! stream bytes, so the module does not depend on pyarrow's version.

use std::io::Cursor;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ipc::reader::StreamReader;
use datafusion::arrow::ipc::writer::StreamWriter;
use datafusion::common::{
    DataFusionError, Result, tree_node::TreeNode, tree_node::TreeNodeRecursion,
};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use quail_datafusion::{ModelRegion, RegionRunner, region_sql, session_context, sql};
use tokio::runtime::Runtime;

/// Runs each model region by calling a Python function with the
/// region's SQL; the function returns the rows as IPC stream bytes.
#[derive(Debug)]
struct PythonRunner {
    run_region: Py<PyAny>,
}

impl RegionRunner for PythonRunner {
    fn run(&self, region: &ModelRegion) -> Result<Vec<RecordBatch>> {
        let text = region_sql(region)?;
        let bytes: Vec<u8> = Python::attach(|py| {
            self.run_region
                .call1(py, (text,))
                .and_then(|answer| answer.extract::<Vec<u8>>(py))
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
        let schema = Arc::clone(region.plan.schema().inner());
        read_ipc(&bytes)?
            .into_iter()
            .map(|batch| conform(batch, &schema))
            .collect()
    }
}

/// Give a batch the region's schema: its columns by position, cast to
/// the region's types.
fn conform(batch: RecordBatch, schema: &SchemaRef) -> Result<RecordBatch> {
    if batch.num_columns() != schema.fields().len() {
        let got: Vec<_> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let want: Vec<_> = schema.fields().iter().map(|f| f.name().clone()).collect();
        return Err(DataFusionError::Execution(format!(
            "Quail returned columns {got:?} for a region with columns {want:?}"
        )));
    }
    let columns = batch
        .columns()
        .iter()
        .zip(schema.fields())
        .map(|(column, field)| Ok(cast(column, field.data_type())?))
        .collect::<Result<Vec<ArrayRef>>>()?;
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
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
#[pyclass(name = "NativeContext", module = "_quail_datafusion")]
struct NativeContext {
    context: SessionContext,
    runtime: Runtime,
}

#[pymethods]
impl NativeContext {
    /// Create a context whose model regions run through `run_region`.
    #[new]
    fn new(run_region: Py<PyAny>) -> PyResult<Self> {
        let runtime = Runtime::new()?;
        let context = session_context(Arc::new(PythonRunner { run_region }));
        Ok(Self { context, runtime })
    }

    /// Register a table given as IPC stream bytes.
    fn register(&self, name: &str, table: &[u8]) -> PyResult<()> {
        let batches = read_ipc(table).map_err(python_error)?;
        let reader = StreamReader::try_new(Cursor::new(table), None)
            .map_err(|error| python_error(error.into()))?;
        let memory = MemTable::try_new(reader.schema(), vec![batches]).map_err(python_error)?;
        self.context
            .register_table(name, Arc::new(memory))
            .map_err(python_error)?;
        Ok(())
    }

    /// Run a query and return its rows as IPC stream bytes.
    fn sql<'py>(&self, py: Python<'py>, query: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| {
                self.runtime.block_on(async {
                    let frame = sql(&self.context, query).await?;
                    let schema = Arc::new(frame.schema().as_arrow().clone());
                    let batches = frame.collect().await?;
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

#[pymodule]
fn _quail_datafusion(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeContext>()
}
