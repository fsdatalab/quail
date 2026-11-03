//! The `_quail_datafusion` Python module.
//!
//! Tables and results cross between Python and Rust as Arrow IPC
//! stream bytes, so the module does not depend on pyarrow's version.

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
    ModelRegion, RegionRunner, quail_column_names, quail_schema, region_sql, session_context, sql,
};
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
