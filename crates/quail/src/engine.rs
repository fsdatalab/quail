//! The model region runner that calls Quail's Python executor, and the
//! tables whose rows a Python provider returns.

use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{Constraints, DataFusionError, Result};
use datafusion::datasource::TableType;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use quail_datafusion::{
    InputRows, ModelRegion, RegionBatches, RegionRunner, bind, quail_column_names, region_sql,
};
use quail_planner::prompts::Turn;
use tokio::sync::mpsc::UnboundedSender;

use crate::pyarrow::{from_pyarrow, stream_from_pyarrow, to_pyarrow};

/// Runs each model region by calling the engine's
/// `run_region(kind, payload, tables, fetch)`, which returns the rows
/// as a pyarrow Table or RecordBatchReader.
#[derive(Debug)]
pub(crate) struct PythonRunner {
    pub(crate) engine: Py<PyAny>,
    /// The model's chat-turn text and prompt layout, to bind regions in
    /// Rust; None sends each region's SQL instead.
    pub(crate) prompting: Option<(Turn, String)>,
    /// While a query runs: the thread that called DataFusion, and where
    /// a worker sends it a region to run.
    pub(crate) caller: Mutex<Option<(ThreadId, UnboundedSender<RegionCall>)>>,
}

/// A region a DataFusion worker hands to the calling thread to run.
#[derive(Debug)]
pub(crate) struct RegionCall {
    pub(crate) region: ModelRegion,
    pub(crate) inputs: Vec<InputRows>,
    pub(crate) fetch: Option<usize>,
    pub(crate) rows: std::sync::mpsc::Sender<Result<RegionBatches>>,
}

impl PythonRunner {
    /// The request for one region: ("plan", its logical plan as JSON),
    /// or ("sql", its SQL) for a region the Rust binder does not read.
    fn request(&self, region: &ModelRegion) -> Result<(&'static str, String)> {
        let Some((turn, layout)) = &self.prompting else {
            return Ok(("sql", region_sql(region)?));
        };
        match bind::bind_model_region(region, turn, layout) {
            Ok(plan) => Ok((
                "plan",
                quail_planner::write_logical_plan(&plan)
                    .map_err(|error| DataFusionError::External(Box::new(error)))?,
            )),
            Err(error) if error.0.contains(bind::UNSUPPORTED) => Ok(("sql", region_sql(region)?)),
            Err(error) => Err(DataFusionError::Plan(format!("CompileError: {}", error.0))),
        }
    }

    /// Run the region on this thread.
    pub(crate) fn run_here(
        &self,
        region: &ModelRegion,
        inputs: &[InputRows],
        fetch: Option<usize>,
    ) -> Result<RegionBatches> {
        let (kind, payload) = self.request(region)?;
        let reader = Python::attach(|py| {
            // each input as a pyarrow Table, by the table name the region reads
            let mut tables = Vec::new();
            for ((name, batches), input) in inputs.iter().zip(&region.inputs) {
                let schema = Arc::new(input.schema().as_arrow().clone());
                tables.push((name.clone(), to_pyarrow(py, schema, batches.clone())?));
            }
            let tables = tables.into_py_dict(py)?;
            let answer =
                self.engine
                    .call_method1(py, "run_region", (kind, payload, tables, fetch))?;
            stream_from_pyarrow(answer.bind(py))
        })
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
        let schema = Arc::clone(region.bound.schema());
        let names = quail_column_names(&schema);
        // DataFusion reads each batch as the Python stream yields it
        Ok(Box::new(
            reader.map(move |batch| conform(batch?, &names, schema.inner())),
        ))
    }
}

impl RegionRunner for PythonRunner {
    /// Run the region on the thread that called DataFusion, so the GPU
    /// process's per-thread state is set up once, not on each worker.
    fn run(
        &self,
        region: &ModelRegion,
        inputs: &[InputRows],
        fetch: Option<usize>,
    ) -> Result<RegionBatches> {
        let caller = self.caller.lock().expect("the caller lock").clone();
        match caller {
            Some((thread, sender)) if thread != std::thread::current().id() => {
                let (rows, answer) = std::sync::mpsc::channel();
                let call = RegionCall {
                    region: region.clone(),
                    inputs: inputs.to_vec(),
                    fetch,
                    rows,
                };
                sender
                    .send(call)
                    .map_err(|_| DataFusionError::Execution("the query ended".into()))?;
                answer
                    .recv()
                    .map_err(|_| DataFusionError::Execution("the region did not run".into()))?
            }
            _ => self.run_here(region, inputs, fetch),
        }
    }
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
