use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::TableProvider;
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::file_format::arrow::ArrowFormat;
use datafusion::datasource::file_format::csv::CsvFormat;
use datafusion::datasource::file_format::json::JsonFormat;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::{MemTable, TableType};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::LexOrdering;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::{PartitionStream, StreamingTableExec};
use datafusion::prelude::SessionContext;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use quail::planning::{Planner, ScanInputs, Settings};
use quail::{EngineRegion, Executor, RegionEngine};

use crate::pyarrow::{from_pyarrow, stream_from_pyarrow, to_pyarrow};
use crate::python_failure;

#[derive(Debug)]
struct PythonEngine(Py<PyAny>);

impl RegionEngine for PythonEngine {
    fn run(&self, region: EngineRegion<'_>) -> Result<Vec<RecordBatch>> {
        let EngineRegion {
            region: model_region,
            inputs,
            planned,
            payload,
            fetch,
            hint,
            sink,
        } = region;
        Python::attach(|py| {
            let mut tables = Vec::new();
            for ((name, batches), input) in inputs.iter().zip(&model_region.inputs) {
                let schema = Arc::new(input.schema().as_arrow().clone());
                tables.push((name.clone(), to_pyarrow(py, schema, batches.clone())?));
            }
            let tables = tables.into_py_dict(py)?;
            let answer = match planned {
                Some((found, scans)) => {
                    let found = (found, scans_to_pyarrow(py, &scans)?);
                    let sink = match model_region.has_sorted_input() {
                        true => py.None(),
                        false => Bound::new(py, RegionSink(sink))?.into_any().unbind(),
                    };
                    let hint = hint
                        .map(|hint| Bound::new(py, InputHint(hint)))
                        .transpose()?;
                    (self.0).call_method1(py, "execute", (found, tables, sink, hint))?
                }
                None => {
                    (self.0).call_method1(py, "run_region", (payload.as_deref(), tables, fetch))?
                }
            };
            let answer = answer.bind(py);
            if answer.is_none() {
                return Ok(Vec::new());
            }
            from_pyarrow(answer)
        })
        .map_err(python_failure)
    }
}

fn runtime_error(error: DataFusionError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pyclass(name = "InputHint", module = "_quail_native", frozen)]
struct InputHint(quail::InputHint);

#[pymethods]
impl InputHint {
    fn generation(&self) -> u64 {
        self.0.generation()
    }

    fn tables(&self) -> Vec<String> {
        self.0.tables()
    }

    fn keep(&self, table: &str) -> PyResult<Vec<bool>> {
        self.0.keep(table).map_err(runtime_error)
    }

    fn reads_answer(&self, answer: &str) -> bool {
        self.0.reads_answer(answer)
    }

    #[pyo3(signature = (answer, prefix, complete=false, may_be_null=false, tails=None, max_chars=None))]
    fn rejects_answer(
        &self,
        answer: &str,
        prefix: &str,
        complete: bool,
        may_be_null: bool,
        tails: Option<Vec<String>>,
        max_chars: Option<usize>,
    ) -> PyResult<bool> {
        let answers = quail_datafusion::derive::partial::Completions {
            prefix: prefix.to_string(),
            complete,
            may_be_null,
            tails,
            max_chars,
        };
        (self.0)
            .rejects_answer(answer, &answers)
            .map_err(runtime_error)
    }
}

#[pyclass(name = "RegionSink", module = "_quail_native", frozen)]
struct RegionSink(quail::RegionSink);

#[pymethods]
impl RegionSink {
    fn __call__(&self, table: &Bound<'_, PyAny>) -> PyResult<()> {
        for batch in stream_from_pyarrow(table)? {
            let batch = batch.map_err(|error| runtime_error(error.into()))?;
            self.0.send(batch).map_err(runtime_error)?;
        }
        Ok(())
    }
}

fn optional_string_attribute(engine: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    match engine.getattr(name) {
        Ok(found) => found.extract(),
        Err(error) if error.is_instance_of::<PyAttributeError>(engine.py()) => Ok(None),
        Err(error) => Err(error),
    }
}

pub fn executor_with_engine(engine: Py<PyAny>) -> Result<Executor> {
    let tokenizer_file =
        Python::attach(|py| optional_string_attribute(engine.bind(py), "tokenizer_file"))
            .map_err(python_failure)?;
    let planner = tokenizer_file
        .map(|path| Planner::new(std::path::Path::new(&path)))
        .transpose()
        .map_err(quail_datafusion::external)?
        .map(Arc::new);
    executor_with_planner(engine, planner)
}

pub fn executor_with_planner(engine: Py<PyAny>, planner: Option<Arc<Planner>>) -> Result<Executor> {
    let (prompting, rules, settings) = Python::attach(|py| {
        let bound = engine.bind(py);
        let turn: (String, String) = bound.getattr("turn")?.extract()?;
        let layout: String = bound.getattr("layout")?.extract()?;
        let rules: Option<Vec<String>> = bound.getattr("rules")?.extract()?;
        let settings = optional_string_attribute(bound, "settings")?;
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
    Ok(Executor::with_engine(
        Arc::new(PythonEngine(engine)),
        prompting,
        rules,
        planning,
    ))
}

pub fn scans_to_pyarrow<'py>(
    py: Python<'py>,
    scans: &[ScanInputs],
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let found = pyo3::types::PyDict::new(py);
    for scan in scans {
        let (schema, batches) = scan.tokens.to_table();
        let tokens = to_pyarrow(py, schema, batches)?;
        let (schema, batches) = scan.columns.clone();
        let columns = to_pyarrow(py, schema, batches)?;
        found.set_item(&scan.alias, (tokens, columns))?;
    }
    Ok(found)
}

pub fn register_python_table(
    context: &SessionContext,
    name: &str,
    schema: SchemaRef,
    scan: Py<PyAny>,
    id_column: Option<&str>,
    identity: &str,
) -> Result<()> {
    let table = PythonTable {
        schema,
        scan: Arc::new(scan),
    };
    quail::register_scanned_table(context, name, Arc::new(table), id_column, identity)
}

#[derive(Debug)]
struct PythonTable {
    schema: SchemaRef,
    scan: Arc<Py<PyAny>>,
}

#[async_trait::async_trait]
impl TableProvider for PythonTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn datafusion::catalog::Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => Arc::clone(&self.schema),
        };
        let partition = PythonScan {
            schema: Arc::clone(&schema),
            scan: Arc::clone(&self.scan),
        };
        Ok(Arc::new(StreamingTableExec::try_new(
            schema,
            vec![Arc::new(partition)],
            None,
            Vec::<LexOrdering>::new(),
            false,
            limit,
        )?))
    }
}

#[derive(Debug)]
struct PythonScan {
    schema: SchemaRef,
    scan: Arc<Py<PyAny>>,
}

impl PartitionStream for PythonScan {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _context: Arc<TaskContext>) -> SendableRecordBatchStream {
        let columns: Vec<String> = self
            .schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let reader = Python::attach(|py| {
            let answer = self.scan.call1(py, (columns,))?;
            stream_from_pyarrow(answer.bind(py))
        })
        .map_err(python_failure);
        let batches: Box<dyn Iterator<Item = Result<RecordBatch>> + Send> = match reader {
            Ok(reader) => Box::new(reader.map(|batch| batch.map_err(DataFusionError::from))),
            Err(error) => Box::new(std::iter::once(Err(error))),
        };
        Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            futures::stream::iter(batches),
        ))
    }
}

pub fn register_batches(
    context: &SessionContext,
    name: &str,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    id_column: Option<&str>,
    identity: &str,
) -> Result<()> {
    let table = MemTable::try_new(schema, vec![batches])?;
    quail::register_scanned_table(context, name, Arc::new(table), id_column, identity)
}

pub fn register_files(
    context: &SessionContext,
    name: &str,
    format: &str,
    paths: &[String],
    schema: SchemaRef,
    id_column: Option<&str>,
    identity: &str,
) -> Result<()> {
    let format: Arc<dyn FileFormat> = match format {
        "parquet" => Arc::new(ParquetFormat::default()),
        "arrow" => Arc::new(ArrowFormat),
        "json" => Arc::new(JsonFormat::default()),
        "csv" => Arc::new(CsvFormat::default()),
        other => return datafusion::common::plan_err!("Quail reads no file format {other:?}"),
    };
    let urls = paths
        .iter()
        .map(ListingTableUrl::parse)
        .collect::<Result<Vec<_>>>()?;
    let options = ListingOptions::new(format).with_file_extension("");
    let config = ListingTableConfig::new_with_multi_paths(urls)
        .with_listing_options(options)
        .with_schema(schema);
    let table = ListingTable::try_new(config)?;
    quail::register_scanned_table(context, name, Arc::new(table), id_column, identity)
}
