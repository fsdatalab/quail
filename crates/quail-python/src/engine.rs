//! A Python engine as Quail's [`RegionEngine`], and the Python objects
//! it is handed: a region's hint and its sink.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result};
use pyo3::exceptions::{PyAttributeError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use quail::planning::{Planner, ScanInputs, Settings};
use quail::{EngineRegion, Executor, RegionEngine, Session};

use crate::pyarrow::{from_pyarrow, stream_from_pyarrow, to_pyarrow};
use crate::python_failure;

/// Runs each model region by calling a Python engine: its
/// `execute(planned, tables, sink, hint)` runs a region Rust planned,
/// and `run_region(payload, tables, fetch)` one it did not. `planned`
/// is the planned region's JSON and its scans, as [`scans_to_pyarrow`]
/// hands them over. `execute` hands the sink each batch of finished
/// rows as the region runs and returns None, or returns the rows;
/// `run_region` returns them, as a pyarrow Table or RecordBatchReader.
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
            // each input as a pyarrow Table, by the table name the region reads
            let mut tables = Vec::new();
            for ((name, batches), input) in inputs.iter().zip(&model_region.inputs) {
                let schema = Arc::new(input.schema().as_arrow().clone());
                tables.push((name.clone(), to_pyarrow(py, schema, batches.clone())?));
            }
            let tables = tables.into_py_dict(py)?;
            let answer = match planned {
                Some((found, scans)) => {
                    let found = (found, scans_to_pyarrow(py, &scans)?);
                    let sink = Bound::new(py, RegionSink(sink))?;
                    let hint = hint
                        .map(|hint| Bound::new(py, InputHint(hint)))
                        .transpose()?;
                    (self.0).call_method1(py, "execute", (found, tables, sink, hint))?
                }
                None => {
                    (self.0).call_method1(py, "run_region", (payload.as_deref(), tables, fetch))?
                }
            };
            // an engine that sent every row to the sink returns None
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

/// A region's hint over the rows of the tables the executor reads, as
/// [`quail::InputHint`]: whether each row may still pass the tests
/// above the region, as they tighten while the region runs.
#[pyclass(name = "InputHint", module = "_quail_native", frozen)]
struct InputHint(quail::InputHint);

#[pymethods]
impl InputHint {
    /// A number that changes whenever a test tightens.
    fn generation(&self) -> u64 {
        self.0.generation()
    }

    /// The tables the hint holds rows of.
    fn tables(&self) -> Vec<String> {
        self.0.tables()
    }

    /// Whether each row of `table`, in the order the executor reads them,
    /// may still pass every test.
    fn keep(&self, table: &str) -> PyResult<Vec<bool>> {
        self.0.keep(table).map_err(runtime_error)
    }

    /// Whether a test reads the answer column `answer`, an operator's
    /// name such as `vendor` or `contract.law`, and no other column.
    fn reads_answer(&self, answer: &str) -> bool {
        self.0.reads_answer(answer)
    }

    /// Whether some test, as the tests are now, is TRUE for no complete
    /// answer that starts with `prefix`; the arguments are those of
    /// `Expression.decide`.
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

/// The callable a region's engine hands each batch of finished rows,
/// as a pyarrow Table or RecordBatchReader, while the region runs.
#[pyclass(name = "RegionSink", module = "_quail_native", frozen)]
struct RegionSink(quail::RegionSink);

#[pymethods]
impl RegionSink {
    /// Send a table's rows on to DataFusion, in the region's schema.
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

/// An executor that runs model regions on `engine`, a Python object
/// with `run_region(payload, tables, fetch)` and the attributes `turn`,
/// `layout`, and `rules`, planning each region in Rust when the engine
/// has a `tokenizer_file` attribute.
///
/// `run_region` takes a region's operators and DataFusion plan as
/// [`quail::region_json`] writes them, the rows of each input the
/// region reads by table name, and the region's row limit, and returns
/// a pyarrow Table or RecordBatchReader. An engine with a `settings`
/// attribute also has `execute(planned, tables, sink, hint)`. `turn`
/// and `layout` are the model's chat-turn text and prompt layout.
/// `rules` names the optimizer rules each model region runs, in order,
/// or is None for the built-in rules.
///
/// # Errors
///
/// An attribute is missing or malformed.
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

/// An executor as [`executor_with_engine`] describes, whose regions
/// `planner` plans in Rust when the engine has a `settings` attribute;
/// without a planner every region goes to `run_region`.
///
/// # Errors
///
/// An attribute is missing or malformed.
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

/// Register a table whose rows the Python function `scan(columns)`
/// returns as a pyarrow Table, the same rows whenever it is scanned,
/// which `identity` names. `id_column` names the column whose values
/// are unique.
///
/// # Errors
///
/// DataFusion refuses the name.
pub fn register_python_table(
    session: &Session,
    name: &str,
    schema: SchemaRef,
    scan: Py<PyAny>,
    id_column: Option<&str>,
    identity: &str,
) -> Result<()> {
    let rows = move |columns: Vec<String>| {
        Python::attach(|py| {
            let answer = scan.call1(py, (columns,))?;
            from_pyarrow(answer.bind(py))
        })
        .map_err(python_failure)
    };
    session.register_scanned_table(name, schema, Arc::new(rows), id_column, identity)
}
