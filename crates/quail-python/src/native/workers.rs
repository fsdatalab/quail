//! The GPU workers ([`crate::workers`]) as Python sees them: `Workers`
//! in the process that runs a region, `WorkerEnd` in each worker.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use datafusion::common::DataFusionError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};
use quail_planner::physical::PhysicalGraph;
use quail_planner::stages::Stage;

use super::{from_json, python_error};
use crate::pyarrow::{stream_from_pyarrow, to_pyarrow};
use crate::shared::Table;
use crate::workers::{Answer, Message, Region};

fn table_from_python(value: &Bound<'_, PyAny>) -> PyResult<Table> {
    let reader = stream_from_pyarrow(value)?;
    let schema = datafusion::arrow::array::RecordBatchReader::schema(&reader);
    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| python_error(error.into()))?;
    Ok((schema, batches))
}

fn tables_from_python(values: &[Bound<'_, PyAny>]) -> PyResult<Vec<Table>> {
    values.iter().map(table_from_python).collect()
}

fn tables_to_python<'py>(py: Python<'py>, tables: Vec<Table>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    tables
        .into_iter()
        .map(|(schema, batches)| to_pyarrow(py, schema, batches))
        .collect()
}

/// A session's GPU workers: one process per GPU, as
/// `crate::workers::Workers`.
#[pyclass(name = "Workers", module = "_quail_native")]
pub(super) struct Workers {
    workers: Option<Arc<Mutex<crate::workers::Workers>>>,
}

impl Workers {
    /// The workers, which a session may share with this object.
    pub(super) fn shared(&self) -> PyResult<Arc<Mutex<crate::workers::Workers>>> {
        self.workers
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("the GPU workers are closed"))
    }
}

#[pymethods]
impl Workers {
    /// Listen for `gpus` workers, one socket each.
    #[new]
    fn new(gpus: usize) -> PyResult<Self> {
        let workers = crate::workers::Workers::bind(gpus).map_err(python_error)?;
        Ok(Workers {
            workers: Some(Arc::new(Mutex::new(workers))),
        })
    }

    /// Each worker's socket, by GPU.
    #[getter]
    fn paths(&mut self) -> PyResult<Vec<PathBuf>> {
        let shared = self.shared()?;
        let workers = shared.lock().expect("the workers lock");
        Ok(workers.paths().to_vec())
    }

    /// Start one process per GPU: `command`, then the worker's socket
    /// and its GPU's index.
    fn spawn(&mut self, command: Vec<String>) -> PyResult<()> {
        let shared = self.shared()?;
        let mut workers = shared.lock().expect("the workers lock");
        workers.spawn(&command).map_err(python_error)
    }

    /// Wait for every worker to connect.
    fn connect(&mut self, py: Python<'_>) -> PyResult<()> {
        let shared = self.shared()?;
        py.detach(|| shared.lock().expect("the workers lock").connect())
            .map_err(python_error)
    }

    /// Whether the workers are open and every process is running.
    fn alive(&mut self) -> bool {
        self.workers
            .as_ref()
            .is_some_and(|shared| shared.lock().expect("the workers lock").alive())
    }

    /// What Quail plans the regions of `config` with; both are JSON.
    fn describe(&mut self, py: Python<'_>, config: &str) -> PyResult<String> {
        let config: serde_json::Value = from_json(config)?;
        let shared = self.shared()?;
        let described = py
            .detach(|| shared.lock().expect("the workers lock").describe(&config))
            .map_err(python_error)?;
        Ok(described.to_string())
    }

    /// Prepare every worker's GPU for a plan; `setup` and each
    /// worker's boot record are JSON.
    fn boot(&mut self, py: Python<'_>, setup: &str) -> PyResult<Vec<String>> {
        let setup: serde_json::Value = from_json(setup)?;
        let shared = self.shared()?;
        let booted = py
            .detach(|| shared.lock().expect("the workers lock").boot(&setup))
            .map_err(python_error)?;
        Ok(booted.iter().map(ToString::to_string).collect())
    }

    /// Run a region's stages on the workers.
    ///
    /// `region` describes the region to each worker, as JSON, and
    /// `tables` holds its documents and columns as pyarrow Tables, by
    /// name. `graph` is the region's compute graph as JSON; `stages`
    /// is the plan's `stages` setting as JSON. `report(table)` takes each finished
    /// table as a pyarrow Table over the pages its worker shared;
    /// `ask(question)` takes a worker's question as JSON and returns
    /// its answer, as JSON or a pyarrow Table.
    ///
    /// Returns, per worker, what it ran as JSON and its nodes' output
    /// tables. No worker hears of a region whose graph or stages do
    /// not decode.
    #[pyo3(signature = (region, tables, graph, stages, report, ask))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn run_region<'py>(
        &mut self,
        py: Python<'py>,
        region: &str,
        tables: Vec<(String, Bound<'py, PyAny>)>,
        graph: &str,
        stages: &str,
        report: Py<PyAny>,
        ask: Py<PyAny>,
    ) -> PyResult<Vec<(String, Vec<Bound<'py, PyAny>>)>> {
        let graph: PhysicalGraph = from_json(graph)?;
        let cut: Vec<Stage> = from_json(stages)?;
        quail_planner::physical::validate_graph(&graph.nodes, &graph.root)
            .map_err(PyValueError::new_err)?;
        let region: serde_json::Value = from_json(region)?;
        let tables = tables
            .iter()
            .map(|(name, table)| Ok((name.clone(), table_from_python(table)?)))
            .collect::<PyResult<Vec<_>>>()?;
        // a table the plan deduplicates or stops by a key spreads its
        // documents over the GPUs by group
        let deduplicated: Vec<String> = match region["query"].get("deduplicate") {
            Some(serde_json::Value::Object(found)) => found.keys().cloned().collect(),
            _ => Vec::new(),
        };
        let groupings = crate::request::stage_groupings(&graph, &deduplicated, &tables)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let external = |error: PyErr| DataFusionError::External(Box::new(error));
        let calls = Region {
            report: Box::new(move |(schema, batches)| {
                Python::attach(|py| {
                    report.call1(py, (to_pyarrow(py, schema, batches)?,))?;
                    Ok::<_, PyErr>(())
                })
                .map_err(external)
            }),
            ask: Box::new(move |question| {
                Python::attach(|py| {
                    let answer = ask.call1(py, (question.to_string(),))?;
                    let answer = answer.bind(py);
                    if let Ok(text) = answer.extract::<String>() {
                        return Ok(Answer::Json(from_json(&text)?));
                    }
                    Ok::<_, PyErr>(Answer::Table(table_from_python(answer)?))
                })
                .map_err(external)
            }),
        };
        let shared = self.shared()?;
        let ran = py
            .detach(|| {
                let mut workers = shared.lock().expect("the workers lock");
                workers.run_region(&region, &tables, |connections| {
                    crate::stages::run_stages(&graph, cut, groupings, connections, calls)
                })
            })
            .map_err(python_error)?;
        ran.into_iter()
            .map(|(ran, tables)| Ok((ran.to_string(), tables_to_python(py, tables)?)))
            .collect()
    }

    /// Stop the workers: tell each to exit, then kill what is left.
    fn close(&mut self, py: Python<'_>) {
        let workers = self.workers.take();
        py.detach(|| drop(workers));
    }
}

/// A worker's end of its connection, as `crate::workers::WorkerEnd`:
/// it connects to the socket at `path` once made.
#[pyclass(name = "WorkerEnd", module = "_quail_native")]
struct WorkerEnd {
    end: crate::workers::WorkerEnd,
}

#[pymethods]
impl WorkerEnd {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let end = py
            .detach(|| crate::workers::WorkerEnd::connect(&path))
            .map_err(python_error)?;
        Ok(WorkerEnd { end })
    }

    /// The next message, or None once Quail is done with the worker:
    /// ("describe", configuration JSON), ("boot", setup JSON),
    /// ("region", region JSON, tables by name), ("stage", index, input
    /// tables), or ("end",).
    fn next<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let end = &mut self.end;
        let message = py.detach(|| end.read()).map_err(python_error)?;
        let text = |value: serde_json::Value| PyString::new(py, &value.to_string()).into_any();
        let kind = |name: &str| PyString::new(py, name).into_any();
        let parts = match message {
            crate::workers::Message::Shutdown => return Ok(None),
            Message::Describe(config) => vec![kind("describe"), text(config)],
            Message::Boot(setup) => vec![kind("boot"), text(setup)],
            Message::Region(region, tables) => {
                let named = PyDict::new(py);
                for (name, (schema, batches)) in tables {
                    named.set_item(name, to_pyarrow(py, schema, batches)?)?;
                }
                vec![kind("region"), text(region), named.into_any()]
            }
            Message::Stage(stage, inputs) => vec![
                kind("stage"),
                stage.into_pyobject(py)?.into_any(),
                tables_to_python(py, inputs)?.into_pyobject(py)?.into_any(),
            ],
            Message::End => vec![kind("end")],
        };
        Ok(Some(PyTuple::new(py, parts)?))
    }

    /// Answer a describe with what Quail plans with, as JSON.
    fn described(&mut self, py: Python<'_>, description: &str) -> PyResult<()> {
        let description: serde_json::Value = from_json(description)?;
        let end = &mut self.end;
        py.detach(|| end.described(&description))
            .map_err(python_error)
    }

    /// Answer a boot with its record, as JSON.
    fn booted(&mut self, py: Python<'_>, record: &str) -> PyResult<()> {
        let record: serde_json::Value = from_json(record)?;
        let end = &mut self.end;
        py.detach(|| end.booted(&record)).map_err(python_error)
    }

    /// Say the region's stages can start.
    fn ready(&mut self, py: Python<'_>) -> PyResult<()> {
        let end = &mut self.end;
        py.detach(|| end.ready()).map_err(python_error)
    }

    /// Send a stage's output tables, which ends the stage.
    fn outputs(&mut self, py: Python<'_>, outputs: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let outputs = tables_from_python(&outputs)?;
        let end = &mut self.end;
        py.detach(|| end.outputs(&outputs)).map_err(python_error)
    }

    /// Send a table the running stage finished.
    fn finished(&mut self, py: Python<'_>, table: &Bound<'_, PyAny>) -> PyResult<()> {
        let table = table_from_python(table)?;
        let end = &mut self.end;
        py.detach(|| end.finished(&table)).map_err(python_error)
    }

    /// Ask the region a question, as JSON, while a stage runs; returns
    /// its answer as JSON text or a pyarrow Table.
    fn ask<'py>(&mut self, py: Python<'py>, question: &str) -> PyResult<Bound<'py, PyAny>> {
        let question: serde_json::Value = from_json(question)?;
        let end = &mut self.end;
        match py.detach(|| end.ask(&question)).map_err(python_error)? {
            Answer::Json(value) => Ok(PyString::new(py, &value.to_string()).into_any()),
            Answer::Table((schema, batches)) => to_pyarrow(py, schema, batches),
        }
    }

    /// Answer the region's end with what the worker ran, as JSON, and
    /// its nodes' output tables.
    fn ran(&mut self, py: Python<'_>, ran: &str, tables: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let ran: serde_json::Value = from_json(ran)?;
        let tables = tables_from_python(&tables)?;
        let end = &mut self.end;
        py.detach(|| end.ran(&ran, &tables)).map_err(python_error)
    }

    /// Say why the message the worker is answering failed.
    fn fail(&mut self, py: Python<'_>, error: &str) -> PyResult<()> {
        let end = &mut self.end;
        py.detach(|| end.fail(error)).map_err(python_error)
    }
}

/// The tables a planned region's workers read, by name, as
/// `crate::request::request_tables` builds them.
///
/// `envelope` is the plan's envelope as JSON; `operators` the region's
/// operators as JSON; `scans` gives, by alias, each scan's token store
/// and the columns of its rows, as pyarrow Tables.
#[pyfunction]
fn request_tables<'py>(
    py: Python<'py>,
    envelope: &str,
    operators: &str,
    scans: HashMap<String, (Bound<'py, PyAny>, Bound<'py, PyAny>)>,
) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    let envelope: serde_json::Value = from_json(envelope)?;
    let operators: quail_planner::operators::Operators = from_json(operators)?;
    let scans = scans
        .into_iter()
        .map(|(alias, (tokens, columns))| {
            Ok((
                alias,
                table_from_python(&tokens)?,
                table_from_python(&columns)?,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let tables = py
        .detach(|| crate::request::request_tables(&envelope, &operators, &scans))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    tables
        .into_iter()
        .map(|(name, (schema, batches))| Ok((name, to_pyarrow(py, schema, batches)?)))
        .collect()
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Workers>()?;
    module.add_class::<WorkerEnd>()?;
    module.add_function(wrap_pyfunction!(request_tables, module)?)
}
