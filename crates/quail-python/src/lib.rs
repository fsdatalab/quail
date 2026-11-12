//! Quail's Python bindings: the `_quail_native` module, and Quail's
//! [`Session`] over a Python engine.
//!
//! - [`native`] is the `_quail_native` Python module, built by maturin
//!   from this crate. Python's `quail.Session` runs its SQL on the
//!   `quail` crate's [`Session`] through it.
//! - [`session_with_engine`] makes a session whose model regions a
//!   Python engine runs, for a backend that is not Quail's own.
//!
//! The `quail` crate is Quail without Python: its session, executor,
//! and GPU workers.

mod engine;
pub mod native;
pub mod pyarrow;

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::common::{DataFusionError, Result};
use pyo3::prelude::*;
pub use quail::{EngineConfig, Executor, Session, region_json};
pub use quail::{finished, planning, region_run, request, results, shared, stages, workers};

pub use crate::engine::{
    executor_with_engine, executor_with_planner, register_python_table, scans_to_pyarrow,
};

/// A Python exception as a DataFusion error with its traceback and
/// chained causes: these are otherwise lost once the error leaves Python.
pub(crate) fn python_failure(error: PyErr) -> DataFusionError {
    let text = Python::attach(|py| {
        // a refusal or a compile error is Quail's answer to the query, in
        // one sentence; any other failure carries its traceback
        let kind = error.get_type(py).name().map(|name| name.to_string());
        if let Ok(kind) = kind
            && (kind == "RefusalError" || kind == "CompileError")
        {
            return format!("{kind}: {}", error.value(py));
        }
        let formatted = py
            .import("traceback")
            .and_then(|module| module.call_method1("format_exception", (error.value(py),)))
            .and_then(|lines| lines.extract::<Vec<String>>());
        match formatted {
            Ok(lines) => lines.concat().trim_end().to_string(),
            Err(_) => error.to_string(),
        }
    });
    DataFusionError::External(text.into())
}

/// Add the site-packages of the virtual environment `VIRTUAL_ENV`
/// names, which an embedded interpreter does not read by itself.
fn add_virtual_env(py: Python<'_>) -> PyResult<()> {
    let Some(root) = std::env::var_os("VIRTUAL_ENV").map(PathBuf::from) else {
        return Ok(());
    };
    let Ok(entries) = std::fs::read_dir(root.join("lib")) else {
        return Ok(());
    };
    let site = py.import("site")?;
    for entry in entries.flatten() {
        let packages = entry.path().join("site-packages");
        if packages.is_dir() {
            site.call_method1("addsitedir", (packages.to_string_lossy().to_string(),))?;
        }
    }
    Ok(())
}

/// Start Python in this process if it has not started, and add the
/// virtual environment `VIRTUAL_ENV` names to its path.
///
/// # Errors
///
/// The environment's site-packages do not load.
pub fn start_python() -> Result<()> {
    Python::initialize();
    Python::attach(add_virtual_env).map_err(python_failure)
}

/// A session whose model regions run on `engine`, a Python object as
/// [`executor_with_engine`] describes.
///
/// # Errors
///
/// An attribute is missing or a rule name is unknown.
pub fn session_with_engine(engine: Py<PyAny>) -> Result<Session> {
    Session::with_executor(Arc::new(executor_with_engine(engine)?))
}

/// A session as [`session_with_engine`] describes, whose regions
/// `planner` plans in Rust when the engine has a `settings` attribute;
/// without a planner every region goes to `run_region`.
///
/// # Errors
///
/// An attribute is missing or malformed, or a rule name is unknown.
pub fn session_with_planner(
    engine: Py<PyAny>,
    planner: Option<Arc<planning::Planner>>,
) -> Result<Session> {
    Session::with_executor(Arc::new(executor_with_planner(engine, planner)?))
}
