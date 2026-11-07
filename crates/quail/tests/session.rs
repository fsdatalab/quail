//! A Rust program runs a query through Quail's Python executor, with
//! the model answered by the test oracle in `tests/helpers/oracle.py`.

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use quail::Session;

/// An engine on the test configuration whose model keeps reviews r0, r2,
/// and r3 for the question `q1:`.
const ENGINE: &std::ffi::CStr = c"
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.api.engine import Engine

session = quail.Session(CONFIG, tokenizer=TOKENIZER)
engine = Engine(session, oracle_executor(Oracle(filters={'r': {'q1:': [1, 0, 1, 1]}})))
";

fn engine() -> Py<PyAny> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if std::env::var_os("VIRTUAL_ENV").is_none() {
        // SAFETY: set before Python starts, while no other thread reads it
        unsafe { std::env::set_var("VIRTUAL_ENV", root.join(".venv")) };
    }
    quail::start_python().expect("Python starts");
    let tests = root.join("tests");
    Python::attach(|py| {
        py.import("sys")?
            .getattr("path")?
            .call_method1("insert", (0, tests.to_string_lossy().to_string()))?;
        let module = PyModule::from_code(py, ENGINE, c"engine.py", c"engine")?;
        Ok::<_, PyErr>(module.getattr("engine")?.unbind())
    })
    .expect("the test engine")
}

fn reviews() -> Result<MemTable> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["r0", "r1", "r2", "r3"])),
            Arc::new(StringArray::from(vec![
                "good ending",
                "slow pacing",
                "great cast",
                "weak ending",
            ])),
        ],
    )?;
    MemTable::try_new(schema, vec![vec![batch]])
}

#[test]
fn a_rust_program_runs_an_ai_filter_over_its_own_table() -> Result<()> {
    let session = Session::with_engine(engine())?;
    session.register_table("reviews", Arc::new(reviews()?))?;
    quail_datafusion::derive::set_shadow_check(true);
    let batches =
        session.sql("SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('q1: {0}', r.body))")?;
    let log = quail_datafusion::derive::take_shadow_log().expect("a shadow check");
    quail_datafusion::derive::set_shadow_check(false);
    // the region's operators derived from its optimized plan are the binder's
    assert!(log.num_comparisons > 0);
    assert!(log.differences.is_empty(), "{:?}", log.differences);
    let mut ids: Vec<String> = Vec::new();
    for batch in &batches {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string ids");
        ids.extend(column.iter().flatten().map(str::to_string));
    }
    ids.sort();
    assert_eq!(ids, ["r0", "r2", "r3"]);
    Ok(())
}
