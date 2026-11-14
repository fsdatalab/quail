//! A Rust program runs a query through Quail's Python executor, with
//! the model answered by the test oracle in `tests/helpers/oracle.py`.

use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use pyo3::prelude::*;
use pyo3::types::PyModule;

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
    crate::start_python();
    let tests = crate::root().join("tests");
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
fn a_rust_program_creates_a_table_from_model_results() -> Result<()> {
    let session = quail_python::session_with_engine(engine())?;
    session.register_table("reviews", Arc::new(reviews()?))?;
    session.sql(
        "CREATE TABLE kept AS SELECT r.id FROM reviews r \
         WHERE AI.IF(PROMPT('q1: {0}', r.body))",
    )?;
    let batches = session.sql("SELECT id FROM kept ORDER BY id")?;
    let ids: Vec<&str> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string ids")
                .iter()
                .flatten()
        })
        .collect();
    assert_eq!(ids, ["r0", "r2", "r3"]);
    session.sql("DROP TABLE kept")?;
    assert!(session.sql("SELECT * FROM kept").is_err());
    Ok(())
}

#[test]
fn a_rust_program_runs_an_ai_filter_over_its_own_table() -> Result<()> {
    let session = quail_python::session_with_engine(engine())?;
    session.register_table("reviews", Arc::new(reviews()?))?;
    let batches =
        session.sql("SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('q1: {0}', r.body))")?;
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
