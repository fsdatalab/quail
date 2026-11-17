//! A table another library exports through DataFusion's FFI, as
//! datafusion-python's table protocol hands it over, answers an AI
//! filter in a Python session.

use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyModule};

/// A table with `__datafusion_table_provider__`, as Delta Lake and
/// Iceberg export one.
#[pyclass]
struct Exported {
    table: Arc<MemTable>,
}

#[pymethods]
impl Exported {
    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = session.call_method0("__datafusion_logical_extension_codec__")?;
        let codec = codec.cast::<PyCapsule>()?;
        let pointer = codec.pointer_checked(Some(c"datafusion_logical_extension_codec"))?;
        // the session's capsule keeps its codec; the table takes a clone
        let codec = unsafe { &*pointer.as_ptr().cast::<FFI_LogicalExtensionCodec>() }.clone();
        let provider =
            FFI_TableProvider::new_with_ffi_codec(Arc::clone(&self.table) as _, false, None, codec);
        PyCapsule::new_with_value(py, provider, c"datafusion_table_provider")
    }
}

/// Registers `reviews` and returns the ids the oracle keeps for `q1:`.
const QUERY: &std::ffi::CStr = c"
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER

import quail

def kept(reviews):
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register('reviews', reviews)
        oracle = Oracle(filters={'r': {'q1:': [1, 0, 1, 1]}})
        result = session.sql(
            \"SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('q1: {0}', r.body)) \"
            \"ORDER BY r.id\").run(physical_executor=oracle_executor(oracle))
        return result.collect().column(0).to_pylist()
";

#[test]
fn a_table_exported_through_ffi_answers_an_ai_filter() -> datafusion::common::Result<()> {
    crate::start_python();
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
    let table = Arc::new(MemTable::try_new(schema, vec![vec![batch]])?);
    let tests = crate::root().join("tests");
    let kept: Vec<String> = Python::attach(|py| {
        py.import("sys")?
            .getattr("path")?
            .call_method1("insert", (0, tests.to_string_lossy().to_string()))?;
        let module = PyModule::from_code(py, QUERY, c"foreign.py", c"foreign")?;
        module
            .getattr("kept")?
            .call1((Exported { table },))?
            .extract()
    })
    .expect("the query runs");
    assert_eq!(kept, ["r0", "r2", "r3"]);
    Ok(())
}
