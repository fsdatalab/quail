//! Run an AI filter over a small table.
//!
//! Quail starts one GPU worker per GPU: `python -m quail.worker`, from
//! the Python environment `VIRTUAL_ENV` names. Install Quail's Python
//! package there first, on a machine with the GPU:
//!
//! ```sh
//! pip install quail-engine
//! cargo run --example quickstart
//! ```

// links DataFusion and Quail from the shared library
use quail_dylib as _;

use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::util::pretty::print_batches;
use datafusion::datasource::MemTable;

fn main() -> datafusion::common::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let reviews = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["1", "2", "3"])),
            Arc::new(StringArray::from(vec![
                "The ending was perfect and the cast was great.",
                "Too long, and the ending made no sense.",
                "Great music, but I want my money back.",
            ])),
        ],
    )?;

    let config = quail::EngineConfig::new("qwen3-4b-fp8", "h100-sxm");
    let session = quail::Session::new(&config)?;
    session.register_table(
        "reviews",
        Arc::new(MemTable::try_new(schema, vec![vec![reviews]])?),
    )?;
    let rows = session.sql(
        "SELECT r.id FROM reviews r \
         WHERE AI.IF(PROMPT('Does the reviewer ask for a refund? {0}', r.body))",
    )?;
    print_batches(&rows)?;
    Ok(())
}
