//! Filter IMDB reviews from an Arrow IPC file with a language model.

const SQL: &str = r#"
SELECT r.id
FROM reviews AS r
WHERE AI.IF(PROMPT(
    'Does this review mention a positive aspect of the movie? {0}',
    r.body
))
"#;

use std::fs::File;
use std::sync::Arc;

use datafusion::arrow::ipc::reader::FileReader;
use datafusion::arrow::util::pretty::print_batches;
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use quail::{EngineConfig, Session};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "reviews.arrow".to_string());
    let reader = FileReader::try_new(File::open(path)?, None)?;
    let schema = reader.schema();
    let batches = reader.collect::<std::result::Result<Vec<_>, _>>()?;
    let reviews = MemTable::try_new(schema, vec![batches])?;

    let config = EngineConfig::new("qwen3-4b-fp8", "h100-sxm");
    let session = Session::new(&config)?;
    session.register_table("reviews", Arc::new(reviews))?;
    let (_, batches) = session.sql_in_dialect(SQL, Some("BigQuery"))?;

    let num_reviews: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    println!("Matching reviews: {num_reviews}");
    if let Some(batch) = batches.first() {
        print_batches(&[batch.slice(0, batch.num_rows().min(10))])?;
    }
    Ok(())
}
