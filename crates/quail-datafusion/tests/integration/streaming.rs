//! `ModelRegionExec` hands DataFusion each batch as the runner yields it.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use datafusion::arrow::array::{Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{Constraint, Constraints, DataFusionError, Result};
use datafusion::datasource::MemTable;
use futures::StreamExt;
use quail_datafusion::{
    IdentifiedTable, InputStream, ModelRegion, RegionBatches, RegionRunner, session_context,
};

/// Stands in for Quail's executor: it yields one row per review, and
/// yields the second only once the test has read the first.
#[derive(Debug)]
struct WaitsForTheReader {
    read: Mutex<Option<Receiver<()>>>,
}

impl RegionRunner for WaitsForTheReader {
    fn run(
        &self,
        region: &ModelRegion,
        _inputs: Vec<InputStream>,
        _fetch: Option<usize>,
        _hint: &quail_datafusion::RegionHint,
    ) -> Result<RegionBatches> {
        let schema = Arc::new(region.plan.schema().as_arrow().clone());
        let read = self.read.lock().expect("the lock").take().expect("one run");
        // each column, such as a row number, read from the given text
        let batch = move |value: &str| {
            let columns = schema
                .fields()
                .iter()
                .map(|field| {
                    let column: Arc<dyn Array> = Arc::new(StringArray::from(vec![value]));
                    datafusion::arrow::compute::cast(&column, field.data_type())
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
        };
        let first = batch("0");
        let mut second = Some(batch);
        let rest = std::iter::from_fn(move || {
            let batch = second.take()?;
            Some(match read.recv_timeout(Duration::from_secs(10)) {
                Ok(()) => batch("1"),
                Err(_) => Err(DataFusionError::Execution(
                    "the reader did not get the first batch".into(),
                )),
            })
        });
        Ok(Box::new(std::iter::once(first).chain(rest)))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_first_batch_reaches_datafusion_before_the_region_finishes() -> Result<()> {
    let (signal, read): (Sender<()>, Receiver<()>) = channel();
    let context = session_context(Arc::new(WaitsForTheReader {
        read: Mutex::new(Some(read)),
    }));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let reviews = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["r1", "r2"])),
            Arc::new(StringArray::from(vec!["Too slow.", "Lovely."])),
        ],
    )?;
    // the id column as a primary key: the region returns the ids itself
    let reviews = MemTable::try_new(schema, vec![vec![reviews]])?.with_constraints(
        Constraints::new_unverified(vec![Constraint::PrimaryKey(vec![0])]),
    );
    context.register_table(
        "reviews",
        Arc::new(IdentifiedTable::new(Arc::new(reviews), "reviews")),
    )?;
    let frame = context
        .sql("SELECT r.id FROM reviews r WHERE ai_filter(prompt('Negative? {0}', r.body))")
        .await?;
    let mut stream = frame.execute_stream().await?;
    let first = stream.next().await.expect("a first batch")?;
    assert_eq!(first.num_rows(), 1);
    signal.send(()).expect("the runner waits");
    let second = stream.next().await.expect("a second batch")?;
    assert_eq!(second.num_rows(), 1);
    assert!(stream.next().await.is_none());
    Ok(())
}

/// Stands in for an executor that yields one row, then panics.
#[derive(Debug)]
struct PanicsAfterOneRow;

impl RegionRunner for PanicsAfterOneRow {
    fn run(
        &self,
        region: &ModelRegion,
        _inputs: Vec<InputStream>,
        _fetch: Option<usize>,
        _hint: &quail_datafusion::RegionHint,
    ) -> Result<RegionBatches> {
        let schema = Arc::new(region.plan.schema().as_arrow().clone());
        let columns = schema
            .fields()
            .iter()
            .map(|field| {
                let column: Arc<dyn Array> = Arc::new(StringArray::from(vec!["0"]));
                datafusion::arrow::compute::cast(&column, field.data_type())
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let first = RecordBatch::try_new(Arc::clone(&schema), columns)?;
        let rest = std::iter::from_fn(|| panic!("the executor failed"));
        Ok(Box::new(std::iter::once(Ok(first)).chain(rest)))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_runner_that_panics_fails_the_query() -> Result<()> {
    let context = session_context(Arc::new(PanicsAfterOneRow));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let reviews = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["r1", "r2"])),
            Arc::new(StringArray::from(vec!["Too slow.", "Lovely."])),
        ],
    )?;
    let reviews = MemTable::try_new(schema, vec![vec![reviews]])?.with_constraints(
        Constraints::new_unverified(vec![Constraint::PrimaryKey(vec![0])]),
    );
    context.register_table(
        "reviews",
        Arc::new(IdentifiedTable::new(Arc::new(reviews), "reviews")),
    )?;
    let frame = context
        .sql("SELECT r.id FROM reviews r WHERE ai_filter(prompt('Negative? {0}', r.body))")
        .await?;
    // the stream re-raises the runner's panic on the reading task
    let collected = tokio::spawn(frame.collect()).await;
    assert!(collected.is_err_and(|error| error.is_panic()));
    Ok(())
}
