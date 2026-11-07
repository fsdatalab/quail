//! Dynamic join filters around a model region's rows.
//!
//! A hash join sends the keys of its build side to its probe side as a
//! dynamic filter, which an operator there, such as a sort, applies to
//! the rows it reads. DataFusion 55.2 moves the filter into a child
//! by column name (`FilterRemapper::try_remap` in
//! `datafusion-physical-plan` `filter_pushdown.rs`, which a hash join
//! calls from `gather_filters_for_pushdown`). When the child has two
//! columns of one name, the filter reads the first, whichever column
//! the join reads (apache/datafusion#25244). A model region over two
//! tables returns two id columns of one name, such as `r.id` and
//! `p.id`, so Quail's session turns dynamic filter pushdown off
//! (`lib.rs`).
//!
//! The first test reproduces the bug with DataFusion alone. It fails
//! once DataFusion maps the columns by position (apache/datafusion#25259,
//! on its `branch-56`); Quail's session can then keep the filters on.
//! The other tests turn the filters on to check the join back, and
//! check that Quail's own setting keeps the rows of joins above a region.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{Constraint, Constraints, Result};
use datafusion::datasource::MemTable;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::SessionContext;
use quail_datafusion::{
    IdentifiedTable, InputRows, ModelRegion, ModelRegionExec, RegionBatches, RegionRunner,
    session_context, sql,
};

fn text_schema(columns: &[&str]) -> Arc<Schema> {
    Arc::new(Schema::new(
        columns
            .iter()
            .map(|name| Field::new(*name, DataType::Utf8, false))
            .collect::<Vec<_>>(),
    ))
}

fn text_batch(columns: &[&str], values: &[&[&str]]) -> Result<RecordBatch> {
    let arrays = values
        .iter()
        .map(|column| Arc::new(StringArray::from(column.to_vec())) as Arc<dyn Array>)
        .collect();
    Ok(RecordBatch::try_new(text_schema(columns), arrays)?)
}

/// A table of text columns with exact row counts, as DataFusion's
/// statistics give a `MemTable`.
fn text_table(columns: &[&str], values: &[&[&str]]) -> Result<Arc<MemTable>> {
    let batch = text_batch(columns, values)?;
    Ok(Arc::new(MemTable::try_new(
        batch.schema(),
        vec![vec![batch]],
    )?))
}

#[tokio::test]
async fn datafusion_filters_the_first_of_two_columns_of_one_name() -> Result<()> {
    let context = SessionContext::new();
    // eight pairs of a review id and a product id; four are of reviews
    // and products the tables hold
    let pairs = text_table(
        &["k", "rid", "pid"],
        &[
            &["0", "1", "2", "3", "4", "5", "6", "7"],
            &["r0", "r0", "r2", "r2", "r8", "r8", "r9", "r9"],
            &["p0", "p1", "p0", "p1", "p8", "p8", "p9", "p9"],
        ],
    )?;
    context.register_table("pairs", pairs)?;
    let reviews = text_table(&["id", "body"], &[&["r0", "r1", "r2"], &["a", "b", "c"]])?;
    context.register_table("reviews", reviews)?;
    let products = text_table(&["id", "text"], &[&["p0", "p1"], &["x", "y"]])?;
    context.register_table("products", products)?;
    // the pairs' join returns two columns named `id`. JoinSelection
    // builds both outer joins on the smaller tables, so their filters
    // reach the sort over the pairs, and the products' filter reads the
    // first `id`, the review's
    let query = "SELECT a.id, b.id, r.body, p.text \
        FROM (SELECT k, rid AS id FROM pairs) a \
        JOIN (SELECT k, pid AS id FROM pairs) b ON a.k = b.k \
        JOIN reviews r ON r.id = a.id JOIN products p ON p.id = b.id \
        ORDER BY a.id, b.id";
    let mut found = Vec::new();
    for setting in ["true", "false"] {
        let set = format!("SET datafusion.optimizer.enable_dynamic_filter_pushdown = {setting}");
        context.sql(&set).await?.collect().await?;
        let plan = context.sql(query).await?.create_physical_plan().await?;
        let batches = collect(Arc::clone(&plan), context.task_ctx()).await?;
        found.push(batches.iter().map(RecordBatch::num_rows).sum::<usize>());
    }
    assert_eq!(
        found,
        [0, 4],
        "DataFusion now keeps the pairs; Quail's session can keep dynamic filters on (lib.rs)"
    );
    Ok(())
}

/// Stands in for Quail's executor: the model keeps the pairs of reviews
/// r0 and r2 with every product, or reviews r0 and r2 alone.
#[derive(Debug)]
struct KeepsTwoReviews;

impl RegionRunner for KeepsTwoReviews {
    fn run(
        &self,
        region: &ModelRegion,
        _inputs: &[InputRows],
        _fetch: Option<usize>,
    ) -> Result<RegionBatches> {
        let schema = region.plan.schema();
        let pairs = schema
            .iter()
            .any(|(qualifier, _)| qualifier.is_some_and(|found| found.table() == "p"));
        let columns = schema
            .iter()
            .map(|(qualifier, _)| {
                let values = match qualifier.map(|found| found.table()) {
                    Some("p") => vec!["p0", "p1", "p0", "p1"],
                    _ if pairs => vec!["r0", "r0", "r2", "r2"],
                    _ => vec!["r0", "r2"],
                };
                Arc::new(StringArray::from(values)) as Arc<dyn Array>
            })
            .collect();
        let batch = RecordBatch::try_new(Arc::new(schema.as_arrow().clone()), columns)?;
        Ok(Box::new(std::iter::once(Ok(batch))))
    }
}

fn identified(name: &str, table: MemTable) -> Arc<IdentifiedTable> {
    let table = table.with_constraints(Constraints::new_unverified(vec![Constraint::PrimaryKey(
        vec![0],
    )]));
    Arc::new(IdentifiedTable::new(Arc::new(table), name))
}

async fn context(settings: &[&str]) -> Result<SessionContext> {
    let context = session_context(Arc::new(KeepsTwoReviews));
    for setting in settings {
        context.sql(setting).await?.collect().await?;
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
        Field::new("stars", DataType::Int64, false),
    ]));
    let reviews = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["r0", "r1", "r2"])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Int64Array::from(vec![5, 2, 4])),
        ],
    )?;
    let reviews = MemTable::try_new(schema, vec![vec![reviews]])?;
    context.register_table("reviews", identified("reviews", reviews))?;
    let products = text_batch(&["id", "text"], &[&["p0", "p1"], &["x", "y"]])?;
    let products = MemTable::try_new(products.schema(), vec![vec![products]])?;
    context.register_table("products", identified("products", products))?;
    let tags = text_batch(
        &["rid", "pid", "tag"],
        &[
            &["r0", "r1", "r2", "r9"],
            &["p1", "p0", "p0", "p9"],
            &["t0", "t1", "t2", "t9"],
        ],
    )?;
    context.register_table(
        "tags",
        Arc::new(MemTable::try_new(tags.schema(), vec![vec![tags]])?),
    )?;
    Ok(context)
}

/// The plan of `query` and its rows, each as text.
async fn run(context: &SessionContext, query: &str) -> Result<(Arc<dyn ExecutionPlan>, String)> {
    let plan = sql(context, query).await?.create_physical_plan().await?;
    let batches = collect(Arc::clone(&plan), context.task_ctx()).await?;
    let rows = datafusion::arrow::util::pretty::pretty_format_batches(&batches)?.to_string();
    Ok((plan, rows))
}

/// Whether a hash join in `plan` builds its table from the region's rows.
fn builds_on_the_region(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let contains_region = |plan: &Arc<dyn ExecutionPlan>| {
        let mut found = false;
        plan.apply(|node| {
            found |= node.is::<ModelRegionExec>();
            Ok(TreeNodeRecursion::Continue)
        })
        .expect("a plan walk");
        found
    };
    let mut found = false;
    plan.apply(|node| {
        if let Some(join) = node.downcast_ref::<HashJoinExec>() {
            found |= contains_region(join.left()) && !contains_region(join.right());
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .expect("a plan walk");
    found
}

const DYNAMIC_FILTERS: &str = "SET datafusion.optimizer.enable_dynamic_filter_pushdown = true";

#[tokio::test]
async fn an_ai_join_returns_its_pairs_with_the_region_on_either_side_of_the_join_back() -> Result<()>
{
    let query = "SELECT r.id, r.stars, p.id, p.text FROM reviews r JOIN products p \
        ON AI_FILTER(PROMPT('m {0} {1}', r.body, p.text)) ORDER BY r.id, p.id";
    let expected = [
        "+----+-------+----+------+",
        "| id | stars | id | text |",
        "+----+-------+----+------+",
        "| r0 | 5     | p0 | x    |",
        "| r0 | 5     | p1 | y    |",
        "| r2 | 4     | p0 | x    |",
        "| r2 | 4     | p1 | y    |",
        "+----+-------+----+------+",
    ]
    .join("\n");
    let mut sides = Vec::new();
    // with its statistics, DataFusion builds on the small tables; without
    // them, on the region's rows, the join back's left side
    for reordering in [true, false] {
        let setting = format!("SET datafusion.optimizer.join_reordering = {reordering}");
        let context = context(&[DYNAMIC_FILTERS, &setting]).await?;
        let (plan, rows) = run(&context, query).await?;
        assert_eq!(
            rows,
            expected,
            "{}",
            displayable(plan.as_ref()).indent(true)
        );
        sides.push(builds_on_the_region(&plan));
    }
    assert_eq!(sides, [false, true]);
    Ok(())
}

#[tokio::test]
async fn a_join_with_a_plain_table_filters_the_regions_rows_by_the_tables_keys() -> Result<()> {
    let context = context(&[DYNAMIC_FILTERS]).await?;
    let (plan, rows) = run(
        &context,
        "SELECT r.id, p.id FROM reviews r JOIN products p \
            ON AI_FILTER(PROMPT('m {0} {1}', r.body, p.text)) \
            JOIN tags t ON t.pid = p.id ORDER BY r.id, p.id",
    )
    .await?;
    let text = displayable(plan.as_ref()).indent(true).to_string();
    // the sort below the join applies the tags' product ids to the
    // region's rows
    assert!(text.contains("FilterExec: DynamicFilter"), "{text}");
    let expected = [
        "+----+----+",
        "| id | id |",
        "+----+----+",
        "| r0 | p0 |",
        "| r0 | p0 |",
        "| r0 | p1 |",
        "| r2 | p0 |",
        "| r2 | p0 |",
        "| r2 | p1 |",
        "+----+----+",
    ]
    .join("\n");
    assert_eq!(rows, expected, "{text}");
    Ok(())
}

#[tokio::test]
async fn joins_of_tables_above_a_region_keep_its_pairs_in_quails_session() -> Result<()> {
    let context = context(&[]).await?;
    // the region returns `r.id` and `p.id`; the joins above it read each
    let (plan, rows) = run(
        &context,
        "SELECT r2.stars, p2.text FROM reviews r JOIN products p \
            ON AI_FILTER(PROMPT('m {0} {1}', r.body, p.text)) \
            JOIN reviews r2 ON r2.id = r.id JOIN products p2 ON p2.id = p.id \
            ORDER BY r.id, p.id",
    )
    .await?;
    let expected = [
        "+-------+------+",
        "| stars | text |",
        "+-------+------+",
        "| 5     | x    |",
        "| 5     | y    |",
        "| 4     | x    |",
        "| 4     | y    |",
        "+-------+------+",
    ]
    .join("\n");
    assert_eq!(
        rows,
        expected,
        "{}",
        displayable(plan.as_ref()).indent(true)
    );
    Ok(())
}
