//! Dynamic join filters around a model region's rows.
//!
//! A hash join sends the keys of its build side to its probe side as a
//! dynamic filter, which an operator there, such as a sort, applies to
//! the rows it reads. DataFusion 55.2 moves the filter into a child
//! by column name (`FilterRemapper::try_remap` in
//! `datafusion-physical-plan` `filter_pushdown.rs`, which a hash join
//! calls from `gather_filters_for_pushdown`). When the child has two
//! columns of one name, the filter reads the first, whichever column
//! the join reads. A model region over two tables returns two id
//! columns of one name, such as `r.id` and `p.id`, so the join that adds
//! the tables' other columns back gives each id a name of its own
//! (`join_back.rs`).
//!
//! The first test reproduces the bug with DataFusion's operators alone.
//! It fails once DataFusion maps the columns by position; the renaming
//! in `join_back.rs` can then go.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{Constraint, Constraints, JoinType, NullEquality, Result};
use datafusion::datasource::MemTable;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::filter_pushdown::FilterPushdown;
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::sorts::sort::SortExec;
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

fn inner_join(
    build: Arc<dyn ExecutionPlan>,
    probe: Arc<dyn ExecutionPlan>,
    keys: (Column, Column),
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(HashJoinExec::try_new(
        build,
        probe,
        vec![(Arc::new(keys.0), Arc::new(keys.1))],
        None,
        &JoinType::Inner,
        None,
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false,
    )?))
}

#[tokio::test]
async fn datafusion_filters_the_first_of_two_columns_of_one_name() -> Result<()> {
    // each row pairs a review id with a product id, under one name
    let pairs = text_batch(&["id", "id"], &[&["r0", "r2"], &["p0", "p1"]])?;
    let pairs = MemorySourceConfig::try_new_exec(&[vec![pairs.clone()]], pairs.schema(), None)?;
    let ordering = LexOrdering::new(vec![PhysicalSortExpr::new_default(Arc::new(Column::new(
        "id", 0,
    )))])
    .expect("one sort key");
    let sorted = Arc::new(SortExec::new(ordering, pairs));
    let reviews = text_batch(&["k"], &[&["r0", "r2"]])?;
    let reviews =
        MemorySourceConfig::try_new_exec(&[vec![reviews.clone()]], reviews.schema(), None)?;
    let products = text_batch(&["k"], &[&["p0", "p1"]])?;
    let products =
        MemorySourceConfig::try_new_exec(&[vec![products.clone()]], products.schema(), None)?;
    // the review join's output is k, id, id; the product join reads the
    // second id, the product's
    let by_review = inner_join(reviews, sorted, (Column::new("k", 0), Column::new("id", 0)))?;
    let by_product = inner_join(
        products,
        by_review,
        (Column::new("k", 0), Column::new("id", 2)),
    )?;
    let plan =
        FilterPushdown::new_post_optimization().optimize(by_product, &ConfigOptions::default())?;
    let batches = collect(Arc::clone(&plan), SessionContext::new().task_ctx()).await?;
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    // both pairs match; the product filter tests the review ids instead
    assert_eq!(
        rows,
        0,
        "DataFusion now keeps the pairs; drop the renaming in join_back.rs:\n{}",
        displayable(plan.as_ref()).indent(true)
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
