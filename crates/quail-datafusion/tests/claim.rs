use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use quail_datafusion::{ModelRegion, RegionRunner, session_context, sql};

/// Answers every region with the rows it holds, in the region's schema.
#[derive(Debug)]
struct FixedRows(Vec<RecordBatch>);

impl RegionRunner for FixedRows {
    fn run(&self, _region: &ModelRegion) -> Result<Vec<RecordBatch>> {
        Ok(self.0.clone())
    }
}

fn text_table(columns: &[&str], rows: &[&[&str]]) -> Result<MemTable> {
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|name| Field::new(*name, DataType::Utf8, false))
            .collect::<Vec<_>>(),
    ));
    let arrays = (0..columns.len())
        .map(|index| {
            Arc::new(StringArray::from(
                rows.iter().map(|row| row[index]).collect::<Vec<_>>(),
            )) as _
        })
        .collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    MemTable::try_new(schema, vec![vec![batch]])
}

fn context(runner: FixedRows) -> Result<SessionContext> {
    let context = session_context(Arc::new(runner));
    context.register_table(
        "reviews",
        Arc::new(text_table(
            &["id", "body"],
            &[&["r1", "A moving ending."], &["r2", "Too long."]],
        )?),
    )?;
    context.register_table(
        "aspects",
        Arc::new(text_table(&["aspect"], &[&["ending"], &["pacing"]])?),
    )?;
    Ok(context)
}

async fn optimized(context: &SessionContext, query: &str) -> Result<LogicalPlan> {
    sql(context, query).await?.into_optimized_plan()
}

fn regions(plan: &LogicalPlan) -> Vec<ModelRegion> {
    let mut found = Vec::new();
    plan.apply(|node| {
        if let LogicalPlan::Extension(extension) = node
            && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
        {
            found.push(region.clone());
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })
    .unwrap();
    found
}

use datafusion::common::tree_node::TreeNode;

const FILTER_QUERY: &str = "SELECT r.id FROM reviews r \
    WHERE AI.IF(PROMPT('Does this review discuss the ending?\n\n{0}', r.body))";

#[tokio::test]
async fn a_filter_query_becomes_one_model_region() -> Result<()> {
    let context = context(FixedRows(vec![]))?;
    let plan = optimized(&context, FILTER_QUERY).await?;
    assert!(
        matches!(plan, LogicalPlan::Extension(_)),
        "{}",
        plan.display_indent()
    );
    assert_eq!(regions(&plan).len(), 1);
    Ok(())
}

#[tokio::test]
async fn a_query_without_ai_calls_has_no_model_region() -> Result<()> {
    let context = context(FixedRows(vec![]))?;
    let plan = optimized(&context, "SELECT id FROM reviews WHERE id = 'r1'").await?;
    assert!(regions(&plan).is_empty());
    Ok(())
}

#[tokio::test]
async fn an_aggregate_over_ai_answers_stays_in_the_region() -> Result<()> {
    let context = context(FixedRows(vec![]))?;
    let query = "SELECT AI.CLASSIFY(r.body, ['good', 'bad']) AS label, COUNT(*) AS n \
        FROM reviews r GROUP BY label";
    let plan = optimized(&context, query).await?;
    assert!(
        matches!(plan, LogicalPlan::Extension(_)),
        "{}",
        plan.display_indent()
    );
    Ok(())
}

#[tokio::test]
async fn an_ai_join_and_its_distinct_stay_in_one_region() -> Result<()> {
    let context = context(FixedRows(vec![]))?;
    let query = "SELECT DISTINCT r.id FROM reviews r JOIN aspects a \
        ON AI.IF(PROMPT('Does the review discuss {1}?\n\n{0}', r.body, a.aspect))";
    let plan = optimized(&context, query).await?;
    let found = regions(&plan);
    assert_eq!(found.len(), 1, "{}", plan.display_indent());
    let inner = found[0].plan.display_indent().to_string();
    assert!(inner.contains("Distinct"), "{inner}");
    assert!(inner.contains("Join"), "{inner}");
    Ok(())
}

#[tokio::test]
async fn a_union_keeps_one_region_per_branch() -> Result<()> {
    let context = context(FixedRows(vec![]))?;
    let query = format!("{FILTER_QUERY} UNION ALL SELECT id FROM reviews WHERE id = 'r2'");
    let plan = optimized(&context, &query).await?;
    assert!(
        matches!(plan, LogicalPlan::Union(_)),
        "{}",
        plan.display_indent()
    );
    assert_eq!(regions(&plan).len(), 1);
    Ok(())
}

#[tokio::test]
async fn an_exists_subquery_becomes_a_join_inside_the_region() -> Result<()> {
    let context = context(FixedRows(vec![]))?;
    let query = "SELECT r.id FROM reviews r WHERE EXISTS (SELECT 1 FROM aspects a \
        WHERE AI.IF(PROMPT('Does the review discuss {1}?\n\n{0}', r.body, a.aspect)))";
    let plan = optimized(&context, query).await?;
    let found = regions(&plan);
    assert_eq!(found.len(), 1, "{}", plan.display_indent());
    assert!(
        found[0]
            .plan
            .display_indent()
            .to_string()
            .contains("LeftSemi")
    );
    Ok(())
}

#[tokio::test]
async fn a_region_returns_its_runners_rows() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
    let answer = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["r1"]))],
    )?;
    let context = context(FixedRows(vec![answer.clone()]))?;
    let batches = sql(&context, FILTER_QUERY).await?.collect().await?;
    assert_eq!(batches, vec![answer]);
    Ok(())
}
