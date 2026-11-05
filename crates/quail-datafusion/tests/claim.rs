use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{Result, exec_err};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use quail_datafusion::{ModelRegion, RegionRunner, session_context, sql};

/// A runner for planning tests, which never run a region. Execution is
/// covered end to end in `tests/api/test_session_context.py`.
#[derive(Debug)]
struct NotRun;

impl RegionRunner for NotRun {
    fn run(
        &self,
        _region: &ModelRegion,
        _inputs: &[quail_datafusion::InputRows],
    ) -> Result<Vec<RecordBatch>> {
        exec_err!("planning tests do not run model regions")
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

fn context() -> Result<SessionContext> {
    let context = session_context(Arc::new(NotRun));
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
    let context = context()?;
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
    let context = context()?;
    let plan = optimized(&context, "SELECT id FROM reviews WHERE id = 'r1'").await?;
    assert!(regions(&plan).is_empty());
    Ok(())
}

#[tokio::test]
async fn datafusion_groups_the_answers_a_region_returns() -> Result<()> {
    let context = context()?;
    let query = "SELECT AI.CLASSIFY(r.body, ['good', 'bad']) AS label, COUNT(*) AS n \
        FROM reviews r GROUP BY label";
    let plan = optimized(&context, query).await?;
    let shape = lines(&plan);
    assert!(
        shape.iter().any(|line| line.starts_with("Aggregate:")),
        "{shape:?}"
    );
    let found = regions(&plan);
    assert_eq!(found.len(), 1, "{shape:?}");
    let inner = lines(&found[0].plan);
    assert!(
        !inner.iter().any(|line| line.starts_with("Aggregate")),
        "{inner:?}"
    );
    assert!(
        inner.iter().any(|line| line.starts_with("SemanticColumn")),
        "{inner:?}"
    );
    Ok(())
}

#[tokio::test]
async fn an_ai_join_and_its_distinct_stay_in_one_region() -> Result<()> {
    let context = context()?;
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
    let context = context()?;
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
async fn an_exists_gate_becomes_a_semi_join_inside_the_region() -> Result<()> {
    let context = context()?;
    let query = "SELECT r.id FROM reviews r WHERE EXISTS (SELECT 1 FROM aspects a \
        WHERE AI.IF(PROMPT('Does the review discuss {1}?\n\n{0}', r.body, a.aspect)))";
    let plan = optimized(&context, query).await?;
    let found = regions(&plan);
    assert_eq!(found.len(), 1, "{}", plan.display_indent());
    let shape = lines(&found[0].plan);
    assert!(shape[1].starts_with("SemanticJoin: semi"), "{shape:?}");
    assert_eq!(
        shape.last().map(String::as_str),
        Some("TableScan: aspects projection=[aspect]")
    );
    Ok(())
}

fn lines(plan: &LogicalPlan) -> Vec<String> {
    plan.display_indent()
        .to_string()
        .lines()
        .map(|line| line.trim().to_string())
        .collect()
}

#[tokio::test]
async fn a_where_clause_asks_the_model_above_its_plain_terms() -> Result<()> {
    let context = context()?;
    let query = "SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('Kind? {0}', r.body)) \
        AND r.id <> 'r9' AND AI.IF(PROMPT('Long? {0}', r.body))";
    let [region] =
        <[ModelRegion; 1]>::try_from(regions(&optimized(&context, query).await?)).unwrap();
    let plan = lines(&region.plan);
    assert!(plan[1].starts_with("SemanticFilter: ai_filter"), "{plan:?}");
    assert!(
        plan[1].contains("Kind?") && plan[1].contains("Long?"),
        "{plan:?}"
    );
    // DataFusion pushed the plain test below the model's and the alias
    let alias = plan
        .iter()
        .position(|line| line == "SubqueryAlias: r")
        .unwrap();
    let test = plan.iter().position(|line| line.contains("r9")).unwrap();
    assert!(test > alias, "{plan:?}");
    Ok(())
}

#[tokio::test]
async fn an_ai_join_asks_the_model_above_datafusions_join() -> Result<()> {
    let context = context()?;
    let query = "SELECT r.id, a.aspect FROM reviews r JOIN aspects a \
        ON AI.IF(PROMPT('Does {0} discuss {1}?', r.body, a.aspect))";
    let [region] =
        <[ModelRegion; 1]>::try_from(regions(&optimized(&context, query).await?)).unwrap();
    let plan = lines(&region.plan);
    assert!(
        plan[1].starts_with("SemanticJoin: full ai_filter"),
        "{plan:?}"
    );
    // the ON has no other term, so DataFusion's join pairs every row
    assert_eq!(plan[2], "Cross Join:", "{plan:?}");
    Ok(())
}

#[tokio::test]
async fn a_classification_is_a_column_the_projection_reads() -> Result<()> {
    let context = context()?;
    let query = "SELECT r.id, AI.CLASSIFY(PROMPT('Tone of {0}?', r.body), ['good', 'bad']) \
        AS tone FROM reviews r";
    let [region] =
        <[ModelRegion; 1]>::try_from(regions(&optimized(&context, query).await?)).unwrap();
    let plan = lines(&region.plan);
    assert_eq!(plan[0], "Projection: r.id, tone");
    assert!(
        plan[1].starts_with("SemanticColumn: tone = ai_classify"),
        "{plan:?}"
    );
    // the binder reads the SELECT list as DataFusion bound it
    assert!(lines(&region.bound)[0].contains("AS tone"));
    Ok(())
}

#[tokio::test]
async fn a_test_of_a_label_reads_the_column_its_classification_adds() -> Result<()> {
    let context = context()?;
    let query = "SELECT r.id, AI.CLASSIFY(PROMPT('Tone of {0}?', r.body), ['good', 'bad']) \
        AS tone FROM reviews r WHERE tone = 'good'";
    let [region] =
        <[ModelRegion; 1]>::try_from(regions(&optimized(&context, query).await?)).unwrap();
    let plan = lines(&region.plan);
    let test = plan
        .iter()
        .position(|line| line == "Filter: tone = Utf8(\"good\")");
    let column = plan
        .iter()
        .position(|line| line.starts_with("SemanticColumn: tone = ai_classify"));
    assert!(test.unwrap() < column.unwrap(), "{plan:?}");
    // one classification serves the test and the SELECT list
    let columns = plan
        .iter()
        .filter(|line| line.starts_with("SemanticColumn"))
        .count();
    assert_eq!(columns, 1, "{plan:?}");
    Ok(())
}
