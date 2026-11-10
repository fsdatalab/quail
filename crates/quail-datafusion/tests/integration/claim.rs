use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{Constraint, Constraints, Result, exec_err};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use quail_datafusion::{ModelRegion, RegionRunner, session_context};

/// A runner for planning tests, which never run a region. Execution is
/// covered end to end in `tests/api/test_session_context.py`.
#[derive(Debug)]
struct NotRun;

impl RegionRunner for NotRun {
    fn run(
        &self,
        _region: &ModelRegion,
        _inputs: Vec<quail_datafusion::InputStream>,
        _fetch: Option<usize>,
        _hint: &quail_datafusion::RegionHint,
    ) -> Result<quail_datafusion::RegionBatches> {
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
    let context = session_context(Arc::new(NotRun))?;
    // a region returns each review's id, its primary key
    let reviews = text_table(
        &["id", "body"],
        &[&["r1", "A moving ending."], &["r2", "Too long."]],
    )?
    .with_constraints(Constraints::new_unverified(vec![Constraint::PrimaryKey(
        vec![0],
    )]));
    context.register_table("reviews", Arc::new(reviews))?;
    context.register_table(
        "aspects",
        Arc::new(text_table(&["aspect"], &[&["ending"], &["pacing"]])?),
    )?;
    Ok(context)
}

async fn optimized(context: &SessionContext, query: &str) -> Result<LogicalPlan> {
    claimed(context, context.sql(query).await?.into_optimized_plan()?)
}

/// An optimized plan with its model regions claimed, as the query
/// planner sees it.
fn claimed(context: &SessionContext, plan: LogicalPlan) -> Result<LogicalPlan> {
    quail_datafusion::claim_model_regions(plan, &context.state())
}

/// Each model region of a plan, each checked to derive its operators.
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
    let turn = (String::new(), String::new());
    for region in &found {
        if let Err(error) = quail_datafusion::derive::derive_region(region, &turn, "ai-if") {
            panic!("{}\n{}", region.plan.display_indent(), error.message());
        }
    }
    found
}

use datafusion::common::tree_node::TreeNode;

const FILTER_QUERY: &str = "SELECT r.id FROM reviews r \
    WHERE AI.IF(PROMPT('Does this review discuss the ending?\n\n{0}', r.body))";

#[tokio::test]
async fn outer_ai_on_predicates_plan_without_leaving_calls_in_datafusion() -> Result<()> {
    let context = context()?;
    for join_type in ["LEFT", "RIGHT", "FULL"] {
        let query = format!(
            "SELECT r.body, a.aspect FROM reviews r {join_type} JOIN aspects a \
             ON AI.IF(PROMPT('Does {{0}} discuss {{1}}?', r.body, a.aspect))"
        );
        let plan = optimized(&context, &query).await?;
        assert_eq!(regions(&plan).len(), 1, "{}", plan.display_indent());
        assert_eq!(plan.schema().fields().len(), 2);
        context.sql(&query).await?.create_physical_plan().await?;
    }
    Ok(())
}

#[tokio::test]
async fn outer_ai_joins_precompute_nullable_scalar_on_keys() -> Result<()> {
    let context = context()?;
    for join_type in ["LEFT", "RIGHT", "FULL"] {
        let query = format!(
            "SELECT r.body, a.aspect FROM reviews r {join_type} JOIN aspects a \
             ON NULLIF(r.body, 'Too long.') = a.aspect \
             AND AI.IF(PROMPT('Does {{0}} discuss {{1}}?', r.body, a.aspect))"
        );
        let plan = optimized(&context, &query).await?;
        assert_eq!(regions(&plan).len(), 1, "{}", plan.display_indent());
        context.sql(&query).await?.create_physical_plan().await?;
    }
    Ok(())
}

#[tokio::test]
async fn conjunctive_ai_on_predicates_plan_without_scalar_model_udfs() -> Result<()> {
    let context = context()?;
    for join_type in ["INNER", "LEFT", "RIGHT", "FULL"] {
        let query = format!(
            "SELECT r.body, a.aspect FROM reviews r {join_type} JOIN aspects a \
             ON AI.IF(PROMPT('first {{0}} {{1}}', r.body, a.aspect)) \
             AND AI.IF(PROMPT('second {{0}} {{1}}', r.body, a.aspect))"
        );
        let plan = optimized(&context, &query).await?;
        assert!(!regions(&plan).is_empty(), "{}", plan.display_indent());
        assert_eq!(plan.schema().fields().len(), 2);
        context.sql(&query).await?.create_physical_plan().await?;
    }
    Ok(())
}

#[test]
fn malformed_environment_options_return_errors() {
    const CHILD: &str = "QUAIL_TEST_INVALID_SESSION_ENV";
    if std::env::var_os(CHILD).is_some() {
        let error = session_context(Arc::new(NotRun))
            .err()
            .expect("invalid environment");
        assert!(error.to_string().contains("not-a-number"), "{error}");
        return;
    }
    // Isolate the environment from parallel tests without mutating this process.
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "claim::malformed_environment_options_return_errors",
        ])
        .env(CHILD, "1")
        .env("DATAFUSION_EXECUTION_TARGET_PARTITIONS", "not-a-number")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

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
async fn datafusion_tests_score_values_with_other_expressions() -> Result<()> {
    let context = context()?;
    for predicate in [
        "AI.SCORE(PROMPT('Rate {0}', r.body)) > length(r.body) / 100.0",
        "AI.SCORE(PROMPT('Rate {0}', r.body)) > 0.5 OR r.id = 'r0'",
    ] {
        let frame = context
            .sql(&format!(
                "SELECT r.id FROM reviews r WHERE {predicate} ORDER BY r.id"
            ))
            .await?;
        let plan = claimed(&context, frame.clone().into_optimized_plan()?)?;
        assert_eq!(regions(&plan).len(), 1, "{}", plan.display_indent());
        frame.create_physical_plan().await?;
    }
    Ok(())
}

#[tokio::test]
async fn a_scalar_subquery_claims_its_model_work() -> Result<()> {
    let context = context()?;
    let frame = context
        .sql(
            "SELECT (SELECT COUNT(*) FROM reviews r WHERE AI.IF(PROMPT('Keep {0}?', r.body))) AS n",
        )
        .await?;
    frame.create_physical_plan().await?;
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
    let query = "SELECT AI.CLASSIFY(PROMPT('Tone of {0}?', r.body), ['good', 'bad']) AS label, \
        COUNT(*) AS n FROM reviews r GROUP BY label ORDER BY n DESC, label";
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
    // the classification's column keeps the name SELECT gives it
    assert!(
        inner
            .iter()
            .any(|line| line.starts_with("AiNode: label = ai_classify")),
        "{inner:?}"
    );
    Ok(())
}

#[tokio::test]
async fn an_unnamed_classification_keeps_its_sql_output_name() -> Result<()> {
    let context = context()?;
    for function in ["AI.CLASSIFY", "ai_classify"] {
        let query = format!(
            "SELECT {function}(PROMPT('Tone of {{0}}?', r.body), ['good', 'bad']) \
             FROM reviews r"
        );
        let frame = context.sql(&query).await?;
        let name = frame.schema().field(0).name().clone();
        let plan = claimed(&context, frame.into_optimized_plan()?)?;
        assert_eq!(plan.schema().field(0).name(), &name);
        let found = regions(&plan);
        assert_eq!(found.len(), 1);
        let (operators, _) = quail_datafusion::derive::derive_region(
            &found[0],
            &(String::new(), String::new()),
            "ai-if",
        )
        .unwrap();
        assert_eq!(operators.classifies.len(), 1);
        assert!(operators.filters.is_empty());
    }
    Ok(())
}

#[tokio::test]
async fn a_projected_ai_if_returns_a_boolean_without_filtering_rows() -> Result<()> {
    let context = context()?;
    for expression in [
        "AI.IF(PROMPT('Does this discuss the ending? {0}', r.body)) AS has_ending",
        "AI.IF(PROMPT('Does this discuss the ending? {0}', r.body))",
        "ai_filter(PROMPT('Does this discuss the ending? {0}', r.body))",
    ] {
        let query = format!("SELECT r.id, {expression} FROM reviews r");
        let frame = context.sql(&query).await?;
        let name = frame.schema().field(1).name().clone();
        let plan = claimed(&context, frame.into_optimized_plan()?)?;
        assert_eq!(plan.schema().field(1).name(), &name);
        assert_eq!(plan.schema().field(1).data_type(), &DataType::Boolean);
        let found = regions(&plan);
        assert_eq!(found.len(), 1);
        let (operators, _) = quail_datafusion::derive::derive_region(
            &found[0],
            &(String::new(), String::new()),
            "ai-if",
        )
        .unwrap();
        let predicates: Vec<_> = operators
            .filters
            .iter()
            .flat_map(|(_, predicates)| predicates)
            .collect();
        assert_eq!(predicates.len(), 1);
        let answer = predicates[0].answer.as_ref().expect("a retained answer");
        assert!(
            operators
                .scans
                .iter()
                .all(|scan| !scan.columns.contains(answer))
        );
        assert!(
            quail_planner::results::source_columns(&operators)
                .iter()
                .all(|(name, _)| name != answer)
        );
        assert!(
            quail_planner::results::answer_columns(&operators)
                .iter()
                .any(|(name, _, kind, _, _)| name == answer && kind == "boolean")
        );
        assert!(predicates[0].alternative.is_none());
        assert!(predicates[0].guard.is_none());
    }
    Ok(())
}

#[tokio::test]
async fn a_negated_ai_filter_is_one_filter_that_keeps_the_no_answers() -> Result<()> {
    let context = context()?;
    let query = "SELECT r.id FROM reviews r \
        WHERE NOT AI.IF(PROMPT('Does this discuss the ending? {0}', r.body))";
    let plan = optimized(&context, query).await?;
    let found = regions(&plan);
    assert_eq!(found.len(), 1);
    let (operators, _) = quail_datafusion::derive::derive_region(
        &found[0],
        &(String::new(), String::new()),
        "ai-if",
    )
    .unwrap();
    let predicates: Vec<_> = operators
        .filters
        .iter()
        .flat_map(|(_, predicates)| predicates)
        .collect();
    assert_eq!(predicates.len(), 1);
    assert!(predicates[0].negated && predicates[0].answer.is_none());
    assert!(!lines(&plan).iter().any(|line| line.starts_with("Filter:")));
    Ok(())
}

#[tokio::test]
async fn ai_if_can_read_a_literal_without_a_from_clause() -> Result<()> {
    let context = context()?;
    let query = "SELECT AI.IF(PROMPT('Does this discuss the ending? {0}', 'hello')) AS answer";
    let plan = optimized(&context, query).await?;
    assert_eq!(plan.schema().field(0).name(), "answer");
    assert_eq!(plan.schema().field(0).data_type(), &DataType::Boolean);
    let found = regions(&plan);
    assert_eq!(found.len(), 1);
    Ok(())
}

#[tokio::test]
async fn an_ai_join_stays_in_one_region_under_its_distinct() -> Result<()> {
    let context = context()?;
    let query = "SELECT DISTINCT r.id FROM reviews r JOIN aspects a \
        ON AI.IF(PROMPT('Does the review discuss {1}?\n\n{0}', r.body, a.aspect))";
    let plan = optimized(&context, query).await?;
    let found = regions(&plan);
    assert_eq!(found.len(), 1, "{}", plan.display_indent());
    let inner = found[0].plan.display_indent().to_string();
    assert!(!inner.contains("Distinct"), "{inner}");
    assert!(inner.contains("Join"), "{inner}");
    // DataFusion's DISTINCT, a GROUP BY, reads the region's rows
    let shape = lines(&plan);
    assert!(
        shape.iter().any(|line| line.starts_with("Aggregate:")),
        "{shape:?}"
    );
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
    assert!(shape[1].starts_with("AiNode: semi"), "{shape:?}");
    // the gate's table is an input DataFusion scans for the region
    assert_eq!(
        shape.last().map(String::as_str),
        Some("TableScan: aspects@a")
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
    // one node per AI call, the first written lowest
    assert!(plan[1].starts_with("AiNode: "), "{plan:?}");
    assert!(plan[1].contains("Long?"), "{plan:?}");
    assert!(plan[2].starts_with("AiNode: "), "{plan:?}");
    assert!(plan[2].contains("Kind?"), "{plan:?}");
    // the plain test runs in DataFusion, in the region's input
    assert!(!plan.iter().any(|line| line.contains("r9")), "{plan:?}");
    assert!(
        region
            .inputs
            .iter()
            .any(|input| lines(input).iter().any(|line| line.contains("r9")))
    );
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
        plan[1].starts_with("AiNode: __quail_answer_1 = ai_filter"),
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
        plan[1].starts_with("AiNode: tone = ai_classify"),
        "{plan:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_test_of_a_label_is_absorbed_into_its_classifications_filter() -> Result<()> {
    let context = context()?;
    let query = "SELECT r.id, AI.CLASSIFY(PROMPT('Tone of {0}?', r.body), ['good', 'bad']) \
        AS tone FROM reviews r \
        WHERE AI.CLASSIFY(PROMPT('Tone of {0}?', r.body), ['good', 'bad']) = 'good'";
    let [region] =
        <[ModelRegion; 1]>::try_from(regions(&optimized(&context, query).await?)).unwrap();
    let plan = lines(&region.plan);
    // one classification serves the test and the SELECT list
    let columns: Vec<&String> = plan
        .iter()
        .filter(|line| line.starts_with("AiNode"))
        .collect();
    let [column] = columns.as_slice() else {
        panic!("{plan:?}");
    };
    assert!(column.starts_with("AiNode: tone = ai_classify"), "{plan:?}");
    assert!(
        column.ends_with(", filter: tone = Utf8(\"good\")"),
        "{plan:?}"
    );
    assert!(
        !plan.iter().any(|line| line.starts_with("Filter")),
        "{plan:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_plain_filter_runs_in_datafusion_as_the_regions_input() -> Result<()> {
    let context = context()?;
    let query = "SELECT r.id FROM reviews r WHERE r.id <> 'r9' \
        AND AI.IF(PROMPT('Is {0} kind?', r.body))";
    let [region] =
        <[ModelRegion; 1]>::try_from(regions(&optimized(&context, query).await?)).unwrap();
    // the model reads only the rows the plain test keeps
    assert!(
        !lines(&region.plan).iter().any(|line| line.contains("r9")),
        "{:?}",
        lines(&region.plan)
    );
    let [input] = <[LogicalPlan; 1]>::try_from(region.inputs.clone()).unwrap();
    assert!(
        lines(&input).iter().any(|line| line.contains("r9")),
        "{:?}",
        lines(&input)
    );
    Ok(())
}

#[tokio::test]
async fn a_plain_expression_in_select_runs_in_datafusion_as_an_input_column() -> Result<()> {
    let context = context()?;
    let query = "SELECT r.id, UPPER(r.body) AS loud FROM reviews r \
        WHERE AI.IF(PROMPT('Is {0} kind?', r.body))";
    let [region] =
        <[ModelRegion; 1]>::try_from(regions(&optimized(&context, query).await?)).unwrap();
    // the region reads the computed column; DataFusion computes it
    assert!(
        !lines(&region.plan)
            .iter()
            .any(|line| line.contains("upper(")),
        "{:?}",
        lines(&region.plan)
    );
    let [input] = <[LogicalPlan; 1]>::try_from(region.inputs.clone()).unwrap();
    assert!(
        lines(&input).iter().any(|line| line.contains("upper(")),
        "{:?}",
        lines(&input)
    );
    Ok(())
}

#[tokio::test]
async fn a_table_without_a_key_is_numbered_and_joined_back_on_its_rows() -> Result<()> {
    let context = context()?;
    let query = "SELECT r.id, a.aspect FROM reviews r \
        JOIN aspects a ON AI.IF(PROMPT('Does {0} discuss {1}?', r.body, a.aspect))";
    let plan = optimized(&context, query).await?;
    let [region] = <[ModelRegion; 1]>::try_from(regions(&plan)).unwrap();
    // the region returns the review's key and the aspect's row number
    assert_eq!(
        lines(&region.plan)[0],
        "Projection: r.id, a.__quail_row",
        "{}",
        plan.display_indent()
    );
    // the join above reads the aspects through the region's shared input
    let text = plan.display_indent().to_string();
    assert!(
        text.contains("Inner Join: __quail_key_a = a.__quail_row"),
        "{text}"
    );
    let shared: Vec<&str> = lines(&plan)
        .iter()
        .filter(|line| line.starts_with("SharedInput: "))
        .map(|line| text.matches(line.as_str()).next().unwrap_or_default())
        .collect();
    assert_eq!(shared.len(), 2, "{text}");
    assert_eq!(shared[0], shared[1], "{text}");
    Ok(())
}
