//! A host's own DataFusion session with Quail added to it.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::catalog::Session;
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::QueryPlanner;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use datafusion::prelude::SessionContext;
use quail_datafusion::{InputStream, ModelRegion, QuailExtension, RegionBatches, RegionRunner};

/// Stands in for Quail's executor: the model keeps the first review.
#[derive(Debug)]
struct KeepsFirstReview;

impl RegionRunner for KeepsFirstReview {
    fn run(
        &self,
        region: &ModelRegion,
        _inputs: Vec<InputStream>,
        _fetch: Option<usize>,
        _hint: &quail_datafusion::RegionHint,
    ) -> Result<RegionBatches> {
        let schema = Arc::new(region.plan.schema().as_arrow().clone());
        // one row: each id column, such as a row number, read from "0"
        let columns = schema
            .fields()
            .iter()
            .map(|field| {
                let value: Arc<dyn Array> = Arc::new(StringArray::from(vec!["0"]));
                datafusion::arrow::compute::cast(&value, field.data_type())
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let batch = RecordBatch::try_new(schema, columns)?;
        Ok(Box::new(std::iter::once(Ok(batch))))
    }
}

fn reviews() -> Result<MemTable> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["r1", "r2", "r3"])),
            Arc::new(StringArray::from(vec!["Too slow.", "Lovely.", "Fine."])),
        ],
    )?;
    MemTable::try_new(schema, vec![vec![batch]])
}

const QUERY: &str = "SELECT COUNT(*) AS n FROM reviews r \
    WHERE ai_filter(prompt('Is this review negative? {0}', r.body))";

async fn count(context: &SessionContext) -> Result<i64> {
    context.register_table("reviews", Arc::new(reviews()?))?;
    let batches = context.sql(QUERY).await?.collect().await?;
    let counts = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("a count");
    Ok(counts.value(0))
}

#[tokio::test]
async fn a_host_session_with_quail_runs_ai_functions_in_its_own_sql() -> Result<()> {
    let state = SessionStateBuilder::new().with_default_features().build();
    let state = QuailExtension::new(Arc::new(KeepsFirstReview)).register(state)?;
    // the host's DataFusion counts the rows the model region returns
    assert_eq!(count(&SessionContext::new_with_state(state)).await?, 1);
    Ok(())
}

/// A host's own query planner: it claims the model regions and has
/// Quail's extension planner.
struct HostPlanner {
    extensions: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>,
}

impl std::fmt::Debug for HostPlanner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HostPlanner")
    }
}

#[async_trait]
impl QueryPlanner for HostPlanner {
    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
        state: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let claimed = quail_datafusion::claim_model_regions(plan.clone(), state)?;
        DefaultPhysicalPlanner::with_extension_planners(self.extensions.clone())
            .create_physical_plan(&claimed, state)
            .await
    }
}

/// A host's own function rewrite, which changes nothing.
#[derive(Debug)]
struct HostRewrite;

impl datafusion::logical_expr::expr_rewriter::FunctionRewrite for HostRewrite {
    fn name(&self) -> &str {
        "host_rewrite"
    }

    fn rewrite(
        &self,
        expr: datafusion::logical_expr::Expr,
        _schema: &datafusion::common::DFSchema,
        _config: &datafusion::config::ConfigOptions,
    ) -> Result<datafusion::common::tree_node::Transformed<datafusion::logical_expr::Expr>> {
        Ok(datafusion::common::tree_node::Transformed::no(expr))
    }
}

#[test]
fn adding_quail_keeps_the_hosts_session_id_and_function_rewrites() -> Result<()> {
    use datafusion::execution::FunctionRegistry;
    let mut host = SessionStateBuilder::new()
        .with_default_features()
        .with_session_id("host".to_string())
        .build();
    host.register_function_rewrite(Arc::new(HostRewrite))?;
    let extension = QuailExtension::new(Arc::new(KeepsFirstReview));
    // adding Quail twice adds each of its hooks once
    let state = extension.register(extension.register(host)?)?;
    assert_eq!(state.session_id(), "host");
    let rewrites: Vec<&str> = state
        .analyzer()
        .function_rewrites
        .iter()
        .map(|rewrite| rewrite.name())
        .collect();
    assert_eq!(rewrites, ["host_rewrite", "snowflake_regexp_like"]);
    Ok(())
}

#[tokio::test]
async fn a_host_with_its_own_query_planner_claims_regions_and_adds_quails_planner() -> Result<()> {
    let quail = QuailExtension::new(Arc::new(KeepsFirstReview));
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_query_planner(Arc::new(HostPlanner {
            extensions: vec![quail.extension_planner()],
        }))
        .build();
    let state = quail.register_rules(state)?;
    assert_eq!(count(&SessionContext::new_with_state(state)).await?, 1);
    Ok(())
}

#[tokio::test]
async fn a_host_writes_ai_functions_by_their_dotted_names() -> Result<()> {
    let state = SessionStateBuilder::new().with_default_features().build();
    let state = QuailExtension::new(Arc::new(KeepsFirstReview)).register(state)?;
    let context = SessionContext::new_with_state(state);
    context.register_table("reviews", Arc::new(reviews()?))?;
    let batches = context
        .sql("SELECT COUNT(*) AS n FROM reviews r WHERE AI.IF(PROMPT('Negative? {0}', r.body))")
        .await?
        .collect()
        .await?;
    let counts = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("a count");
    assert_eq!(counts.value(0), 1);
    Ok(())
}

#[test]
fn an_expression_that_calls_a_hosts_function_is_refused_by_name() -> Result<()> {
    use datafusion::logical_expr::{ColumnarValue, Volatility, col, create_udf};
    use quail_datafusion::derive::expressions::{PhysicalExpression, expression};
    use quail_planner::physical::ColumnRef;

    // a host function Quail's executor does not have
    let shout = create_udf(
        "quail_test_shout",
        vec![DataType::Utf8],
        DataType::Utf8,
        Volatility::Immutable,
        Arc::new(|args: &[ColumnarValue]| Ok(args[0].clone())),
    );
    let mut state = SessionStateBuilder::new().with_default_features().build();
    datafusion::execution::FunctionRegistry::register_udf(&mut state, Arc::new(shout.clone()))?;
    QuailExtension::new(Arc::new(KeepsFirstReview)).register(state)?;
    let test = shout.call(vec![col("body")]);
    let encoded = expression(&test, |column| {
        let reference = ColumnRef {
            alias: "r".into(),
            provider: "reviews".into(),
            column: column.name.clone(),
        };
        Ok((reference, DataType::Utf8))
    })
    .expect("the test encodes");
    let refused = PhysicalExpression::new(&encoded).expect_err("an unknown function");
    assert!(refused.message().contains("quail_test_shout"), "{refused}");
    Ok(())
}
