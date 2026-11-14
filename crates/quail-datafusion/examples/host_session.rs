//! Add Quail to a DataFusion session that has its own function and rule.
//!
//! The host here is a program with its own DataFusion extensions: a
//! function `shout` and an optimizer rule that rewrites `shout(x)` to
//! `upper(x)`. `QuailExtension::register` adds the AI functions beside
//! them, and one SQL statement uses both.
//!
//! `FirstRow` stands in for the model: it keeps the first row of each
//! region. A program with a GPU passes `quail::Executor` instead:
//! `QuailExtension::new(Arc::new(quail::Executor::open(&config)?))`.
//!
//! Run with `cargo run --example host_session`.

use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::functions::string::expr_fn::upper;
use datafusion::logical_expr::{ColumnarValue, Expr, LogicalPlan, Volatility, create_udf};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::prelude::SessionContext;
use quail_datafusion::{
    InputStream, ModelRegion, QuailExtension, RegionBatches, RegionHint, RegionRunner,
    collect_inputs,
};

/// The query: a host function and an AI function in one statement.
pub const SQL: &str = "SELECT r.id, shout(r.body) AS loud FROM reviews r \
                       WHERE AI.IF(PROMPT('Is the review negative? {0}', r.body))";

/// Stands in for Quail's executor: the model keeps the first row.
#[derive(Debug)]
struct FirstRow;

impl RegionRunner for FirstRow {
    fn run(
        &self,
        region: &ModelRegion,
        inputs: Vec<InputStream>,
        _fetch: Option<usize>,
        _hint: &RegionHint,
    ) -> Result<RegionBatches> {
        let schema = Arc::new(region.plan.schema().as_arrow().clone());
        let (_, rows) = collect_inputs(inputs)?.remove(0);
        let first = rows[0].slice(0, 1);
        // the region's columns are its input's, by name
        let columns = schema
            .fields()
            .iter()
            .map(|field| {
                first
                    .column_by_name(field.name())
                    .cloned()
                    .expect("an input column")
            })
            .collect();
        let batch = RecordBatch::try_new(schema, columns)?;
        Ok(Box::new(std::iter::once(Ok(batch))))
    }
}

/// The host's optimizer rule: `shout(x)` is `upper(x)`.
#[derive(Debug)]
struct ShoutIsUpper;

impl OptimizerRule for ShoutIsUpper {
    fn name(&self) -> &str {
        "shout_is_upper"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform(|node| {
            node.map_expressions(|expr| {
                expr.transform(|found| match found {
                    Expr::ScalarFunction(call) if call.name() == "shout" => {
                        Ok(Transformed::yes(upper(call.args[0].clone())))
                    }
                    other => Ok(Transformed::no(other)),
                })
            })
        })
    }
}

/// A host session with `shout`, the host's rule, and Quail.
///
/// # Errors
///
/// DataFusion or Quail refuses the session.
pub fn host_context() -> Result<SessionContext> {
    let shout = create_udf(
        "shout",
        vec![DataType::Utf8],
        DataType::Utf8,
        Volatility::Immutable,
        // the host's rule replaces every call before a row reaches it
        Arc::new(|args: &[ColumnarValue]| Ok(args[0].clone())),
    );
    let host = SessionStateBuilder::new()
        .with_default_features()
        .with_optimizer_rule(Arc::new(ShoutIsUpper))
        .build();
    let state = QuailExtension::new(Arc::new(FirstRow)).register(host)?;
    let context = SessionContext::new_with_state(state);
    context.register_udf(shout);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let reviews = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["r1", "r2", "r3"])),
            Arc::new(StringArray::from(vec!["Too slow.", "Lovely.", "Fine."])),
        ],
    )?;
    context.register_table(
        "reviews",
        Arc::new(MemTable::try_new(schema, vec![vec![reviews]])?),
    )?;
    Ok(context)
}

/// Run [`SQL`] in the host session and return the optimized plan and the
/// rows, as text.
///
/// # Errors
///
/// The query fails.
pub async fn run() -> Result<(String, String)> {
    let context = host_context()?;
    let frame = context.sql(SQL).await?;
    let plan = frame.clone().into_optimized_plan()?;
    let rows = frame.collect().await?;
    Ok((
        plan.display_indent().to_string(),
        pretty_format_batches(&rows)?.to_string(),
    ))
}

#[allow(dead_code)]
#[tokio::main]
async fn main() -> Result<()> {
    let (plan, rows) = run().await?;
    println!("{plan}\n{rows}");
    Ok(())
}
