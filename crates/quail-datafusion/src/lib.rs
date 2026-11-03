//! Quail as a DataFusion extension.
//!
//! A query's AI function calls, and the operators around them that
//! Quail plans, become one `ModelRegion` node. DataFusion plans and
//! runs everything else; a [`RegionRunner`] plans and runs each model
//! region.
//!
//! ```ignore
//! let context = quail_datafusion::session_context(runner);
//! context.register_table("reviews", reviews)?;
//! let rows = quail_datafusion::sql(&context, "SELECT r.id FROM reviews r \
//!     WHERE AI.IF(PROMPT('Does the review discuss the ending?\n\n{0}', r.body))")
//!     .await?
//!     .collect()
//!     .await?;
//! ```

mod aliases;
mod claim;
mod exec;
mod functions;
mod region;
mod unparse;

use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::common::{DFSchema, Result};
use datafusion::dataframe::DataFrame;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::{SessionConfig, SessionContext, SessionState};
use datafusion::optimizer::{Optimizer, OptimizerRule};
use datafusion::sql::parser::Statement;

pub use aliases::{
    SPAN_SUFFIX, add_extract_spans, inline_select_aliases, remove_select_calls,
    restore_select_aliases,
};
pub use claim::ClaimModelRegions;
pub use exec::{ModelRegionExec, ModelRegionPlanner, RegionRunner};
pub use functions::{AI_FUNCTION_NAMES, is_ai_call, normalize_ai_calls, quail_functions};
pub use region::ModelRegion;
pub use unparse::region_sql;

/// The optimizer rule after which `ClaimModelRegions` runs.
///
/// Rules up to and including this one rewrite the query without
/// choosing an order: subqueries become joins and join conditions
/// split into keys and filters.
const CLAIM_AFTER: &str = "extract_equijoin_predicate";

/// DataFusion rules that must not see an operator Quail plans, so
/// they run after `ClaimModelRegions`.
const AFTER_CLAIM: [&str; 1] = ["replace_distinct_aggregate"];

/// Return DataFusion's optimizer rules with `ClaimModelRegions` placed
/// among them.
pub fn optimizer_rules() -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    let mut rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> = Vec::new();
    let mut delayed = Vec::new();
    for rule in Optimizer::new().rules {
        if AFTER_CLAIM.contains(&rule.name()) {
            delayed.push(rule);
            continue;
        }
        let claim_next = rule.name() == CLAIM_AFTER;
        rules.push(rule);
        if claim_next {
            rules.push(Arc::new(ClaimModelRegions::new()));
            rules.append(&mut delayed);
        }
    }
    rules.append(&mut delayed);
    rules
}

/// Build a DataFusion session state with Quail registered.
///
/// SQL is read in the Snowflake dialect, as Quail's `Session.sql`
/// reads it; `SET datafusion.sql_parser.dialect` changes it.
pub fn session_state(runner: Arc<dyn RegionRunner>) -> SessionState {
    let config = SessionConfig::new().set_str("datafusion.sql_parser.dialect", "Snowflake");
    SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_optimizer_rules(optimizer_rules())
        .with_query_planner(Arc::new(exec::QuailQueryPlanner::new(runner)))
        .build()
}

/// Build a DataFusion session context with Quail registered.
pub fn session_context(runner: Arc<dyn RegionRunner>) -> SessionContext {
    let context = SessionContext::new_with_state(session_state(runner));
    for function in quail_functions() {
        context.register_udf(function.as_ref().clone());
    }
    context
}

/// Plan a SQL query that may call AI functions as `AI.<name>(...)`.
///
/// The query is parsed in the session's SQL dialect. Before DataFusion
/// binds it, each extraction gets its answer positions column (see
/// [`add_extract_spans`]), and `SELECT` aliases named in `WHERE` are
/// replaced by their expressions (see [`inline_select_aliases`]).
pub async fn sql(context: &SessionContext, query: &str) -> Result<DataFrame> {
    let state = context.state();
    let dialect = state.config().options().sql_parser.dialect;
    let mut statement = state.sql_to_statement(&normalize_ai_calls(query), &dialect)?;
    if let Statement::Statement(inner) = &mut statement {
        add_extract_spans(inner);
        inline_select_aliases(inner);
    }
    let plan = state.statement_to_plan(statement).await?;
    Ok(DataFrame::new(state, plan))
}

/// Quail's name for each column of a schema: `alias.column` for a
/// table's column, as in `r.id`, and the bare name otherwise.
pub fn quail_column_names(schema: &DFSchema) -> Vec<String> {
    schema
        .iter()
        .map(|(qualifier, field)| match qualifier {
            Some(qualifier) => format!("{qualifier}.{}", field.name()),
            None => field.name().clone(),
        })
        .collect()
}

/// An Arrow schema with Quail's column names and the schema's types.
pub fn quail_schema(schema: &DFSchema) -> SchemaRef {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .zip(quail_column_names(schema))
        .map(|(field, name)| field.as_ref().clone().with_name(name))
        .collect();
    Arc::new(Schema::new(fields))
}
