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
pub mod bind;
mod claim;
mod exec;
mod functions;
mod joins;
mod lower;
mod optimize;
mod placement;
mod region;
pub mod semantic;
mod translate;
mod unnest;
mod unparse;

use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::common::{DFSchema, Result};
use datafusion::dataframe::DataFrame;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::{SessionConfig, SessionContext, SessionState};
use datafusion::optimizer::{Analyzer, AnalyzerRule};
use datafusion::sql::parser::Statement;

pub use aliases::{
    SPAN_SUFFIX, add_extract_spans, distinct_order_ordinals, inline_select_aliases,
    positional_ai_arguments, remove_select_calls, restore_select_aliases,
};
pub use claim::ClaimModelRegions;
pub use exec::{ModelRegionExec, ModelRegionPlanner, RegionRunner};
pub use functions::{
    AI_FUNCTION_NAMES, arrow_functions, brace_structs, is_ai_call, normalize_ai_calls,
    quail_functions,
};
pub use lower::{LowerAiCalls, lower_ai_calls};
pub use optimize::{OptimizeModelRegions, optimize_region};
pub use placement::PlaceSemanticNodes;
pub use region::ModelRegion;
pub use translate::translate_relation;
pub use unnest::{UNNEST_FIELD, UNNEST_TEST, take_unnests};
pub use unparse::region_sql;

/// Return DataFusion's analyzer rules with `ClaimModelRegions` and
/// `LowerAiCalls` first.
///
/// A region then holds the plan as DataFusion's SQL planner bound it,
/// before type coercion adds casts: its conditions as written, its ON
/// clauses whole, and an EXISTS gate as a subquery, as Quail's own SQL
/// front end binds them.
pub fn analyzer_rules() -> Vec<Arc<dyn AnalyzerRule + Send + Sync>> {
    let mut rules: Vec<Arc<dyn AnalyzerRule + Send + Sync>> =
        vec![Arc::new(ClaimModelRegions::new()), Arc::new(LowerAiCalls)];
    rules.extend(Analyzer::new().rules);
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
        .with_analyzer_rules(analyzer_rules())
        .with_optimizer_rule(Arc::new(OptimizeModelRegions))
        .with_query_planner(Arc::new(exec::QuailQueryPlanner::new(runner)))
        .build()
}

/// Build a DataFusion session context with Quail registered.
pub fn session_context(runner: Arc<dyn RegionRunner>) -> SessionContext {
    let context = SessionContext::new_with_state(session_state(runner));
    for function in quail_functions() {
        context.register_udf(function.as_ref().clone());
    }
    let state = context.state();
    let taken = |name: &str| state.scalar_functions().contains_key(name);
    for function in arrow_functions(&taken) {
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
    sql_in_dialect(context, query, None).await
}

/// Plan a SQL query as [`sql`] does, read in `dialect` (a DataFusion
/// dialect name, such as `BigQuery`) instead of the session's.
pub async fn sql_in_dialect(
    context: &SessionContext,
    query: &str,
    dialect: Option<&str>,
) -> Result<DataFrame> {
    let mut state = context.state();
    let dialect = match dialect {
        Some(name) => name.parse()?,
        None => state.config().options().sql_parser.dialect,
    };
    if matches!(dialect, datafusion::config::Dialect::BigQuery) {
        // BigQuery sorts NULLs first in ascending order, as the front end reads it
        state
            .config_mut()
            .options_mut()
            .sql_parser
            .default_null_ordering = "nulls_min".to_string();
    }
    let mut text = normalize_ai_calls(query);
    if matches!(dialect, datafusion::config::Dialect::BigQuery) {
        text = brace_structs(&text);
    }
    let mut statement = state.sql_to_statement(&text, &dialect)?;
    if let Statement::Statement(inner) = &mut statement {
        joins::flatten_joins(inner);
        take_unnests(inner);
        positional_ai_arguments(inner);
        add_extract_spans(inner);
        inline_select_aliases(inner);
        distinct_order_ordinals(inner);
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
