//! Quail as a DataFusion extension.
//!
//! A query's AI function calls become `AiNode`s, which DataFusion's
//! optimizer rules move and prune around as they do any other node (see
//! [`rules`]). Then the `AiNode`s, and the operators around them that
//! Quail plans, become one `ModelRegion` node. DataFusion plans and runs
//! everything else; a [`RegionRunner`] plans and runs each model region.
//!
//! ```ignore
//! let context = quail_datafusion::session_context(runner);
//! context.register_table("reviews", reviews)?;
//! let rows = context
//!     .sql("SELECT r.id FROM reviews r \
//!         WHERE AI.IF(PROMPT('Does the review discuss the ending?\n\n{0}', r.body))")
//!     .await?
//!     .collect()
//!     .await?;
//! ```

mod absorb;
pub mod ai;
mod binding;
mod claim;
pub mod derive;
mod distinct;
mod exec;
pub mod expressions;
mod extension;
mod field_access;
mod functions;
mod guards;
mod hints;
mod identified;
mod join_back;
mod keyed;
mod lift;
mod limit;
mod lower;
mod or_guards;
pub mod partial;
mod placement;
mod projection;
mod regexps;
mod region;
pub mod region_hint;
mod repeated;
pub mod rules;
mod shared;
mod substring;

use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::common::{DFSchema, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::{SessionConfig, SessionContext, SessionState};

pub use absorb::AbsorbIntoAiNodes;
pub use claim::ClaimModelRegions;
pub use distinct::{EliminateDistinct, PushDownDistinct};
pub use exec::{
    InputBatches, InputRows, InputStream, ModelRegionExec, ModelRegionPlanner, RegionBatches,
    RegionRunner, collect_inputs,
};
pub use extension::QuailExtension;
pub use field_access::StructFieldAccess;
pub use functions::{AI_FUNCTION_NAMES, SPAN_SUFFIX, is_ai_call, quail_functions};
pub use guards::PushDownCaseGuards;
pub use identified::IdentifiedTable;
pub use keyed::{PushDownCountThreshold, PushDownSemiJoinFetch};
pub use limit::PushDownLimitIntoAiNodes;
pub use lower::{LowerAiCalls, lower_ai_calls};
pub use or_guards::PushDownOrGuards;
pub use placement::PlaceAiNodes;
pub use projection::PushDownProjectionIntoAiNodes;
pub use regexps::SnowflakeRegexpLike;
pub use region::ModelRegion;
pub use region_hint::{RegionColumn, RegionColumns, RegionHint};
pub use repeated::MergeRepeatedAiCalls;
pub use rules::{LOGICAL_RULES, logical_rule};
pub use shared::{ROW_COLUMN, SharedInput, SharedInputExec};
pub use substring::PushDownSubstringPrechecks;

/// Build a DataFusion session state with Quail registered.
///
/// SQL is read in the Snowflake dialect, as Quail's `Session.sql`
/// reads it; `SET datafusion.sql_parser.dialect` changes it.
pub fn session_state(runner: Arc<dyn RegionRunner>) -> SessionState {
    session_state_with_rules(
        runner,
        LOGICAL_RULES.iter().map(ToString::to_string).collect(),
    )
    .expect("Quail's rules and functions register on a new session")
}

/// Build a DataFusion session state with Quail registered, whose
/// optimizer runs `rules`, names [`logical_rule`] knows, in order, then
/// [`ClaimModelRegions`].
///
/// DataFusion's `DATAFUSION_*` environment variables set its options,
/// such as `DATAFUSION_EXECUTION_TARGET_PARTITIONS`.
///
/// # Errors
///
/// A rule name is unknown, or an environment variable holds a value its
/// option does not take.
pub fn session_state_with_rules(
    runner: Arc<dyn RegionRunner>,
    rules: Vec<String>,
) -> Result<SessionState> {
    let config = SessionConfig::from_env()?.set_str("datafusion.sql_parser.dialect", "Snowflake");
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .build();
    QuailExtension::new(runner)
        .with_logical_rules(rules)?
        .register(state)
}

/// Build a DataFusion session context with Quail registered.
pub fn session_context(runner: Arc<dyn RegionRunner>) -> SessionContext {
    SessionContext::new_with_state(session_state(runner))
}

/// Build a DataFusion session context with Quail registered, whose
/// optimizer runs `rules`, as [`session_state_with_rules`] explains.
///
/// # Errors
///
/// A rule name is unknown.
pub fn session_context_with_rules(
    runner: Arc<dyn RegionRunner>,
    rules: Vec<String>,
) -> Result<SessionContext> {
    Ok(SessionContext::new_with_state(session_state_with_rules(
        runner, rules,
    )?))
}

/// Quail's name for each column of a schema: `alias.column` for a
/// table's column, as in `r.id`, and the bare name otherwise.
pub fn quail_column_names(schema: &DFSchema) -> Vec<String> {
    schema
        .iter()
        .map(|(qualifier, field)| match qualifier {
            Some(qualifier) => format!("{qualifier}.{}", field.name()),
            None => quail_name(field.name()),
        })
        .collect()
}

/// Quail's name for a column DataFusion names by its expression: a
/// struct field `r.details[body]` is `r.details.body`. Any other name is
/// unchanged.
fn quail_name(name: &str) -> String {
    let Some((column, fields)) = name.split_once('[') else {
        return name.to_string();
    };
    let word = |text: &str| {
        !text.is_empty()
            && text
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
    };
    let fields: Vec<&str> = fields
        .strip_suffix(']')
        .map(|inner| inner.split("][").collect())
        .unwrap_or_default();
    if !word(column) || fields.is_empty() || !fields.iter().all(|field| word(field)) {
        return name.to_string();
    }
    format!("{column}.{}", fields.join("."))
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
