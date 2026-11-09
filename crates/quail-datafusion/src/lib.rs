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
pub mod similar;
mod substring;

use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::common::{Result, ScalarValue};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::{SessionConfig, SessionContext, SessionState};
use datafusion::logical_expr::{Expr, LogicalPlan};

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

/// Quail's name for each column `plan` returns: `alias.column` for a
/// table's column, as in `r.id`; `r.details.body` for a struct field
/// the SELECT list reads without an `AS` name; and otherwise the name
/// DataFusion gives it, such as an `AS` name, as written.
pub fn quail_column_names(plan: &LogicalPlan) -> Vec<String> {
    let selected = select_list(plan);
    plan.schema()
        .iter()
        .enumerate()
        .map(|(index, (qualifier, field))| {
            let path = selected
                .and_then(|(exprs, input)| Some((exprs.get(index)?, input)))
                .and_then(|(expr, input)| field_path(expr, input));
            match (path, qualifier) {
                (Some(path), _) => path,
                (None, Some(qualifier)) => format!("{qualifier}.{}", field.name()),
                (None, None) => field.name().clone(),
            }
        })
        .collect()
}

/// The SELECT list whose columns `plan` returns as they are, through the
/// operators that keep their input's columns, and the plan it reads.
fn select_list(plan: &LogicalPlan) -> Option<(&[Expr], &LogicalPlan)> {
    match plan {
        LogicalPlan::Projection(projection) => Some((&projection.expr, &projection.input)),
        LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Filter(_)
        | LogicalPlan::Distinct(datafusion::logical_expr::Distinct::All(_)) => {
            select_list(plan.inputs()[0])
        }
        _ => None,
    }
}

/// `column.field.inner` for a SELECT list item that reads a struct
/// field of a column, `get_field(column, 'field', 'inner')`, without an
/// `AS` name, or that reads such a field `input` computes, as a `GROUP
/// BY` key; None for any other item.
fn field_path(expr: &Expr, input: &LogicalPlan) -> Option<String> {
    match expr {
        Expr::ScalarFunction(call) if call.name() == "get_field" => {
            let (base, names) = call.args.split_first()?;
            let mut path = match base {
                Expr::Column(column) => column.flat_name(),
                nested => field_path(nested, input)?,
            };
            for name in names {
                let Expr::Literal(ScalarValue::Utf8(Some(name)), _) = name else {
                    return None;
                };
                path.push('.');
                path.push_str(name);
            }
            Some(path)
        }
        Expr::Column(column) if column.relation.is_none() => {
            let (computed, below) = computed_by(input, &column.name)?;
            field_path(computed, below)
        }
        _ => None,
    }
}

/// The expression of `plan`, or of a plan below it that keeps its
/// columns, that computes the column `name`, and the plan it reads.
fn computed_by<'a>(plan: &'a LogicalPlan, name: &str) -> Option<(&'a Expr, &'a LogicalPlan)> {
    let named = |expr: &&Expr| expr.schema_name().to_string() == name;
    match plan {
        LogicalPlan::Aggregate(aggregate) => aggregate
            .group_expr
            .iter()
            .find(named)
            .map(|expr| (expr, aggregate.input.as_ref())),
        LogicalPlan::Projection(projection) => projection
            .expr
            .iter()
            .find(named)
            .map(|expr| (expr, projection.input.as_ref())),
        LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Filter(_)
        | LogicalPlan::Window(_)
        | LogicalPlan::Distinct(_) => computed_by(plan.inputs()[0], name),
        _ => None,
    }
}

/// An Arrow schema of `plan`'s columns with Quail's names.
pub fn quail_schema(plan: &LogicalPlan) -> SchemaRef {
    let fields: Vec<Field> = plan
        .schema()
        .fields()
        .iter()
        .zip(quail_column_names(plan))
        .map(|(field, name)| field.as_ref().clone().with_name(name))
        .collect();
    Arc::new(Schema::new(fields))
}
