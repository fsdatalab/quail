#[cfg(test)]
use quail_dylib as _;

pub mod ai;
mod extension;
pub mod functions;
mod plan;
pub mod region;
pub mod rules;
mod shared;

use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::common::{Result, ScalarValue};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::{SessionConfig, SessionContext, SessionState};
use datafusion::logical_expr::{Expr, LogicalPlan};

pub use extension::QuailExtension;
pub use functions::field_access::StructFieldAccess;
pub use functions::regexps::SnowflakeRegexpLike;
pub use functions::{AI_FUNCTION_NAMES, is_ai_call, quail_functions};
pub use quail_planner::constants::SPAN_SUFFIX;

pub const PLANNING_STACK_BYTES: usize = 64 << 20;

pub fn external(error: quail_planner::error::CompileError) -> datafusion::error::DataFusionError {
    datafusion::error::DataFusionError::External(Box::new(error))
}
pub use region::ModelRegion;
pub use region::claim::claim_model_regions;
pub use region::derive;
pub use region::exec::{
    InputBatches, InputRows, InputStream, ModelRegionExec, ModelRegionPlanner, RegionBatches,
    RegionRunner, collect_inputs,
};
pub use region::filter_guard::GuardRegionFilters;
pub use region::identified::IdentifiedTable;
pub use region::region_hint::{RegionColumn, RegionColumns, RegionHint};
pub use rules::derive::{Derive, DeriveOption};
pub use rules::lower::{LowerAiCalls, lower_ai_calls};
pub use rules::placement::PlaceAiNodes;
pub use rules::repeated::MergeRepeatedAiCalls;
pub use rules::{default_rule_names, logical_rule};
pub use shared::{ROW_COLUMN, SharedInput, SharedInputExec};

pub fn session_state(runner: Arc<dyn RegionRunner>) -> Result<SessionState> {
    session_state_with_rules(runner, default_rule_names())
}

pub fn session_state_with_rules(
    runner: Arc<dyn RegionRunner>,
    rules: Vec<String>,
) -> Result<SessionState> {
    QuailExtension::new(runner)
        .with_logical_rules(rules)?
        .register(base_state()?)
}

pub fn base_state() -> Result<SessionState> {
    let config = SessionConfig::from_env()?.set_str("datafusion.sql_parser.dialect", "Snowflake");
    Ok(SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .build())
}

pub fn session_context(runner: Arc<dyn RegionRunner>) -> Result<SessionContext> {
    Ok(SessionContext::new_with_state(session_state(runner)?))
}

pub fn session_context_with_rules(
    runner: Arc<dyn RegionRunner>,
    rules: Vec<String>,
) -> Result<SessionContext> {
    Ok(SessionContext::new_with_state(session_state_with_rules(
        runner, rules,
    )?))
}

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
