//! The logical optimizer rules of a session with Quail.
//!
//! DataFusion's optimizer runs its own rules over the whole plan, each
//! `AiNode` included, with Quail's rules among them. [`default_rule_names`]
//! gives the default order: DataFusion's default rules, with Quail's three
//! before `replace_distinct_aggregate`, which turns a `DISTINCT` into a
//! GROUP BY, and `lower_ai_calls` after `decorrelate_lateral_join`.
//! [`crate::claim_model_regions`] claims the model regions of the plan
//! the optimizer leaves, before physical planning.
//!
//! [`crate::LowerAiCalls`] runs twice: as an analyzer rule on the plan
//! the query wrote, and as the optimizer rule `lower_ai_calls` on the
//! joins DataFusion makes of correlated subqueries, which hold the
//! subquery's AI calls in their ON.
//!
//! Each `AiNode` tells DataFusion's rules what they may do around it:
//! which filters may move below it and which columns it reads (see
//! [`crate::ai::AiNode`]). A limit does not move below a node: it
//! becomes the node's fetch ([`crate::Derive`]).

pub(crate) mod derive;
pub(crate) mod distinct;
pub(crate) mod guards;

pub(crate) mod lower;
pub(crate) mod or_guards;
pub(crate) mod placement;
pub(crate) mod projection;
pub(crate) mod repeated;
pub(crate) mod substring;

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Result, plan_err};
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::{ApplyOrder, Optimizer, OptimizerConfig, OptimizerRule};

/// Quail's own rules, in the order they run.
pub const QUAIL_RULES: [&str; 3] = ["merge_repeated_ai_calls", "derive", "place_ai_nodes"];

/// The rule Quail's rules run before.
const QUAIL_RULES_BEFORE: &str = "replace_distinct_aggregate";

/// The rule that lowers the AI calls DataFusion's rules leave in a join.
const LOWERING_RULE: &str = "lower_ai_calls";

/// The last of DataFusion's rules that make a join of a subquery.
const LOWERING_RULE_AFTER: &str = "decorrelate_lateral_join";

#[cfg(test)]
const EXPECTED_DEFAULT_RULES: [&str; 29] = [
    "rewrite_set_comparison",
    "optimize_unions",
    "unions_to_filter",
    "simplify_expressions",
    "merge_repeated_ai_calls",
    "derive",
    "place_ai_nodes",
    "replace_distinct_aggregate",
    "eliminate_join",
    "decorrelate_predicate_subquery",
    "scalar_subquery_to_join",
    "decorrelate_lateral_join",
    "lower_ai_calls",
    "extract_equijoin_predicate",
    "eliminate_duplicated_expr",
    "eliminate_filter",
    "eliminate_cross_join",
    "eliminate_limit",
    "propagate_empty_relation",
    "filter_null_join_keys",
    "eliminate_outer_join",
    "push_down_limit",
    "push_down_filter",
    "single_distinct_aggregation_to_group_by",
    "eliminate_group_by_constant",
    "common_sub_expression_eliminate",
    "extract_leaf_expressions",
    "push_down_leaf_projections",
    "optimize_projections",
];

/// A rule of Quail's, followed by a recomputation of every operator's
/// schema from its inputs.
///
/// A rule that moves or reorders `AiNode`s changes the columns, or their
/// order, of the operators it rebuilds. An operator above them keeps
/// the schema it was built with, and DataFusion's rules read columns of
/// that schema by position.
#[derive(Debug)]
struct Recomputed(Arc<dyn OptimizerRule + Send + Sync>);

impl OptimizerRule for Recomputed {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        self.0.apply_order()
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // PREPARE optimizes a plan whose parameters EXECUTE fills in later,
        // through the expressions each operator lists. An `AiNode` holds a
        // test it absorbed outside that list, so the rules wait for values.
        if !plan.get_parameter_names()?.is_empty() {
            return Ok(Transformed::no(plan));
        }
        let rewritten = self.0.rewrite(plan, config)?;
        if !rewritten.transformed {
            return Ok(rewritten);
        }
        let plan = rewritten
            .data
            .transform_up(|node| node.recompute_schema().map(Transformed::yes))?
            .data;
        Ok(Transformed::yes(plan))
    }
}

fn quail_rule(name: &str) -> Option<Arc<dyn OptimizerRule + Send + Sync>> {
    unchecked_rule(name).map(|rule| Arc::new(Recomputed(rule)) as _)
}

fn unchecked_rule(name: &str) -> Option<Arc<dyn OptimizerRule + Send + Sync>> {
    Some(match name {
        "place_ai_nodes" => Arc::new(crate::rules::placement::PlaceAiNodes),
        name if name.starts_with("derive") => Arc::new(crate::rules::derive::Derive::named(name)?),
        "merge_repeated_ai_calls" => Arc::new(crate::rules::repeated::MergeRepeatedAiCalls),
        LOWERING_RULE => Arc::new(crate::rules::lower::LowerAiCalls),
        _ => return None,
    })
}

/// The optimizer rule named `name`: one of Quail's, or one of
/// DataFusion's default rules.
///
/// # Errors
///
/// No such rule is known.
pub fn logical_rule(name: &str) -> Result<Arc<dyn OptimizerRule + Send + Sync>> {
    if let Some(rule) = quail_rule(name) {
        return Ok(rule);
    }
    match Optimizer::new()
        .rules
        .into_iter()
        .find(|rule| rule.name() == name)
    {
        Some(rule) => Ok(rule),
        None => plan_err!("unknown logical optimizer rule {name:?}"),
    }
}

/// The rules named `names`, in order.
///
/// # Errors
///
/// A name is unknown.
pub fn logical_rules(names: &[String]) -> Result<Vec<Arc<dyn OptimizerRule + Send + Sync>>> {
    let mut rules = Vec::new();
    for name in names {
        rules.push(logical_rule(name)?);
    }
    Ok(rules)
}

/// `existing`, a session's rules, with Quail's rules before
/// `replace_distinct_aggregate`, or first when it has none, and
/// `lower_ai_calls` after `decorrelate_lateral_join`, or last when it
/// has none. A rule of Quail's already in the list is moved there.
pub(crate) fn with_quail_rules(
    existing: &[Arc<dyn OptimizerRule + Send + Sync>],
) -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    let quail = || {
        QUAIL_RULES
            .iter()
            .map(|name| quail_rule(name).expect("a rule of Quail's"))
    };
    let lowering = || quail_rule(LOWERING_RULE).expect("a rule of Quail's");
    let kept = existing
        .iter()
        .filter(|rule| !QUAIL_RULES.contains(&rule.name()) && rule.name() != LOWERING_RULE);
    let mut rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> = Vec::new();
    let mut placed = false;
    let mut lowered = false;
    for rule in kept {
        if rule.name() == QUAIL_RULES_BEFORE && !placed {
            rules.extend(quail());
            placed = true;
        }
        rules.push(Arc::clone(rule));
        if rule.name() == LOWERING_RULE_AFTER && !lowered {
            rules.push(lowering());
            lowered = true;
        }
    }
    if !placed {
        rules.splice(0..0, quail());
    }
    if !lowered {
        rules.push(lowering());
    }
    rules
}

/// The default rules without the claim, which a trial of a model region
/// runs over its operators.
pub(crate) fn default_rules() -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    with_quail_rules(&Optimizer::new().rules)
}

/// The default logical optimizer rule names.
pub fn default_rule_names() -> Vec<String> {
    default_rules()
        .iter()
        .map(|rule| rule.name().to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_rules_are_datafusions_with_quails_before_replacing_distinct() {
        let names = default_rule_names();
        assert_eq!(names, EXPECTED_DEFAULT_RULES);
        assert_eq!(logical_rules(&names).unwrap().len(), names.len());
    }
}
