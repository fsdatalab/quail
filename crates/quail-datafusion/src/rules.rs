//! The logical optimizer rules of a session with Quail.
//!
//! DataFusion's optimizer runs its own rules over the whole plan, each
//! `AiNode` included, with Quail's rules among them. [`LOGICAL_RULES`]
//! is the default order: DataFusion's default rules, with Quail's seven
//! before `replace_distinct_aggregate`, which turns a `DISTINCT` into a
//! GROUP BY. Then
//! [`crate::ClaimModelRegions`] runs last: it runs the listed rules
//! until the plan stops changing, then claims the model regions.
//!
//! Each `AiNode` tells DataFusion's rules what they may do around it:
//! which filters may move below it and which columns it reads (see
//! [`crate::ai::AiNode`]). A limit does not move below a node: it
//! becomes the node's fetch ([`crate::PushDownLimitIntoAiNodes`]).

use std::sync::Arc;

use datafusion::common::{Result, plan_err};
use datafusion::optimizer::{Optimizer, OptimizerRule};

/// Quail's own rules, in the order they run.
pub const QUAIL_RULES: [&str; 7] = [
    "eliminate_distinct",
    "place_ai_nodes",
    "absorb_into_ai_nodes",
    "merge_repeated_ai_calls",
    "push_down_distinct",
    "push_down_limit_into_ai_nodes",
    "push_down_projection_into_ai_nodes",
];

/// The rule Quail's rules run before.
const QUAIL_RULES_BEFORE: &str = "replace_distinct_aggregate";

/// The logical optimizer rules a session runs by default, in order: each
/// name is an `OptimizerRule`'s `name()`.
pub const LOGICAL_RULES: [&str; 32] = [
    "rewrite_set_comparison",
    "optimize_unions",
    "unions_to_filter",
    "simplify_expressions",
    "eliminate_distinct",
    "place_ai_nodes",
    "absorb_into_ai_nodes",
    "merge_repeated_ai_calls",
    "push_down_distinct",
    "push_down_limit_into_ai_nodes",
    "push_down_projection_into_ai_nodes",
    "replace_distinct_aggregate",
    "eliminate_join",
    "decorrelate_predicate_subquery",
    "scalar_subquery_to_join",
    "decorrelate_lateral_join",
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

fn quail_rule(name: &str) -> Option<Arc<dyn OptimizerRule + Send + Sync>> {
    Some(match name {
        "eliminate_distinct" => Arc::new(crate::distinct::EliminateDistinct),
        "place_ai_nodes" => Arc::new(crate::placement::PlaceAiNodes),
        "absorb_into_ai_nodes" => Arc::new(crate::absorb::AbsorbIntoAiNodes),
        "merge_repeated_ai_calls" => Arc::new(crate::repeated::MergeRepeatedAiCalls),
        "push_down_distinct" => Arc::new(crate::distinct::PushDownDistinct),
        "push_down_limit_into_ai_nodes" => Arc::new(crate::limit::PushDownLimitIntoAiNodes),
        "push_down_projection_into_ai_nodes" => {
            Arc::new(crate::projection::PushDownProjectionIntoAiNodes)
        }
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

/// The rules named `names`, in order, then [`crate::ClaimModelRegions`].
///
/// # Errors
///
/// A name is unknown, or names the claim, which always runs last.
pub fn logical_rules(names: &[String]) -> Result<Vec<Arc<dyn OptimizerRule + Send + Sync>>> {
    let mut rules = Vec::new();
    for name in names {
        if name == crate::claim::CLAIM_RULE {
            return plan_err!("{name} always runs last; leave it out of the list");
        }
        rules.push(logical_rule(name)?);
    }
    rules.push(Arc::new(crate::ClaimModelRegions::new(rules.clone())));
    Ok(rules)
}

/// `existing`, a session's rules, with Quail's rules before
/// `replace_distinct_aggregate`, or first when it has none, and
/// [`crate::ClaimModelRegions`] last. A rule of Quail's already in the
/// list is moved there.
pub(crate) fn with_quail_rules(
    existing: &[Arc<dyn OptimizerRule + Send + Sync>],
) -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    let quail = || {
        QUAIL_RULES
            .iter()
            .map(|name| quail_rule(name).expect("a rule of Quail's"))
    };
    let kept = existing.iter().filter(|rule| {
        !QUAIL_RULES.contains(&rule.name()) && rule.name() != crate::claim::CLAIM_RULE
    });
    let mut rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> = Vec::new();
    let mut placed = false;
    for rule in kept {
        if rule.name() == QUAIL_RULES_BEFORE && !placed {
            rules.extend(quail());
            placed = true;
        }
        rules.push(Arc::clone(rule));
    }
    if !placed {
        rules.splice(0..0, quail());
    }
    rules.push(Arc::new(crate::ClaimModelRegions::new(rules.clone())));
    rules
}

/// The default rules without the claim, which a trial of a model region
/// runs over its operators.
pub(crate) fn default_rules() -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    let mut rules = with_quail_rules(&Optimizer::new().rules);
    rules.pop();
    rules
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_rules_are_datafusions_with_quails_before_replacing_distinct() {
        let names: Vec<String> = default_rules()
            .iter()
            .map(|rule| rule.name().to_string())
            .collect();
        assert_eq!(names, LOGICAL_RULES);
        let listed: Vec<String> = LOGICAL_RULES.iter().map(ToString::to_string).collect();
        let rules = logical_rules(&listed).unwrap();
        assert_eq!(rules.last().unwrap().name(), crate::claim::CLAIM_RULE);
    }
}
