//! The `ClaimModelRegions` analyzer rule.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan};
use datafusion::optimizer::AnalyzerRule;

use crate::functions::is_ai_call;
use crate::region::ModelRegion;

/// Replace each largest subtree Quail can plan that calls an AI
/// function with one `ModelRegion`.
///
/// The rule walks the plan from the root. The first operator whose
/// whole subtree Quail can plan, and which calls an AI function
/// somewhere in that subtree, becomes a model region; the rule does
/// not look below it. DataFusion keeps every operator above.
#[derive(Debug, Default)]
pub struct ClaimModelRegions;

impl ClaimModelRegions {
    /// Create the rule.
    pub fn new() -> Self {
        Self
    }
}

impl AnalyzerRule for ClaimModelRegions {
    fn name(&self) -> &str {
        "claim_model_regions"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        Ok(plan
            .transform_down(|node| {
                if calls_ai(&node)? && quail_plans(&node)? {
                    let region = LogicalPlan::Extension(Extension {
                        node: Arc::new(ModelRegion::new(node)),
                    });
                    Ok(Transformed::new(region, true, TreeNodeRecursion::Jump))
                } else {
                    Ok(Transformed::no(node))
                }
            })?
            .data)
    }
}

/// Whether any operator in the subtree, or an EXISTS subquery in it,
/// calls an AI function.
fn calls_ai(plan: &LogicalPlan) -> Result<bool> {
    plan.exists(|node| {
        for expr in node.expressions() {
            let found = expr.exists(|inner| match inner {
                Expr::Exists(exists) => calls_ai(&exists.subquery.subquery),
                other => Ok(is_ai_call(other)),
            })?;
            if found {
                return Ok(true);
            }
        }
        Ok(false)
    })
}

/// Whether Quail can plan every operator in the subtree.
fn quail_plans(plan: &LogicalPlan) -> Result<bool> {
    let unsupported = plan.exists(|node| {
        if !is_quail_operator(node) {
            return Ok(true);
        }
        for expr in node.expressions() {
            if expr.exists(|inner| Ok(is_subquery(inner)))? {
                return Ok(true);
            }
        }
        Ok(false)
    })?;
    Ok(!unsupported)
}

/// Whether Quail's planner has an operator for this node.
fn is_quail_operator(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Projection(_)
            | LogicalPlan::Filter(_)
            | LogicalPlan::Window(_)
            | LogicalPlan::Aggregate(_)
            | LogicalPlan::Sort(_)
            | LogicalPlan::Join(_)
            | LogicalPlan::Limit(_)
            | LogicalPlan::Distinct(_)
            | LogicalPlan::SubqueryAlias(_)
            | LogicalPlan::TableScan(_)
            | LogicalPlan::Unnest(_)
    )
}

/// Whether an expression is a subquery Quail does not plan: anything
/// but an EXISTS gate, whose subquery Quail's binder checks.
fn is_subquery(expr: &Expr) -> bool {
    match expr {
        Expr::Exists(exists) => !quail_plans(&exists.subquery.subquery).unwrap_or(false),
        Expr::InSubquery(_) | Expr::ScalarSubquery(_) => true,
        _ => false,
    }
}
