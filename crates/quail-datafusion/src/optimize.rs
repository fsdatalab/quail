//! DataFusion's own rules inside a model region.
//!
//! DataFusion's optimizer treats a model region as a leaf. Inside it,
//! [`OptimizeModelRegions`] runs a list of optimizer rules over the
//! lowered region, by default [`REGION_RULES`]: Quail's DISTINCT,
//! placement, and absorption rules, then DataFusion's `PushDownFilter`
//! and `OptimizeProjections`, which move plain filters down to the scans
//! around Quail's `AiNode`s and prune columns no operator reads, as each
//! node's `prevent_predicate_push_down_columns` and
//! `necessary_children_exprs` allow.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Result, plan_err};
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::optimize_projections::OptimizeProjections;
use datafusion::optimizer::push_down_filter::PushDownFilter;
use datafusion::optimizer::{Optimizer, OptimizerConfig, OptimizerContext, OptimizerRule};

use crate::region::ModelRegion;

/// The rules a model region runs by default, in order: each name is an
/// `OptimizerRule`'s `name()`.
pub const REGION_RULES: [&str; 6] = [
    "eliminate_distinct",
    "place_ai_nodes",
    "absorb_into_ai_nodes",
    "push_down_distinct",
    "push_down_filter",
    "optimize_projections",
];

/// The optimizer rule named `name`, of the rules a model region can run.
///
/// # Errors
///
/// No such rule is known.
pub fn region_rule(name: &str) -> Result<Arc<dyn OptimizerRule + Send + Sync>> {
    Ok(match name {
        "eliminate_distinct" => Arc::new(crate::distinct::EliminateDistinct),
        "place_ai_nodes" => Arc::new(crate::placement::PlaceAiNodes),
        "absorb_into_ai_nodes" => Arc::new(crate::absorb::AbsorbIntoAiNodes),
        "push_down_distinct" => Arc::new(crate::distinct::PushDownDistinct),
        "push_down_filter" => Arc::new(PushDownFilter::new()),
        "optimize_projections" => Arc::new(OptimizeProjections::new()),
        other => return plan_err!("unknown model region rule {other:?}"),
    })
}

/// Run the named optimizer rules inside each model region, which
/// DataFusion's optimizer treats as a leaf, and keep the result beside
/// the region's bound plan.
#[derive(Debug)]
pub struct OptimizeModelRegions {
    rules: Vec<String>,
}

impl OptimizeModelRegions {
    /// Run `rules`, names that [`region_rule`] knows, in order.
    pub fn new(rules: Vec<String>) -> Self {
        Self { rules }
    }
}

impl OptimizeModelRegions {
    /// The names of the rules each region runs, in order.
    pub fn rules(&self) -> &[String] {
        &self.rules
    }
}

impl Default for OptimizeModelRegions {
    fn default() -> Self {
        Self::new(REGION_RULES.iter().map(ToString::to_string).collect())
    }
}

impl OptimizerRule for OptimizeModelRegions {
    fn name(&self) -> &str {
        "optimize_model_regions"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up(|node| {
            if let LogicalPlan::Extension(extension) = &node
                && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
                && !region.optimized
            {
                // a region DataFusion's rules cannot optimize keeps its lowered
                // plan, and the binder reports the rule's error
                let Ok(optimized) = optimize_region(&region.plan, &self.rules) else {
                    return Ok(Transformed::no(node));
                };
                let mut region = region.clone();
                region.plan = optimized;
                region.optimized = true;
                region.rules.clone_from(&self.rules);
                return Ok(Transformed::yes(LogicalPlan::Extension(
                    datafusion::logical_expr::Extension {
                        node: Arc::new(region),
                    },
                )));
            }
            Ok(Transformed::no(node))
        })
    }
}

/// Run the named rules, in order, inside a lowered model region.
///
/// # Errors
///
/// A rule name is unknown, or a rule fails.
pub fn optimize_region(plan: &LogicalPlan, rules: &[String]) -> Result<LogicalPlan> {
    let rules = rules
        .iter()
        .map(|name| region_rule(name))
        .collect::<Result<Vec<_>>>()?;
    Optimizer::with_rules(rules).optimize(plan.clone(), &OptimizerContext::new(), |_, _| {})
}
