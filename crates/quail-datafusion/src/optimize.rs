//! DataFusion's own rules inside a model region.
//!
//! DataFusion's optimizer treats a model region as a leaf. Inside it,
//! [`OptimizeModelRegions`] runs `OptimizeProjections` and
//! `PushDownFilter` over the lowered region: columns no operator reads
//! are pruned, and plain filters move down to the scans, around Quail's
//! plan nodes, as each node's `prevent_predicate_push_down_columns` and
//! `necessary_children_exprs` allow.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::optimize_projections::OptimizeProjections;
use datafusion::optimizer::push_down_filter::PushDownFilter;
use datafusion::optimizer::{Optimizer, OptimizerConfig, OptimizerContext, OptimizerRule};

use crate::region::ModelRegion;

/// Run DataFusion's rules inside each model region, which DataFusion's
/// optimizer treats as a leaf, and keep the result beside the region's
/// bound plan.
#[derive(Debug, Default)]
pub struct OptimizeModelRegions;

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
                let mut region = region.clone();
                // a region DataFusion's rules cannot optimize keeps its lowered plan
                if let Ok(optimized) = optimize_region(&region.plan) {
                    region.plan = optimized;
                }
                region.optimized = true;
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

/// Prune columns and push filters down inside a lowered model region.
///
/// # Errors
///
/// A DataFusion rule fails.
pub fn optimize_region(plan: &LogicalPlan) -> Result<LogicalPlan> {
    let rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> = vec![
        Arc::new(crate::placement::PlaceSemanticNodes),
        Arc::new(PushDownFilter::new()),
        Arc::new(OptimizeProjections::new()),
    ];
    Optimizer::with_rules(rules).optimize(plan.clone(), &OptimizerContext::new(), |_, _| {})
}
