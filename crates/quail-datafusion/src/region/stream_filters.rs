use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::config::ConfigOptions;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::Partitioning;
use datafusion::physical_plan::execution_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::repartition::RepartitionExec;

use super::exec::ModelRegionExec;

#[derive(Debug, Default)]
pub struct StreamRegionRows;

impl PhysicalOptimizerRule for StreamRegionRows {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(streamed(plan)?.0)
    }

    fn name(&self) -> &str {
        "stream_region_rows"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn streamed(plan: Arc<dyn ExecutionPlan>) -> Result<(Arc<dyn ExecutionPlan>, bool)> {
    let region = plan.downcast_ref::<ModelRegionExec>().is_some();
    let children: Vec<Arc<dyn ExecutionPlan>> = plan.children().into_iter().cloned().collect();
    if children.is_empty() {
        return Ok((plan, region));
    }
    let mut reads_region = false;
    let mut changed = false;
    let mut found = Vec::with_capacity(children.len());
    for child in children {
        let (new, reads) = streamed(Arc::clone(&child))?;
        reads_region |= reads;
        changed |= !Arc::ptr_eq(&new, &child);
        found.push(new);
    }
    let plan = if changed {
        plan.replace_children(
            found,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )?
    } else {
        plan
    };
    if region {
        return Ok((plan, true));
    }
    if !reads_region {
        return Ok((plan, false));
    }
    if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        return Ok((Arc::new(filter.with_batch_size(1)?), true));
    }
    if let Some(repartition) = plan.downcast_ref::<RepartitionExec>()
        && matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_))
    {
        return Ok((Arc::clone(repartition.input()), true));
    }
    Ok((plan, reads_region))
}
