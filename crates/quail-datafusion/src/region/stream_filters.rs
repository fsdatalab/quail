//! The operators above a model region pass its rows on as they arrive.
//!
//! In DataFusion 55.2, a `FilterExec` and a `RepartitionExec` hold the
//! rows they pass until they have a whole batch,
//! `datafusion.execution.batch_size` rows, or their input ends. A region
//! sends its rows a few at a time as the model finishes them, so a
//! filter above it, such as the one that keeps the rows whose AI answer
//! is TRUE, would hold them all until the region ends, and a streaming
//! reader would see nothing before then.
//!
//! [`StreamRegionRows`] changes the operators above each
//! [`ModelRegionExec`]:
//!
//! - a `FilterExec` gets a batch size of one row, so its coalescer
//!   passes every batch on as it is, without splitting or holding it;
//! - a round-robin `RepartitionExec` goes: it only spreads the region's
//!   few rows over threads, and a repartition sizes its batches by the
//!   session alone.
//!
//! The same holds between two regions, where one reads the other's
//! rows: the region above numbers its input's rows as they arrive, so
//! it reads them in the order the region below sent them.

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

/// Lets the operators above a model region pass rows on as they
/// arrive, as the module explains.
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

/// The plan with the operators above each region passing rows on, and
/// whether the plan reads a region.
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
            // a child without its repartition has fewer partitions
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
