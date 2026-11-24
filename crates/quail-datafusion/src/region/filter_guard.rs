use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use datafusion::arrow::datatypes::Schema;
use datafusion::common::Result;
use datafusion::common::config::ConfigOptions;
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::execution_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions};
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;

use super::exec::ModelRegionExec;

pub(crate) const BEFORE_RULE: &str = "FilterPushdown(Post)";

type Arriving = BTreeMap<String, bool>;

#[derive(Debug, Default)]
pub struct GuardRegionFilters;

impl PhysicalOptimizerRule for GuardRegionFilters {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        guard(plan, &Arriving::new())
    }

    fn name(&self) -> &str {
        "guard_region_filters"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn shared(schema: &Schema, name: &str) -> bool {
    schema
        .fields()
        .iter()
        .filter(|field| field.name() == name)
        .count()
        > 1
}

fn guard(plan: Arc<dyn ExecutionPlan>, arriving: &Arriving) -> Result<Arc<dyn ExecutionPlan>> {
    let children: Vec<Arc<dyn ExecutionPlan>> = plan.children().into_iter().cloned().collect();
    if let Some(region) = plan.downcast_ref::<ModelRegionExec>() {
        let schema = region.schema();
        let safe: BTreeSet<String> = arriving
            .iter()
            .filter(|(name, wrong)| !**wrong && !shared(&schema, name))
            .map(|(name, _)| name.clone())
            .collect();
        let inputs = children
            .into_iter()
            .map(|input| guard(input, &Arriving::new()))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Arc::new(region.with_filter_columns(safe, inputs)));
    }
    if children.is_empty() {
        return Ok(plan);
    }
    let guarded = children
        .iter()
        .enumerate()
        .map(|(index, child)| guard(Arc::clone(child), &below(&plan, index, child, arriving)))
        .collect::<Result<Vec<_>>>()?;
    plan.replace_children(
        guarded,
        ReplaceChildrenOptions::new(ChildrenPropertiesMode::Keep),
    )
}

fn below(
    plan: &Arc<dyn ExecutionPlan>,
    index: usize,
    child: &Arc<dyn ExecutionPlan>,
    arriving: &Arriving,
) -> Arriving {
    let schema = child.schema();
    let mut found = Arriving::new();
    let mut pass = |name: &str, wrong: bool| {
        if schema.field_with_name(name).is_ok() {
            *found.entry(name.to_string()).or_insert(false) |= wrong || shared(&schema, name);
        }
    };
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        let output = projection.schema();
        for (name, wrong) in arriving {
            let wrong = *wrong || shared(&output, name);
            let Ok(position) = output.index_of(name) else {
                continue;
            };
            for column in collect_columns(&projection.expr()[position].expr) {
                pass(column.name(), wrong);
            }
        }
    } else {
        for (name, wrong) in arriving {
            pass(name, *wrong);
        }
    }
    let mut start = |name: &str| {
        found.entry(name.to_string()).or_insert(false);
    };
    if let Some(sort) = plan.downcast_ref::<SortExec>() {
        if sort.fetch().is_some() {
            for column in sort
                .expr()
                .iter()
                .flat_map(|key| collect_columns(&key.expr))
            {
                start(column.name());
            }
        }
    } else if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
        for (left, right) in join.on() {
            let key = if index == 0 { left } else { right };
            for column in collect_columns(key) {
                start(column.name());
            }
        }
    } else if plan.downcast_ref::<AggregateExec>().is_some() {
        for field in schema.fields() {
            start(field.name());
        }
    }
    found
}
