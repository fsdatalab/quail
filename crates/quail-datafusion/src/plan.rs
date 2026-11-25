use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::{Extension, LogicalPlan, UserDefinedLogicalNode};

pub(crate) fn with_inputs(plan: &LogicalPlan, inputs: Vec<LogicalPlan>) -> Result<LogicalPlan> {
    let mut inputs = inputs.into_iter();
    plan.clone()
        .map_children(|input| Ok(Transformed::yes(inputs.next().unwrap_or(input))))?
        .data
        .recompute_schema()
}

pub(crate) fn extension_node(node: impl UserDefinedLogicalNode + 'static) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    })
}

pub(crate) fn as_node<T: 'static>(plan: &LogicalPlan) -> Option<&T> {
    match plan {
        LogicalPlan::Extension(extension) => extension.node.as_any().downcast_ref::<T>(),
        _ => None,
    }
}
