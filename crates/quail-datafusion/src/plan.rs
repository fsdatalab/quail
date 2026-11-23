//! Quail's extension nodes in a DataFusion plan, and operators rebuilt
//! over new inputs: the plan helpers every layer of the crate shares.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::{Extension, LogicalPlan, UserDefinedLogicalNode};

/// `plan`'s operator over other inputs, its expressions as they are.
///
/// `LogicalPlan::with_new_exprs` cannot rebuild a `DISTINCT ON` with an
/// `ORDER BY`; this keeps the operator and replaces its inputs alone.
pub(crate) fn with_inputs(plan: &LogicalPlan, inputs: Vec<LogicalPlan>) -> Result<LogicalPlan> {
    let mut inputs = inputs.into_iter();
    plan.clone()
        .map_children(|input| Ok(Transformed::yes(inputs.next().unwrap_or(input))))?
        .data
        .recompute_schema()
}

/// A plan whose root is `node`, one of Quail's extension nodes.
pub(crate) fn extension_node(node: impl UserDefinedLogicalNode + 'static) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    })
}

/// The plan's root as the extension node `T`, if it is one.
pub(crate) fn as_node<T: 'static>(plan: &LogicalPlan) -> Option<&T> {
    match plan {
        LogicalPlan::Extension(extension) => extension.node.as_any().downcast_ref::<T>(),
        _ => None,
    }
}
