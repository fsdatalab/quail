//! Remove each `AiNode` whose columns no operator reads.
//!
//! DataFusion's `OptimizeProjections` prunes the columns a node reads
//! from its input, but never removes the node. A column node that keeps
//! every row and whose added columns nothing above reads asks the model
//! for answers the query drops, so the claim removes it first.

use std::collections::BTreeSet;

use datafusion::common::Result;
use datafusion::common::tree_node::TreeNode;
use datafusion::logical_expr::{Expr, LogicalPlan};

use crate::ai::{AiNode, Semantics};
use crate::extension::as_node;

/// `plan` without the `AiNode`s that filter no row and add only columns
/// no operator above them reads.
///
/// # Errors
///
/// An operator does not fit its inputs once a node is gone.
pub(super) fn drop_unread_nodes(plan: LogicalPlan) -> Result<LogicalPlan> {
    drop_unread(plan, None)
}

/// The column names `exprs` read, or None when one holds a subquery,
/// whose outer columns the search does not follow.
fn read_names(exprs: &[Expr]) -> Option<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for expr in exprs {
        let subquery = expr
            .exists(|inner| {
                Ok(matches!(
                    inner,
                    Expr::ScalarSubquery(_) | Expr::InSubquery(_) | Expr::Exists(_)
                ))
            })
            .unwrap_or(true);
        if subquery {
            return None;
        }
        names.extend(
            expr.column_refs()
                .into_iter()
                .map(|column| column.name.clone()),
        );
    }
    Some(names)
}

/// [`drop_unread_nodes`], where the operators above read only the
/// columns named `read` of `plan`, or every column when None.
fn drop_unread(plan: LogicalPlan, read: Option<&BTreeSet<String>>) -> Result<LogicalPlan> {
    if let Some(node) = as_node::<AiNode>(&plan)
        && let Some(read) = read
        && node.semantics == Semantics::Rows
        && node.filter.is_none()
        && node.tests.is_empty()
        && node.fetch.is_none()
        && !node.fields.iter().any(|field| read.contains(field.name()))
    {
        return drop_unread(node.input.as_ref().clone(), Some(read));
    }
    // a node's filter and tests read columns too, beside its expressions
    let mut exprs = plan.expressions();
    if let Some(node) = as_node::<AiNode>(&plan) {
        exprs.extend(node.filter.clone());
        exprs.extend(node.tests.iter().map(|found| found.test.clone()));
    }
    let own = read_names(&exprs);
    let below: Option<BTreeSet<String>> = match &plan {
        // these return only what their expressions compute
        LogicalPlan::Projection(_) | LogicalPlan::Aggregate(_) => own,
        // these return their inputs' columns under the same names
        LogicalPlan::Filter(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Join(_)
        | LogicalPlan::Window(_)
        | LogicalPlan::Repartition(_)
        | LogicalPlan::SubqueryAlias(_) => match (read, own) {
            (Some(read), Some(own)) => Some(read.union(&own).cloned().collect()),
            _ => None,
        },
        _ if as_node::<AiNode>(&plan).is_some() => match (read, own) {
            (Some(read), Some(own)) => Some(read.union(&own).cloned().collect()),
            _ => None,
        },
        _ => None,
    };
    let inputs = plan
        .inputs()
        .into_iter()
        .map(|input| drop_unread(input.clone(), below.as_ref()))
        .collect::<Result<Vec<_>>>()?;
    plan.with_new_exprs(plan.expressions(), inputs)
}
