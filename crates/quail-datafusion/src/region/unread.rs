use std::collections::BTreeSet;

use datafusion::common::Result;
use datafusion::common::tree_node::TreeNode;
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

use crate::ai::AiNode;
use crate::plan::as_node;

pub(super) fn drop_unread_nodes(plan: LogicalPlan) -> Result<LogicalPlan> {
    drop_unread(plan, None)
}

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

fn drop_unread(plan: LogicalPlan, read: Option<&BTreeSet<String>>) -> Result<LogicalPlan> {
    if let Some(node) = as_node::<AiNode>(&plan)
        && let Some(read) = read
        && node.only_adds_columns()
        && !node.added_names().iter().any(|name| read.contains(name))
    {
        return drop_unread(node.inputs()[0].clone(), Some(read));
    }
    let mut exprs = plan.expressions();
    if let Some(node) = as_node::<AiNode>(&plan) {
        exprs.extend(node.predicate_expressions());
    }
    let own = read_names(&exprs);
    let below: Option<BTreeSet<String>> = match &plan {
        LogicalPlan::Projection(_) | LogicalPlan::Aggregate(_) => own,
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
    crate::plan::with_inputs(&plan, inputs)
}
