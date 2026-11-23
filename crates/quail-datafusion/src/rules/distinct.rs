use std::collections::BTreeSet;

use datafusion::common::Column;
use datafusion::common::Dependency;
use datafusion::logical_expr::{
    Aggregate, Expr, JoinType, LogicalPlan, UserDefinedLogicalNodeCore,
};

use crate::ai::AiNode;
use crate::plan::as_node;

pub(crate) fn is_distinct(aggregate: &Aggregate) -> bool {
    let columns: Vec<Expr> = aggregate
        .input
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    aggregate.aggr_expr.is_empty() && aggregate.group_expr == columns
}

pub(crate) fn is_unique(plan: &LogicalPlan, columns: &BTreeSet<Column>) -> bool {
    let schema = plan.schema();
    let keyed = schema.functional_dependencies().iter().any(|dependency| {
        dependency.mode == Dependency::Single
            && !dependency.nullable
            && (dependency.source_indices.iter())
                .all(|index| columns.contains(&Column::from(schema.qualified_field(*index))))
    });
    if keyed {
        return true;
    }
    match plan {
        LogicalPlan::Projection(projection) => {
            let mut below = BTreeSet::new();
            for (expr, column) in projection.expr.iter().zip(schema.columns()) {
                if let (Expr::Column(input), true) =
                    (expr.clone().unalias(), columns.contains(&column))
                {
                    below.insert(input);
                }
            }
            is_unique(&projection.input, &below)
        }
        LogicalPlan::SubqueryAlias(alias) => {
            let input = alias.input.schema().columns();
            let below = (schema.columns().into_iter().zip(input))
                .filter(|(column, _)| columns.contains(column))
                .map(|(_, input)| input)
                .collect();
            is_unique(&alias.input, &below)
        }
        LogicalPlan::Filter(found) => is_unique(&found.input, columns),
        LogicalPlan::Sort(found) => is_unique(&found.input, columns),
        LogicalPlan::Limit(found) => is_unique(&found.input, columns),
        LogicalPlan::Extension(_) => match as_node::<AiNode>(plan) {
            Some(node) if node.semantics() == Semantics::Rows => {
                is_unique(node.inputs()[0], columns)
            }
            _ => false,
        },
        LogicalPlan::Join(join) => {
            let side = |input: &LogicalPlan| -> BTreeSet<Column> {
                (input.schema().columns().into_iter())
                    .filter(|column| columns.contains(column))
                    .collect()
            };
            match join.join_type {
                JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                    is_unique(&join.left, &side(&join.left))
                        && is_unique(&join.right, &side(&join.right))
                }
                JoinType::LeftSemi | JoinType::LeftAnti => is_unique(&join.left, columns),
                JoinType::RightSemi | JoinType::RightAnti => is_unique(&join.right, columns),
                _ => false,
            }
        }
        _ => false,
    }
}
