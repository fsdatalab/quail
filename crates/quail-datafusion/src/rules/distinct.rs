//! Whether a `DISTINCT` removes a row: [`crate::Derive`] drops one that
//! cannot, so a `LIMIT` above it reaches the `AiNode` below.
//!
//! A `DISTINCT` removes no row when the columns it compares include a
//! unique key of its input. [`is_unique`] reads the keys DataFusion
//! derives (its functional dependencies), and adds what DataFusion does
//! not keep: the keys of each side of a join together are a key of the
//! join.

use std::collections::BTreeSet;

use datafusion::common::Column;
use datafusion::common::Dependency;
use datafusion::logical_expr::{Aggregate, Expr, JoinType, LogicalPlan};

use crate::ai::{AiNode, Semantics};
use crate::extension::as_node;

/// Whether an `Aggregate` is a `DISTINCT`: it groups by each of its
/// input's columns, in order, and computes nothing else.
///
/// DataFusion's `replace_distinct_aggregate` writes a `DISTINCT` so.
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

/// Whether no two rows of `plan` have equal values in `columns`.
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
            // the input columns the SELECT list returns as they are
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
            // a node of rows returns at most one row for each input row
            Some(node) if node.semantics == Semantics::Rows => is_unique(&node.input, columns),
            _ => false,
        },
        LogicalPlan::Join(join) => {
            let side = |input: &LogicalPlan| -> BTreeSet<Column> {
                (input.schema().columns().into_iter())
                    .filter(|column| columns.contains(column))
                    .collect()
            };
            match join.join_type {
                // a row of the join is one row of each side, or one side's
                // row with NULLs, which only that row's key tells apart
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
