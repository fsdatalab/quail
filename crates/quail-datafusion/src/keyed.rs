//! An optimizer rule that gives an AI filter a per-key fetch where the
//! operator above needs only one row of each key.
//!
//! `PushDownSemiJoinFetch`: a semi-join keeps each row of its left input
//! that has one matching row on its right. When the right input is an AI
//! filter's rows and the join condition is `J = K`, with `K` input
//! columns of the AI filter, one passing row of each `K` is enough
//! (R13):
//!
//! ```text
//! S ⋉_{J = K} AI(R) ⇒ S ⋉_{J = K} AI(R; n = 1 per K)
//! ```
//!
//! This covers `IN` and `EXISTS` subqueries that DataFusion's
//! `decorrelate_predicate_subquery` writes as a semi-join.
//!
//! The executor stops asking about a key's documents once `n` of them
//! pass the table's AI filters, and may return more than `n`: the
//! semi-join gives the same rows either way. It honors the stop only in
//! a region that asks one table's AI.IF filters and joins nothing, so the
//! rule applies only there: the node, and every
//! `AiNode` below it, is an AI.IF over the same one table, with no join,
//! fetch, or `HINT`.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::{Expr, JoinType, LogicalPlan, Projection, SubqueryAlias};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Fetch, Semantics};
use crate::extension::{as_node, extension_node};

/// Give the AI filter on a semi-join's right side one row per join key,
/// as the module explains.
#[derive(Debug, Default)]
pub struct PushDownSemiJoinFetch;

impl OptimizerRule for PushDownSemiJoinFetch {
    fn name(&self) -> &str {
        "push_down_semi_join_fetch"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up(semi_join_fetch)
    }
}

fn semi_join_fetch(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Join(join) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let ai_right = match join.join_type {
        JoinType::LeftSemi => true,
        JoinType::RightSemi => false,
        _ => return Ok(Transformed::no(plan)),
    };
    if join.filter.is_some() || join.on.is_empty() {
        return Ok(Transformed::no(plan));
    }
    let keys: Option<Vec<Column>> = join
        .on
        .iter()
        .map(|(left, right)| match if ai_right { right } else { left } {
            Expr::Column(column) => Some(column.clone()),
            _ => None,
        })
        .collect();
    let Some(keys) = keys else {
        return Ok(Transformed::no(plan));
    };
    let side = if ai_right { &join.right } else { &join.left };
    let Some(keyed) = with_key_fetch(side, &keys, 1)? else {
        return Ok(Transformed::no(plan));
    };
    let mut join = join.clone();
    if ai_right {
        join.right = Arc::new(keyed);
    } else {
        join.left = Arc::new(keyed);
    }
    Ok(Transformed::yes(LogicalPlan::Join(join)))
}

/// `plan` with a fetch of `n` rows per `keys` on the AI filter it reads
/// through projections and subquery aliases; None when the module's
/// conditions do not hold.
fn with_key_fetch(plan: &LogicalPlan, keys: &[Column], n: usize) -> Result<Option<LogicalPlan>> {
    match plan {
        LogicalPlan::Projection(projection) => {
            let below: Option<Vec<Column>> = keys
                .iter()
                .map(|key| {
                    let index = projection.schema.index_of_column(key).ok()?;
                    match &projection.expr[index] {
                        Expr::Column(column) => Some(column.clone()),
                        Expr::Alias(alias) => match alias.expr.as_ref() {
                            Expr::Column(column) => Some(column.clone()),
                            _ => None,
                        },
                        _ => None,
                    }
                })
                .collect();
            let Some(below) = below else {
                return Ok(None);
            };
            let Some(input) = with_key_fetch(&projection.input, &below, n)? else {
                return Ok(None);
            };
            Ok(Some(LogicalPlan::Projection(Projection::try_new(
                projection.expr.clone(),
                Arc::new(input),
            )?)))
        }
        LogicalPlan::SubqueryAlias(alias) => {
            let below: Option<Vec<Column>> = keys
                .iter()
                .map(|key| {
                    let index = alias.schema.index_of_column(key).ok()?;
                    Some(Column::from(alias.input.schema().qualified_field(index)))
                })
                .collect();
            let Some(below) = below else {
                return Ok(None);
            };
            let Some(input) = with_key_fetch(&alias.input, &below, n)? else {
                return Ok(None);
            };
            Ok(Some(LogicalPlan::SubqueryAlias(SubqueryAlias::try_new(
                Arc::new(input),
                alias.alias.clone(),
            )?)))
        }
        LogicalPlan::Extension(_) => {
            let Some(node) = as_node::<AiNode>(plan) else {
                return Ok(None);
            };
            if !stops_per_key(node) {
                return Ok(None);
            }
            let input = node.input.schema();
            let mut names: Vec<String> = Vec::new();
            for key in keys {
                if !input.has_column(key) {
                    return Ok(None);
                }
                // the executor groups documents by the key's column names
                let shared = input
                    .fields()
                    .iter()
                    .filter(|field| field.name() == &key.name)
                    .count();
                if shared != 1 {
                    return Ok(None);
                }
                if !names.contains(&key.name) {
                    names.push(key.name.clone());
                }
            }
            let mut keyed = node.clone();
            keyed.fetch = Some(Fetch::PerKey { key: names, n });
            Ok(Some(extension_node(keyed.checked()?)))
        }
        _ => Ok(None),
    }
}

/// Whether the executor stops `node` per key: it and every `AiNode`
/// below it are AI.IF filters of rows over the same one table, with no
/// fetch or `HINT`, and nothing below joins.
fn stops_per_key(node: &AiNode) -> bool {
    let tables = |node: &AiNode| -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        for column in node.call.column_refs() {
            if let Some(table) = &column.relation
                && !found.contains(&table.table().to_string())
            {
                found.push(table.table().to_string());
            }
        }
        found
    };
    let table = tables(node);
    let filters = |found: &AiNode| {
        found.function() == "ai_filter"
            && found.semantics == Semantics::Rows
            && found.fetch.is_none()
            && found.hint.is_none()
            && found.guard.is_none()
            && tables(found) == table
    };
    if table.len() != 1 || !filters(node) {
        return false;
    }
    let mut plain = true;
    let _ = node.input.apply(|below| {
        let joins = matches!(below, LogicalPlan::Join(_));
        let other_call = as_node::<AiNode>(below).is_some_and(|found| !filters(found));
        let unknown =
            matches!(below, LogicalPlan::Extension(_)) && as_node::<AiNode>(below).is_none();
        if joins || other_call || unknown {
            plain = false;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    plain
}
