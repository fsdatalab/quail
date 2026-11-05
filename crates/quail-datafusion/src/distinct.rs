//! The DISTINCT rules of a model region: `EliminateDistinct` and
//! `PushDownDistinct`.
//!
//! Both read only the region's outermost `DISTINCT`, the one under its
//! LIMIT, ORDER BY, and outer SELECT list.
//!
//! - `EliminateDistinct` removes a `DISTINCT` that removes no row: the
//!   SELECT list returns every table's id column, which each table
//!   declares as its primary key, or every grouping key of an
//!   aggregate.
//! - `PushDownDistinct` gives a one-table query's `SemanticFilter` the
//!   selected columns as its stop key: the executor stops asking the
//!   model about a document once a row with the same key has passed.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Constraint, Result};
use datafusion::logical_expr::{
    Distinct, Expr, Extension, LogicalPlan, Projection, UserDefinedLogicalNode,
};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::semantic::{SemanticColumn, SemanticFilter, SemanticJoin};

/// Remove a region's `DISTINCT` when it removes no row, as the module
/// explains.
#[derive(Debug, Default)]
pub struct EliminateDistinct;

impl OptimizerRule for EliminateDistinct {
    fn name(&self) -> &str {
        "eliminate_distinct"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        edit_distinct(plan, |input| {
            removes_no_row(input).then(|| input.as_ref().clone())
        })
    }
}

/// Give a one-table region's `SemanticFilter` the `DISTINCT` columns as
/// its stop key, as the module explains.
#[derive(Debug, Default)]
pub struct PushDownDistinct;

impl OptimizerRule for PushDownDistinct {
    fn name(&self) -> &str {
        "push_down_distinct"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        edit_distinct(plan, |input| {
            let key = stop_key(input)?;
            let marked = input
                .as_ref()
                .clone()
                .transform_down(|node| mark(node, &key))
                .ok()?;
            marked
                .transformed
                .then(|| LogicalPlan::Distinct(Distinct::All(Arc::new(marked.data))))
        })
    }
}

/// Replace the outermost `DISTINCT` with what `edit` returns for its
/// input, if anything.
fn edit_distinct(
    plan: LogicalPlan,
    edit: impl Fn(&Arc<LogicalPlan>) -> Option<LogicalPlan>,
) -> Result<Transformed<LogicalPlan>> {
    fn walk(
        plan: &LogicalPlan,
        edit: &dyn Fn(&Arc<LogicalPlan>) -> Option<LogicalPlan>,
    ) -> Result<Option<LogicalPlan>> {
        let below = match plan {
            LogicalPlan::Distinct(Distinct::All(input)) => return Ok(edit(input)),
            LogicalPlan::Limit(found) => &found.input,
            LogicalPlan::Sort(found) => &found.input,
            LogicalPlan::Projection(found) => &found.input,
            _ => return Ok(None),
        };
        let Some(edited) = walk(below, edit)? else {
            return Ok(None);
        };
        Ok(Some(plan.with_new_exprs(plan.expressions(), vec![edited])?))
    }
    Ok(match walk(&plan, &edit)? {
        Some(edited) => Transformed::yes(edited),
        None => Transformed::no(plan),
    })
}

fn as_node<T: 'static>(plan: &LogicalPlan) -> Option<&T> {
    match plan {
        LogicalPlan::Extension(extension) => extension.node.as_any().downcast_ref::<T>(),
        _ => None,
    }
}

/// The (qualifier, name) of each plain column a SELECT list returns.
fn returned_columns(projection: &Projection) -> BTreeSet<(String, String)> {
    projection
        .expr
        .iter()
        .filter_map(|expr| match expr.clone().unalias() {
            Expr::Column(column) => Some((
                column
                    .relation
                    .map(|r| r.table().to_string())
                    .unwrap_or_default(),
                column.name,
            )),
            _ => None,
        })
        .collect()
}

fn removes_no_row(input: &LogicalPlan) -> bool {
    match input {
        LogicalPlan::Aggregate(aggregate) => true_for_keys(aggregate, None),
        LogicalPlan::Projection(projection) => match projection.input.as_ref() {
            LogicalPlan::Aggregate(aggregate) => true_for_keys(aggregate, Some(projection)),
            _ => {
                let returned = returned_columns(projection);
                match table_keys(&projection.input) {
                    Some(keys) if !keys.is_empty() => keys.iter().all(|key| returned.contains(key)),
                    _ => false,
                }
            }
        },
        _ => false,
    }
}

/// Whether the SELECT list, if any, returns every grouping key.
fn true_for_keys(
    aggregate: &datafusion::logical_expr::Aggregate,
    projection: Option<&Projection>,
) -> bool {
    let Some(projection) = projection else {
        return true;
    };
    let returned = returned_columns(projection);
    (0..aggregate.group_expr.len()).all(|index| {
        let (qualifier, field) = aggregate.schema.qualified_field(index);
        let qualifier = qualifier.map(|q| q.table().to_string()).unwrap_or_default();
        returned.contains(&(qualifier, field.name().clone()))
    })
}

/// (alias, id column) of each table the plan scans, or None when a
/// table has no id column or is read through a derived table.
fn table_keys(plan: &LogicalPlan) -> Option<Vec<(String, String)>> {
    let mut keys = Vec::new();
    let mut known = true;
    plan.apply(|node| {
        let (alias, scan) = match node {
            LogicalPlan::SubqueryAlias(alias) => match alias.input.as_ref() {
                LogicalPlan::TableScan(scan) => (alias.alias.table().to_string(), scan),
                _ => {
                    known = false;
                    return Ok(TreeNodeRecursion::Stop);
                }
            },
            LogicalPlan::TableScan(scan) => (scan.table_name.table().to_string(), scan),
            _ => return Ok(TreeNodeRecursion::Continue),
        };
        let primary = scan.source.constraints().and_then(|constraints| {
            constraints.iter().find_map(|constraint| match constraint {
                Constraint::PrimaryKey(indices) if indices.len() == 1 => Some(indices[0]),
                _ => None,
            })
        });
        let Some(index) = primary else {
            known = false;
            return Ok(TreeNodeRecursion::Stop);
        };
        let column = scan.source.schema().field(index).name().clone();
        if !keys.contains(&(alias.clone(), column.clone())) {
            keys.push((alias, column));
        }
        Ok(TreeNodeRecursion::Jump)
    })
    .ok()?;
    known.then_some(keys)
}

/// The SELECT list's columns, when the region asks only one table's
/// `SemanticFilter` and returns plain columns of that table.
fn stop_key(input: &LogicalPlan) -> Option<Vec<String>> {
    let LogicalPlan::Projection(projection) = input else {
        return None;
    };
    let mut filters: Vec<&SemanticFilter> = Vec::new();
    let mut other = false;
    projection
        .input
        .apply(|node| {
            if let Some(filter) = as_node::<SemanticFilter>(node) {
                filters.push(filter);
            } else if as_node::<SemanticJoin>(node).is_some()
                || as_node::<SemanticColumn>(node).is_some()
            {
                other = true;
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .ok()?;
    let [filter] = filters.as_slice() else {
        return None;
    };
    if other || filter.predicates.iter().any(asks_score) {
        return None;
    }
    let tables: BTreeSet<String> = filter
        .predicates
        .iter()
        .flat_map(|predicate| predicate.column_refs())
        .filter_map(|column| column.relation.as_ref().map(|r| r.table().to_string()))
        .collect();
    let [table] = tables.into_iter().collect::<Vec<_>>().try_into().ok()?;
    let mut key: Vec<String> = Vec::new();
    for expr in &projection.expr {
        let Expr::Column(column) = expr else {
            return None;
        };
        if column.relation.as_ref().map(|r| r.table()) != Some(table.as_str()) {
            return None;
        }
        if !key.contains(&column.name) {
            key.push(column.name.clone());
        }
    }
    (!key.is_empty()).then_some(key)
}

fn asks_score(predicate: &Expr) -> bool {
    predicate
        .exists(|expr| Ok(matches!(expr, Expr::ScalarFunction(call) if call.name() == "ai_score")))
        .unwrap_or(false)
}

fn mark(node: LogicalPlan, key: &[String]) -> Result<Transformed<LogicalPlan>> {
    let Some(filter) = as_node::<SemanticFilter>(&node) else {
        return Ok(Transformed::no(node));
    };
    if filter.stop_key == key {
        return Ok(Transformed::no(node));
    }
    let mut marked = filter.clone();
    marked.stop_key = key.to_vec();
    let node: Arc<dyn UserDefinedLogicalNode> = Arc::new(marked);
    Ok(Transformed::yes(LogicalPlan::Extension(Extension { node })))
}
