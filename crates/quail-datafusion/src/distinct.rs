//! The DISTINCT rules of a model region: `EliminateDistinct` and
//! `PushDownDistinct`.
//!
//! Both read only the region's outermost `DISTINCT`, the one under its
//! LIMIT, ORDER BY, and outer SELECT list.
//!
//! - `EliminateDistinct` removes a `DISTINCT` that removes no row: the
//!   SELECT list returns every table's id column, which each table
//!   declares as its primary key.
//! - `PushDownDistinct` gives a one-table query's last `AI.IF` filter a
//!   fetch of one row per value of the selected columns: the executor
//!   stops asking the model about a document once a row with the same
//!   key has passed every filter. A plain filter right above the node is
//!   a WHERE term under the `DISTINCT`, so it moves below the node: the
//!   fetch counts only rows that pass it. No other filter moves below the
//!   node afterwards, unless it reads only the key columns
//!   ([`AiNode`]'s `prevent_predicate_push_down_columns`).

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Constraint, Result};
use datafusion::logical_expr::{
    Distinct, Expr, Extension, Filter, LogicalPlan, Projection, UserDefinedLogicalNode,
};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Fetch, Semantics};
use crate::extension::as_node;

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

/// Give a one-table region's last `AI.IF` filter a fetch of one row per
/// value of the `DISTINCT` columns, as the module explains.
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
            let mut found = false;
            let marked = input
                .as_ref()
                .clone()
                .transform_down(|node| mark(node, &key, &mut found))
                .ok()?;
            if !marked.transformed {
                return None;
            }
            let lowered = marked.data.transform_up(filter_below_fetch).ok()?.data;
            Some(LogicalPlan::Distinct(Distinct::All(Arc::new(lowered))))
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
    let LogicalPlan::Projection(projection) = input else {
        return false;
    };
    let returned = returned_columns(projection);
    match table_keys(&projection.input) {
        Some(keys) if !keys.is_empty() => keys.iter().all(|key| returned.contains(key)),
        _ => false,
    }
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

/// The SELECT list's columns, when the region asks only `AI.IF` filters
/// of one table and returns plain columns of that table.
fn stop_key(input: &LogicalPlan) -> Option<Vec<String>> {
    let LogicalPlan::Projection(projection) = input else {
        return None;
    };
    let mut filters: Vec<&AiNode> = Vec::new();
    projection
        .input
        .apply(|node| {
            if let Some(ai) = as_node::<AiNode>(node) {
                filters.push(ai);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .ok()?;
    let only_filters = filters
        .iter()
        .all(|ai| ai.function() == "ai_filter" && ai.semantics == Semantics::Rows);
    if filters.is_empty() || !only_filters {
        return None;
    }
    let tables: BTreeSet<String> = filters
        .iter()
        .flat_map(|ai| ai.call.column_refs())
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

/// A plain filter directly above an `AiNode` with a per-key fetch, below
/// the node instead: a term of the WHERE under the `DISTINCT`, which the
/// fetch counts rows after.
fn filter_below_fetch(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Filter(filter) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let Some(node) = as_node::<AiNode>(&filter.input) else {
        return Ok(Transformed::no(plan));
    };
    let reads_answer = filter.predicate.column_refs().iter().any(|column| {
        column.relation.is_none() && node.fields.iter().any(|field| field.name() == &column.name)
    });
    if !matches!(node.fetch, Some(Fetch::PerKey { .. })) || reads_answer {
        return Ok(Transformed::no(plan));
    }
    let below = Filter::try_new(filter.predicate.clone(), Arc::clone(&node.input))?;
    let moved = node.with_input(LogicalPlan::Filter(below))?;
    Ok(Transformed::yes(LogicalPlan::Extension(Extension {
        node: Arc::new(moved),
    })))
}

/// `node` with a fetch of one row per `key` when it is the first
/// `AiNode` found, which `found` records.
fn mark(node: LogicalPlan, key: &[String], found: &mut bool) -> Result<Transformed<LogicalPlan>> {
    if *found {
        return Ok(Transformed::new(node, false, TreeNodeRecursion::Stop));
    }
    let Some(ai) = as_node::<AiNode>(&node) else {
        return Ok(Transformed::no(node));
    };
    *found = true;
    let fetch = Fetch::PerKey {
        key: key.to_vec(),
        n: 1,
    };
    if ai.fetch.as_ref() == Some(&fetch) {
        return Ok(Transformed::new(node, false, TreeNodeRecursion::Stop));
    }
    let mut marked = ai.clone();
    marked.fetch = Some(fetch);
    let node: Arc<dyn UserDefinedLogicalNode> = Arc::new(marked);
    Ok(Transformed::new(
        LogicalPlan::Extension(Extension { node }),
        true,
        TreeNodeRecursion::Stop,
    ))
}
