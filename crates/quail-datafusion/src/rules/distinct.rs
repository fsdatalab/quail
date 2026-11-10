//! The `EliminateDistinct` optimizer rule: a `DISTINCT` that removes no
//! row is dropped.
//!
//! The rule removes the outermost `DISTINCT`, the one under the query's
//! LIMIT, ORDER BY, and outer SELECT list, when the SELECT list returns
//! every table's id column, which each table declares as its primary
//! key. It reads a `DISTINCT` as written or as the GROUP BY that
//! DataFusion's `replace_distinct_aggregate` makes of it.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Constraint, Result};
use datafusion::logical_expr::{Aggregate, Distinct, Expr, LogicalPlan, Projection};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

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
