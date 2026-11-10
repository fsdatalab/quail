//! The DISTINCT rules of a model region: `EliminateDistinct` and
//! `PushDownDistinct`.
//!
//! - `EliminateDistinct` removes the region's outermost `DISTINCT`, the
//!   one under its LIMIT, ORDER BY, and outer SELECT list, when it removes
//!   no row: the SELECT list returns every table's id column, which each
//!   table declares as its primary key.
//! - `PushDownDistinct` gives the `AiNode` under a `DISTINCT` a fetch of
//!   one row per value of the selected columns `K`:
//!
//!   ```text
//!   δ(π_K(AI(R))) ⇒ δ(π_K(AI(R; n = 1 per K)))
//!   ```
//!
//!   `K` is columns of the node's input, or the node's answer alone: a
//!   `DISTINCT` over a classification's label stops once each label has
//!   a row. The `DISTINCT` stays: it removes the rows the executor
//!   returns before it stops a key. The node has no fetch and no `HINT`
//!   yet: its tests are its own filter first, as a fetch counts rows
//!   after them. A plain filter between the SELECT list and the node is
//!   a WHERE term under the `DISTINCT`, so it moves below the node.
//!   No other filter moves below the node afterwards, unless it reads
//!   only the key columns ([`AiNode`]'s
//!   `prevent_predicate_push_down_columns`).
//!
//! The rule reads a `DISTINCT` as written or as the GROUP BY that
//! DataFusion's `replace_distinct_aggregate` makes of it. So a filter on
//! `K` above a `DISTINCT`, which commutes with it and moves below it
//! as DataFusion rewrites the `DISTINCT`, does not stop the rule.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Constraint, Result};
use datafusion::logical_expr::{Aggregate, Distinct, Expr, Filter, LogicalPlan, Projection};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Fetch, Semantics};
use crate::extension::{as_node, extension_node};

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

/// Give the `AiNode` under each `DISTINCT` a fetch of one row per value
/// of the `DISTINCT` columns, as the module explains.
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
        plan.transform_up(|node| {
            let Some(input) = distinct_input(&node) else {
                return Ok(Transformed::no(node));
            };
            let Some(keyed) = with_per_key_fetch(input)? else {
                return Ok(Transformed::no(node));
            };
            let rebuilt = node.with_new_exprs(node.expressions(), vec![keyed])?;
            Ok(Transformed::yes(rebuilt))
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

/// The input of a `DISTINCT`, written as one or as a GROUP BY.
fn distinct_input(plan: &LogicalPlan) -> Option<&LogicalPlan> {
    match plan {
        LogicalPlan::Distinct(Distinct::All(input)) => Some(input),
        LogicalPlan::Aggregate(aggregate) if is_distinct(aggregate) => Some(&aggregate.input),
        _ => None,
    }
}

/// `input`, a `DISTINCT`'s input, with the per-key fetch on its
/// `AiNode` and the plain filters above the node moved below it; None
/// when the module's conditions do not hold.
fn with_per_key_fetch(input: &LogicalPlan) -> Result<Option<LogicalPlan>> {
    let (selected, mut below): (Vec<Expr>, &LogicalPlan) = match input {
        LogicalPlan::Projection(projection) => (projection.expr.clone(), &projection.input),
        other => (
            other
                .schema()
                .columns()
                .into_iter()
                .map(Expr::Column)
                .collect(),
            other,
        ),
    };
    let mut filters: Vec<Expr> = Vec::new();
    while let LogicalPlan::Filter(filter) = below {
        filters.push(filter.predicate.clone());
        below = &filter.input;
    }
    let Some(node) = as_node::<AiNode>(below) else {
        return Ok(None);
    };
    if node.semantics != Semantics::Rows || node.fetch.is_some() || !node.accepts_fetch() {
        return Ok(None);
    }
    let reads_answer = |expr: &Expr| {
        expr.column_refs().iter().any(|column| {
            column.relation.is_none()
                && node.fields.iter().any(|field| field.name() == &column.name)
        })
    };
    if filters.iter().any(reads_answer) {
        return Ok(None);
    }
    let Some(key) = stop_key(node, &selected) else {
        return Ok(None);
    };
    let mut lowered = node.input.as_ref().clone();
    for predicate in filters.into_iter().rev() {
        lowered = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(lowered))?);
    }
    let mut keyed = node.with_input(lowered)?;
    keyed.fetch = Some(Fetch::PerKey { key, n: 1 });
    let keyed = extension_node(keyed.checked()?);
    Ok(Some(match input {
        LogicalPlan::Projection(projection) => LogicalPlan::Projection(Projection::try_new(
            projection.expr.clone(),
            Arc::new(keyed),
        )?),
        _ => keyed,
    }))
}

/// The names of the selected columns, when they are columns of the
/// node's input, each name once there, or the node's returned answer
/// alone.
fn stop_key(node: &AiNode, selected: &[Expr]) -> Option<Vec<String>> {
    let columns: Vec<&Column> = selected
        .iter()
        .map(|expr| match expr {
            Expr::Column(column) => Some(column),
            Expr::Alias(alias) => match alias.expr.as_ref() {
                Expr::Column(column) => Some(column),
                _ => None,
            },
            _ => None,
        })
        .collect::<Option<_>>()?;
    let returns_answer = node
        .returned_fields()
        .iter()
        .any(|field| field.name() == &node.answer);
    if let [column] = columns.as_slice()
        && column.relation.is_none()
        && column.name == node.answer
        && returns_answer
    {
        return Some(vec![node.answer.clone()]);
    }
    let input = node.input.schema();
    let mut key: Vec<String> = Vec::new();
    for column in columns {
        let named = input
            .fields()
            .iter()
            .filter(|field| field.name() == &column.name)
            .count();
        if !input.has_column(column) || named != 1 {
            return None;
        }
        if !key.contains(&column.name) {
            key.push(column.name.clone());
        }
    }
    (!key.is_empty()).then_some(key)
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
