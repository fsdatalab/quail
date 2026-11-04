//! A model region after DataFusion's optimizer, read as the binder reads
//! a bound region.
//!
//! Inside a model region DataFusion's own rules prune columns and push
//! filters down: [`optimize_region`] runs `OptimizeProjections` and
//! `PushDownFilter` over the lowered region. The binder reads a SELECT
//! as DataFusion's SQL planner binds it, so [`as_bound`] reads the
//! optimized region back into that shape:
//!
//! - Quail's plan nodes become the terms and calls DataFusion bound
//!   ([`crate::raise_ai_calls`]).
//! - A filter DataFusion pushed below a table's alias, or into its
//!   `TableScan`, moves above the alias, naming the table's columns by
//!   the alias.
//! - A projection that only passes columns through is removed, except
//!   a SELECT list.
//! - The filters DataFusion pushed into a join's inputs join the
//!   WHERE above the join: the terms left above it first, then each
//!   input's, left to right.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, TableReference};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Expr, Filter, JoinType, LogicalPlan, Operator, Projection, SubqueryAlias, TableScan,
};
use datafusion::optimizer::optimize_projections::OptimizeProjections;
use datafusion::optimizer::push_down_filter::PushDownFilter;
use datafusion::optimizer::{Optimizer, OptimizerConfig, OptimizerContext, OptimizerRule};

use crate::lower::raise_ai_calls;
use crate::region::ModelRegion;

/// Run DataFusion's rules inside each model region, which DataFusion's
/// optimizer treats as a leaf, and keep the result beside the region's
/// bound plan.
#[derive(Debug, Default)]
pub struct OptimizeModelRegions;

impl OptimizerRule for OptimizeModelRegions {
    fn name(&self) -> &str {
        "optimize_model_regions"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up(|node| {
            if let LogicalPlan::Extension(extension) = &node
                && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
                && region.optimized.is_none()
            {
                // a region DataFusion's rules cannot optimize keeps its bound plan
                let Ok(optimized) = optimize_region(&region.plan) else {
                    return Ok(Transformed::no(node));
                };
                let mut region = region.clone();
                region.optimized = Some(optimized);
                return Ok(Transformed::yes(LogicalPlan::Extension(
                    datafusion::logical_expr::Extension {
                        node: Arc::new(region),
                    },
                )));
            }
            Ok(Transformed::no(node))
        })
    }
}

/// Prune columns and push filters down inside a lowered model region.
///
/// # Errors
///
/// A DataFusion rule fails.
pub fn optimize_region(plan: &LogicalPlan) -> Result<LogicalPlan> {
    let rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> = vec![
        Arc::new(OptimizeProjections::new()),
        Arc::new(PushDownFilter::new()),
    ];
    Optimizer::with_rules(rules).optimize(plan.clone(), &OptimizerContext::new(), |_, _| {})
}

/// Read an optimized region in the shape DataFusion's SQL planner
/// binds, as the module explains.
///
/// # Errors
///
/// A rebuilt operator's schema does not fit DataFusion's rules.
pub fn as_bound(plan: &LogicalPlan) -> Result<LogicalPlan> {
    let raised = raise_ai_calls(plan.clone())?;
    read_select(&lift_scan_filters(select_above_alias(raised)?)?)
}

/// A region whose root is a table's alias over its SELECT list, as
/// DataFusion leaves `SELECT d.id FROM documents d WHERE ...` after
/// pushing the projection through the alias: the SELECT list above the
/// alias, naming the table's columns by the alias.
fn select_above_alias(plan: LogicalPlan) -> Result<LogicalPlan> {
    let LogicalPlan::SubqueryAlias(alias) = &plan else {
        return Ok(plan);
    };
    let LogicalPlan::Projection(projection) = alias.input.as_ref() else {
        return Ok(plan);
    };
    let mut below = projection.input.as_ref();
    while let LogicalPlan::Filter(filter) = below {
        below = filter.input.as_ref();
    }
    let LogicalPlan::TableScan(scan) = below else {
        // a CTE merged into its reader: the reader's SELECT list passed
        // the CTE's columns through
        let columns = plan
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        return Ok(LogicalPlan::Projection(Projection::try_new(
            columns,
            Arc::new(plan),
        )?));
    };
    let table = scan.table_name.clone();
    let aliased = LogicalPlan::SubqueryAlias(SubqueryAlias::try_new(
        Arc::clone(&projection.input),
        alias.alias.clone(),
    )?);
    let exprs = projection
        .expr
        .iter()
        .map(|expr| requalified(expr.clone(), &table, &alias.alias))
        .collect::<Result<Vec<_>>>()?;
    Ok(LogicalPlan::Projection(Projection::try_new(
        exprs,
        Arc::new(aliased),
    )?))
}

/// Each column of `table` named by `alias` instead.
fn requalified(expr: Expr, table: &TableReference, alias: &TableReference) -> Result<Expr> {
    Ok(expr
        .transform_up(|node| match &node {
            Expr::Column(column) if column.relation.as_ref() == Some(table) => Ok(
                Transformed::yes(Expr::Column(Column::new(Some(alias.clone()), &column.name))),
            ),
            _ => Ok(Transformed::no(node)),
        })?
        .data)
}

/// Move each filter below a table's alias, or in its `TableScan`, above
/// the alias.
fn lift_scan_filters(plan: LogicalPlan) -> Result<LogicalPlan> {
    Ok(plan
        .transform_up_with_subqueries(|node| {
            let LogicalPlan::SubqueryAlias(alias) = &node else {
                return Ok(Transformed::no(node));
            };
            let mut terms: Vec<Expr> = Vec::new();
            let mut below = alias.input.as_ref().clone();
            loop {
                match &below {
                    LogicalPlan::Filter(filter) => {
                        terms.splice(
                            0..0,
                            split_conjunction(&filter.predicate).into_iter().cloned(),
                        );
                        below = filter.input.as_ref().clone();
                    }
                    LogicalPlan::Projection(projection) if passes_columns(projection) => {
                        below = projection.input.as_ref().clone();
                    }
                    _ => break,
                }
            }
            let LogicalPlan::TableScan(scan) = below else {
                return lift_into_reader(alias);
            };
            let pruned = !matches!(alias.input.as_ref(), LogicalPlan::TableScan(_));
            let mut pushed = scan.filters.clone();
            pushed.extend(terms);
            if pushed.is_empty() && !pruned {
                return Ok(Transformed::no(node));
            }
            let table = scan.table_name.clone();
            let scan = TableScan {
                filters: Vec::new(),
                ..scan
            };
            let aliased = LogicalPlan::SubqueryAlias(SubqueryAlias::try_new(
                Arc::new(LogicalPlan::TableScan(scan)),
                alias.alias.clone(),
            )?);
            let terms = pushed
                .into_iter()
                .map(|term| requalified(term, &table, &alias.alias))
                .collect::<Result<Vec<_>>>()?;
            Ok(Transformed::yes(match conjunction(terms) {
                Some(predicate) => {
                    LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(aliased))?)
                }
                None => aliased,
            }))
        })?
        .data)
}

/// A filter DataFusion pushed into a derived table, directly below its
/// alias and above its LIMIT, read above the alias as the reader's WHERE.
fn lift_into_reader(alias: &SubqueryAlias) -> Result<Transformed<LogicalPlan>> {
    let node = LogicalPlan::SubqueryAlias(alias.clone());
    let LogicalPlan::Filter(filter) = alias.input.as_ref() else {
        return Ok(Transformed::no(node));
    };
    // DataFusion pushes a reader's filter into a derived table until a
    // LIMIT stops it; any other filter there is the table's own WHERE
    if !matches!(filter.input.as_ref(), LogicalPlan::Limit(_)) {
        return Ok(Transformed::no(node));
    }
    let schema = filter.input.schema();
    let renamed = filter
        .predicate
        .clone()
        .transform_up(|expr| match &expr {
            Expr::Column(column) if schema.has_column(column) => Ok(Transformed::yes(
                Expr::Column(Column::new(Some(alias.alias.clone()), &column.name)),
            )),
            _ => Ok(Transformed::no(expr)),
        })?
        .data;
    let aliased = LogicalPlan::SubqueryAlias(SubqueryAlias::try_new(
        Arc::clone(&filter.input),
        alias.alias.clone(),
    )?);
    Ok(Transformed::yes(LogicalPlan::Filter(Filter::try_new(
        renamed,
        Arc::new(aliased),
    )?)))
}

/// Whether an ON term stays in the ON: an AI call, or an equality whose
/// sides each read one table, a different one.
fn is_join_term(term: &Expr) -> bool {
    let mut asks = false;
    let _ = term.apply(|expr| {
        if crate::is_ai_call(expr) {
            asks = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    let tables = |side: &Expr| -> Vec<Option<TableReference>> {
        let mut found: Vec<Option<TableReference>> = side
            .column_refs()
            .into_iter()
            .map(|column| column.relation.clone())
            .collect();
        found.sort_by_key(|table| table.as_ref().map(ToString::to_string));
        found.dedup();
        found
    };
    asks || matches!(term, Expr::BinaryExpr(binary)
    if binary.op == Operator::Eq && {
        let (left, right) = (tables(&binary.left), tables(&binary.right));
        left.len() == 1 && right.len() == 1 && left != right
    })
}

fn passes_columns(projection: &Projection) -> bool {
    projection
        .expr
        .iter()
        .all(|expr| matches!(expr, Expr::Column(_)))
}

fn rebuilt(plan: &LogicalPlan, inputs: Vec<LogicalPlan>) -> Result<LogicalPlan> {
    plan.with_new_exprs(plan.expressions(), inputs)
}

/// Read one SELECT: keep the operators of its SELECT list, ORDER BY,
/// DISTINCT, and LIMIT, and read the relation below them.
fn read_select(plan: &LogicalPlan) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Limit(_) | LogicalPlan::Sort(_) | LogicalPlan::Distinct(_) => {
            let inputs = plan
                .inputs()
                .into_iter()
                .map(read_select)
                .collect::<Result<Vec<_>>>()?;
            rebuilt(plan, inputs)
        }
        LogicalPlan::Projection(projection) => {
            let input = if matches!(
                projection.input.as_ref(),
                LogicalPlan::Sort(_) | LogicalPlan::Distinct(_)
            ) {
                read_select(&projection.input)?
            } else {
                read_body(&projection.input)?
            };
            rebuilt(plan, vec![input])
        }
        // DataFusion removes a SELECT list that passes every column through
        other => {
            let body = read_body(other)?;
            let columns = body
                .schema()
                .columns()
                .into_iter()
                .map(Expr::Column)
                .collect();
            Ok(LogicalPlan::Projection(Projection::try_new(
                columns,
                Arc::new(body),
            )?))
        }
    }
}

/// Read what a SELECT list reads: windows, HAVING, the grouping, then
/// the relation and its WHERE.
fn read_body(plan: &LogicalPlan) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Window(_) | LogicalPlan::Aggregate(_) => {
            let inputs = plan
                .inputs()
                .into_iter()
                .map(read_body)
                .collect::<Result<Vec<_>>>()?;
            rebuilt(plan, inputs)
        }
        LogicalPlan::Filter(filter)
            if matches!(filter.input.as_ref(), LogicalPlan::Aggregate(_)) =>
        {
            rebuilt(plan, vec![read_body(&filter.input)?])
        }
        LogicalPlan::Projection(projection) if passes_columns(projection) => {
            read_body(&projection.input)
        }
        other => read_relation(other),
    }
}

/// Read a relation with its WHERE above it: the terms of every filter in
/// the join tree, in one `Filter`.
fn read_relation(plan: &LogicalPlan) -> Result<LogicalPlan> {
    let mut terms = Vec::new();
    let relation = take_filters(plan, &mut terms)?;
    match conjunction(terms) {
        Some(predicate) => Ok(LogicalPlan::Filter(Filter::try_new(
            predicate,
            Arc::new(relation),
        )?)),
        None => Ok(relation),
    }
}

/// The join tree without its filters and pruning projections; the
/// filters' terms go to `terms` in the order the module gives.
fn take_filters(plan: &LogicalPlan, terms: &mut Vec<Expr>) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Filter(filter) => {
            terms.extend(split_conjunction(&filter.predicate).into_iter().cloned());
            take_filters(&filter.input, terms)
        }
        LogicalPlan::Projection(projection) if passes_columns(projection) => {
            take_filters(&projection.input, terms)
        }
        LogicalPlan::Join(join) => {
            // DataFusion moves a WHERE term over both inputs into an inner
            // join's ON; the binder reads ON for equalities and AI terms
            let mut join = join.clone();
            if join.join_type == JoinType::Inner
                && let Some(filter) = join.filter.take()
            {
                let (kept, moved): (Vec<Expr>, Vec<Expr>) = split_conjunction(&filter)
                    .into_iter()
                    .cloned()
                    .partition(is_join_term);
                terms.extend(moved);
                join.filter = conjunction(kept);
            }
            let plan = LogicalPlan::Join(join);
            let inputs = plan
                .inputs()
                .into_iter()
                .map(|input| take_filters(input, terms))
                .collect::<Result<Vec<_>>>()?;
            rebuilt(&plan, inputs)
        }
        LogicalPlan::SubqueryAlias(_) => read_alias(plan),
        other => Ok(other.clone()),
    }
}

/// A table's alias as it is, or a derived table's alias over its SELECT;
/// an alias of a CTE names the CTE's own alias.
fn read_alias(plan: &LogicalPlan) -> Result<LogicalPlan> {
    let LogicalPlan::SubqueryAlias(alias) = plan else {
        return read_select(plan);
    };
    match alias.input.as_ref() {
        LogicalPlan::TableScan(_) => Ok(plan.clone()),
        inner => rebuilt(plan, vec![read_alias(inner)?]),
    }
}
