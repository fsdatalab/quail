//! The `MergeRepeatedAiCalls` optimizer rule: an AI call the rows below
//! already answered reads that answer.
//!
//! When an `AiNode` makes the same call, with the same guard, as an
//! `AiNode` below it, its answer for each row is the lower node's:
//!
//! ```text
//! AI_{f → b}(AI_{f → a}(R)) ⇒ π_{*, a → b}(AI_{f → a}(R))
//! ```
//!
//! For example, `WHERE AI.IF(q) AND AI.IF(q)` asks the model `q` once.
//! The rule reads down from the upper node through filters, other
//! `AiNode`s of the same rows, and the projections and aliases of
//! derived tables and CTEs, to the first node with the same call, each
//! column the call reads mapped to the column it names below. Across a
//! derived table, the lower node's answers are added to its SELECT list
//! where it does not return them already. For example, in `SELECT
//! AI.CLASSIFY(d.body, ...) FROM (SELECT r.body, AI.CLASSIFY(r.body, ...)
//! AS t FROM reviews r) d` the outer call reads `d.t`.
//!
//! The rule runs before [`crate::rules::derive`], while no node has a
//! filter, guard, fetch, or projection, so one node stands for the call
//! and `derive` gives it what every read of its answer needs together.
//! It leaves a node that has one of those options, and a gate, alone.
//! The upper node's `HINT` estimated the rows of a test the lower node
//! already makes, and goes with it.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::{Expr, Filter, LogicalPlan, Projection, SubqueryAlias};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};

/// Make each repeated AI call read the answer of the same call below it,
/// as the module explains.
#[derive(Debug, Default)]
pub struct MergeRepeatedAiCalls;

impl OptimizerRule for MergeRepeatedAiCalls {
    fn name(&self) -> &str {
        "merge_repeated_ai_calls"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up(merge)
    }
}

fn merge(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let Some(upper) = as_node::<AiNode>(&plan) else {
        return Ok(Transformed::no(plan));
    };
    if upper.semantics != Semantics::Rows || upper.has_options() {
        return Ok(Transformed::no(plan));
    }
    let Some(lower) = same_call_below(upper) else {
        return Ok(Transformed::no(plan));
    };
    let answers = &upper.fields;
    let input = upper.input.as_ref().clone();
    let names: Vec<String> = lower
        .fields
        .iter()
        .map(|field| field.name().clone())
        .collect();
    // the lower node's answers as the upper node's input names them,
    // added to each derived table's SELECT list between them
    let Some((exposed, read)) = expose(&input, lower, &names)? else {
        return Ok(Transformed::no(plan));
    };
    let mut exprs: Vec<Expr> = input
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    exprs.extend(
        answers
            .iter()
            .zip(read)
            .map(|(field, column)| Expr::Column(column).alias(field.name())),
    );
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(exprs, Arc::new(exposed))?,
    )))
}

/// `plan` with the columns `names` of `lower`, a node below it, passed up
/// through each projection and alias between them, and those columns as
/// `plan` names them; None when a projection cannot pass one on.
fn expose(
    plan: &LogicalPlan,
    lower: &AiNode,
    names: &[String],
) -> Result<Option<(LogicalPlan, Vec<Column>)>> {
    if as_node::<AiNode>(plan) == Some(lower) {
        let read = names.iter().map(Column::new_unqualified).collect();
        return Ok(Some((plan.clone(), read)));
    }
    let Some(input) = plan.inputs().first().copied() else {
        return Ok(None);
    };
    let Some((exposed, read)) = expose(input, lower, names)? else {
        return Ok(None);
    };
    Ok(match plan {
        LogicalPlan::Filter(filter) => Some((
            LogicalPlan::Filter(Filter::try_new(
                filter.predicate.clone(),
                Arc::new(exposed),
            )?),
            read,
        )),
        LogicalPlan::Extension(_) => {
            let Some(node) = as_node::<AiNode>(plan) else {
                return Ok(None);
            };
            Some((extension_node(node.with_input(exposed)?), read))
        }
        LogicalPlan::Projection(projection) => {
            let mut exprs = projection.expr.clone();
            let mut passed = Vec::new();
            for column in read {
                let found = projection.expr.iter().position(|expr| {
                    let inner = match expr {
                        Expr::Alias(alias) => alias.expr.as_ref(),
                        other => other,
                    };
                    *inner == Expr::Column(column.clone())
                });
                let index = match found {
                    Some(index) => index,
                    None => {
                        exprs.push(Expr::Column(column));
                        exprs.len() - 1
                    }
                };
                passed.push(index);
            }
            let Ok(rebuilt) = Projection::try_new(exprs, Arc::new(exposed)) else {
                return Ok(None);
            };
            let read = passed
                .into_iter()
                .map(|index| {
                    let (qualifier, field) = rebuilt.schema.qualified_field(index);
                    Column::new(qualifier.cloned(), field.name())
                })
                .collect();
            Some((LogicalPlan::Projection(rebuilt), read))
        }
        LogicalPlan::SubqueryAlias(alias) => {
            let rebuilt = SubqueryAlias::try_new(Arc::new(exposed), alias.alias.clone())?;
            let read = read
                .into_iter()
                .map(|column| Column::new(Some(alias.alias.clone()), column.name))
                .collect();
            Some((LogicalPlan::SubqueryAlias(rebuilt), read))
        }
        _ => None,
    })
}

/// `expr` with each column of `projection`'s output read as the
/// expression that computes it; None when one is not an output column.
pub(crate) fn through_projection(expr: &Expr, projection: &Projection) -> Option<Expr> {
    let mut missing = false;
    let found = expr
        .clone()
        .transform_up(|found| {
            let Expr::Column(column) = &found else {
                return Ok(Transformed::no(found));
            };
            let index = (0..projection.expr.len()).find(|index| {
                let (qualifier, field) = projection.schema.qualified_field(*index);
                field.name() == &column.name
                    && (column.relation.is_none() || column.relation.as_ref() == qualifier)
            });
            match index {
                Some(index) => Ok(Transformed::yes(match &projection.expr[index] {
                    Expr::Alias(alias) => alias.expr.as_ref().clone(),
                    other => other.clone(),
                })),
                None => {
                    missing = true;
                    Ok(Transformed::no(found))
                }
            }
        })
        .ok()?
        .data;
    (!missing).then_some(found)
}

/// `expr` with each column of the derived table `alias` read as its
/// input's column of the same name.
pub(crate) fn through_alias(expr: &Expr, alias: &SubqueryAlias) -> Option<Expr> {
    let mut missing = false;
    let found = expr
        .clone()
        .transform_up(|found| {
            let Expr::Column(column) = &found else {
                return Ok(Transformed::no(found));
            };
            if column
                .relation
                .as_ref()
                .is_some_and(|relation| *relation != alias.alias)
            {
                missing = true;
                return Ok(Transformed::no(found));
            }
            match alias
                .input
                .schema()
                .qualified_field_with_unqualified_name(&column.name)
            {
                Ok((qualifier, field)) => Ok(Transformed::yes(Expr::Column(Column::new(
                    qualifier.cloned(),
                    field.name(),
                )))),
                Err(_) => {
                    missing = true;
                    Ok(Transformed::no(found))
                }
            }
        })
        .ok()?
        .data;
    (!missing).then_some(found)
}

/// The first `AiNode` below `upper`, through filters, other nodes of the
/// same rows, and the projections and aliases of derived tables, that
/// makes `upper`'s call, each column read by the name the node's input
/// gives it, has no options, and adds columns of the same types.
fn same_call_below(upper: &AiNode) -> Option<&AiNode> {
    let mut plan = upper.input.as_ref();
    let mut call = upper.call.clone();
    loop {
        plan = match plan {
            LogicalPlan::Filter(filter) => filter.input.as_ref(),
            LogicalPlan::Projection(projection) => {
                call = through_projection(&call, projection)?;
                projection.input.as_ref()
            }
            LogicalPlan::SubqueryAlias(alias) => {
                call = through_alias(&call, alias)?;
                alias.input.as_ref()
            }
            LogicalPlan::Extension(_) => {
                let node = as_node::<AiNode>(plan)?;
                if node.semantics != Semantics::Rows {
                    return None;
                }
                let same_types = node.fields.len() == upper.fields.len()
                    && node
                        .fields
                        .iter()
                        .zip(&upper.fields)
                        .all(|(found, field)| found.data_type() == field.data_type());
                if node.call == call && same_types && !node.has_options() {
                    return Some(node);
                }
                node.input.as_ref()
            }
            _ => return None,
        };
    }
}
