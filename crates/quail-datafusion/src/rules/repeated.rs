use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::{
    Expr, Filter, LogicalPlan, Projection, SubqueryAlias, UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::AiNode;
use crate::plan::{as_node, extension_node};

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
    if upper.is_join() || upper.has_options() {
        return Ok(Transformed::no(plan));
    }
    let Some(lower) = same_call_below(upper) else {
        return Ok(Transformed::no(plan));
    };
    let answers = upper.added_names();
    let input = upper.inputs()[0].clone();
    let names = lower.added_names();
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
            .map(|(name, column)| Expr::Column(column).alias(name)),
    );
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(exprs, Arc::new(exposed))?,
    )))
}

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

fn same_call_below(upper: &AiNode) -> Option<&AiNode> {
    let mut plan = upper.inputs()[0];
    let mut call = upper.call();
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
                if node.is_join() {
                    return None;
                }
                if node.repeats(&call, upper) {
                    return Some(node);
                }
                node.inputs()[0]
            }
            _ => return None,
        };
    }
}
