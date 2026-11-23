use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Expr, JoinType, LogicalPlan, LogicalPlanBuilder, Projection, UserDefinedLogicalNodeCore,
};

use super::derive::is_join_filter;
use crate::ai::AiNode;
use crate::functions::{is_ai_call, is_quail_call};
use crate::plan::{as_node, extension_node};

#[derive(Debug, Default)]
pub(crate) struct Above {
    steps: Vec<Step>,
}

#[derive(Debug)]
enum Step {
    Filter(Expr),
    Projection(Vec<Expr>),
}

impl Above {
    pub(crate) fn build(self, region: LogicalPlan) -> Result<LogicalPlan> {
        let mut builder = LogicalPlanBuilder::from(region);
        for step in self.steps {
            builder = match step {
                Step::Filter(predicate) => builder.filter(predicate)?,
                Step::Projection(exprs) => builder.project(exprs)?,
            };
        }
        builder.build()
    }
}

pub(crate) fn lift_relational(
    plan: LogicalPlan,
    join_filters: bool,
) -> Result<(LogicalPlan, Above)> {
    let plan = inline_computed_columns(plan)?;
    let LogicalPlan::Projection(root) = &plan else {
        return Ok((plan, Above::default()));
    };
    let answers = answer_tables(&root.input);
    let mut lifted: Vec<Expr> = Vec::new();
    let relation = take_row_tests(
        root.input.as_ref().clone(),
        &answers,
        (false, join_filters),
        &mut lifted,
    )?;
    let tests = lifted;
    let hidden = tests.iter().any(|test| {
        test.column_refs()
            .iter()
            .any(|column| !relation.schema().has_column(column))
    });
    if hidden {
        return Ok((with_a_column(plan)?, Above::default()));
    }
    let mut kept: Vec<Expr> = Vec::new();
    let mut selected: Vec<Expr> = Vec::new();
    for (expr, (qualifier, field)) in root.expr.iter().zip(root.schema.iter()) {
        let calls_model = expr.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(false);
        let column = matches!(expr, Expr::Column(_));
        if calls_model || column {
            if !kept.contains(expr) {
                kept.push(expr.clone());
            }
            selected.push(Expr::Column(Column::new(qualifier.cloned(), field.name())));
            continue;
        }
        let computed = expr.clone().unalias();
        selected.push(computed.alias_qualified(qualifier.cloned(), field.name()));
    }
    let moved = !tests.is_empty() || selected.iter().any(|expr| !matches!(expr, Expr::Column(_)));
    if !moved {
        return Ok((with_a_column(plan)?, Above::default()));
    }
    let returned: Vec<Column> = kept
        .iter()
        .filter_map(|expr| match expr {
            Expr::Column(column) => Some(column.clone()),
            _ => None,
        })
        .collect();
    let mut read: Vec<Expr> = Vec::new();
    let columns_read = tests
        .iter()
        .chain(&selected)
        .flat_map(|expr| expr.column_refs().into_iter().cloned());
    for column in columns_read {
        if relation.schema().has_column(&column) && !returned.contains(&column) {
            let expr = Expr::Column(column);
            if !read.contains(&expr) {
                read.push(expr);
            }
        }
    }
    kept.extend(read);
    let region = with_a_column(LogicalPlan::Projection(Projection::try_new(
        kept,
        Arc::new(relation),
    )?))?;
    let mut above = Above::default();
    if let Some(predicate) = conjunction(tests) {
        above.steps.push(Step::Filter(predicate));
    }
    above.steps.push(Step::Projection(selected));
    Ok((region, above))
}

fn with_a_column(plan: LogicalPlan) -> Result<LogicalPlan> {
    let LogicalPlan::Projection(projection) = &plan else {
        return Ok(plan);
    };
    if !projection.expr.is_empty() {
        return Ok(plan);
    }
    let Some(column) = projection.input.schema().columns().into_iter().next() else {
        return Ok(plan);
    };
    Ok(LogicalPlan::Projection(Projection::try_new(
        vec![Expr::Column(column)],
        Arc::clone(&projection.input),
    )?))
}

fn inline_computed_columns(plan: LogicalPlan) -> Result<LogicalPlan> {
    plan.transform_down(|above| {
        if !matches!(above, LogicalPlan::Projection(_) | LogicalPlan::Filter(_)) {
            return Ok(Transformed::no(above));
        }
        let LogicalPlan::Projection(below) = above.inputs()[0] else {
            return Ok(Transformed::no(above));
        };
        let plain = |expr: &Expr| !expr.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(true);
        let computed: Vec<(String, Expr)> = below
            .expr
            .iter()
            .zip(below.schema.fields())
            .filter(|(expr, _)| !matches!(expr, Expr::Column(_)) && plain(expr))
            .map(|(expr, field)| (field.name().clone(), expr.clone().unalias()))
            .collect();
        if computed.is_empty() || computed.len() != below.expr.len() - passed(below) {
            return Ok(Transformed::no(above));
        }
        let mut columns: Vec<Expr> = below
            .expr
            .iter()
            .filter(|expr| matches!(expr, Expr::Column(_)))
            .cloned()
            .collect();
        for (_, value) in &computed {
            for column in value.column_refs() {
                let column = Expr::Column(column.clone());
                if !columns.contains(&column) {
                    columns.push(column);
                }
            }
        }
        let input =
            LogicalPlan::Projection(Projection::try_new(columns, Arc::clone(&below.input))?);
        let exprs = above
            .expressions()
            .into_iter()
            .map(|expr| {
                let name = expr.schema_name().to_string();
                let read = expr.transform_up(|found| match &found {
                    Expr::Column(column) if column.relation.is_none() => Ok(
                        match computed.iter().find(|(name, _)| *name == column.name) {
                            Some((_, value)) => Transformed::yes(value.clone()),
                            None => Transformed::no(found),
                        },
                    ),
                    _ => Ok(Transformed::no(found)),
                })?;
                Ok(match (&above, read.transformed) {
                    (LogicalPlan::Projection(_), true) => read.data.alias_if_changed(name)?,
                    _ => read.data,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Transformed::yes(above.with_new_exprs(exprs, vec![input])?))
    })
    .map(|inlined| inlined.data)
}

fn passed(projection: &Projection) -> usize {
    projection
        .expr
        .iter()
        .filter(|expr| matches!(expr, Expr::Column(_)))
        .count()
}

type Answers = BTreeMap<String, (String, BTreeSet<String>)>;

fn answer_tables(plan: &LogicalPlan) -> Answers {
    let mut found = BTreeMap::new();
    let _ = plan.apply(|node| {
        if let Some(ai) = as_node::<AiNode>(node) {
            let tables: BTreeSet<String> =
                ai.call_tables().iter().map(ToString::to_string).collect();
            for name in ai.added_names() {
                found.insert(name, (ai.answer().to_string(), tables.clone()));
            }
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    });
    found
}

fn is_row_test(term: &Expr, answers: &Answers, join_key: bool) -> bool {
    let special = term
        .exists(|inner| {
            Ok(is_quail_call(inner)
                || matches!(
                    inner,
                    Expr::Exists(_) | Expr::InSubquery(_) | Expr::ScalarSubquery(_)
                ))
        })
        .unwrap_or(true);
    if special || (join_key && is_column_equality(term)) {
        return false;
    }
    let mut tables: BTreeSet<String> = BTreeSet::new();
    let mut calls: BTreeSet<&str> = BTreeSet::new();
    for column in term.column_refs() {
        match &column.relation {
            Some(table) => {
                tables.insert(table.to_string());
            }
            None => {
                if let Some((owner, read)) = answers.get(&column.name) {
                    calls.insert(owner);
                    tables.extend(read.iter().cloned());
                }
            }
        }
    }
    tables.len() > 1 || calls.len() > 1
}

fn is_column_equality(term: &Expr) -> bool {
    matches!(term, Expr::BinaryExpr(binary)
        if binary.op == datafusion::logical_expr::Operator::Eq
            && matches!((binary.left.as_ref(), binary.right.as_ref()),
                (Expr::Column(left), Expr::Column(right)) if left.relation != right.relation))
}

fn take_row_tests(
    plan: LogicalPlan,
    answers: &Answers,
    (fixed, join_filters): (bool, bool),
    lifted: &mut Vec<Expr>,
) -> Result<LogicalPlan> {
    let lifts = |term: &Expr, join_key: bool| {
        !fixed
            && is_row_test(term, answers, join_key)
            && !(join_key && join_filters && is_join_filter(term))
    };
    let split = |predicate: &Expr, join_key: bool| -> (Vec<Expr>, Vec<Expr>) {
        split_conjunction(predicate)
            .into_iter()
            .cloned()
            .partition(|term| lifts(term, join_key))
    };
    match plan {
        LogicalPlan::Filter(filter) => {
            let input = take_row_tests(
                filter.input.as_ref().clone(),
                answers,
                (fixed, join_filters),
                lifted,
            )?;
            let (moved, kept) = split(&filter.predicate, false);
            lifted.extend(moved);
            match conjunction(kept) {
                Some(predicate) => Ok(LogicalPlan::Filter(
                    datafusion::logical_expr::Filter::try_new(predicate, Arc::new(input))?,
                )),
                None => Ok(input),
            }
        }
        LogicalPlan::Join(join) if join.join_type == JoinType::Inner => {
            let left = take_row_tests(
                join.left.as_ref().clone(),
                answers,
                (fixed, join_filters),
                lifted,
            )?;
            let right = take_row_tests(
                join.right.as_ref().clone(),
                answers,
                (fixed, join_filters),
                lifted,
            )?;
            let mut join = join.clone();
            if let Some(filter) = join.filter.take() {
                let (moved, kept) = split(&filter, true);
                lifted.extend(moved);
                join.filter = conjunction(kept);
            }
            join.left = Arc::new(left);
            join.right = Arc::new(right);
            Ok(LogicalPlan::Join(join))
        }
        LogicalPlan::Extension(_) if let Some(node) = as_node::<AiNode>(&plan) => {
            let below_fixed = fixed || node.limits_rows();
            let input = take_row_tests(
                node.inputs()[0].clone(),
                answers,
                (below_fixed, join_filters),
                lifted,
            )?;
            let mut node = node.with_input(input)?;
            lifted.extend(node.lift_filter(|filter| split(filter, false)));
            Ok(extension_node(node.checked()?))
        }
        other => {
            let inputs = other
                .inputs()
                .into_iter()
                .cloned()
                .map(|input| take_row_tests(input, answers, (fixed, join_filters), lifted))
                .collect::<Result<Vec<_>>>()?;
            crate::plan::with_inputs(&other, inputs)
        }
    }
}
