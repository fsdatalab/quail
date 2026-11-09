//! The operators of a model region's part that DataFusion runs above the
//! region.
//!
//! [`lift_relational`] splits a part [`crate::ClaimModelRegions`] claims
//! into the region, which runs on the GPU, and the operators above it,
//! which run in DataFusion on the rows the region returns:
//!
//! - each test of the result rows: a test of more than one table's
//!   columns, which needs every model call's answer first;
//! - the SELECT list's expressions, other than its columns and model
//!   calls.
//!
//! For example, `SELECT r.id, UPPER(p.name) AS name ... WHERE r.stars <
//! p.stars` over an AI join leaves the SELECT list `r.id`, `p.name`,
//! `r.stars`, and `p.stars` below the operators above; DataFusion keeps
//! the rows with `r.stars < p.stars` and computes `name`.
//! [`crate::region::join_back`] then splits that list into the region's ids
//! and answers and the join that adds `p.name` and the stars back.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use datafusion::common::tree_node::TreeNode;
use datafusion::common::{Column, Result};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{Expr, JoinType, LogicalPlan, LogicalPlanBuilder, Projection};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};
use crate::functions::{is_ai_call, is_quail_call};

/// The operators DataFusion runs above a region, lowest first.
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
    /// Whether the operators keep every row the region returns.
    pub(crate) fn keeps_rows(&self) -> bool {
        self.steps
            .iter()
            .all(|step| matches!(step, Step::Projection(_)))
    }

    /// The operators over `region`.
    ///
    /// # Errors
    ///
    /// An operator does not plan over the region's columns.
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

/// Split a part whose root is its SELECT list into the region and the
/// operators DataFusion runs above it, as the module explains. The
/// operators above return the part's columns.
///
/// # Errors
///
/// A rebuilt operator does not plan.
pub(crate) fn lift_relational(plan: LogicalPlan) -> Result<(LogicalPlan, Above)> {
    let LogicalPlan::Projection(root) = &plan else {
        return Ok((plan, Above::default()));
    };
    let answers = answer_tables(&root.input);
    let mut lifted: Vec<Expr> = Vec::new();
    let relation = take_row_tests(root.input.as_ref().clone(), &answers, false, &mut lifted)?;
    let tests = lifted;
    // each SELECT list item the region returns as it is, by position
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
    // the region also returns the columns the operators above read
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

/// `plan` with one column at its root when it has none, such as the
/// rows a `COUNT(*)` counts: Quail returns at least one column, and a
/// `Projection` above the region drops it.
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

/// The tables each `AiNode` answer column belongs to: those its call
/// reads.
fn answer_tables(plan: &LogicalPlan) -> BTreeMap<String, BTreeSet<String>> {
    let mut found = BTreeMap::new();
    let _ = plan.apply(|node| {
        if let Some(ai) = as_node::<AiNode>(node) {
            let tables: BTreeSet<String> = ai
                .call
                .column_refs()
                .into_iter()
                .filter_map(|column| column.relation.as_ref().map(|table| table.to_string()))
                .collect();
            for field in &ai.fields {
                found.insert(field.name().clone(), tables.clone());
            }
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    });
    found
}

/// Whether a test runs on the result rows: a test of more than one
/// table's columns that calls no model and has no subquery, other than a
/// join's key.
fn is_row_test(term: &Expr, answers: &BTreeMap<String, BTreeSet<String>>, join_key: bool) -> bool {
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
    for column in term.column_refs() {
        match &column.relation {
            Some(table) => {
                tables.insert(table.to_string());
            }
            None => tables.extend(answers.get(&column.name).cloned().unwrap_or_default()),
        }
    }
    tables.len() > 1
}

/// Whether a term is `a.x = b.y`, an equality of two tables' columns.
fn is_column_equality(term: &Expr) -> bool {
    matches!(term, Expr::BinaryExpr(binary)
        if binary.op == datafusion::logical_expr::Operator::Eq
            && matches!((binary.left.as_ref(), binary.right.as_ref()),
                (Expr::Column(left), Expr::Column(right)) if left.relation != right.relation))
}

/// `plan` without its row tests, which are added to `lifted` in plan
/// order, children first.
///
/// `fixed` says an operator above limits or gates the rows, so a test of
/// several tables below it stays.
fn take_row_tests(
    plan: LogicalPlan,
    answers: &BTreeMap<String, BTreeSet<String>>,
    fixed: bool,
    lifted: &mut Vec<Expr>,
) -> Result<LogicalPlan> {
    let lifts = |term: &Expr, join_key: bool| !fixed && is_row_test(term, answers, join_key);
    // the operator's own tests come after the ones below it
    let split = |predicate: &Expr, join_key: bool| -> (Vec<Expr>, Vec<Expr>) {
        split_conjunction(predicate)
            .into_iter()
            .cloned()
            .partition(|term| lifts(term, join_key))
    };
    match plan {
        LogicalPlan::Filter(filter) => {
            let input = take_row_tests(filter.input.as_ref().clone(), answers, fixed, lifted)?;
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
            let left = take_row_tests(join.left.as_ref().clone(), answers, fixed, lifted)?;
            let right = take_row_tests(join.right.as_ref().clone(), answers, fixed, lifted)?;
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
        LogicalPlan::Extension(_) if as_node::<AiNode>(&plan).is_some() => {
            let node = as_node::<AiNode>(&plan).expect("an AiNode").clone();
            let rows = node.semantics == Semantics::Rows;
            let below_fixed = fixed || node.fetch.is_some() || !rows;
            let input = take_row_tests(node.input.as_ref().clone(), answers, below_fixed, lifted)?;
            let mut node = node.with_input(input)?;
            if rows
                && !node.is_predicate()
                && let Some(filter) = node.filter.take()
            {
                let (moved, kept) = split(&filter, false);
                if !moved.is_empty() && node.fetch.is_some() {
                    // a fetch counts the rows that pass the whole filter
                    node.filter = Some(filter);
                } else {
                    node.tests.retain(|test| {
                        !split_conjunction(&test.test)
                            .iter()
                            .any(|term| moved.contains(term))
                    });
                    lifted.extend(moved);
                    node.filter = conjunction(kept);
                }
            }
            Ok(extension_node(node.checked()?))
        }
        other => {
            let inputs = other
                .inputs()
                .into_iter()
                .cloned()
                .map(|input| take_row_tests(input, answers, fixed, lifted))
                .collect::<Result<Vec<_>>>()?;
            other.with_new_exprs(other.expressions(), inputs)
        }
    }
}
