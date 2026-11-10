//! A guard step of [`crate::Derive`]: a `CASE` that reads an AI
//! answer only when a test of the input passes becomes the node's guard.
//!
//! An `AiNode`'s guard `g` says which rows the model is asked about; the
//! answer of every other row is NULL, `f_g = CASE WHEN g THEN f END`. A
//! SELECT list that reads the answer only under a test `c` of the input
//! needs the model only where `c` is TRUE:
//!
//! ```text
//! π_{B, (CASE WHEN c THEN a END) → z}(AI_{f → a}(R))
//!     ⇒ π_{B, a → z}(AI_{f → a}(R; g = g ∧ c))
//! ```
//!
//! For example, `SELECT r.id, CASE WHEN r.long THEN AI.MAP(...) END AS
//! summary` writes summaries of the long reviews only. The step rewrites
//! a projection above a classification, extraction, map, or SAE features
//! node, through the SELECT lists and derived-table aliases between them
//! that pass the node's columns on as they are, when:
//!
//! - every item that reads a column the node adds reads it as `CASE WHEN
//!   c THEN <column> END`, with the same `c`, and no other item reads
//!   one;
//! - `c` reads only columns of the one table the call reads;
//! - the node has no filter, `HINT`, or fetch.
//!
//! The executor asks the model about the rows where the guard is TRUE
//! and gives every other row a NULL answer. An extraction keeps its
//! `CASE` above the guarded node: its answer is a struct of one answer
//! per question, and the executor gives an unasked row a struct of NULL
//! answers, which the `CASE` makes NULL as a whole:
//!
//! ```text
//! π_{B, (CASE WHEN c THEN e END) → z}(AI_{extract → e}(R))
//!     ⇒ π_{B, (CASE WHEN c THEN e END) → z}(AI_{extract → e}(R; g = g ∧ c))
//! ```

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::expr::Case;
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::logical_expr::{Expr, LogicalPlan, Projection};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};
use crate::rules::repeated::{through_alias, through_projection};

/// Make each `CASE` test above a column node the node's guard, as the
/// module explains.
pub(super) fn case_guards(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    plan.transform_up(push_down_case)
}

fn push_down_case(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Projection(projection) = &plan else {
        return Ok(Transformed::no(plan));
    };
    // each test `c` with the columns the SELECT list reads only under it
    let mut tests: Vec<&Expr> = Vec::new();
    for expr in &projection.expr {
        if let Some((when, _)) = guarded_column(expr)
            && !tests.contains(&when)
        {
            tests.push(when);
        }
    }
    for test in tests {
        let mut answers: Vec<Column> = Vec::new();
        let mut others: BTreeSet<Column> = BTreeSet::new();
        for expr in &projection.expr {
            match guarded_column(expr) {
                Some((when, column)) if when == test => answers.push(column),
                _ => others.extend(expr.column_refs().into_iter().cloned()),
            }
        }
        let Some((input, extracts)) = guard_below(&projection.input, test, &answers, &others)?
        else {
            continue;
        };
        let exprs = projection
            .expr
            .iter()
            .enumerate()
            .map(|(index, expr)| match guarded_column(expr) {
                // an extraction's unasked row is a struct of NULL answers,
                // which its CASE makes NULL as a whole
                Some((when, column)) if when == test && !extracts => {
                    let (qualifier, field) = projection.schema.qualified_field(index);
                    Expr::Column(column).alias_qualified(qualifier.cloned(), field.name())
                }
                _ => expr.clone(),
            })
            .collect();
        return Ok(Transformed::yes(LogicalPlan::Projection(
            Projection::try_new(exprs, Arc::new(input))?,
        )));
    }
    Ok(Transformed::no(plan))
}

/// `(c, a)` when a SELECT list's item is `CASE WHEN c THEN a END` and `a`
/// is a column.
fn guarded_column(expr: &Expr) -> Option<(&Expr, Column)> {
    let inner = match expr {
        Expr::Alias(alias) => alias.expr.as_ref(),
        other => other,
    };
    let Expr::Case(Case {
        expr: None,
        when_then_expr,
        else_expr: None,
    }) = inner
    else {
        return None;
    };
    let [(when, then)] = when_then_expr.as_slice() else {
        return None;
    };
    match then.as_ref() {
        Expr::Column(column) => Some((when.as_ref(), column.clone())),
        _ => None,
    }
}

/// `plan` with `test` as the guard of the `AiNode` that adds `answers`,
/// and whether the node is an extraction; None when the module's
/// conditions do not hold.
///
/// `answers` are columns of `plan` read only where `test` is TRUE, and
/// `others` the columns of `plan` read whatever the test is. The node is
/// reached through the SELECT lists and derived-table aliases that pass
/// the answers on as they are.
fn guard_below(
    plan: &LogicalPlan,
    test: &Expr,
    answers: &[Column],
    others: &BTreeSet<Column>,
) -> Result<Option<(LogicalPlan, bool)>> {
    let rebuilt = |found: Option<(LogicalPlan, bool)>| -> Result<Option<(LogicalPlan, bool)>> {
        match found {
            Some((input, extracts)) => Ok(Some((
                plan.with_new_exprs(plan.expressions(), vec![input])?,
                extracts,
            ))),
            None => Ok(None),
        }
    };
    match plan {
        LogicalPlan::Projection(projection) => {
            let defined = |column: &Column| -> Option<&Expr> {
                let index = projection.schema.index_of_column(column).ok()?;
                Some(&projection.expr[index])
            };
            let passed: Option<Vec<Column>> = answers
                .iter()
                .map(|answer| match defined(answer)?.clone().unalias() {
                    Expr::Column(column) => Some(column),
                    _ => None,
                })
                .collect();
            let (Some(answers), Some(test)) = (passed, through_projection(test, projection)) else {
                return Ok(None);
            };
            let mut read = BTreeSet::new();
            for other in others {
                let Some(expr) = defined(other) else {
                    return Ok(None);
                };
                read.extend(expr.column_refs().into_iter().cloned());
            }
            rebuilt(guard_below(&projection.input, &test, &answers, &read)?)
        }
        LogicalPlan::SubqueryAlias(alias) => {
            let below = |column: &Column| -> Option<Column> {
                let index = alias.schema.index_of_column(column).ok()?;
                Some(Column::from(alias.input.schema().qualified_field(index)))
            };
            let passed: Option<Vec<Column>> = answers.iter().map(below).collect();
            let read: Option<BTreeSet<Column>> = others.iter().map(below).collect();
            let (Some(answers), Some(read), Some(test)) =
                (passed, read, through_alias(test, alias))
            else {
                return Ok(None);
            };
            rebuilt(guard_below(&alias.input, &test, &answers, &read)?)
        }
        LogicalPlan::Extension(_) => {
            let Some(node) = as_node::<AiNode>(plan) else {
                return Ok(None);
            };
            let adds = |column: &Column| column.relation.is_none() && node.is_field(&column.name);
            let plain = node.semantics == Semantics::Rows
                && !node.is_predicate()
                && node.filter.is_none()
                && node.tests.is_empty()
                && node.hint.is_none()
                && node.fetch.is_none();
            if !plain
                || answers.is_empty()
                || !answers.iter().all(adds)
                || others.iter().any(adds)
                || !reads_call_table(test, node)
            {
                return Ok(None);
            }
            // an extraction's CASE stays above it, once its test is the guard
            let extracts = node.function() == "ai_extract";
            let guarded_already = node
                .guard
                .as_ref()
                .is_some_and(|guard| split_conjunction(guard).contains(&test));
            if extracts && guarded_already {
                return Ok(None);
            }
            let mut guarded = node.clone();
            guarded.guard = Some(match guarded.guard.take() {
                Some(guard) => guard.and(test.clone()),
                None => test.clone(),
            });
            Ok(Some((extension_node(guarded.checked()?), extracts)))
        }
        _ => Ok(None),
    }
}

/// Whether `test` reads columns of the node's input, of the one table its
/// call reads, and calls no AI function.
pub(crate) fn reads_call_table(test: &Expr, node: &AiNode) -> bool {
    let tables = |expr: &Expr| -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        for column in expr.column_refs() {
            let table = column
                .relation
                .as_ref()
                .map(|table| table.table().to_string());
            if let Some(table) = table
                && !found.contains(&table)
            {
                found.push(table);
            }
        }
        found
    };
    let call = tables(&node.call);
    let read = tables(test);
    let input = node.input.schema();
    call.len() == 1
        && read == call
        && test
            .column_refs()
            .iter()
            .all(|column| input.has_column(column))
        && !test
            .exists(|inner| Ok(crate::is_ai_call(inner)))
            .unwrap_or(true)
}

/// The estimated fraction of rows where `guard` is TRUE, as the planner
/// estimates a filter's: a `HINT`'s selectivity where the guard gives
/// one, the selectivity of an earlier `AI.IF` whose answer it reads, by
/// the answer's name in `answers`, and the default selectivity for any
/// other test. `x IS NOT TRUE` and `NOT x` pass the rest of `x`'s rows,
/// an AND the product of its terms', and an OR the rows any term passes,
/// each term taken as independent.
pub(crate) fn guard_selectivity(guard: &Expr, answers: &[(String, Option<f64>)]) -> f64 {
    use datafusion::logical_expr::{BinaryExpr, Operator};
    use quail_planner::statistics::effective_selectivity;
    match guard {
        Expr::IsNotTrue(inner) | Expr::Not(inner) => 1.0 - guard_selectivity(inner, answers),
        Expr::BinaryExpr(BinaryExpr { left, op, right }) if *op == Operator::And => {
            guard_selectivity(left, answers) * guard_selectivity(right, answers)
        }
        Expr::BinaryExpr(BinaryExpr { left, op, right }) if *op == Operator::Or => {
            1.0 - (1.0 - guard_selectivity(left, answers))
                * (1.0 - guard_selectivity(right, answers))
        }
        Expr::Column(column) if column.relation.is_none() => effective_selectivity(
            answers
                .iter()
                .find(|(name, _)| *name == column.name)
                .and_then(|(_, selectivity)| *selectivity),
        ),
        Expr::ScalarFunction(call) if call.name() == "hint" => {
            effective_selectivity(call.args.get(1).and_then(hinted_selectivity))
        }
        _ => effective_selectivity(None),
    }
}

/// The selectivity a `HINT`'s options give, `{'selectivity': s}`.
fn hinted_selectivity(options: &Expr) -> Option<f64> {
    let Expr::ScalarFunction(call) = options else {
        return None;
    };
    if call.name() != "named_struct" {
        return None;
    }
    call.args.chunks(2).find_map(|pair| match pair {
        [Expr::Literal(key, _), Expr::Literal(value, _)]
            if key.try_as_str().flatten() == Some("selectivity") =>
        {
            value.to_string().parse().ok()
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::expr::ScalarFunction;
    use datafusion::logical_expr::{LogicalPlanBuilder, col, lit, when};

    use super::*;

    /// A classification of `reviews.body`, adding `tone`.
    fn tone() -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("body", DataType::Utf8, true),
            Field::new("long", DataType::Boolean, true),
        ]));
        let reviews = LogicalPlanBuilder::scan(
            "reviews",
            provider_as_source(Arc::new(EmptyTable::new(schema))),
            None,
        )
        .unwrap()
        .build()
        .unwrap();
        let call = Expr::ScalarFunction(ScalarFunction::new_udf(
            crate::functions::quail_functions()
                .into_iter()
                .find(|found| found.name() == "ai_classify")
                .unwrap(),
            vec![lit("Tone of {0}?"), col("reviews.body")],
        ));
        let node = AiNode::try_new(
            Arc::new(reviews),
            call,
            "tone".into(),
            vec![Field::new("tone", DataType::Utf8, true)],
        )
        .unwrap();
        extension_node(node)
    }

    fn pushed(plan: LogicalPlan) -> String {
        case_guards(plan).unwrap().data.display_indent().to_string()
    }

    #[test]
    fn a_case_over_an_answer_becomes_the_nodes_guard() {
        let guarded = when(col("reviews.long"), col("tone")).end().unwrap();
        let plan = LogicalPlanBuilder::from(tone())
            .project(vec![col("reviews.id"), guarded.alias("z")])
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(text.contains("Projection: reviews.id, tone AS z"), "{text}");
        assert!(text.contains("guard: reviews.long"), "{text}");
    }

    #[test]
    fn a_case_stays_when_another_item_reads_the_answer() {
        let guarded = when(col("reviews.long"), col("tone")).end().unwrap();
        let plan = LogicalPlanBuilder::from(tone())
            .project(vec![guarded.alias("z"), col("tone")])
            .unwrap()
            .build()
            .unwrap();
        assert!(!pushed(plan).contains("guard:"));
        let otherwise = when(col("reviews.long"), col("tone"))
            .otherwise(lit("none"))
            .unwrap();
        let plan = LogicalPlanBuilder::from(tone())
            .project(vec![otherwise.alias("z")])
            .unwrap()
            .build()
            .unwrap();
        assert!(!pushed(plan).contains("guard:"));
    }
}
