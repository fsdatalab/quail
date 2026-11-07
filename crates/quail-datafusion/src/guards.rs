//! The `PushDownCaseGuards` optimizer rule: a `CASE` that reads an AI
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
//! summary` writes summaries of the long reviews only. The rule rewrites
//! a projection directly above a classification, map, or SAE features
//! node when:
//!
//! - every item that reads a column the node adds reads it as `CASE WHEN
//!   c THEN <column> END`, with the same `c`, and no other item reads
//!   one;
//! - `c` reads only columns of the one table the call reads;
//! - the node has no filter, `HINT`, or fetch.
//!
//! The executor asks the model about the rows where the guard is TRUE
//! and gives every other row a NULL answer. An extraction is left
//! alone: its answer is a struct of one answer per question, and an
//! unasked row would read as a struct of NULL answers, not as NULL.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::expr::Case;
use datafusion::logical_expr::{Expr, LogicalPlan, Projection};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};

/// Make each `CASE` test above a column node the node's guard, as the
/// module explains.
#[derive(Debug, Default)]
pub struct PushDownCaseGuards;

impl OptimizerRule for PushDownCaseGuards {
    fn name(&self) -> &str {
        "push_down_case_guards"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up(push_down_case)
    }
}

fn push_down_case(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Projection(projection) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let Some(node) = as_node::<AiNode>(&projection.input) else {
        return Ok(Transformed::no(plan));
    };
    let plain = node.semantics == Semantics::Rows
        && !node.is_predicate()
        && node.function() != "ai_extract"
        && node.filter.is_none()
        && node.tests.is_empty()
        && node.hint.is_none()
        && node.fetch.is_none();
    if !plain {
        return Ok(Transformed::no(plan));
    }
    let mut test: Option<Expr> = None;
    let mut exprs = Vec::new();
    for (index, expr) in projection.expr.iter().enumerate() {
        let inner = match expr {
            Expr::Alias(alias) => alias.expr.as_ref(),
            other => other,
        };
        let reads_answer = inner
            .column_refs()
            .iter()
            .any(|column| column.relation.is_none() && node.is_field(&column.name));
        if !reads_answer {
            exprs.push(expr.clone());
            continue;
        }
        let Some((when, answer)) = guarded_answer(inner, node) else {
            return Ok(Transformed::no(plan));
        };
        match &test {
            Some(found) if found != when => return Ok(Transformed::no(plan)),
            _ => test = Some(when.clone()),
        }
        let (qualifier, field) = projection.schema.qualified_field(index);
        exprs.push(Expr::Column(answer).alias_qualified(qualifier.cloned(), field.name()));
    }
    let Some(test) = test else {
        return Ok(Transformed::no(plan));
    };
    if !reads_call_table(&test, node) {
        return Ok(Transformed::no(plan));
    }
    let mut guarded = node.clone();
    guarded.guard = Some(match guarded.guard.take() {
        Some(guard) => guard.and(test),
        None => test,
    });
    let input = extension_node(guarded.checked()?);
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(exprs, Arc::new(input))?,
    )))
}

/// `(c, a)` when `expr` is `CASE WHEN c THEN a END` and `a` is a column
/// the node adds.
fn guarded_answer<'a>(expr: &'a Expr, node: &AiNode) -> Option<(&'a Expr, Column)> {
    let Expr::Case(Case {
        expr: None,
        when_then_expr,
        else_expr: None,
    }) = expr
    else {
        return None;
    };
    let [(when, then)] = when_then_expr.as_slice() else {
        return None;
    };
    match then.as_ref() {
        Expr::Column(column) if column.relation.is_none() && node.is_field(&column.name) => {
            Some((when.as_ref(), column.clone()))
        }
        _ => None,
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

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::expr::ScalarFunction;
    use datafusion::logical_expr::{LogicalPlanBuilder, col, lit, when};
    use datafusion::optimizer::{Optimizer, OptimizerContext};

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
        Optimizer::with_rules(vec![Arc::new(PushDownCaseGuards)])
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
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
