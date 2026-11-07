//! The `PushDownOrGuards` optimizer rule: an AI test in an OR is asked
//! only where the rest of the OR is not TRUE.
//!
//! A filter `a OR b` keeps every row where `a` is TRUE, whatever `b` is.
//! When `b` is an AI call's answer and `a` tests the input alone, the
//! model need not answer for those rows: with the guard `a IS NOT TRUE`,
//! their answer is NULL, and `a OR NULL` is still TRUE.
//!
//! ```text
//! π_Q(σ_{a ∨ b}(AI_{f → b}(R)))
//!     ⇒ π_Q(σ_{a ∨ b}(AI_{f → b}(R; g = g ∧ (a IS NOT TRUE))))
//! ```
//!
//! For example, `WHERE r.flagged OR AI.IF(...)` asks the model only about
//! the rows that are not flagged. The rule rewrites a projection, over a
//! filter, over an `AiNode` of rows, when:
//!
//! - the filter is an OR, one of whose terms is the node's answer `b`;
//! - the other terms, `a`, read only columns of the one table the call
//!   reads;
//! - the projection does not read `b`, which the guard makes NULL;
//! - the node has no filter, `HINT`, or fetch, and returns every column
//!   it adds.
//!
//! The rule is off by default: the lowering refuses an AI call in an OR,
//! so no plan has this shape yet, and the physical planner has no AI
//! filter that passes the rows its guard leaves out. Derive refuses a
//! guarded AI filter rather than ignoring the guard.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::utils::disjunction;
use datafusion::logical_expr::{Expr, Filter, LogicalPlan, Projection};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};
use crate::guards::reads_call_table;

/// Guard the AI call that is one term of an OR with the other terms, as
/// the module explains.
#[derive(Debug, Default)]
pub struct PushDownOrGuards;

impl OptimizerRule for PushDownOrGuards {
    fn name(&self) -> &str {
        "push_down_or_guards"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up(push_down_or)
    }
}

/// The terms of an OR, outermost ORs split.
fn split_disjunction(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::BinaryExpr(binary) if binary.op == datafusion::logical_expr::Operator::Or => {
            let mut terms = split_disjunction(&binary.left);
            terms.extend(split_disjunction(&binary.right));
            terms
        }
        Expr::Alias(alias) => split_disjunction(&alias.expr),
        other => vec![other],
    }
}

fn push_down_or(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Projection(projection) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let LogicalPlan::Filter(filter) = projection.input.as_ref() else {
        return Ok(Transformed::no(plan));
    };
    let Some(node) = as_node::<AiNode>(&filter.input) else {
        return Ok(Transformed::no(plan));
    };
    let plain = node.semantics == Semantics::Rows
        && node.filter.is_none()
        && node.tests.is_empty()
        && node.hint.is_none()
        && node.fetch.is_none()
        && node.projection.is_none();
    if !plain {
        return Ok(Transformed::no(plan));
    }
    let is_answer = |term: &Expr| matches!(term, Expr::Column(column) if column.relation.is_none() && column.name == node.answer);
    let terms = split_disjunction(&filter.predicate);
    let (answers, others): (Vec<&Expr>, Vec<&Expr>) =
        terms.into_iter().partition(|term| is_answer(term));
    if answers.is_empty() || others.is_empty() {
        return Ok(Transformed::no(plan));
    }
    let Some(rest) = disjunction(others.into_iter().cloned()) else {
        return Ok(Transformed::no(plan));
    };
    let reads_answer = projection.expr.iter().any(|expr| {
        expr.column_refs()
            .iter()
            .any(|column| column.relation.is_none() && node.is_field(&column.name))
    });
    if reads_answer || !reads_call_table(&rest, node) {
        return Ok(Transformed::no(plan));
    }
    let test = rest.is_not_true();
    if node.guard.as_ref() == Some(&test) {
        return Ok(Transformed::no(plan));
    }
    let mut guarded = node.clone();
    guarded.guard = Some(match guarded.guard.take() {
        Some(guard) => guard.and(test),
        None => test,
    });
    let input = extension_node(guarded.checked()?);
    let filtered = Filter::try_new(filter.predicate.clone(), Arc::new(input))?;
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(
            projection.expr.clone(),
            Arc::new(LogicalPlan::Filter(filtered)),
        )?,
    )))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::expr::ScalarFunction;
    use datafusion::logical_expr::{LogicalPlanBuilder, col, lit};
    use datafusion::optimizer::{Optimizer, OptimizerContext};

    use super::*;

    /// An AI.IF of `reviews.body` returning its answer `kind`.
    fn kind() -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("body", DataType::Utf8, true),
            Field::new("flagged", DataType::Boolean, true),
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
                .find(|found| found.name() == "ai_filter")
                .unwrap(),
            vec![lit("Is {0} kind?"), col("reviews.body")],
        ));
        let node = AiNode::try_new(
            Arc::new(reviews),
            call,
            "kind".into(),
            vec![Field::new("kind", DataType::Boolean, true)],
        )
        .unwrap();
        extension_node(node)
    }

    fn pushed(plan: LogicalPlan) -> String {
        Optimizer::with_rules(vec![Arc::new(PushDownOrGuards)])
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
    }

    #[test]
    fn an_ai_test_in_an_or_is_asked_where_the_other_terms_are_not_true() {
        let plan = LogicalPlanBuilder::from(kind())
            .filter(col("reviews.flagged").or(col("kind")))
            .unwrap()
            .project(vec![col("reviews.id")])
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(
            text.contains("guard: reviews.flagged IS NOT TRUE"),
            "{text}"
        );
        assert!(text.contains("Filter: reviews.flagged OR kind"), "{text}");
    }

    #[test]
    fn an_ai_test_in_an_or_is_asked_everywhere_when_the_select_list_reads_it() {
        let plan = LogicalPlanBuilder::from(kind())
            .filter(col("reviews.flagged").or(col("kind")))
            .unwrap()
            .project(vec![col("reviews.id"), col("kind")])
            .unwrap()
            .build()
            .unwrap();
        assert!(!pushed(plan).contains("guard:"));
    }
}
