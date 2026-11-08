//! The `PushDownOrGuards` optimizer rule: an AI test in an OR is asked
//! only where the rest of the OR is not TRUE.
//!
//! A filter `a OR b` keeps every row where `a` is TRUE, whatever `b` is.
//! When `b` is an `AI.IF` node's answer and `a` tests the input alone, the
//! model need not answer for those rows: with the guard `a IS NOT TRUE`,
//! their answer is NULL, and `a OR NULL` is still TRUE.
//!
//! ```text
//! AI_{f → b}(R; p = a ∨ b) ⇒ AI_{f → b}(R; g = (a IS NOT TRUE), p = a ∨ b)
//! ```
//!
//! The lowering writes `WHERE r.flagged OR AI.IF(...)` as such a node:
//! its filter is the OR, with the answer in the call's place, and it
//! returns none of the columns it adds. The rewrite is the logical form
//! of `π_Q(σ_{a ∨ b}(AI_{f → b}(R))) ⇒ π_Q(σ_{a ∨ b}(AI_{f → b}(R; g =
//! g ∧ (a IS NOT TRUE))))`, with `g` TRUE, once `absorb_into_ai_nodes` has made the OR the
//! node's filter. The rule guards an `AiNode` when:
//!
//! - the node is an `AI.IF` of rows with no guard, `HINT`, or fetch,
//!   and it returns no column it adds, so nothing above it reads `b`;
//! - a term of its filter is an OR, one of whose terms is `b`, and no
//!   other term of the filter reads `b`;
//! - the other terms of the OR, `a`, read only columns of the one table
//!   the call reads.
//!
//! The executor's AI filter asks the model about the rows where the
//! guard is TRUE and passes each row where `a` is TRUE without asking.

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::utils::{disjunction, split_binary, split_conjunction};
use datafusion::logical_expr::{Expr, LogicalPlan, Operator};
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

/// The terms of an `AI.IF` node's filter OR other than its answer `b`,
/// when the filter is `a ∨ b` as the module describes.
pub(crate) fn or_terms(node: &AiNode) -> Option<Expr> {
    let filter = node.filter.as_ref()?;
    let is_answer = |term: &Expr| {
        matches!(term, Expr::Column(column)
            if column.relation.is_none() && column.name == node.answer)
    };
    let reads_answer = |term: &Expr| {
        term.column_refs()
            .iter()
            .any(|column| column.relation.is_none() && node.is_field(&column.name))
    };
    let conjuncts = split_conjunction(filter);
    let mut found = None;
    for conjunct in &conjuncts {
        let terms = split_binary(conjunct, Operator::Or);
        let (answers, others): (Vec<&Expr>, Vec<&Expr>) =
            terms.into_iter().partition(|term| is_answer(term));
        if answers.is_empty() || others.is_empty() {
            if reads_answer(conjunct) {
                return None;
            }
            continue;
        }
        if found.is_some() || others.iter().any(|term| reads_answer(term)) {
            return None;
        }
        found = disjunction(others.into_iter().cloned());
    }
    found
}

fn push_down_or(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let Some(node) = as_node::<AiNode>(&plan) else {
        return Ok(Transformed::no(plan));
    };
    let plain = node.semantics == Semantics::Rows
        && node.function() == "ai_filter"
        && node.guard.is_none()
        && node.hint.is_none()
        && node.fetch.is_none()
        && node.returned_fields().is_empty();
    if !plain {
        return Ok(Transformed::no(plan));
    }
    let Some(rest) = or_terms(node) else {
        return Ok(Transformed::no(plan));
    };
    if !reads_call_table(&rest, node) {
        return Ok(Transformed::no(plan));
    }
    let mut guarded = node.clone();
    guarded.guard = Some(rest.is_not_true());
    Ok(Transformed::yes(extension_node(guarded.checked()?)))
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
    use std::sync::Arc;

    /// An AI.IF of `reviews.body` answering `kind`, with `filter`, and
    /// returning `returned` of its added columns.
    fn kind(filter: Expr, returned: Vec<usize>) -> LogicalPlan {
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
        let mut node = AiNode::try_new(
            Arc::new(reviews),
            call,
            "kind".into(),
            vec![Field::new("kind", DataType::Boolean, true)],
        )
        .unwrap();
        node.filter = Some(filter);
        node.projection = Some(returned);
        extension_node(node.checked().unwrap())
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
        let text = pushed(kind(col("reviews.flagged").or(col("kind")), Vec::new()));
        assert!(
            text.contains("guard: reviews.flagged IS NOT TRUE, filter: reviews.flagged OR kind"),
            "{text}"
        );
    }

    #[test]
    fn an_ai_test_in_an_or_is_asked_everywhere_when_its_answer_is_returned() {
        let text = pushed(kind(col("reviews.flagged").or(col("kind")), vec![0]));
        assert!(!text.contains("guard:"), "{text}");
    }

    #[test]
    fn an_ai_test_is_asked_everywhere_when_another_term_reads_its_answer() {
        let filter = col("reviews.flagged")
            .or(col("kind"))
            .and(col("kind").is_not_null());
        assert!(!pushed(kind(filter, Vec::new())).contains("guard:"));
        let alone = pushed(kind(col("kind"), Vec::new()));
        assert!(!alone.contains("guard:"), "{alone}");
    }
}
