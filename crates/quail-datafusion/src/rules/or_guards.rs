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
//!   the call reads, or are answers of the earlier `AI.IF`s of the OR.
//!
//! An OR of several `AI.IF`s, `a ∨ f_1 ∨ … ∨ f_k`, lowers to one node per
//! call: each `f_j` but the last returns its answer `b_j` and drops no
//! row, in written order, below the last, whose filter is `a ∨ b_1 ∨ … ∨
//! b_k`. The rule guards each earlier node with the terms before it,
//!
//! ```text
//! AI_{f_j → b_j}(R) ⇒ AI_{f_j → b_j}(R; g = a IS NOT TRUE ∧ b_1 IS NOT TRUE ∧ … ∧ b_{j-1} IS NOT TRUE)
//! ```
//!
//! leaving the first unguarded when there is no `a`, and the last with
//! every other term. A guard writes `t_1 OR t_2 IS NOT TRUE` as `t_1 IS
//! NOT TRUE AND t_2 IS NOT TRUE`, which is the same test.
//!
//! A term of the OR may read an answer as `NOT b`; the guards are the
//! same, with `NOT b` in the place of `b`.
//!
//! The executor's AI filter asks the model about the rows where the
//! guard is TRUE and passes each row where `a` is TRUE without asking.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::utils::{conjunction, disjunction, split_binary, split_conjunction};
use datafusion::logical_expr::{Expr, LogicalPlan, Operator};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};
use crate::rules::guards::reads_call_table;

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
/// when the filter is `a ∨ b` as the module describes, and whether the
/// OR reads the answer as `NOT b`.
pub(crate) fn or_terms(node: &AiNode) -> Option<(Expr, bool)> {
    let filter = node.filter.as_ref()?;
    let reads_answer = |term: &Expr| {
        term.column_refs()
            .iter()
            .any(|column| column.relation.is_none() && node.is_field(&column.name))
    };
    let conjuncts = split_conjunction(filter);
    let mut found = None;
    for conjunct in &conjuncts {
        let terms = split_binary(conjunct, Operator::Or);
        let (answers, others): (Vec<&Expr>, Vec<&Expr>) = terms
            .into_iter()
            .partition(|term| node.answer_term(term).is_some());
        if answers.is_empty() || others.is_empty() {
            if reads_answer(conjunct) {
                return None;
            }
            continue;
        }
        if found.is_some() || others.iter().any(|term| reads_answer(term)) {
            return None;
        }
        let negated = answers
            .iter()
            .any(|term| node.answer_term(term) == Some(true));
        found = disjunction(others.into_iter().cloned()).map(|rest| (rest, negated));
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
    let Some((rest, _)) = or_terms(node) else {
        return Ok(Transformed::no(plan));
    };
    // the earlier AI.IFs of the OR, which return their answers below
    let answers = answer_nodes(&node.input);
    // the answer an OR's term reads, as the column or its NOT
    let answer_of = |term: &Expr| match term {
        Expr::Not(inner) => inner.try_as_col().cloned(),
        other => other.try_as_col().cloned(),
    };
    let is_answer = |term: &Expr| {
        answer_of(term)
            .is_some_and(|column| column.relation.is_none() && answers.contains(&column.name))
    };
    let (read, plain): (Vec<Expr>, Vec<Expr>) = split_binary(&rest, Operator::Or)
        .into_iter()
        .cloned()
        .partition(|term| is_answer(term));
    if let Some(plain) = disjunction(plain.iter().cloned())
        && !reads_call_table(&plain, node)
    {
        return Ok(Transformed::no(plan));
    }
    let mut seen = plain;
    let input = Arc::unwrap_or_clone(Arc::clone(&node.input))
        .transform_up(|below| {
            let Some(asked) = as_node::<AiNode>(&below) else {
                return Ok(Transformed::no(below));
            };
            let answer = Column::from_name(asked.answer.clone());
            let Some(term) = read
                .iter()
                .find(|term| answer_of(term).as_ref() == Some(&answer))
            else {
                return Ok(Transformed::no(below));
            };
            let mut guarded = asked.clone();
            guarded.guard = not_true(&seen);
            seen.push(term.clone());
            Ok(Transformed::yes(extension_node(guarded.checked()?)))
        })?
        .data;
    let mut guarded = node.with_input(input)?;
    guarded.guard = not_true(&seen);
    Ok(Transformed::yes(extension_node(guarded.checked()?)))
}

/// `t1 IS NOT TRUE AND t2 IS NOT TRUE ...`, which is TRUE exactly where
/// `t1 OR t2 ...` is not TRUE; None for no terms.
fn not_true(terms: &[Expr]) -> Option<Expr> {
    conjunction(terms.iter().cloned().map(Expr::is_not_true))
}

/// The answers of the `AI.IF` nodes in `plan` that return their answer
/// and drop no row, with no guard, `HINT`, or fetch: the earlier `AI.IF`s
/// of an OR.
fn answer_nodes(plan: &LogicalPlan) -> Vec<String> {
    let mut found = Vec::new();
    let _ = plan.apply(|below| {
        if let Some(asked) = as_node::<AiNode>(below)
            && asked.semantics == Semantics::Rows
            && asked.function() == "ai_filter"
            && asked.filter.is_none()
            && asked.guard.is_none()
            && asked.hint.is_none()
            && asked.fetch.is_none()
        {
            found.push(asked.answer.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
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
