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
//! The rule reads down from the upper node through filters and other
//! `AiNode`s of the same rows, to the first node with the same call. It
//! replaces the upper node only when:
//!
//! - both nodes return rows, not a gate's;
//! - the upper node has no fetch;
//! - each term of the upper node's filter, read on the lower node's
//!   answer, is already a term of the lower node's filter, so every row
//!   that reaches the upper node passes it;
//! - the lower node returns each answer column the upper node returns.
//!
//! The upper node's `HINT` estimated the rows of a test the lower node
//! already makes, and goes with it.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::logical_expr::{Expr, LogicalPlan, Projection};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};
use crate::extension::as_node;

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
    if upper.semantics != Semantics::Rows || upper.fetch.is_some() {
        return Ok(Transformed::no(plan));
    }
    let Some(lower) = same_call_below(upper) else {
        return Ok(Transformed::no(plan));
    };
    // each of the upper node's answer columns, by the lower node's name
    let renamed: HashMap<&String, &String> = upper
        .fields
        .iter()
        .map(|field| field.name())
        .zip(lower.fields.iter().map(|field| field.name()))
        .collect();
    let on_lower = |expr: &Expr| -> Result<Expr> {
        expr.clone()
            .transform_up(|expr| match &expr {
                Expr::Column(column) if column.relation.is_none() => {
                    match renamed.get(&column.name) {
                        Some(name) => Ok(Transformed::yes(Expr::Column(Column::new_unqualified(
                            (*name).clone(),
                        )))),
                        None => Ok(Transformed::no(expr)),
                    }
                }
                _ => Ok(Transformed::no(expr)),
            })
            .map(|read| read.data)
    };
    let lower_terms: Vec<&Expr> = lower.filter.iter().flat_map(split_conjunction).collect();
    for term in upper.filter.iter().flat_map(split_conjunction) {
        if !lower_terms.contains(&&on_lower(term)?) {
            return Ok(Transformed::no(plan));
        }
    }
    let returned: Vec<String> = lower
        .returned_fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let answers = upper.returned_fields();
    if answers
        .iter()
        .any(|field| !returned.contains(renamed[field.name()]))
    {
        return Ok(Transformed::no(plan));
    }
    let input = upper.input.as_ref().clone();
    if answers.is_empty() {
        return Ok(Transformed::yes(input));
    }
    let mut exprs: Vec<Expr> = input
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    exprs.extend(answers.iter().map(|field| {
        Expr::Column(Column::new_unqualified(renamed[field.name()].clone())).alias(field.name())
    }));
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(exprs, Arc::new(input))?,
    )))
}

/// The first `AiNode` below `upper`, through filters and other nodes of
/// the same rows, that makes `upper`'s call with its guard and adds
/// columns of the same types.
fn same_call_below(upper: &AiNode) -> Option<&AiNode> {
    let mut plan = upper.input.as_ref();
    loop {
        plan = match plan {
            LogicalPlan::Filter(filter) => filter.input.as_ref(),
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
                if node.call == upper.call && node.guard == upper.guard && same_types {
                    return Some(node);
                }
                node.input.as_ref()
            }
            _ => return None,
        };
    }
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
    use crate::ai::Fetch;
    use crate::extension::extension_node;

    fn reviews() -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("body", DataType::Utf8, true),
        ]));
        LogicalPlanBuilder::scan(
            "reviews",
            provider_as_source(Arc::new(EmptyTable::new(schema))),
            None,
        )
        .unwrap()
        .build()
        .unwrap()
    }

    fn call(function: &str, template: &str) -> Expr {
        Expr::ScalarFunction(ScalarFunction::new_udf(
            crate::functions::quail_functions()
                .into_iter()
                .find(|found| found.name() == function)
                .unwrap(),
            vec![lit(template), col("reviews.body")],
        ))
    }

    /// An `AI.IF` of `template` over `input`, adding `answer`.
    fn ai_filter(input: LogicalPlan, template: &str, answer: &str) -> AiNode {
        let mut node = AiNode::try_new(
            Arc::new(input),
            call("ai_filter", template),
            answer.into(),
            vec![Field::new(answer, DataType::Boolean, true)],
        )
        .unwrap();
        node.filter = Some(col(answer));
        node.projection = Some(Vec::new());
        node.checked().unwrap()
    }

    fn tone(input: LogicalPlan, answer: &str) -> AiNode {
        AiNode::try_new(
            Arc::new(input),
            call("ai_classify", "Tone of {0}?"),
            answer.into(),
            vec![Field::new(answer, DataType::Utf8, true)],
        )
        .unwrap()
    }

    fn merged(plan: LogicalPlan) -> Vec<String> {
        Optimizer::with_rules(vec![Arc::new(MergeRepeatedAiCalls)])
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
            .lines()
            .map(|line| line.trim().to_string())
            .collect()
    }

    fn num_nodes(lines: &[String]) -> usize {
        lines
            .iter()
            .filter(|line| line.starts_with("AiNode"))
            .count()
    }

    #[test]
    fn a_repeated_ai_filter_asks_the_model_once() {
        let kind = extension_node(ai_filter(reviews(), "Is {0} kind?", "a"));
        let short = extension_node(ai_filter(kind, "Is {0} short?", "b"));
        let again = extension_node(ai_filter(short, "Is {0} kind?", "c"));
        let lines = merged(again);
        assert_eq!(num_nodes(&lines), 2, "{lines:?}");
        assert!(lines[0].contains("short"), "{lines:?}");
    }

    #[test]
    fn a_repeated_classification_reads_the_label_below_it() {
        let first = extension_node(tone(reviews(), "a"));
        let plan = LogicalPlanBuilder::from(extension_node(tone(first, "b")))
            .project(vec![col("reviews.id"), col("a"), col("b")])
            .unwrap()
            .build()
            .unwrap();
        let schema = plan.schema().clone();
        let optimized = Optimizer::with_rules(vec![Arc::new(MergeRepeatedAiCalls)])
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap();
        assert_eq!(optimized.schema(), &schema);
        let lines: Vec<String> = optimized
            .display_indent()
            .to_string()
            .lines()
            .map(|line| line.trim().to_string())
            .collect();
        assert_eq!(num_nodes(&lines), 1, "{lines:?}");
        assert!(
            lines.iter().any(|line| line.contains("a AS b")),
            "{lines:?}"
        );
    }

    #[test]
    fn a_repeated_call_with_a_new_test_or_a_fetch_is_asked_again() {
        // the lower classification keeps every row; the upper one's test
        // would drop rows the lower one returns
        let first = extension_node(tone(reviews(), "a"));
        let mut tested = tone(first, "b");
        tested.filter = Some(col("b").eq(lit("kind")));
        let tested = extension_node(tested.checked().unwrap());
        assert_eq!(num_nodes(&merged(tested)), 2);
        let first = extension_node(tone(reviews(), "a"));
        let mut fetched = tone(first, "b");
        fetched.fetch = Some(Fetch::Global(2));
        let fetched = extension_node(fetched.checked().unwrap());
        assert_eq!(num_nodes(&merged(fetched)), 2);
    }
}
