//! The `PushDownHintTests` optimizer rule: a test of an AI column's answer
//! above the node also becomes the node's hint test.
//!
//! An `AiNode`'s hint test `h` lets the node drop rows where `h` is not
//! TRUE. A filter `q` directly above a node of rows with no fetch drops
//! those rows anyway, so the node may drop them first:
//!
//! ```text
//! σ_q(AI(R; h)) ⇒ σ_q(AI(R; h ∧ q))
//! ```
//!
//! The filter stays: the node may keep a row `q` rejects. The executor
//! uses the hint test to stop an answer once the test is decided not
//! TRUE on the tokens decoded so far ([`crate::partial`]). A term `q` of the
//! filter moves when it reads a column the node adds and otherwise only
//! columns of the node's input.
//!
//! `absorb_into_ai_nodes` (R2) makes each such term the node's exact
//! filter first, so in the default rule order this rule finds a test
//! only when R2 is turned off. The executor reads a column node's exact
//! tests of its answer as it reads a hint test.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{Expr, Filter, LogicalPlan};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};

/// Add each test of an `AiNode`'s answer above it to the node's hint
/// test, as the module explains.
#[derive(Debug, Default)]
pub struct PushDownHintTests;

impl OptimizerRule for PushDownHintTests {
    fn name(&self) -> &str {
        "push_down_hint_tests"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up(push_down_hint)
    }
}

fn push_down_hint(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Filter(filter) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let Some(node) = as_node::<AiNode>(&filter.input) else {
        return Ok(Transformed::no(plan));
    };
    if node.semantics != Semantics::Rows || node.is_predicate() || node.fetch.is_some() {
        return Ok(Transformed::no(plan));
    }
    let input = node.input.schema();
    let held: Vec<&Expr> = node.hint_test.iter().flat_map(split_conjunction).collect();
    let tests: Vec<Expr> = split_conjunction(&filter.predicate)
        .into_iter()
        .filter(|term| !held.contains(term))
        .filter(|term| {
            let columns = term.column_refs();
            let added = |column: &&datafusion::common::Column| {
                column.relation.is_none() && node.is_field(&column.name)
            };
            columns.iter().any(added)
                && columns
                    .iter()
                    .all(|column| added(column) || input.has_column(column))
                && !term
                    .exists(|inner| Ok(crate::is_ai_call(inner)))
                    .unwrap_or(true)
        })
        .cloned()
        .collect();
    if tests.is_empty() {
        return Ok(Transformed::no(plan));
    }
    let mut hinted = node.clone();
    hinted.hint_test = conjunction(hinted.hint_test.take().into_iter().chain(tests));
    let kept = Filter::try_new(
        filter.predicate.clone(),
        Arc::new(extension_node(hinted.checked()?)),
    )?;
    Ok(Transformed::yes(LogicalPlan::Filter(kept)))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::expr::ScalarFunction;
    use datafusion::logical_expr::{LogicalPlanBuilder, col, lit};
    use datafusion::optimizer::{Optimizer, OptimizerContext};
    use datafusion::prelude::left;

    use super::*;

    /// A map of `reviews.body` adding `summary`, with a fetch when
    /// `fetch` says so.
    fn summary(fetch: bool) -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("body", DataType::Utf8, true),
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
                .find(|found| found.name() == "ai_map")
                .unwrap(),
            vec![col("reviews.body"), lit("Summarize.")],
        ));
        let mut node = AiNode::try_new(
            Arc::new(reviews),
            call,
            "summary".into(),
            vec![Field::new("summary", DataType::Utf8, true)],
        )
        .unwrap();
        if fetch {
            node.fetch = Some(crate::ai::Fetch::Global(2));
        }
        extension_node(node.checked().unwrap())
    }

    fn hinted(plan: LogicalPlan) -> String {
        Optimizer::with_rules(vec![Arc::new(PushDownHintTests)])
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
    }

    #[test]
    fn a_test_of_an_answer_above_its_node_becomes_the_nodes_hint_test() {
        let test = left(col("summary"), lit(3_i64)).eq(lit("yes"));
        let plan = LogicalPlanBuilder::from(summary(false))
            .filter(test.and(col("reviews.id").gt(lit(1_i64))))
            .unwrap()
            .build()
            .unwrap();
        let text = hinted(plan);
        assert!(
            text.contains("hint test: left(summary, Int64(3)) = Utf8(\"yes\")\n"),
            "{text}"
        );
        // the filter stays
        assert!(text.starts_with("Filter: left(summary"), "{text}");
    }

    #[test]
    fn a_node_with_a_fetch_takes_no_hint_test() {
        let test = left(col("summary"), lit(3_i64)).eq(lit("yes"));
        let plan = LogicalPlanBuilder::from(summary(true))
            .filter(test)
            .unwrap()
            .build()
            .unwrap();
        assert!(!hinted(plan).contains("hint test"));
    }
}
