//! The `PushDownLimitIntoAiNodes` optimizer rule: a `LIMIT` as the fetch of
//! the `AiNode` whose rows it reads.
//!
//! An `AiNode`'s fetch `n` keeps the first `n` rows that pass its filter
//! `p`. A `LIMIT` of `k` rows after an `OFFSET` of `m` needs only the
//! first `k + m` rows of its input:
//!
//! ```text
//! Limit_k(AI(R; n))     ⇒ AI(R; n = min(k, n))
//! Limit_{k,m}(AI(R; n)) ⇒ Limit_{k,m}(AI(R; n = min(k + m, n)))
//! ```
//!
//! The executor then stops asking the model once that many rows pass.
//! The rule reads the node through the projections and subquery aliases
//! between them, which keep one row per row. It leaves the `LIMIT` alone
//! when:
//!
//! - another operator is between them, such as a filter, a join, a sort,
//!   or a `DISTINCT`, which can change how many rows there are;
//! - the node has a `HINT`, whose options estimate the rows of its test;
//! - the node has a per-key fetch, which `Fetch` cannot combine with a
//!   global one;
//! - the `LIMIT` keeps no row, which DataFusion's `eliminate_limit`
//!   replaces with an empty relation.
//!
//! A test of the node's answer is part of `p` before the rule runs:
//! `AbsorbIntoAiNodes` takes it while the filter is still between the
//! `LIMIT` and the node, and absorbs nothing into a node with a fetch.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::{FetchType, Limit, LogicalPlan, SkipType};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Fetch};
use crate::extension::{as_node, extension_node};

/// Make each `LIMIT` above an `AiNode` the node's fetch, as the module
/// explains.
#[derive(Debug, Default)]
pub struct PushDownLimitIntoAiNodes;

impl OptimizerRule for PushDownLimitIntoAiNodes {
    fn name(&self) -> &str {
        "push_down_limit_into_ai_nodes"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_down(push_down)
    }
}

fn push_down(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Limit(limit) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let (Ok(SkipType::Literal(skip)), Ok(FetchType::Literal(Some(fetch)))) =
        (limit.get_skip_type(), limit.get_fetch_type())
    else {
        return Ok(Transformed::no(plan));
    };
    // DataFusion's `eliminate_limit` replaces a LIMIT 0 with no rows
    if fetch == 0 {
        return Ok(Transformed::no(plan));
    }
    let Some((input, changed)) = with_fetch(&limit.input, skip + fetch)? else {
        return Ok(Transformed::no(plan));
    };
    if skip == 0 {
        return Ok(Transformed::yes(input));
    }
    if !changed {
        return Ok(Transformed::no(plan));
    }
    Ok(Transformed::yes(LogicalPlan::Limit(Limit {
        skip: limit.skip.clone(),
        fetch: limit.fetch.clone(),
        input: Arc::new(input),
    })))
}

/// `plan` with its `AiNode` returning at most `rows` rows, and whether
/// that changed the node; None when the node is out of reach or cannot
/// take the fetch, as the module explains.
fn with_fetch(plan: &LogicalPlan, rows: usize) -> Result<Option<(LogicalPlan, bool)>> {
    match plan {
        LogicalPlan::Projection(_) | LogicalPlan::SubqueryAlias(_) => {
            let Some((input, changed)) = with_fetch(plan.inputs()[0], rows)? else {
                return Ok(None);
            };
            let rebuilt = plan.with_new_exprs(plan.expressions(), vec![input])?;
            Ok(Some((rebuilt, changed)))
        }
        LogicalPlan::Extension(_) => {
            let Some(node) = as_node::<AiNode>(plan) else {
                return Ok(None);
            };
            let rows = match &node.fetch {
                _ if node.hint.is_some() => return Ok(None),
                Some(Fetch::PerKey { .. }) => return Ok(None),
                Some(Fetch::Global(found)) if *found <= rows => {
                    return Ok(Some((plan.clone(), false)));
                }
                Some(Fetch::Global(_)) | None => rows,
            };
            let mut fetched: AiNode = node.clone();
            fetched.fetch = Some(Fetch::Global(rows));
            Ok(Some((extension_node(fetched.checked()?), true)))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::expr::ScalarFunction;
    use datafusion::logical_expr::{Expr, LogicalPlanBuilder, col, lit};
    use datafusion::optimizer::{Optimizer, OptimizerContext};

    use super::*;

    fn kind_reviews(hint: Option<Expr>) -> LogicalPlan {
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
        node.filter = Some(col("kind"));
        node.projection = Some(Vec::new());
        node.hint = hint;
        extension_node(node.checked().unwrap())
    }

    fn pushed(plan: LogicalPlan) -> String {
        Optimizer::with_rules(vec![Arc::new(PushDownLimitIntoAiNodes)])
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
    }

    #[test]
    fn a_limit_over_an_ai_filter_becomes_its_fetch() {
        let plan = LogicalPlanBuilder::from(kind_reviews(None))
            .project(vec![col("reviews.id")])
            .unwrap()
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(!text.contains("Limit"), "{text}");
        assert!(text.contains("fetch: 3"), "{text}");
    }

    #[test]
    fn a_limit_with_an_offset_stays_and_fetches_its_offset_rows_too() {
        let plan = LogicalPlanBuilder::from(kind_reviews(None))
            .limit(2, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(text.contains("Limit: skip=2, fetch=3"), "{text}");
        assert!(text.contains("fetch: 5"), "{text}");
    }

    #[test]
    fn a_limit_stays_above_a_filter_or_a_hint() {
        let filtered = LogicalPlanBuilder::from(kind_reviews(None))
            .filter(col("reviews.id").gt(lit(3)))
            .unwrap()
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let hinted = LogicalPlanBuilder::from(kind_reviews(Some(lit(0.5))))
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        for plan in [filtered, hinted] {
            let text = pushed(plan);
            assert!(text.contains("Limit: skip=0, fetch=3"), "{text}");
            assert!(!text.contains("fetch: 3"), "{text}");
        }
    }
}
