//! The `PushDownProjectionIntoAiNodes` optimizer rule: an `AiNode` returns
//! only the added columns the projection above it reads.
//!
//! An `AiNode`'s projection `P` names the added columns it returns. A
//! projection `Q` above the node that reads only some of them needs only
//! those:
//!
//! ```text
//! π_Q(AI(R; P)) ⇒ π_Q(AI(R; P = Q[P]))
//! ```
//!
//! `Q[P]` is the columns of `P` that `Q` reads. For example, a SELECT
//! list over a FROM subquery that returns an extraction reads the
//! extraction's answer but not its spans: the node then returns the
//! answer alone. The node always returns its answer: a column node's
//! answer is the reason the model is asked. The node's filter and tests
//! still read every added column, as `P` applies after them.
//!
//! DataFusion's `OptimizeProjections` prunes the columns a node reads
//! from its input (`AiNode::necessary_children_exprs`), but not the
//! columns the node adds.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::{LogicalPlan, Projection};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};

/// Make each `AiNode` under a projection return only the added columns
/// the projection reads, as the module explains.
#[derive(Debug, Default)]
pub struct PushDownProjectionIntoAiNodes;

impl OptimizerRule for PushDownProjectionIntoAiNodes {
    fn name(&self) -> &str {
        "push_down_projection_into_ai_nodes"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up(push_down)
    }
}

fn push_down(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Projection(projection) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let Some(node) = as_node::<AiNode>(&projection.input) else {
        return Ok(Transformed::no(plan));
    };
    if node.semantics != Semantics::Rows || node.is_predicate() {
        return Ok(Transformed::no(plan));
    }
    let read: Vec<String> = projection
        .expr
        .iter()
        .flat_map(|expr| expr.column_refs())
        .filter(|column| column.relation.is_none())
        .map(|column| column.name.clone())
        .collect();
    let returned: Vec<usize> = match &node.projection {
        Some(positions) => positions.clone(),
        None => (0..node.fields.len()).collect(),
    };
    let kept: Vec<usize> = returned
        .iter()
        .copied()
        .filter(|index| *index == 0 || read.contains(node.fields[*index].name()))
        .collect();
    if kept.len() == returned.len() {
        return Ok(Transformed::no(plan));
    }
    let mut pruned = node.clone();
    pruned.projection = Some(kept);
    let input = extension_node(pruned.checked()?);
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(projection.expr.clone(), Arc::new(input))?,
    )))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::expr::ScalarFunction;
    use datafusion::logical_expr::{Expr, LogicalPlanBuilder, col, lit};
    use datafusion::optimizer::{Optimizer, OptimizerContext};

    use super::*;

    /// An extraction of `reviews.body`, adding `e` and `e_span`.
    fn extraction() -> LogicalPlan {
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
                .find(|found| found.name() == "ai_extract")
                .unwrap(),
            vec![col("reviews.body"), lit("who?")],
        ));
        let answer = Fields::from(vec![Field::new("who", DataType::Utf8, true)]);
        let node = AiNode::try_new(
            Arc::new(reviews),
            call,
            "e".into(),
            vec![
                Field::new("e", DataType::Struct(answer.clone()), true),
                Field::new("e_span", DataType::Struct(answer), true),
            ],
        )
        .unwrap();
        extension_node(node)
    }

    fn pushed(plan: LogicalPlan) -> String {
        Optimizer::with_rules(vec![Arc::new(PushDownProjectionIntoAiNodes)])
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
    }

    #[test]
    fn a_node_returns_only_the_added_columns_the_projection_reads() {
        let plan = LogicalPlanBuilder::from(extraction())
            .project(vec![col("reviews.id"), col("e")])
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(text.contains("returns: [e]"), "{text}");
    }

    #[test]
    fn a_node_keeps_the_columns_the_projection_reads_and_its_answer() {
        let spans = LogicalPlanBuilder::from(extraction())
            .project(vec![col("reviews.id"), col("e"), col("e_span")])
            .unwrap()
            .build()
            .unwrap();
        assert!(!pushed(spans).contains("returns:"));
        let ids = LogicalPlanBuilder::from(extraction())
            .project(vec![col("reviews.id")])
            .unwrap()
            .build()
            .unwrap();
        assert!(pushed(ids).contains("returns: [e]"));
    }
}
