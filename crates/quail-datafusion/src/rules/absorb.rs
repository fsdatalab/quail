//! The `AbsorbIntoAiNodes` optimizer rule: a test of an AI column's answer
//! as the `AiNode`'s own filter.
//!
//! [`crate::PlaceAiNodes`] leaves each test of an answer, such as
//! `sentiment IN ('negative')` or a test of an extraction's field, in a
//! `Filter` right above the `AiNode` that adds the answer. The rule moves
//! each such term into the node's filter, `σ_q(AI(R; p)) ⇒ AI(R; p ∧ q)`,
//! keeping the terms in the order they stood. A node with a fetch keeps
//! the `Filter` above it: limiting the rows before the test is not the
//! same as limiting them after.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{Expr, Extension, Filter, LogicalPlan};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};

/// Move each test of an `AiNode`'s answer into the node's filter, as the
/// module explains.
#[derive(Debug, Default)]
pub struct AbsorbIntoAiNodes;

impl OptimizerRule for AbsorbIntoAiNodes {
    fn name(&self) -> &str {
        "absorb_into_ai_nodes"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_up(absorb)
    }
}

fn absorb(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Filter(filter) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let LogicalPlan::Extension(extension) = filter.input.as_ref() else {
        return Ok(Transformed::no(plan));
    };
    let Some(node) = extension.node.as_any().downcast_ref::<AiNode>() else {
        return Ok(Transformed::no(plan));
    };
    if node.fetch.is_some() || node.semantics != Semantics::Rows {
        return Ok(Transformed::no(plan));
    }
    let answers: BTreeSet<String> = node
        .returned_fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let (tests, kept): (Vec<Expr>, Vec<Expr>) = split_conjunction(&filter.predicate)
        .into_iter()
        .cloned()
        .partition(|term| {
            term.column_refs()
                .iter()
                .any(|column| column.relation.is_none() && answers.contains(&column.name))
        });
    if tests.is_empty() {
        return Ok(Transformed::no(plan));
    }
    let mut absorbed = node.clone();
    absorbed.filter = conjunction(absorbed.filter.take().into_iter().chain(tests));
    let mut plan = LogicalPlan::Extension(Extension {
        node: Arc::new(absorbed.checked()?),
    });
    if let Some(predicate) = conjunction(kept) {
        plan = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(plan))?);
    }
    Ok(Transformed::yes(plan))
}
