//! The `AbsorbIntoAiNodes` optimizer rule: a test of an AI column's answer
//! as the `AiNode`'s own filter.
//!
//! [`crate::LowerAiCalls`] and [`crate::PlaceAiNodes`] leave each test of
//! an answer, such as `sentiment IN ('negative')` or a test of an
//! extraction's field, in a `Filter` right above the `AiNode` that adds
//! the answer. The rule moves
//! each such term into the node's filter, `σ_q(AI(R; p)) ⇒ AI(R; p ∧ q)`,
//! keeping the terms in the order they stood. A node with a fetch keeps
//! the `Filter` above it: limiting the rows before the test is not the
//! same as limiting them after.
//!
//! A test of an `AI.IF` answer `b` is absorbed the same way, whatever
//! its form once DataFusion has simplified it: `b`, `NOT b`, or `a OR
//! b`. The node then returns `b` only when an operator above reads it,
//! `π_∅` in `AI_{f → b}(R; p = b, π_∅)`: an `AI.IF` in a WHERE alone is
//! a filter that adds no column.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{Expr, Extension, Filter, LogicalPlan, lit};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};

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
        // a subquery's filters are absorbed where the query wrote them,
        // before DataFusion makes the subquery a join
        let absorbed = plan.transform_up_with_subqueries(absorb)?;
        let transformed = absorbed.transformed;
        let mut dropped = drop_unread_answers(absorbed.data)?;
        dropped.transformed |= transformed;
        Ok(dropped)
    }
}

/// `plan` with each filtering `AI.IF` node returning no column when no
/// operator above it reads its answer; a subquery returns its own columns.
fn drop_unread_answers(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let plan = plan.transform_up(|below| below.map_subqueries(drop_unread_answers))?;
    let (read, _) = read_kept_answers(plan.data.clone())?;
    let mut dropped = drop_in(read.data)?;
    dropped.transformed |= plan.transformed || read.transformed;
    Ok(dropped)
}

/// An `AI.IF` answer and the value every row its node passes has.
type Kept = Vec<(String, bool)>;

/// `plan` with each read of an answer above the `AI.IF` node that filters
/// on it as the value the filter keeps: TRUE above `σ_b`, FALSE above
/// `σ_{NOT b}`. For example, `SELECT AI.IF(q) AS k ... WHERE AI.IF(q)`
/// returns `k` as TRUE without asking again. Also the answers that hold
/// such a value in the rows `plan` returns.
///
/// Only operators that return rows of their input unchanged carry a
/// value up: an outer join's unmatched rows hold NULL instead.
fn read_kept_answers(plan: LogicalPlan) -> Result<(Transformed<LogicalPlan>, Kept)> {
    let mut kept = Kept::new();
    let mut changed = false;
    let mut inputs = Vec::new();
    for input in plan.inputs() {
        let (read, below) = read_kept_answers(input.clone())?;
        changed |= read.transformed;
        kept.extend(below);
        inputs.push(read.data);
    }
    let own = as_node::<AiNode>(&plan).and_then(|node| {
        let filters = node.function() == "ai_filter"
            && node.semantics == Semantics::Rows
            && !node.returned_fields().is_empty();
        let negated = node.answer_term(node.filter.as_ref().filter(|_| filters)?)?;
        Some((node.answer.clone(), !negated))
    });
    let carries = match &plan {
        LogicalPlan::Join(join) => join.join_type == datafusion::logical_expr::JoinType::Inner,
        LogicalPlan::Filter(_)
        | LogicalPlan::Projection(_)
        | LogicalPlan::SubqueryAlias(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_) => true,
        other => as_node::<AiNode>(other).is_some(),
    };
    let preserver = NamePreserver::new(&plan);
    let mut exprs = Vec::new();
    for expr in plan.expressions() {
        let saved = preserver.save(&expr);
        let read = expr.transform_up(|found| {
            let value = match &found {
                Expr::Column(column) if column.relation.is_none() => kept
                    .iter()
                    .find(|(name, _)| *name == column.name)
                    .map(|(_, value)| *value),
                _ => None,
            };
            Ok(match value {
                Some(value) => Transformed::yes(lit(value)),
                None => Transformed::no(found),
            })
        })?;
        changed |= read.transformed;
        exprs.push(saved.restore(read.data));
    }
    let plan = if changed {
        plan.with_new_exprs(exprs, inputs)?
    } else {
        plan
    };
    if !carries {
        kept.clear();
    }
    kept.extend(own);
    Ok((Transformed::new_transformed(plan, changed), kept))
}

fn drop_in(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let unread = |node: &AiNode, read: &BTreeSet<String>| {
        node.function() == "ai_filter"
            && node.semantics == Semantics::Rows
            && node.filter.is_some()
            && !node.returned_fields().is_empty()
            && !read.contains(&node.answer)
    };
    // the plan's own columns, then the columns each operator reads, top down
    let mut read: BTreeSet<String> = plan
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let mut found = false;
    plan.apply(|below| {
        let node = as_node::<AiNode>(below);
        found = found || node.is_some_and(|node| unread(node, &read));
        // a node's own filter and tests read its answer before it is dropped
        let own = node.map(|node| node.answer.as_str());
        for expr in below.expressions() {
            for column in expr.column_refs() {
                if Some(column.name.as_str()) != own {
                    read.insert(column.name.clone());
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if !found {
        return Ok(Transformed::no(plan));
    }
    plan.transform_up(|below| {
        let dropped = match as_node::<AiNode>(&below) {
            Some(node) if unread(node, &read) => {
                let mut node = node.clone();
                node.projection = Some(Vec::new());
                extension_node(node.checked()?)
            }
            _ => below.recompute_schema()?,
        };
        Ok(Transformed::yes(dropped))
    })
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
