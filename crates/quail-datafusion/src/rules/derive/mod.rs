//! The `Derive` optimizer rule: what the operators above an `AiNode`
//! need of it, as the node's own options.
//!
//! An `AiNode` with no options asks the model about every row of its
//! input and returns every answer. The operators above it usually need
//! less: a `Filter` keeps only some answers, a `LIMIT` only some rows.
//! The rule gives each node what it can use of that, so the executor
//! skips the rest:
//!
//! - **filters.** A test of a node's answer that the node can decide
//!   ([`AiNode::accepts_test`]) becomes its filter,
//!   `σ_q(AI(R; p)) ⇒ AI(R; p ∧ q)`, and leaves the `Filter`.
//! - **fetch.** How many rows are enough: a `LIMIT`, one row of each
//!   key of a `DISTINCT` or a semi-join, or `k` of each group under
//!   `HAVING COUNT(*) >= k`, `Limit_k(AI(R; n)) ⇒ AI(R; n = min(k, n))`.
//! - **guard.** Where a column node's answer is read: a `CASE` reads a
//!   branch only where its condition holds.
//! - **projection.** An `AI.IF` node whose answer only its own filter
//!   reads returns no column, and a read of the answer above `σ_b` is
//!   the constant the filter keeps.
//!
//! The first three come from one pass down the plan ([`demand`]): each
//! operator says what the plan above needs of its input's rows, so a
//! need reaches a node through any operators that pass it on. The
//! projection steps, the guard of an `AI.IF` in an OR
//! ([`super::or_guards`]), and the substring precheck
//! ([`super::substring`]) follow it.

use std::collections::BTreeSet;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::{Expr, LogicalPlan, lit};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

mod demand;
mod fetch;
use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};

/// One kind of option [`Derive`] sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeriveOption {
    /// The rows a node drops: tests of its answer, and a document test
    /// an extraction's answer implies.
    Filters,
    /// How many rows are enough: a `LIMIT`, or one or `k` rows for each
    /// key of a `DISTINCT`, a semi-join, or a count threshold.
    Fetch,
    /// The added columns and extraction fields a node returns.
    Projection,
    /// The rows a node need not ask about: those a `CASE` does not read
    /// its answer for, or another term of an OR already passes.
    Guard,
}

impl DeriveOption {
    const ALL: [Self; 4] = [Self::Filters, Self::Fetch, Self::Projection, Self::Guard];

    /// The name of the rule that derives this option alone.
    pub const fn rule_name(self) -> &'static str {
        match self {
            Self::Filters => "derive.filters",
            Self::Fetch => "derive.fetch",
            Self::Projection => "derive.projection",
            Self::Guard => "derive.guard",
        }
    }
}

/// Give each `AiNode` what the operators above it need of it, as the
/// module explains: every option, or `only` one.
#[derive(Debug, Default)]
pub struct Derive {
    only: Option<DeriveOption>,
}

impl Derive {
    /// The rule that derives `option` and no other. A session lists such
    /// rules in place of `derive` to leave an option out.
    pub fn only(option: DeriveOption) -> Self {
        Self { only: Some(option) }
    }

    /// The rule named `name`: `derive`, or one option's own.
    pub fn named(name: &str) -> Option<Self> {
        if name == "derive" {
            return Some(Self::default());
        }
        DeriveOption::ALL
            .into_iter()
            .find(|option| option.rule_name() == name)
            .map(Self::only)
    }
}

/// A step of the rule after the demand pass: the option it derives and
/// its rewrite.
type Step = (
    DeriveOption,
    fn(LogicalPlan) -> Result<Transformed<LogicalPlan>>,
);

/// The steps after the demand pass, in the order they run. A later step
/// reads what an earlier one set: a guard follows the filters.
const STEPS: [Step; 4] = [
    (DeriveOption::Projection, drop_unread_answers),
    (DeriveOption::Projection, super::projection::projections),
    (DeriveOption::Guard, super::or_guards::or_guards),
    (DeriveOption::Filters, super::substring::substring_prechecks),
];

impl OptimizerRule for Derive {
    fn name(&self) -> &str {
        match self.only {
            Some(option) => option.rule_name(),
            None => "derive",
        }
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let wants = |option: DeriveOption| self.only.is_none_or(|only| only == option);
        let options = demand::Options {
            filters: wants(DeriveOption::Filters),
            fetch: wants(DeriveOption::Fetch),
            guard: wants(DeriveOption::Guard),
        };
        let done = demand::demand_pass(plan, options)?;
        let mut changed = done.transformed;
        let mut plan = done.data;
        for (option, step) in STEPS {
            if self.only.is_none_or(|only| only == option) {
                let done = step(plan)?;
                changed |= done.transformed;
                plan = done.data;
            }
        }
        Ok(Transformed::new_transformed(plan, changed))
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
