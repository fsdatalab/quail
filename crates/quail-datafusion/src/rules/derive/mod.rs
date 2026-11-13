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
//! All four come from one pass down the plan ([`demand`]): each operator
//! says what the plan above needs of its input's rows, so a need reaches
//! a node through any operators that pass it on. An extraction asks only
//! the fields of its answer the plan reads.

use datafusion::common::Result;
use datafusion::common::tree_node::Transformed;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

mod demand;
mod fetch;

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
            projection: wants(DeriveOption::Projection),
        };
        demand::demand_pass(plan, options)
    }
}
