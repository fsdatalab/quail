use datafusion::common::Result;
use datafusion::common::tree_node::Transformed;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

mod demand;
mod node;
mod row_limit;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeriveOption {
    InlinePredicates,
    RowLimit,
    FieldPruning,
    Precondition,
}

impl DeriveOption {
    const ALL: [Self; 4] = [
        Self::InlinePredicates,
        Self::RowLimit,
        Self::FieldPruning,
        Self::Precondition,
    ];

    pub const fn rule_name(self) -> &'static str {
        match self {
            Self::InlinePredicates => "derive.inline_predicates",
            Self::RowLimit => "derive.row_limit",
            Self::FieldPruning => "derive.field_pruning",
            Self::Precondition => "derive.precondition",
        }
    }
}

#[derive(Debug, Default)]
pub struct Derive {
    only: Option<DeriveOption>,
}

impl Derive {
    pub fn only(option: DeriveOption) -> Self {
        Self { only: Some(option) }
    }

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
            inline_predicates: wants(DeriveOption::InlinePredicates),
            row_limit: wants(DeriveOption::RowLimit),
            precondition: wants(DeriveOption::Precondition),
            field_pruning: wants(DeriveOption::FieldPruning),
        };
        demand::demand_pass(plan, options)
    }
}
