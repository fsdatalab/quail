use datafusion::common::Result;
use datafusion::common::tree_node::Transformed;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

mod demand;
mod fetch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeriveOption {
    Filters,
    Fetch,
    Projection,
    Guard,
}

impl DeriveOption {
    const ALL: [Self; 4] = [Self::Filters, Self::Fetch, Self::Projection, Self::Guard];

    pub const fn rule_name(self) -> &'static str {
        match self {
            Self::Filters => "derive.filters",
            Self::Fetch => "derive.fetch",
            Self::Projection => "derive.projection",
            Self::Guard => "derive.guard",
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
            filters: wants(DeriveOption::Filters),
            fetch: wants(DeriveOption::Fetch),
            guard: wants(DeriveOption::Guard),
            projection: wants(DeriveOption::Projection),
        };
        demand::demand_pass(plan, options)
    }
}
