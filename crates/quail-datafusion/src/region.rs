//! The `ModelRegion` logical node.

use std::fmt;

use datafusion::common::{DFSchemaRef, Result};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// Connected logical model work that Quail plans as one unit.
///
/// The node is a leaf: the claimed operators are held in `plan`, not
/// as inputs, so DataFusion's optimizer rules never rewrite them.
/// Quail's planner reads `plan` when the node is planned physically.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct ModelRegion {
    /// The claimed operators, from the region's root down to its scans.
    pub plan: LogicalPlan,
}

impl ModelRegion {
    /// Create a model region over the given operators.
    pub fn new(plan: LogicalPlan) -> Self {
        Self { plan }
    }
}

impl UserDefinedLogicalNodeCore for ModelRegion {
    fn name(&self) -> &str {
        "ModelRegion"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![]
    }

    fn schema(&self) -> &DFSchemaRef {
        self.plan.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ModelRegion\n{}", self.plan.display_indent())
    }

    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, _inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(self.clone())
    }
}
