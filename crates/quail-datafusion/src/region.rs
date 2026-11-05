//! The `ModelRegion` logical node.

use std::fmt;

use datafusion::common::{DFSchemaRef, Result};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// Connected logical model work that Quail plans as one unit.
///
/// The claimed operators are held in `plan`, not as inputs, so
/// DataFusion's optimizer rules never rewrite them. The node's inputs
/// are the relations DataFusion computes for the region: a CTE or FROM
/// subquery that groups or windows rows, which `bound` reads as a table
/// named in `input_names`.
/// Quail's planner reads `plan` when the node is planned physically.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct ModelRegion {
    /// The claimed operators as DataFusion's SQL planner bound them, from
    /// the region's root down to its scans.
    pub bound: LogicalPlan,
    /// The same operators with the AI calls as Quail's plan nodes
    /// ([`crate::LowerAiCalls`]) and, once [`crate::OptimizeModelRegions`]
    /// has run, after DataFusion's rules inside the region.
    pub plan: LogicalPlan,
    /// Whether `plan` has been through DataFusion's rules.
    pub optimized: bool,
    /// The names of the rules [`crate::OptimizeModelRegions`] ran, which
    /// the binder also runs on each CTE stage's body.
    pub rules: Vec<String>,
    /// The relations DataFusion computes for the region, in order.
    pub inputs: Vec<LogicalPlan>,
    /// The table name `bound` reads each input by.
    pub input_names: Vec<String>,
}

impl ModelRegion {
    /// Create a model region over the given operators.
    pub fn new(plan: LogicalPlan) -> Self {
        Self {
            bound: plan.clone(),
            plan,
            optimized: false,
            rules: Vec::new(),
            inputs: Vec::new(),
            input_names: Vec::new(),
        }
    }
}

impl UserDefinedLogicalNodeCore for ModelRegion {
    fn name(&self) -> &str {
        "ModelRegion"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        self.inputs.iter().collect()
    }

    fn schema(&self) -> &DFSchemaRef {
        self.bound.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Quail's explain shows the region's plan
        write!(f, "ModelRegion")
    }

    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(Self {
            inputs,
            ..self.clone()
        })
    }
}
