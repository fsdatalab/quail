//! The `ModelRegion` logical node.

use std::collections::HashSet;
use std::fmt;

use datafusion::common::{DFSchemaRef, Result};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// Connected logical model work that Quail plans as one unit.
///
/// The claimed operators are held in `plan`, not as inputs, so
/// DataFusion's optimizer rules never rewrite them again. The node's
/// inputs are the relations DataFusion computes for the region, such as
/// a table's plain filters or a CTE that groups rows, which `plan` reads
/// as tables named in `input_names`. Quail's planner reads `plan` when
/// the node is planned physically.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct ModelRegion {
    /// The claimed operators after DataFusion's optimizer, with the AI
    /// calls as `AiNode`s, from the region's root down to its scans.
    pub plan: LogicalPlan,
    /// The relations DataFusion computes for the region, in order.
    pub inputs: Vec<LogicalPlan>,
    /// The table name `plan` reads each input by.
    pub input_names: Vec<String>,
    /// The most rows the operators above read, when a LIMIT above passes
    /// them on unchanged; the region may stop the model once it has them.
    pub fetch: Option<usize>,
}

impl ModelRegion {
    /// Create a model region over the given operators.
    pub fn new(plan: LogicalPlan) -> Self {
        Self {
            plan,
            inputs: Vec::new(),
            input_names: Vec::new(),
            fetch: None,
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
        self.plan.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    // the region's columns are not its inputs' columns
    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        self.plan
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Quail's explain shows the region's plan
        write!(f, "ModelRegion")?;
        if let Some(fetch) = self.fetch {
            write!(f, ": fetch={fetch}")?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(Self {
            inputs,
            ..self.clone()
        })
    }
}
