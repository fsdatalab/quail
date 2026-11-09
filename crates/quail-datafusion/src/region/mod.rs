//! Model regions: the part of a plan that Quail runs as one unit.
//!
//! - `claim`: the `ClaimModelRegions` rule, which cuts each region out
//!   of the optimized plan.
//! - `lift`, `join_back`, `shared`: the operators DataFusion keeps above
//!   a region, and the inputs that a region and its parent both read.
//! - `region_hint`: the tests above a region that may drop its rows.
//! - `exec`: `ModelRegionExec`, which hands a region to a `RegionRunner`.
//! - this module: the `ModelRegion` logical node.

pub(crate) mod claim;
pub(crate) mod exec;
pub(crate) mod identified;
mod inputs;
mod join;
pub(crate) mod join_back;
pub(crate) mod lift;
mod lowering;
pub mod region_hint;
pub(crate) mod shared;

use std::collections::HashSet;
use std::fmt;

use datafusion::common::{DFSchemaRef, Result, internal_err};
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
    /// Query-local identity assigned in logical traversal order after claiming.
    /// Preserved through physical planning, independent of execution order.
    pub id: Option<usize>,
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
            id: None,
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

    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        if !exprs.is_empty() || inputs.len() != self.inputs.len() {
            return internal_err!(
                "ModelRegion has no expressions and {} inputs, got {} and {}",
                self.inputs.len(),
                exprs.len(),
                inputs.len()
            );
        }
        Ok(Self {
            inputs,
            ..self.clone()
        })
    }
}
