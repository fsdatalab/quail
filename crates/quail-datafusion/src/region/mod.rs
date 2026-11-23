//! Model regions: the part of a plan that Quail runs as one unit.
//!
//! - `claim`: `claim_model_regions`, which cuts each region out
//!   of the optimized plan.
//! - `lift`: the operators DataFusion keeps above a region.
//! - `inputs`: the relations DataFusion computes for a region.
//! - `derive`: a region's operators, read from its DataFusion plan.
//! - `region_hint`: the tests above a region that may drop its rows.
//! - `exec`: `ModelRegionExec`, which hands a region to a `RegionRunner`.
//! - this module: the `ModelRegion` logical node.

pub(crate) mod claim;
pub mod derive;
pub(crate) mod exec;
pub(crate) mod filter_guard;
pub(crate) mod identified;
mod inputs;
pub(crate) mod lift;
pub mod region_hint;
pub(crate) mod stream_filters;
mod unread;

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use std::collections::HashSet;
use std::fmt;

use datafusion::common::{DFSchemaRef, Result, internal_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

use crate::ai::AiNode;
use crate::functions::is_ai_call;
use crate::plan::as_node;

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
}

impl ModelRegion {
    /// Create a model region over the given operators.
    pub fn new(plan: LogicalPlan) -> Self {
        Self {
            id: None,
            plan,
            inputs: Vec::new(),
            input_names: Vec::new(),
        }
    }
}

impl ModelRegion {
    /// Whether an input of the region is sorted: the region then returns
    /// its rows in its input's order, as DataFusion reads them.
    pub fn has_sorted_input(&self) -> bool {
        self.inputs.iter().any(|input| {
            input
                .exists(|node| Ok(matches!(node, LogicalPlan::Sort(_))))
                .unwrap_or(true)
        })
    }

    /// The region's AI calls and the tables it reads, for EXPLAIN:
    /// `calls=[ai_filter], tables=[reviews]`.
    pub fn summary(&self) -> String {
        let (mut calls, mut tables) = (Vec::new(), Vec::new());
        let _ = self.plan.apply(|node| {
            if let Some(found) = as_node::<AiNode>(node) {
                calls.extend(
                    Some(found.function())
                        .filter(|name| !name.is_empty())
                        .map(str::to_string),
                );
            }
            if let LogicalPlan::TableScan(scan) = node {
                tables.push(scan.table_name.to_string());
            }
            Ok(TreeNodeRecursion::Continue)
        });
        format!(
            "calls=[{}], tables=[{}]",
            calls.join(", "),
            tables.join(", ")
        )
    }
}

/// Whether any operator in the subtree is an `AiNode`, or calls an AI
/// function in its expressions or in an EXISTS or IN subquery.
fn calls_ai(plan: &LogicalPlan) -> Result<bool> {
    let mut found = false;
    plan.apply_with_subqueries(|node| {
        found = as_node::<AiNode>(node).is_some()
            || node
                .expressions()
                .iter()
                .any(|expr| expr.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(false));
        Ok(if found {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    })?;
    Ok(found)
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
        write!(f, "ModelRegion: {}", self.summary())
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
