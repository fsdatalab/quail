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

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct ModelRegion {
    pub id: Option<usize>,
    pub plan: LogicalPlan,
    pub inputs: Vec<LogicalPlan>,
    pub input_names: Vec<String>,
}

impl ModelRegion {
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
    pub fn has_sorted_input(&self) -> bool {
        self.inputs.iter().any(|input| {
            input
                .exists(|node| Ok(matches!(node, LogicalPlan::Sort(_))))
                .unwrap_or(true)
        })
    }

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

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        self.plan
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
