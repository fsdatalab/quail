//! The physical graph and the plan envelope.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::physical::nodes::PhysicalNode;
use crate::physical::ports::PortRef;

/// A physical graph: its nodes, and the port that returns the result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalGraph {
    pub nodes: Vec<PhysicalNode>,
    pub root: PortRef,
}

/// The physical plan as it crosses a process boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanEnvelope {
    pub backend: String,
    pub model: String,
    pub device: String,
    pub gpus: i64,
    /// Settings the planner chose for the run, keyed by name.
    pub settings: Map<String, Value>,
    pub graph: PhysicalGraph,
}
