//! The physical graph, its ports, and the plan envelope.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::nodes::{NestedNode, PhysicalNode};
use crate::tagged::tagged_struct;

/// What a port carries between two nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueType {
    FilterAnswers,
    JoinAnswers,
    LabelAnswers,
    ExtractAnswers,
    Pairs,
    Rows,
    Any,
}

tagged_struct! {
    /// One output port of another node.
    PortRef(PortRefFields, PortRefTagged) = "quail.physical.base.PortRef" {
        node_id: String,
        port: String,
    }
}

tagged_struct! {
    /// One input of a node: its name, what it carries, and where from.
    InputPort(InputPortFields, InputPortTagged) = "quail.physical.base.InputPort" {
        name: String,
        value_type: ValueType,
        source: PortRef,
        schema: Vec<String>,
    }
}

/// A physical graph: its nodes, and the port that returns the result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalGraph {
    pub nodes: Vec<PhysicalNode>,
    pub root: PortRef,
}

tagged_struct! {
    /// A graph nested inside a node, which carries a class tag.
    NestedGraph(NestedGraphFields, NestedGraphTagged) = "quail.physical.base.PhysicalGraph" {
        nodes: Vec<NestedNode>,
        root: PortRef,
    }
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
