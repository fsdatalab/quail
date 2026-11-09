//! The physical graph, its ports, and the plan envelope.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::physical::nodes::PhysicalNode;
use crate::tagged_struct;

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

tagged_struct! {
    /// A model call's answer columns, as a node's output reports them
    /// before later row operations.
    AnswerOutput(AnswerOutputFields, AnswerOutputTagged) = "quail.physical.base.AnswerOutput" {
        kind: String,
        name: String,
        columns: Vec<String>,
    }
}

tagged_struct! {
    /// One output of a node: its name, what it carries, its columns, and
    /// the model answer it holds, if any.
    OutputPort(OutputPortFields, OutputPortTagged) = "quail.physical.base.OutputPort" {
        name: String,
        value_type: ValueType,
        schema: Vec<String>,
        answer: Option<AnswerOutput>,
    }
}

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
