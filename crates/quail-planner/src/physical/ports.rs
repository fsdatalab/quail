//! The ports that join nodes: what each carries, and where from.

use serde::{Deserialize, Serialize};

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
