//! Quail's physical plan types, as `quail/physical/codec.py` writes them.
//!
//! A plan crosses from the planner to the GPU executor as a JSON
//! envelope. These types read and write that envelope, so the Rust
//! planner can produce plans the Python executor runs.

mod expressions;
mod graph;
mod nodes;
mod specs;
mod tagged;
mod topology;

pub use expressions::{
    AggregateCall, Alias, Asked, ColumnRef, Condition, Connective, Equality, ExtractStruct,
    FilterPredicate, GetField, HavingTest, Literal, ModelCall, Not, Projected, Prompt,
    RegularPredicate, ScalarFunction, SortExpression, SortKey, TaggedModelCall, ValueExpression,
    WindowCall,
};
pub use graph::{InputPort, NestedGraph, PhysicalGraph, PlanEnvelope, PortRef, ValueType};
pub use nodes::*;
pub use specs::{ClassifySpec, ExtractSpec, MapSpec, SaeFeaturesSpec, ScoreSpec, TokenIds};
pub use topology::{
    Pipelines, build_pipelines, groups_acyclic, input_ports, port, resident_inputs,
    topological_nodes, value_type_for_port,
};

/// Read a plan envelope from JSON.
pub fn read_plan(text: &str) -> serde_json::Result<PlanEnvelope> {
    serde_json::from_str(text)
}

/// Write a plan envelope as JSON.
pub fn write_plan(plan: &PlanEnvelope) -> serde_json::Result<String> {
    serde_json::to_string(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tagged_struct_reads_and_writes_its_tag() {
        let text = r#"{"type":"quail.physical.base.PortRef","node_id":"scan:r","port":"rows:r"}"#;
        let port: PortRef = serde_json::from_str(text).unwrap();
        assert_eq!(port.node_id, "scan:r");
        assert_eq!(serde_json::to_string(&port).unwrap(), text);
    }

    #[test]
    fn a_struct_with_another_class_tag_is_refused() {
        let text = r#"{"type":"quail.physical.base.InputPort","node_id":"a","port":"b"}"#;
        assert!(serde_json::from_str::<PortRef>(text).is_err());
    }

    #[test]
    fn an_expression_saved_before_the_move_reads() {
        let text = r#"{"type":"quail.logical.expressions.ColumnRef","alias":"r","provider":"reviews","column":"id"}"#;
        let expression: ValueExpression = serde_json::from_str(text).unwrap();
        assert!(matches!(expression, ValueExpression::ColumnRef(_)));
    }
}
