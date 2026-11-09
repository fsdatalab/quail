//! Quail's physical plan types, which the GPU executor runs.
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
    Alias, Asked, ColumnRef, Equality, Expression, ExtractStruct, FilterPredicate, GetField,
    ModelCall, Projected, Prompt, ScoreTest, TaggedModelCall, ValueExpression,
};
pub use graph::{
    AnswerOutput, InputPort, OutputPort, PhysicalGraph, PlanEnvelope, PortRef, ValueType,
};
pub use nodes::*;
pub use specs::{
    ClassifySpec, ExtractSpec, MapSpec, RequestClassifySpec, RequestExtractSpec, RequestFilterSpec,
    RequestJoinSpec, RequestMapSpec, SaeFeaturesSpec, ScoreSpec, TokenIds,
};
pub use topology::{
    Pipelines, build_pipelines, groups_acyclic, input_ports, port, resident_inputs, same_decoding,
    topological_nodes, validate_graph, value_type_for_port, value_type_name,
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
    fn malformed_schema_payloads_return_errors() {
        use crate::operators::Schema;
        for (ipc, message) in [
            ("not base64!", "invalid schema base64"),
            ("", "invalid IPC schema"),
            ("AAAAAA==", "invalid IPC schema"),
        ] {
            let schema = Schema { ipc: ipc.into() };
            assert!(schema.names().unwrap_err().message().starts_with(message));
        }
    }

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

    #[test]
    fn a_plan_saved_before_guards_and_key_survivors_reads_with_their_defaults() {
        let filter = PhysicalNode::AiFilter(Node {
            id: "ai_filter:r".into(),
            inputs: Vec::new(),
            attributes: AiFilterAttributes {
                alias: "r".into(),
                arena_writes: false,
                keep_kv: false,
                written_position: 0,
                question_tokens: 4,
                preamble_tokens: 2,
                selectivity: None,
                expected_rows: 6.0,
                question_token_ids: Vec::new(),
                document_head: Vec::new(),
                share_prefixes: false,
                attention: String::new(),
                stop_key: Vec::new(),
                pass_group: -1,
                state_tail_tokens: 0,
                question_header_token_ids: Vec::new(),
                document_ends: serde_json::Value::Array(Vec::new()),
                num_key_survivors: 1,
                guard: None,
                alternative: None,
                answer: None,
            },
        });
        let map = PhysicalNode::AiMap(Node {
            id: "ai-map:0".into(),
            inputs: Vec::new(),
            attributes: AiMapAttributes {
                backend_name: "quail".into(),
                model: String::new(),
                spec: None,
                keep_kv: false,
                pass_group: -1,
                document_ends: serde_json::Value::Array(Vec::new()),
                attention: String::new(),
                guard: None,
                stop_tests: Vec::new(),
                stop_when_true: false,
                hint_tests: Vec::new(),
            },
        });
        let added = [
            "num_key_survivors",
            "guard",
            "alternative",
            "answer",
            "stop_tests",
            "stop_when_true",
            "hint_tests",
        ];
        for node in [filter, map] {
            let mut saved = serde_json::to_value(&node).unwrap();
            let attributes = saved["attributes"].as_object_mut().unwrap();
            for name in added {
                attributes.remove(name);
            }
            let read: PhysicalNode = serde_json::from_value(saved).unwrap();
            assert_eq!(read, node);
        }
        let predicate = r#"{"type":"quail.common.expressions.FilterPredicate","expression":
            {"type":"quail.common.expressions.ModelCall","prompt":{"type":
            "quail.common.expressions.Prompt","template":"t","args":[],"preamble":"",
            "tail":"","frame":"","frame_text":"","kind":"boolean","layout":"",
            "tail_segments":[],"label_texts":[],"label_prefix":"","categories":[],
            "descriptions":[],"task_description":"","turn":["",""],"source":"",
            "letters":[]},"kind":"boolean","labels":null,"descriptions":null,
            "probabilities":false,"question":"","pooling":"","max_answer_tokens":0},
            "selectivity":null}"#;
        let predicate: FilterPredicate = serde_json::from_str(predicate).unwrap();
        assert_eq!(
            (predicate.guard, predicate.alternative, predicate.answer),
            (None, None, None)
        );
        assert_eq!(predicate.guard_selectivity, None);
        let text = r#"{"type":"quail.common.expressions.Expression","encoded":"",
            "columns":[],"text":"TRUE"}"#;
        let ValueExpression::Value(expression) = serde_json::from_str(text).unwrap() else {
            panic!("an expression");
        };
        assert!(expression.types.is_empty());
    }

    #[test]
    fn an_expression_names_its_columns_in_its_text() {
        let column = |alias: &str, name: &str| ColumnRef {
            alias: alias.into(),
            provider: String::new(),
            column: name.into(),
        };
        let expression = Expression {
            encoded: String::new(),
            columns: vec![column("r", "stars"), column("", "label")],
            types: vec!["Int64".into(), "Utf8".into()],
            text: "{0} > 3 AND {1} = '{{x}}'".into(),
        };
        assert_eq!(expression.text_with(true), "r.stars > 3 AND label = '{x}'");
        assert_eq!(expression.text_with(false), "stars > 3 AND label = '{x}'");
    }
}
