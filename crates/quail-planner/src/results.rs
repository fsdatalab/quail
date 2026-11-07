//! The result nodes a physical plan ends with: the Project that reads the
//! region's rows, and the Limit of the region's fetch.

use quail_physical::{
    LimitAttributes, ModelCall, Node, PhysicalNode, PortRef, ProjectAttributes, Projected,
    ValueExpression, input_ports,
};
use serde_json::Value;

use crate::expressions::{
    PROBABILITIES_SUFFIX, SPAN_SUFFIX, call_aliases, column_name, column_references,
    projected_call, projected_name,
};
use crate::operators::Operators;

/// The struct columns a Project packs, as `Project.structs` encodes them.
pub fn structs_value(structs: &[(String, Vec<(String, String)>)]) -> Value {
    Value::Array(
        structs
            .iter()
            .map(|(name, members)| {
                Value::Array(vec![
                    Value::from(name.clone()),
                    Value::Array(
                        members
                            .iter()
                            .map(|(field, member)| {
                                Value::Array(vec![
                                    Value::from(field.clone()),
                                    Value::from(member.clone()),
                                ])
                            })
                            .collect(),
                    ),
                ])
            })
            .collect(),
    )
}

/// A column a model call adds: its name, the call's table aliases, kind,
/// and labels, and whether it is the call's answer.
pub type AnswerColumn = (String, Vec<String>, String, Value, bool);

/// The columns a plan's result Project binds: each source expression
/// column and each model answer column, as `ProjectAttributes` holds
/// them.
pub fn project_bindings(found: &Operators) -> (Vec<(String, ValueExpression)>, Vec<AnswerColumn>) {
    (source_columns(found), answer_columns(found))
}

/// Each result column a source expression computes, by name.
fn source_columns(found: &Operators) -> Vec<(String, ValueExpression)> {
    let projected = found
        .projection
        .columns
        .iter()
        .filter_map(|column| match column {
            Projected::Expression(expression) => Some(expression),
            Projected::Name(_) => None,
        });
    let mut found: Vec<(String, ValueExpression)> = Vec::new();
    for expression in projected {
        if column_references(expression).is_empty() {
            continue;
        }
        let name = column_name(expression, true);
        match found.iter_mut().find(|(known, _)| *known == name) {
            Some(entry) => entry.1 = expression.clone(),
            None => found.push((name, expression.clone())),
        }
    }
    found
}

/// Each column a model call adds: its name, the call's aliases, kind,
/// and labels, and whether it is the call's answer, not a label's
/// probabilities or an answer's span.
fn answer_columns(found: &Operators) -> Vec<AnswerColumn> {
    let mut columns: Vec<AnswerColumn> = Vec::new();
    let mut put = |name: String, call: &ModelCall, answer: bool| {
        let entry = (
            name.clone(),
            call_aliases(call),
            call.kind.clone(),
            call.labels.clone(),
            answer,
        );
        match columns.iter_mut().find(|known| known.0 == name) {
            Some(known) => *known = entry,
            None => columns.push(entry),
        }
    };
    for column in &found.classifies {
        let call = &column.call;
        put(column.name.clone(), call, true);
        if call.probabilities {
            put(
                format!("{}{PROBABILITIES_SUFFIX}", column.name),
                call,
                false,
            );
        }
    }
    for column in &found.extracts {
        let call = &column.call;
        put(column.name.clone(), call, true);
        put(format!("{}{SPAN_SUFFIX}", column.name), call, false);
    }
    for column in found.sae_features.iter().chain(&found.maps) {
        put(column.name.clone(), &column.call, true);
    }
    for column in &found.projection.columns {
        if let Some(call) =
            projected_call(column, &["label", "extract", "score", "features", "map"])
        {
            put(projected_name(column), call, true);
        }
    }
    columns
}

fn node<A>(id: &str, inputs: &[PortRef], attributes: A) -> Node<A> {
    Node {
        id: id.to_string(),
        inputs: input_ports(inputs),
        attributes,
    }
}

/// The Project that reads the region's rows, and the Limit of the
/// region's fetch, which `should_push_down_limit` stops the model at.
pub fn result_nodes(
    found: &Operators,
    inputs: &[PortRef],
    columns: &[String],
    structs: &[(String, Vec<(String, String)>)],
) -> Vec<PhysicalNode> {
    let mut nodes = vec![PhysicalNode::Project(node(
        "project",
        inputs,
        ProjectAttributes {
            columns: columns.to_vec(),
            structs: structs_value(structs),
            sources: source_columns(found),
            answers: answer_columns(found),
        },
    ))];
    if let Some(count) = found.fetch {
        nodes.push(PhysicalNode::Limit(node(
            "limit",
            &[quail_physical::port("project", "rows")],
            LimitAttributes { count },
        )));
    }
    nodes
}
