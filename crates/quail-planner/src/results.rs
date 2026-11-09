//! What a region returns: the columns its rows hold, as its operators
//! name them, and the Limit of the region's fetch that a physical plan
//! ends with.

use crate::physical::{
    LimitAttributes, ModelCall, Node, PhysicalNode, PortRef, Projected, ValueExpression,
    input_ports,
};
use serde_json::Value;

use crate::expressions::{
    PROBABILITIES_SUFFIX, SPAN_SUFFIX, call_aliases, column_name, column_references, packing,
    projected_call, projected_name,
};
use crate::operators::Operators;

/// A column a model call adds: its name, the call's table aliases, kind,
/// and labels, and whether it is the call's answer.
pub type AnswerColumn = (String, Vec<String>, String, Value, bool);

/// The columns of a region's rows, in order, before struct packing.
///
/// Each SELECT list column, then a label's probabilities and an
/// extraction's spans, when the SELECT list returns them, after the
/// column they belong to.
pub fn region_columns(found: &Operators) -> Vec<String> {
    let mut columns = Vec::new();
    for column in &found.projection.columns {
        let name = projected_name(column);
        columns.push(name.clone());
        if let Some(call) = projected_call(column, &[]) {
            if call.probabilities {
                columns.push(format!("{name}{PROBABILITIES_SUFFIX}"));
            }
            let spanned = found.projection.spans.iter().any(|spanned| {
                name.strip_prefix(spanned.as_str())
                    .is_some_and(|rest| rest.starts_with('.'))
            });
            if call.kind == "extract" && spanned {
                columns.push(format!("{name}{SPAN_SUFFIX}"));
            }
        }
    }
    columns
}

/// The struct columns of a region's rows: (struct column, [(field,
/// member column)]).
pub fn region_structs(found: &Operators) -> Vec<(String, Vec<(String, String)>)> {
    packing(&found.projection.structs, &found.projection.spans)
}

/// Each region column a source expression computes, by name.
pub fn source_columns(found: &Operators) -> Vec<(String, ValueExpression)> {
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
pub fn answer_columns(found: &Operators) -> Vec<AnswerColumn> {
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

/// The Limit of the region's fetch, reading the region's rows at
/// `input`; `should_push_down_limit` stops the model at it.
pub fn limit_node(found: &Operators, input: &PortRef) -> Option<PhysicalNode> {
    found.fetch.map(|count| {
        PhysicalNode::Limit(Node {
            id: "limit".to_string(),
            inputs: input_ports(std::slice::from_ref(input)),
            attributes: LimitAttributes { count },
        })
    })
}
