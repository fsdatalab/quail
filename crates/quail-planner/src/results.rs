//! The result nodes a physical plan ends with, as
//! `quail/planner/graph/results.py` builds them.

use quail_physical::{
    FilterAttributes, LimitAttributes, ModelCall, Node, PhysicalNode, PortRef, ProjectAttributes,
    Projected, SortAttributes, SortExpression, UnnestAttributes, ValueExpression, input_ports,
    port,
};
use serde_json::Value;

use crate::LogicalPlan;
use crate::expressions::{
    PROBABILITIES_SUFFIX, SPAN_SUFFIX, call_aliases, column_name, column_references,
    projected_call, projected_name, sort_name,
};
use crate::logical::{LogicalNode, ResultNode};
use crate::nodes::{above_projection, column_call, packed_columns, result_columns};
use crate::operators::{operators, projection};

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
pub fn project_bindings(plan: &LogicalPlan) -> (Vec<(String, ValueExpression)>, Vec<AnswerColumn>) {
    let root = match &plan.root {
        LogicalNode::Result(result) => result.clone(),
        other => ResultNode {
            input: Box::new(other.clone()),
            limit: None,
            order: Vec::new(),
            offset: 0,
            distinct: false,
        },
    };
    (source_columns(plan, &root), answer_columns(plan))
}

/// Each result column a source expression computes, by name: the
/// projected columns and the sort keys that read a table's columns.
fn source_columns(plan: &LogicalPlan, root: &ResultNode) -> Vec<(String, ValueExpression)> {
    let LogicalNode::Project(project) = projection(plan) else {
        return Vec::new();
    };
    let projected = project.columns.iter().filter_map(|column| match column {
        Projected::Expression(expression) => Some(expression),
        Projected::Name(_) => None,
    });
    let sorted = root.order.iter().map(|key| {
        let SortExpression::Value(value) = &key.expression;
        value
    });
    let mut found: Vec<(String, ValueExpression)> = Vec::new();
    for expression in projected.chain(sorted) {
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
fn answer_columns(plan: &LogicalPlan) -> Vec<AnswerColumn> {
    let found = operators(plan);
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
    for node in &found.classifies {
        let call = column_call(node);
        put(node.name.clone(), call, true);
        if call.probabilities {
            put(format!("{}{PROBABILITIES_SUFFIX}", node.name), call, false);
        }
    }
    for node in &found.extracts {
        let call = column_call(node);
        put(node.name.clone(), call, true);
        put(format!("{}{SPAN_SUFFIX}", node.name), call, false);
    }
    for node in found.sae_features.iter().chain(&found.maps) {
        put(node.name.clone(), column_call(node), true);
    }
    if let LogicalNode::Project(project) = projection(plan) {
        for column in &project.columns {
            if let Some(call) =
                projected_call(column, &["label", "extract", "score", "features", "map"])
            {
                put(projected_name(column), call, true);
            }
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

/// The Project, its unnests, and the nodes ending a plan, as `result_nodes`.
///
/// # Panics
///
/// The root's input is not a projection.
pub fn result_nodes(
    plan: &LogicalPlan,
    root: &ResultNode,
    inputs: &[PortRef],
    columns: &[String],
    structs: &[(String, Vec<(String, String)>)],
) -> Vec<PhysicalNode> {
    let mut chain: Vec<&LogicalNode> = Vec::new();
    let mut walked: &LogicalNode = &root.input;
    loop {
        let next = match walked {
            LogicalNode::Unnest(unnest) => &unnest.input,
            LogicalNode::Filter(filter) => &filter.input,
            LogicalNode::Project(project) => {
                if !above_projection(&project.input) {
                    break;
                }
                &project.input
            }
            _ => break,
        };
        chain.insert(0, walked);
        walked = next;
    }
    let mut hidden: Vec<String> = Vec::new();
    if chain.is_empty() {
        for key in &root.order {
            let name = sort_name(&key.expression);
            if !columns.contains(&name) && !hidden.contains(&name) {
                hidden.push(name);
            }
        }
    }
    let with_hidden = |names: Vec<String>| {
        let mut names = names;
        names.extend(hidden.iter().cloned());
        names
    };
    let mut nodes = vec![PhysicalNode::Project(node(
        "project",
        inputs,
        ProjectAttributes {
            columns: with_hidden(columns.to_vec()),
            structs: structs_value(structs),
            expressions: None,
            sources: source_columns(plan, root),
            answers: answer_columns(plan),
        },
    ))];
    let mut rows = port("project", "rows");
    let mut result = packed_columns(columns.iter().cloned(), structs);
    for (index, logical) in chain.iter().enumerate() {
        result = result_columns(logical);
        let node_id = match logical {
            LogicalNode::Unnest(unnest) => {
                let node_id = format!("unnest_{index}");
                nodes.push(PhysicalNode::Unnest(node(
                    &node_id,
                    std::slice::from_ref(&rows),
                    UnnestAttributes {
                        lists: unnest.columns.clone(),
                        outer: unnest.outer,
                        columns: with_hidden(result.clone()),
                    },
                )));
                node_id
            }
            LogicalNode::Project(project) => {
                let node_id = format!("project_{index}");
                let mut expressions = project.columns.clone();
                let available = result_columns(&project.input);
                for key in &root.order {
                    let name = sort_name(&key.expression);
                    if result.contains(&name) {
                        continue;
                    }
                    expressions.push(if available.contains(&name) {
                        Projected::Name(name)
                    } else {
                        let SortExpression::Value(value) = &key.expression;
                        Projected::Expression(value.clone())
                    });
                }
                nodes.push(PhysicalNode::Project(node(
                    &node_id,
                    std::slice::from_ref(&rows),
                    ProjectAttributes {
                        columns: expressions.iter().map(projected_name).collect(),
                        structs: Value::Array(Vec::new()),
                        expressions: Some(expressions),
                        sources: Vec::new(),
                        answers: Vec::new(),
                    },
                )));
                node_id
            }
            LogicalNode::Filter(filter) => {
                let node_id = format!("filter_{index}");
                nodes.push(PhysicalNode::Filter(node(
                    &node_id,
                    std::slice::from_ref(&rows),
                    FilterAttributes {
                        predicate: Some(filter.condition.clone()),
                        aliases: Vec::new(),
                        selectivity: None,
                        written_position: 0,
                        columns: with_hidden(result.clone()),
                    },
                )));
                node_id
            }
            _ => unreachable!("the chain holds unnests, filters, and projects"),
        };
        rows = port(&node_id, "rows");
    }
    if !root.order.is_empty() || root.distinct || root.offset != 0 {
        nodes.push(PhysicalNode::Sort(node(
            "sort",
            std::slice::from_ref(&rows),
            SortAttributes {
                keys: root
                    .order
                    .iter()
                    .map(|key| (sort_name(&key.expression), key.descending, key.nulls_first))
                    .collect(),
                columns: result,
                distinct: root.distinct,
                offset: root.offset,
                fetch: root.limit,
            },
        )));
    } else if let Some(count) = root.limit {
        nodes.push(PhysicalNode::Limit(node(
            "limit",
            std::slice::from_ref(&rows),
            LimitAttributes { count },
        )));
    }
    nodes
}
