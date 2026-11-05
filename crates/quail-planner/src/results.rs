//! The result nodes a physical plan ends with, as
//! `quail/planner/graph/results.py` builds them.

use quail_physical::{
    FilterAttributes, LimitAttributes, Node, PhysicalNode, PortRef, ProjectAttributes, Projected,
    SortAttributes, SortExpression, UnnestAttributes, input_ports, port,
};
use serde_json::Value;

use crate::expressions::{projected_name, sort_name};
use crate::logical::{LogicalNode, ResultNode};
use crate::nodes::{above_projection, packed_columns, result_columns};

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
