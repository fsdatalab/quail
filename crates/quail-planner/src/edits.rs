//! Edits to a planned graph: put a node on an edge, take a Barrier out,
//! or move one to another edge.
//!
//! [`edit_graph`] applies an edit, then sets KV retention again and
//! validates the graph, as `PhysicalPlan.insert`, `remove`, and `move`
//! do before they price the result.

use std::collections::BTreeSet;

use crate::physical::{
    InputPort, PhysicalGraph, PhysicalNode, PortRef, ValueType, build_pipelines,
};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::physical_rules::plan_kv_retention;

/// One edit of a planned graph.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Edit {
    /// Put `node`, which has no inputs yet, on the edge from the first
    /// node of `between` to the second.
    Insert {
        node: Box<PhysicalNode>,
        between: (String, String),
    },
    /// Take a Barrier out; its consumers read its producer.
    Remove { node_id: String },
    /// Take a node out and put it on another edge.
    Move {
        node_id: String,
        between: (String, String),
    },
}

/// Apply `edit` to `graph`, set KV retention again, and validate.
///
/// # Errors
///
/// The edit names a node or edge the graph does not have, removes a node
/// that changes what the query means, or leaves an invalid graph.
pub fn edit_graph(graph: &PhysicalGraph, edit: &Edit) -> Result<PhysicalGraph, String> {
    validate_graph(graph)?;
    let edited = match edit {
        Edit::Insert { node, between } => insert(graph, node.as_ref().clone(), between)?,
        Edit::Remove { node_id } => remove(graph, node_id)?,
        Edit::Move { node_id, between } => {
            let mut node = find(graph, node_id)?.clone();
            node.inputs_mut().clear();
            insert(&remove(graph, node_id)?, node, between)?
        }
    };
    validate_graph(&edited)?;
    let edited = plan_kv_retention(&edited);
    validate_graph(&edited)?;
    Ok(edited)
}

/// The seconds an edit adds by separating filters from the join that
/// reads their survivors: each filter whose alias's KV the plan does not
/// already retain, and whose path no longer ends in a join, recomputes
/// its documents at the join.
pub fn recompute_seconds(
    graph: &PhysicalGraph,
    estimates: &Map<String, Value>,
    retained: &BTreeSet<String>,
) -> f64 {
    let pipelines = build_pipelines(&graph.nodes, &[]);
    graph
        .nodes
        .iter()
        .filter_map(|node| match node {
            PhysicalNode::AiFilter(filter) => Some((node, filter)),
            _ => None,
        })
        .filter(|(_, filter)| !retained.contains(&filter.attributes.alias))
        .filter(|(node, _)| {
            pipelines
                .get(node.id())
                .and_then(|members| members.last())
                .is_none_or(|last| !matches!(graph.nodes[*last], PhysicalNode::AiJoin(_)))
        })
        .filter_map(|(node, _)| {
            estimates
                .get(node.id())?
                .get("release_recompute_seconds")?
                .as_f64()
        })
        .sum()
}

/// Check the graph as [`crate::physical::validate_graph`] does.
///
/// # Errors
///
/// The graph breaks one of its rules.
pub fn validate_graph(graph: &PhysicalGraph) -> Result<(), String> {
    crate::physical::validate_graph(&graph.nodes, &graph.root)
}

fn type_name(value_type: ValueType) -> String {
    crate::physical::value_type_name(value_type).to_string()
}

fn find<'a>(graph: &'a PhysicalGraph, node_id: &str) -> Result<&'a PhysicalNode, String> {
    graph
        .nodes
        .iter()
        .find(|node| node.id() == node_id)
        .ok_or_else(|| format!("no node {node_id:?} in the plan"))
}

/// The one input port of `consumer` that reads `producer`.
fn edge(graph: &PhysicalGraph, producer: &str, consumer: &str) -> Result<InputPort, String> {
    find(graph, producer)?;
    let target = find(graph, consumer)?;
    let ports: Vec<&InputPort> = target
        .inputs()
        .iter()
        .filter(|port| port.source.node_id == producer)
        .collect();
    match ports.as_slice() {
        [port] => Ok((*port).clone()),
        _ => Err(format!(
            "{consumer:?} reads {} outputs of {producer:?}; an edit needs exactly one edge \
             between them",
            ports.len()
        )),
    }
}

fn insert(
    graph: &PhysicalGraph,
    mut node: PhysicalNode,
    (producer, consumer): &(String, String),
) -> Result<PhysicalGraph, String> {
    let port = edge(graph, producer, consumer)?;
    if graph
        .nodes
        .iter()
        .any(|existing| existing.id() == node.id())
    {
        return Err(format!("the plan already has a node {:?}", node.id()));
    }
    if !node.inputs().is_empty() {
        return Err(format!(
            "{:?} already has inputs; insert wires its one input itself",
            node.id()
        ));
    }
    let outputs = node.outputs();
    let output = match outputs.as_slice() {
        [(name, value_type)] if *value_type == port.value_type => name.clone(),
        _ => {
            return Err(format!(
                "{:?} needs exactly one output of type {} to sit on the edge from {producer:?} \
                 to {consumer:?}",
                node.id(),
                type_name(port.value_type)
            ));
        }
    };
    node.inputs_mut().push(InputPort {
        name: "input:0".into(),
        value_type: port.value_type,
        source: port.source.clone(),
        schema: port.schema.clone(),
    });
    let new_source = PortRef {
        node_id: node.id().to_string(),
        port: output,
    };
    let mut nodes = Vec::with_capacity(graph.nodes.len() + 1);
    let mut wired = Some(node);
    for existing in &graph.nodes {
        if existing.id() == consumer.as_str() {
            nodes.extend(wired.take());
            let mut rewired = existing.clone();
            for input in rewired.inputs_mut() {
                if input.name == port.name && input.source == port.source {
                    input.source = new_source.clone();
                }
            }
            nodes.push(rewired);
        } else {
            nodes.push(existing.clone());
        }
    }
    Ok(PhysicalGraph {
        nodes,
        root: graph.root.clone(),
    })
}

fn remove(graph: &PhysicalGraph, node_id: &str) -> Result<PhysicalGraph, String> {
    let node = find(graph, node_id)?;
    if !matches!(node, PhysicalNode::Barrier(_)) {
        return Err(format!(
            "{node_id:?} is a {}; removing it would change what the query means (only Barrier \
             nodes can be removed)",
            node.type_name()
        ));
    }
    let outputs = node.outputs();
    let (input, output) = match (node.inputs(), outputs.as_slice()) {
        ([input], [(output, value_type)]) if input.value_type == *value_type => (input, output),
        (inputs, outputs) => {
            return Err(format!(
                "{node_id:?} needs one input and one output of the same type to be removed; it \
                 has {} and {}",
                inputs.len(),
                outputs.len()
            ));
        }
    };
    let upstream = input.source.clone();
    let gone = PortRef {
        node_id: node_id.to_string(),
        port: output.clone(),
    };
    let nodes = graph
        .nodes
        .iter()
        .filter(|existing| existing.id() != node_id)
        .map(|existing| {
            let mut existing = existing.clone();
            for input in existing.inputs_mut() {
                if input.source == gone {
                    input.source = upstream.clone();
                }
            }
            existing
        })
        .collect();
    Ok(PhysicalGraph {
        nodes,
        root: graph.root.clone(),
    })
}
