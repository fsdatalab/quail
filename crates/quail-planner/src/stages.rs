//! Cut a region's graph into stages, one DataFusion partition per GPU.
//!
//! A stage is a run of nodes whose rows are spread over the GPUs by the
//! ids of one table, its key: partition p of the stage runs on GPU p
//! over the rows whose key id hashes to p. A stage's inputs are the
//! ports it reads from scans or other stages: a port carrying the key's
//! rows is hashed by the key, so each document is asked once and on the
//! GPU that holds its KV; every other port is broadcast whole to every
//! partition, after the stage that writes it has finished.
//!
//! A table's filters run in the stage of the join anchored on it; a
//! partner's filters are a stage of their own, so they run once, not
//! once per partition. A Barrier starts a stage, keyed by the next
//! anchor, that reads the join answers so far whole.

use std::collections::{BTreeMap, BTreeSet};

use quail_physical::{PhysicalGraph, PhysicalNode, PortRef, topological_nodes};
use serde::Serialize;

/// One port a stage reads from outside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StageInput {
    pub node_id: String,
    pub port: String,
    /// The stage that writes the port, or None for a scan's rows.
    pub stage: Option<usize>,
    /// Whether each partition reads the rows whose key id hashes to it;
    /// otherwise every partition reads the whole port.
    pub hashed: bool,
}

/// A run of nodes spread over the GPUs by the ids of one table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Stage {
    /// The alias whose ids pick each row's partition.
    pub key: String,
    /// The stage's node ids, in run order.
    pub nodes: Vec<String>,
    pub inputs: Vec<StageInput>,
    /// The ports later stages read, as (node id, port).
    pub outputs: Vec<(String, String)>,
}

/// What the cut reads of one node.
#[derive(Debug, Clone)]
pub struct CutNode {
    pub id: String,
    pub role: Role,
    pub inputs: Vec<PortRef>,
}

/// How a node takes part in the cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    /// A scan: its rows are an input, not part of a stage.
    Scan,
    /// A join that asks no model: it runs in the stage that reads it.
    HashJoin,
    /// A Barrier: it starts a stage keyed by the next anchor.
    Barrier(String),
    /// A node over one table's documents, or a join's anchor.
    Keyed(String),
    /// A node over rows of several tables: it runs with its input.
    Follows(String),
}

impl CutNode {
    /// What the cut reads of a physical node.
    pub fn of(node: &PhysicalNode) -> Self {
        let role = match node {
            PhysicalNode::Scan(_) => Role::Scan,
            PhysicalNode::HashJoin(_) => Role::HashJoin,
            PhysicalNode::Barrier(barrier) => Role::Barrier(barrier.attributes.next_anchor.clone()),
            PhysicalNode::AiJoin(join) => Role::Keyed(join.attributes.anchor.clone()),
            _ => {
                let mut aliases = node.document_aliases();
                if aliases.is_empty() {
                    aliases = node.spec_aliases();
                }
                match aliases.as_slice() {
                    [alias] => Role::Keyed(alias.clone()),
                    _ => Role::Follows(aliases.first().cloned().unwrap_or_default()),
                }
            }
        };
        CutNode {
            id: node.id().to_string(),
            role,
            inputs: node
                .inputs()
                .iter()
                .map(|input| input.source.clone())
                .collect(),
        }
    }
}

/// Cut a graph's compute nodes into stages.
///
/// # Errors
///
/// A node reads a port no scan or stage writes.
pub fn stages(graph: &PhysicalGraph) -> Result<Vec<Stage>, String> {
    let nodes: Vec<CutNode> = topological_nodes(&graph.nodes)
        .into_iter()
        .map(|index| CutNode::of(&graph.nodes[index]))
        .collect();
    cut(&nodes)
}

/// Cut nodes, in topological order, into stages.
///
/// # Errors
///
/// A node reads a port no scan or stage writes.
pub fn cut(nodes: &[CutNode]) -> Result<Vec<Stage>, String> {
    let mut stages: Vec<Stage> = Vec::new();
    // stage -> the stages it reads
    let mut reads: Vec<BTreeSet<usize>> = Vec::new();
    let mut stage_of: BTreeMap<&str, usize> = BTreeMap::new();
    let rows = |key: &str| format!("rows:{key}");
    for node in nodes {
        let producers: Vec<usize> = node
            .inputs
            .iter()
            .filter_map(|input| stage_of.get(input.node_id.as_str()).copied())
            .collect();
        let (key, joined) = match &node.role {
            Role::Scan | Role::HashJoin => continue,
            Role::Barrier(key) => (key.clone(), None),
            Role::Keyed(key) => (
                key.clone(),
                node.inputs
                    .iter()
                    .filter(|input| input.port == rows(key))
                    .find_map(|input| stage_of.get(input.node_id.as_str()).copied())
                    .filter(|stage| stages[*stage].key == *key),
            ),
            Role::Follows(alias) => match producers.first() {
                Some(stage) => (stages[*stage].key.clone(), Some(*stage)),
                None => (alias.clone(), None),
            },
        };
        // a node joins its producer's stage unless that stage would then
        // read a stage that reads it
        let joined = joined.filter(|stage| {
            producers
                .iter()
                .all(|producer| producer == stage || !reaches(&reads, *producer, *stage))
        });
        let stage = joined.unwrap_or_else(|| {
            stages.push(Stage {
                key,
                nodes: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            });
            reads.push(BTreeSet::new());
            stages.len() - 1
        });
        reads[stage].extend(producers.into_iter().filter(|producer| *producer != stage));
        stages[stage].nodes.push(node.id.clone());
        stage_of.insert(&node.id, stage);
    }
    // a join that asks no model runs in the stage of its first reader
    for node in nodes.iter().filter(|node| node.role == Role::HashJoin) {
        let reader = nodes
            .iter()
            .find(|reader| reader.inputs.iter().any(|input| input.node_id == node.id));
        if let Some(stage) = reader.and_then(|reader| stage_of.get(reader.id.as_str()).copied()) {
            stage_of.insert(&node.id, stage);
            stages[stage].nodes.push(node.id.clone());
        }
    }
    let position: BTreeMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id.as_str(), index))
        .collect();
    let scans: BTreeSet<&str> = nodes
        .iter()
        .filter(|node| node.role == Role::Scan)
        .map(|node| node.id.as_str())
        .collect();
    for index in 0..stages.len() {
        stages[index].nodes.sort_by_key(|id| position[id.as_str()]);
        let mut inputs: Vec<StageInput> = Vec::new();
        for id in stages[index].nodes.clone() {
            let node = &nodes[position[id.as_str()]];
            for source in &node.inputs {
                let from = stage_of.get(source.node_id.as_str()).copied();
                if from == Some(index) {
                    continue;
                }
                if from.is_none() && !scans.contains(source.node_id.as_str()) {
                    return Err(format!(
                        "{:?} reads {:?}.{:?}, which no scan or stage writes",
                        id, source.node_id, source.port
                    ));
                }
                let input = StageInput {
                    node_id: source.node_id.clone(),
                    port: source.port.clone(),
                    stage: from,
                    hashed: source.port == rows(&stages[index].key),
                };
                if !inputs.contains(&input) {
                    inputs.push(input);
                }
            }
        }
        for input in &inputs {
            if let Some(from) = input.stage {
                let output = (input.node_id.clone(), input.port.clone());
                if !stages[from].outputs.contains(&output) {
                    stages[from].outputs.push(output);
                }
            }
        }
        stages[index].inputs = inputs;
    }
    Ok(stages)
}

/// Whether stage `from` reads stage `to`, directly or through others.
fn reaches(reads: &[BTreeSet<usize>], from: usize, to: usize) -> bool {
    let mut pending = vec![from];
    let mut seen = BTreeSet::new();
    while let Some(stage) = pending.pop() {
        if stage == to {
            return true;
        }
        if seen.insert(stage) {
            pending.extend(reads[stage].iter().copied());
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, role: Role, inputs: &[(&str, &str)]) -> CutNode {
        CutNode {
            id: id.into(),
            role,
            inputs: inputs
                .iter()
                .map(|(node_id, port)| quail_physical::port(node_id, *port))
                .collect(),
        }
    }

    fn scan(alias: &str) -> CutNode {
        node(&format!("scan:{alias}"), Role::Scan, &[])
    }

    fn filter(alias: &str, source: &str) -> CutNode {
        node(
            &format!("ai_filter:{alias}"),
            Role::Keyed(alias.into()),
            &[(source, &format!("rows:{alias}"))],
        )
    }

    /// A stage as its key, its nodes, and its inputs as (node, hashed).
    type Shape<'a> = (&'a str, Vec<&'a str>, Vec<(&'a str, bool)>);

    fn shape(stages: &[Stage]) -> Vec<Shape<'_>> {
        stages
            .iter()
            .map(|stage| {
                (
                    stage.key.as_str(),
                    stage.nodes.iter().map(String::as_str).collect(),
                    stage
                        .inputs
                        .iter()
                        .map(|input| (input.node_id.as_str(), input.hashed))
                        .collect(),
                )
            })
            .collect()
    }

    #[test]
    fn a_partner_filter_is_a_stage_of_its_own_and_the_anchor_filter_runs_with_its_join() {
        // IMDB-4: two filters on reviews, one on aspects, then a join
        // anchored on reviews
        let nodes = vec![
            scan("r"),
            scan("a"),
            filter("r", "scan:r"),
            filter("a", "scan:a"),
            node(
                "ai_join:r",
                Role::Keyed("r".into()),
                &[("ai_filter:r", "rows:r"), ("ai_filter:a", "rows:a")],
            ),
        ];
        let stages = cut(&nodes).expect("stages");
        assert_eq!(
            shape(&stages),
            vec![
                (
                    "r",
                    vec!["ai_filter:r", "ai_join:r"],
                    vec![("scan:r", true), ("ai_filter:a", false)]
                ),
                ("a", vec!["ai_filter:a"], vec![("scan:a", true)]),
            ]
        );
        assert_eq!(
            stages[1].outputs,
            vec![("ai_filter:a".to_string(), "rows:a".to_string())]
        );
    }

    #[test]
    fn a_join_without_filters_hashes_its_anchors_and_broadcasts_its_partners() {
        // IMDB-2: reviews joined with aspects
        let nodes = vec![
            scan("r"),
            scan("a"),
            node(
                "ai_join:r",
                Role::Keyed("r".into()),
                &[("scan:r", "rows:r"), ("scan:a", "rows:a")],
            ),
        ];
        assert_eq!(
            shape(&cut(&nodes).expect("stages")),
            vec![(
                "r",
                vec!["ai_join:r"],
                vec![("scan:r", true), ("scan:a", false)]
            )]
        );
    }

    #[test]
    fn a_barrier_starts_a_stage_keyed_by_the_next_anchor() {
        // FEV-9: e1 joins c1 and c2, then c2 joins e2 after a barrier
        let nodes = vec![
            scan("c1"),
            scan("e1"),
            scan("c2"),
            scan("e2"),
            filter("c1", "scan:c1"),
            filter("c2", "scan:c2"),
            filter("e1", "scan:e1"),
            filter("e2", "scan:e2"),
            node(
                "ai_join:e1",
                Role::Keyed("e1".into()),
                &[("ai_filter:e1", "rows:e1"), ("ai_filter:c1", "rows:c1")],
            ),
            node(
                "ai_join:e1:predicate:1",
                Role::Keyed("e1".into()),
                &[("ai_join:e1", "rows:e1"), ("ai_filter:c2", "rows:c2")],
            ),
            node(
                "barrier:c2",
                Role::Barrier("c2".into()),
                &[
                    ("ai_join:e1", "join_answers:0"),
                    ("ai_join:e1:predicate:1", "join_answers:1"),
                    ("ai_filter:c1", "rows:c1"),
                    ("ai_join:e1:predicate:1", "rows:e1"),
                    ("ai_filter:c2", "rows:c2"),
                    ("ai_filter:e2", "rows:e2"),
                ],
            ),
            node(
                "ai_join:c2",
                Role::Keyed("c2".into()),
                &[("barrier:c2", "rows:c2"), ("barrier:c2", "rows:e2")],
            ),
        ];
        let stages = cut(&nodes).expect("stages");
        assert_eq!(
            shape(&stages),
            vec![
                ("c1", vec!["ai_filter:c1"], vec![("scan:c1", true)]),
                ("c2", vec!["ai_filter:c2"], vec![("scan:c2", true)]),
                (
                    "e1",
                    vec!["ai_filter:e1", "ai_join:e1", "ai_join:e1:predicate:1"],
                    vec![
                        ("scan:e1", true),
                        ("ai_filter:c1", false),
                        ("ai_filter:c2", false)
                    ]
                ),
                ("e2", vec!["ai_filter:e2"], vec![("scan:e2", true)]),
                (
                    "c2",
                    vec!["barrier:c2", "ai_join:c2"],
                    vec![
                        ("ai_join:e1", false),
                        ("ai_join:e1:predicate:1", false),
                        ("ai_filter:c1", false),
                        ("ai_join:e1:predicate:1", false),
                        ("ai_filter:c2", true),
                        ("ai_filter:e2", false)
                    ]
                ),
            ]
        );
    }

    #[test]
    fn a_node_does_not_join_a_stage_that_reads_its_other_input() {
        // y's filter reads x's rows; x's join reads y's: the join cannot
        // run in x's filter stage
        let nodes = vec![
            scan("x"),
            filter("x", "scan:x"),
            node(
                "ai_filter:y",
                Role::Keyed("y".into()),
                &[("ai_filter:x", "rows:x")],
            ),
            node(
                "ai_join:x",
                Role::Keyed("x".into()),
                &[("ai_filter:x", "rows:x"), ("ai_filter:y", "rows:y")],
            ),
        ];
        let stages = cut(&nodes).expect("stages");
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[2].nodes, vec!["ai_join:x"]);
        assert!(stages[2].inputs.iter().any(|input| input.hashed));
    }
}
