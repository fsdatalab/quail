//! What each node offers a pipeline, and the pipelines of a graph, as
//! `quail/physical/base.py` and `quail/physical/pipelines.py` define them.

use std::collections::{BTreeMap, BTreeSet};

use crate::expressions::ValueExpression;
use crate::graph::{AnswerOutput, InputPort, OutputPort, PortRef, ValueType};
use crate::nodes::*;
use crate::specs::ExtractSpec;

/// The outputs of a request backend's node: each table's survivors, then
/// each operator's answers.
fn request_outputs(attributes: &RequestExecutionAttributes) -> Vec<OutputPort> {
    let port =
        |name: String, value_type: ValueType, schema: Vec<String>, kind: Option<&str>| OutputPort {
            answer: kind.map(|kind| AnswerOutput {
                kind: kind.into(),
                name: schema.last().cloned().unwrap_or_default(),
                columns: schema.clone(),
            }),
            name,
            value_type,
            schema,
        };
    let answers = |alias: &str| vec![alias.to_string(), "predicate".into(), "answer".into()];
    let mut outputs: Vec<OutputPort> = attributes
        .aliases
        .iter()
        .map(|alias| {
            port(
                format!("rows:{alias}"),
                ValueType::Rows,
                vec![alias.clone()],
                None,
            )
        })
        .collect();
    for spec in &attributes.filters {
        outputs.push(port(
            format!("filter_answers:{}", spec.alias),
            ValueType::FilterAnswers,
            answers(&spec.alias),
            None,
        ));
    }
    for spec in &attributes.joins {
        let mut schema = spec.aliases.clone();
        schema.push("answer".into());
        outputs.push(port(
            format!("join_answers:{}", spec.written_position),
            ValueType::JoinAnswers,
            schema,
            None,
        ));
    }
    for spec in &attributes.classifies {
        let mut schema = vec![spec.alias.clone()];
        schema.extend(spec.partner.iter().cloned());
        schema.push(spec.output.clone());
        outputs.push(port(
            format!("label_answers:{}", spec.output),
            ValueType::LabelAnswers,
            schema,
            Some("classifies"),
        ));
    }
    for spec in attributes
        .classifies
        .iter()
        .filter(|spec| !spec.tests.is_empty())
    {
        outputs.push(port(
            format!("label_in_answers:{}", spec.output),
            ValueType::FilterAnswers,
            answers(&spec.alias),
            None,
        ));
    }
    for spec in &attributes.extracts {
        let mut answer = port(
            format!("extract_answers:{}", spec.output),
            ValueType::ExtractAnswers,
            vec![
                spec.alias.clone(),
                spec.output.clone(),
                format!("{}_span", spec.output),
            ],
            Some("extracts"),
        );
        if let Some(found) = &mut answer.answer {
            found.name = spec.output.clone();
        }
        outputs.push(answer);
    }
    for spec in &attributes.maps {
        outputs.push(port(
            format!("map_answers:{}", spec.output),
            ValueType::Rows,
            vec![spec.alias.clone(), spec.output.clone()],
            Some("maps"),
        ));
    }
    outputs
}

/// The value type a port name carries.
pub fn value_type_for_port(name: &str) -> ValueType {
    if name == "rows" || name.starts_with("rows:") {
        ValueType::Rows
    } else if name.starts_with("filter_answers:") {
        ValueType::FilterAnswers
    } else if name.starts_with("join_answers:") {
        ValueType::JoinAnswers
    } else if name.starts_with("pairs:") {
        ValueType::Pairs
    } else {
        ValueType::Any
    }
}

/// Typed input ports reading the given outputs, named `input:<index>`.
pub fn input_ports(values: &[PortRef]) -> Vec<InputPort> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| InputPort {
            name: format!("input:{index}"),
            value_type: value_type_for_port(&value.port),
            source: value.clone(),
            schema: Vec::new(),
        })
        .collect()
}

/// A port of a node.
pub fn port(node_id: &str, port: impl Into<String>) -> PortRef {
    PortRef {
        node_id: node_id.to_string(),
        port: port.into(),
    }
}

/// Whether two extractions can use the same decoder.
pub fn same_decoding(one: &ExtractSpec, other: &ExtractSpec) -> bool {
    one.max_answer_tokens == other.max_answer_tokens
        && one.draft_tokens == other.draft_tokens
        && one.draft_policy == other.draft_policy
        && one.expected_answer_tokens == other.expected_answer_tokens
        && one.round_cost_per_token == other.round_cost_per_token
        && one.top_tokens == other.top_tokens
        && one.first_window_chars == other.first_window_chars
        && one.window_chars == other.window_chars
}

fn rows_port_of(aliases: &[String]) -> String {
    if aliases.len() == 1 {
        format!("rows:{}", aliases[0])
    } else {
        "rows".to_string()
    }
}

impl PhysicalNode {
    /// The aliases of a model node's spec; empty without a spec.
    pub fn spec_aliases(&self) -> Vec<String> {
        match self {
            PhysicalNode::AiScore(node) => node.attributes.spec.as_ref().map(|s| s.aliases.clone()),
            PhysicalNode::AiClassify(node) => {
                node.attributes.spec.as_ref().map(|s| s.aliases.clone())
            }
            PhysicalNode::AiExtract(node) => {
                node.attributes.spec.as_ref().map(|s| s.aliases.clone())
            }
            PhysicalNode::AiMap(node) => node.attributes.spec.as_ref().map(|s| s.aliases.clone()),
            PhysicalNode::AiSaeFeatures(node) => {
                node.attributes.spec.as_ref().map(|s| s.aliases.clone())
            }
            _ => None,
        }
        .unwrap_or_default()
    }

    /// A model node's prompt token parts; empty without a spec.
    pub fn prompt_token_parts(&self) -> Vec<Vec<i64>> {
        match self {
            PhysicalNode::AiScore(node) => node
                .attributes
                .spec
                .as_ref()
                .map(|s| s.prompt_token_parts.clone()),
            PhysicalNode::AiClassify(node) => node
                .attributes
                .spec
                .as_ref()
                .map(|s| s.prompt_token_parts.clone()),
            PhysicalNode::AiExtract(node) => node
                .attributes
                .spec
                .as_ref()
                .map(|s| s.prompt_token_parts.clone()),
            PhysicalNode::AiMap(node) => node
                .attributes
                .spec
                .as_ref()
                .map(|s| s.prompt_token_parts.clone()),
            PhysicalNode::AiSaeFeatures(node) => node
                .attributes
                .spec
                .as_ref()
                .map(|s| s.prompt_token_parts.clone()),
            _ => None,
        }
        .unwrap_or_default()
    }

    /// Whether the node is a model call carrying a spec.
    fn is_inference(&self) -> bool {
        matches!(
            self,
            PhysicalNode::AiScore(_)
                | PhysicalNode::AiClassify(_)
                | PhysicalNode::AiExtract(_)
                | PhysicalNode::AiMap(_)
                | PhysicalNode::AiSaeFeatures(_)
        )
    }

    fn has_spec(&self) -> bool {
        match self {
            PhysicalNode::AiScore(node) => node.attributes.spec.is_some(),
            PhysicalNode::AiClassify(node) => node.attributes.spec.is_some(),
            PhysicalNode::AiExtract(node) => node.attributes.spec.is_some(),
            PhysicalNode::AiMap(node) => node.attributes.spec.is_some(),
            PhysicalNode::AiSaeFeatures(node) => node.attributes.spec.is_some(),
            _ => false,
        }
    }

    /// The port carrying the rows a model node or a condition keeps.
    pub fn rows_port(&self) -> String {
        match self {
            PhysicalNode::Filter(node) => rows_port_of(&node.attributes.aliases),
            _ => rows_port_of(&self.spec_aliases()),
        }
    }

    /// The kind of answer a model node's rows output reports.
    fn answer_kind(&self) -> &'static str {
        match self {
            PhysicalNode::AiClassify(_) => "classifies",
            PhysicalNode::AiExtract(_) => "extracts",
            PhysicalNode::AiMap(_) => "maps",
            PhysicalNode::AiSaeFeatures(_) => "features",
            _ => "scores",
        }
    }

    /// A model node's spec name; empty without a spec.
    pub fn spec_name(&self) -> String {
        match self {
            PhysicalNode::AiScore(node) => node.attributes.spec.as_ref().map(|s| s.name.clone()),
            PhysicalNode::AiClassify(node) => node.attributes.spec.as_ref().map(|s| s.name.clone()),
            PhysicalNode::AiExtract(node) => node.attributes.spec.as_ref().map(|s| s.name.clone()),
            PhysicalNode::AiMap(node) => node.attributes.spec.as_ref().map(|s| s.name.clone()),
            PhysicalNode::AiSaeFeatures(node) => {
                node.attributes.spec.as_ref().map(|s| s.name.clone())
            }
            _ => None,
        }
        .unwrap_or_default()
    }

    /// The node's outputs: each port's name, value type, columns, and
    /// model answer.
    pub fn output_ports(&self) -> Vec<OutputPort> {
        let output = |name: String, value_type: ValueType, schema: Vec<String>| OutputPort {
            name,
            value_type,
            schema,
            answer: None,
        };
        let rows = |name: String, schema: Vec<String>| output(name, ValueType::Rows, schema);
        let of = |alias: &str| vec![alias.to_string()];
        let answers = |alias: &str| vec![alias.to_string(), "predicate".into(), "answer".into()];
        match self {
            PhysicalNode::Scan(node) => {
                let alias = &node.attributes.alias;
                vec![rows(format!("rows:{alias}"), of(alias))]
            }
            PhysicalNode::AiFilter(node) => {
                let alias = &node.attributes.alias;
                vec![
                    rows(format!("rows:{alias}"), of(alias)),
                    output(
                        format!("filter_answers:{alias}"),
                        ValueType::FilterAnswers,
                        answers(alias),
                    ),
                ]
            }
            PhysicalNode::Filter(node) => {
                let attributes = &node.attributes;
                match attributes.aliases.as_slice() {
                    [alias] => vec![
                        rows(format!("rows:{alias}"), Vec::new()),
                        output(
                            format!("filter_answers:{alias}"),
                            ValueType::FilterAnswers,
                            Vec::new(),
                        ),
                    ],
                    _ => vec![
                        rows("rows".into(), Vec::new()),
                        output(
                            format!("join_answers:{}", attributes.written_position),
                            ValueType::JoinAnswers,
                            Vec::new(),
                        ),
                    ],
                }
            }
            PhysicalNode::Barrier(node) => node
                .attributes
                .aliases
                .iter()
                .map(|alias| rows(format!("rows:{alias}"), of(alias)))
                .collect(),
            PhysicalNode::Exchange(node) => {
                let anchor = &node.attributes.anchor;
                vec![rows(format!("rows:{anchor}"), of(anchor))]
            }
            PhysicalNode::HashJoin(node) => {
                let attributes = &node.attributes;
                vec![output(
                    format!("pairs:{}", attributes.written_position),
                    ValueType::Pairs,
                    vec![attributes.left.clone(), attributes.right.clone()],
                )]
            }
            PhysicalNode::AiJoin(node) => {
                let attributes = &node.attributes;
                let mut schema = of(&attributes.anchor);
                schema.extend(attributes.partners.iter().cloned());
                schema.push("answer".into());
                vec![
                    rows(
                        format!("rows:{}", attributes.anchor),
                        of(&attributes.anchor),
                    ),
                    output(
                        format!("join_answers:{}", attributes.written_position),
                        ValueType::JoinAnswers,
                        schema,
                    ),
                ]
            }
            PhysicalNode::Limit(_) => vec![rows("rows".into(), Vec::new())],
            PhysicalNode::RequestExecution(node) => request_outputs(&node.attributes),
            PhysicalNode::Extension(node) => node.outputs.clone(),
            _ if self.has_spec() => {
                let name = self.spec_name();
                let mut columns = self.spec_aliases();
                columns.push(name.clone());
                if matches!(self, PhysicalNode::AiExtract(_)) {
                    columns.push(format!("{name}_span"));
                }
                vec![OutputPort {
                    name: self.rows_port(),
                    value_type: ValueType::Rows,
                    schema: columns.clone(),
                    answer: Some(AnswerOutput {
                        kind: self.answer_kind().into(),
                        name,
                        columns,
                    }),
                }]
            }
            _ => Vec::new(),
        }
    }

    /// The node's outputs: each port's name and value type.
    pub fn outputs(&self) -> Vec<(String, ValueType)> {
        self.output_ports()
            .into_iter()
            .map(|port| (port.name, port.value_type))
            .collect()
    }

    /// Where the node runs: a model node, or a condition on documents, on
    /// the GPU executor; every other node on the coordinator.
    pub fn location(&self) -> Location {
        match self {
            PhysicalNode::Filter(_)
            | PhysicalNode::AiFilter(_)
            | PhysicalNode::AiJoin(_)
            | PhysicalNode::HashJoin(_)
            | PhysicalNode::RequestExecution(_) => Location::GpuExecutor,
            PhysicalNode::Extension(node) => node.location,
            _ if self.is_inference() => Location::GpuExecutor,
            _ => Location::Coordinator,
        }
    }

    /// The aliases whose documents the node processes one at a time.
    pub fn document_aliases(&self) -> Vec<String> {
        match self {
            PhysicalNode::AiFilter(node) => vec![node.attributes.alias.clone()],
            PhysicalNode::AiJoin(node) => vec![node.attributes.anchor.clone()],
            PhysicalNode::Filter(node) if node.attributes.aliases.len() == 1 => {
                node.attributes.aliases.clone()
            }
            PhysicalNode::AiScore(node) => match &node.attributes.spec {
                Some(spec) if spec.state_tail_tokens != 0 => vec![spec.aliases[0].clone()],
                _ => Vec::new(),
            },
            _ if self.is_inference() => self.spec_aliases().into_iter().take(1).collect(),
            _ => Vec::new(),
        }
    }

    /// The token ids the node shows each document under.
    pub fn document_head(&self) -> Vec<i64> {
        match self {
            PhysicalNode::AiFilter(node) => node.attributes.document_head.clone(),
            PhysicalNode::AiJoin(node) => node.attributes.document_head.clone(),
            _ => self
                .prompt_token_parts()
                .into_iter()
                .next()
                .unwrap_or_default(),
        }
    }

    /// The outputs that can feed the next operator for a document.
    pub fn document_outputs(&self, alias: &str) -> Vec<String> {
        match self {
            PhysicalNode::AiJoin(node) => vec![
                format!("join_answers:{}", node.attributes.written_position),
                format!("rows:{alias}"),
            ],
            _ => vec![format!("rows:{alias}")],
        }
    }

    /// Whether this classification reads a join's surviving pairs.
    pub fn classifies_join(&self, join: &PhysicalNode) -> bool {
        let (PhysicalNode::AiClassify(node), PhysicalNode::AiJoin(join)) = (self, join) else {
            return false;
        };
        let Some(spec) = &node.attributes.spec else {
            return false;
        };
        if spec.aliases.len() != 2 || spec.aliases[0] != join.attributes.anchor {
            return false;
        }
        let mut mine = spec.aliases.clone();
        mine.sort();
        let mut theirs = vec![join.attributes.anchor.clone()];
        theirs.extend(join.attributes.partners.iter().cloned());
        theirs.sort();
        mine == theirs
    }

    /// Whether this node can use the KV the node before it left.
    pub fn can_follow(&self, previous: &PhysicalNode, first: &PhysicalNode) -> bool {
        // the executor asks a guarded node about its guard's rows alone
        if self.guard().is_some() {
            return false;
        }
        match self {
            PhysicalNode::Filter(_) => true,
            PhysicalNode::AiClassify(node) => {
                if matches!(previous, PhysicalNode::AiJoin(_)) {
                    return self.classifies_join(previous);
                }
                node.attributes
                    .spec
                    .as_ref()
                    .is_some_and(|spec| spec.aliases.len() != 2)
                    && self.document_head() == first.document_head()
            }
            PhysicalNode::AiExtract(node) => {
                if let (PhysicalNode::AiExtract(earlier), Some(spec)) =
                    (previous, &node.attributes.spec)
                    && let Some(earlier_spec) = &earlier.attributes.spec
                    && !same_decoding(earlier_spec, spec)
                {
                    return false;
                }
                first.document_head() == self.document_head()
            }
            _ => first.document_head() == self.document_head(),
        }
    }

    /// Whether a consumer can run before other documents finish.
    pub fn can_precede(&self, consumer: &PhysicalNode) -> bool {
        if self.guard().is_some() {
            return false;
        }
        match self {
            PhysicalNode::AiJoin(node) => match consumer {
                PhysicalNode::AiJoin(next) => {
                    node.attributes.anchor == next.attributes.anchor
                        && node.attributes.semantics == "full"
                        && next.attributes.semantics == "full"
                }
                PhysicalNode::AiClassify(_) => consumer.classifies_join(self),
                _ => false,
            },
            _ => true,
        }
    }

    /// A column node's guard: the condition that says which rows the
    /// model is asked about. None for every other node.
    pub fn guard(&self) -> Option<&ValueExpression> {
        match self {
            PhysicalNode::AiClassify(node) => node.attributes.guard.as_ref(),
            PhysicalNode::AiExtract(node) => node.attributes.guard.as_ref(),
            PhysicalNode::AiMap(node) => node.attributes.guard.as_ref(),
            PhysicalNode::AiSaeFeatures(node) => node.attributes.guard.as_ref(),
            _ => None,
        }
    }

    /// The backend of a model node; None for every other node.
    pub fn backend(&self) -> Option<&str> {
        match self {
            PhysicalNode::RequestExecution(node) => Some(&node.attributes.backend_name),
            PhysicalNode::Extension(node) => node.backend.as_deref(),
            PhysicalNode::AiFilter(_) | PhysicalNode::AiJoin(_) => Some("quail"),
            PhysicalNode::AiScore(node) => Some(&node.attributes.backend_name),
            PhysicalNode::AiClassify(node) => Some(&node.attributes.backend_name),
            PhysicalNode::AiExtract(node) => Some(&node.attributes.backend_name),
            PhysicalNode::AiMap(node) => Some(&node.attributes.backend_name),
            PhysicalNode::AiSaeFeatures(node) => Some(&node.attributes.backend_name),
            _ => None,
        }
    }

    /// Whether the node keeps its documents' KV after it finishes.
    pub fn keep_kv(&self) -> bool {
        match self {
            PhysicalNode::AiFilter(node) => node.attributes.keep_kv,
            PhysicalNode::AiJoin(node) => node.attributes.keep_kv,
            PhysicalNode::AiScore(node) => node.attributes.keep_kv,
            PhysicalNode::AiClassify(node) => node.attributes.keep_kv,
            PhysicalNode::AiExtract(node) => node.attributes.keep_kv,
            PhysicalNode::AiMap(node) => node.attributes.keep_kv,
            PhysicalNode::AiSaeFeatures(node) => node.attributes.keep_kv,
            _ => false,
        }
    }

    /// The node's set-attention pass group; -1 for none.
    pub fn pass_group(&self) -> i64 {
        match self {
            PhysicalNode::AiFilter(node) => node.attributes.pass_group,
            PhysicalNode::AiScore(node) => node.attributes.pass_group,
            PhysicalNode::AiClassify(node) => node.attributes.pass_group,
            PhysicalNode::AiExtract(node) => node.attributes.pass_group,
            PhysicalNode::AiMap(node) => node.attributes.pass_group,
            PhysicalNode::AiSaeFeatures(node) => node.attributes.pass_group,
            _ => -1,
        }
    }
}

/// The graph's node positions, each after the nodes it reads.
pub fn topological_nodes(nodes: &[PhysicalNode]) -> Vec<usize> {
    let by_id: BTreeMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id(), index))
        .collect();
    let mut ordered = Vec::with_capacity(nodes.len());
    let mut visited = vec![false; nodes.len()];
    fn visit(
        index: usize,
        nodes: &[PhysicalNode],
        by_id: &BTreeMap<&str, usize>,
        visited: &mut Vec<bool>,
        ordered: &mut Vec<usize>,
    ) {
        if visited[index] {
            return;
        }
        for input in nodes[index].inputs() {
            visit(
                by_id[input.source.node_id.as_str()],
                nodes,
                by_id,
                visited,
                ordered,
            );
        }
        visited[index] = true;
        ordered.push(index);
    }
    for index in 0..nodes.len() {
        visit(index, nodes, &by_id, &mut visited, &mut ordered);
    }
    ordered
}

/// The pipelines of a graph: each pipeline's node positions in order,
/// and node id -> the pipeline holding it.
#[derive(Debug, Clone, Default)]
pub struct Pipelines {
    pub members: Vec<Vec<usize>>,
    pub of: BTreeMap<String, usize>,
}

impl Pipelines {
    /// The positions of the pipeline holding a node, if any.
    pub fn get(&self, node_id: &str) -> Option<&[usize]> {
        self.of
            .get(node_id)
            .map(|index| self.members[*index].as_slice())
    }
}

/// Whether the pipelines and single nodes can run in some order: no
/// two groups read each other's outputs.
pub fn groups_acyclic(nodes: &[PhysicalNode], pipeline_of: &BTreeMap<String, Vec<usize>>) -> bool {
    let mut owners: BTreeMap<&str, &str> = BTreeMap::new();
    for node in nodes {
        let owner = pipeline_of
            .get(node.id())
            .map(|group| nodes[group[0]].id())
            .unwrap_or(node.id());
        owners.insert(node.id(), owner);
    }
    let mut dependencies: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for node in nodes {
        let owner = owners[node.id()];
        let entry = dependencies.entry(owner).or_default();
        for input in node.inputs() {
            let source = owners[input.source.node_id.as_str()];
            if source != owner {
                entry.insert(source);
            }
        }
    }
    // Kahn's algorithm: every group is placed exactly when there is no cycle
    let mut remaining: BTreeMap<&str, usize> = dependencies
        .iter()
        .map(|(owner, needs)| (*owner, needs.len()))
        .collect();
    let mut ready: Vec<&str> = remaining
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(owner, _)| *owner)
        .collect();
    let mut placed = 0;
    while let Some(owner) = ready.pop() {
        placed += 1;
        for (other, needs) in &dependencies {
            if needs.contains(owner) {
                let count = remaining.get_mut(other).expect("a group");
                *count -= 1;
                if *count == 0 {
                    ready.push(other);
                }
            }
        }
    }
    placed == dependencies.len()
}

/// Group compatible per-document operators into pipelines, as
/// `build_pipelines`.
pub fn build_pipelines(nodes: &[PhysicalNode], completed: &[String]) -> Pipelines {
    let mut consumers: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        for input in node.inputs() {
            consumers
                .entry((input.source.node_id.as_str(), input.source.port.as_str()))
                .or_default()
                .push(index);
        }
    }
    let topological = topological_nodes(nodes);
    let mut order = vec![0; nodes.len()];
    for (rank, index) in topological.iter().enumerate() {
        order[*index] = rank;
    }
    let reads = |consumer: usize, producer: usize| {
        nodes[consumer]
            .inputs()
            .iter()
            .any(|input| input.source.node_id == nodes[producer].id())
    };
    let continuation = |node: usize, alias: &str, first: usize| -> Vec<usize> {
        let mut following: Vec<usize> = Vec::new();
        for output in nodes[node].document_outputs(alias) {
            for consumer in consumers
                .get(&(nodes[node].id(), output.as_str()))
                .map(Vec::as_slice)
                .unwrap_or(&[])
            {
                if completed.iter().any(|done| done == nodes[*consumer].id()) {
                    continue;
                }
                if !nodes[node].can_precede(&nodes[*consumer])
                    || !nodes[*consumer].can_follow(&nodes[node], &nodes[first])
                {
                    continue;
                }
                if nodes[*consumer]
                    .document_aliases()
                    .iter()
                    .any(|a| a == alias)
                    && !following.contains(consumer)
                {
                    following.push(*consumer);
                }
            }
        }
        if matches!(nodes[node], PhysicalNode::AiJoin(_)) {
            let next_joins: Vec<usize> = following
                .iter()
                .copied()
                .filter(|consumer| matches!(nodes[*consumer], PhysicalNode::AiJoin(_)))
                .collect();
            if !next_joins.is_empty() {
                following = next_joins;
            }
        }
        following.sort_by_key(|consumer| order[*consumer]);
        for pair in following.windows(2) {
            if !reads(pair[1], pair[0]) {
                return Vec::new();
            }
        }
        following
    };
    let mut pipeline_of: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let can_group = |pipeline_of: &BTreeMap<String, Vec<usize>>, members: &[usize]| {
        let mut proposed = pipeline_of.clone();
        for member in members {
            proposed.insert(nodes[*member].id().to_string(), members.to_vec());
        }
        groups_acyclic(nodes, &proposed)
    };
    let mut followed: BTreeSet<String> = completed.iter().cloned().collect();
    let mut result = Pipelines::default();
    for &node in &topological {
        let aliases = nodes[node].document_aliases();
        if aliases.is_empty() || followed.contains(nodes[node].id()) {
            continue;
        }
        if nodes[node].backend().is_none() {
            continue;
        }
        let alias = aliases[0].clone();
        let mut members = vec![node];
        followed.insert(nodes[node].id().to_string());
        loop {
            let last = *members.last().expect("a member");
            let after: Vec<usize> = continuation(last, &alias, members[0])
                .into_iter()
                .filter(|consumer| !followed.contains(nodes[*consumer].id()))
                .collect();
            if after.is_empty() {
                break;
            }
            let mut proposed = members.clone();
            proposed.extend(after.iter().copied());
            if !can_group(&pipeline_of, &proposed) {
                break;
            }
            members = proposed;
            for consumer in after {
                followed.insert(nodes[consumer].id().to_string());
            }
        }
        let joins: Vec<usize> = members
            .iter()
            .copied()
            .filter(|member| matches!(nodes[*member], PhysicalNode::AiJoin(_)))
            .collect();
        for join in joins {
            let PhysicalNode::AiJoin(join_node) = &nodes[join] else {
                unreachable!()
            };
            let answers = format!("join_answers:{}", join_node.attributes.written_position);
            let candidates: Vec<usize> = consumers
                .get(&(nodes[join].id(), answers.as_str()))
                .map(Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .copied()
                .filter(|consumer| {
                    matches!(nodes[*consumer], PhysicalNode::AiClassify(_))
                        && nodes[*consumer].classifies_join(&nodes[join])
                })
                .collect();
            if candidates.len() == 1 && !followed.contains(nodes[candidates[0]].id()) {
                let mut proposed = members.clone();
                proposed.push(candidates[0]);
                if can_group(&pipeline_of, &proposed) {
                    members = proposed;
                    followed.insert(nodes[candidates[0]].id().to_string());
                }
            }
        }
        let position = result.members.len();
        for member in &members {
            pipeline_of.insert(nodes[*member].id().to_string(), members.clone());
            result.of.insert(nodes[*member].id().to_string(), position);
        }
        result.members.push(members);
    }
    result
}

/// The ids of the model nodes whose documents' KV is already resident,
/// as `resident_inputs`.
pub fn resident_inputs(nodes: &[PhysicalNode], pipelines: &Pipelines) -> BTreeSet<String> {
    let mut resident = BTreeSet::new();
    let mut kept: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    for index in topological_nodes(nodes) {
        let node = &nodes[index];
        let Some(members) = pipelines.get(node.id()) else {
            continue;
        };
        if members[0] != index {
            resident.insert(node.id().to_string());
            continue;
        }
        let alias = node.document_aliases()[0].clone();
        let head = node.document_head();
        if kept.get(&alias) == Some(&head) {
            resident.insert(node.id().to_string());
        }
        if members.iter().any(|member| nodes[*member].keep_kv()) {
            kept.insert(alias, head);
        } else {
            kept.remove(&alias);
        }
    }
    resident
}

/// The name of a value type, as plans write it.
pub fn value_type_name(value_type: ValueType) -> &'static str {
    match value_type {
        ValueType::FilterAnswers => "filter_answers",
        ValueType::JoinAnswers => "join_answers",
        ValueType::LabelAnswers => "label_answers",
        ValueType::ExtractAnswers => "extract_answers",
        ValueType::Pairs => "pairs",
        ValueType::Rows => "rows",
        ValueType::Any => "any",
    }
}

/// Check node ids, that every input reads an output of its type and
/// columns, that the root exists, and that the graph has no cycle.
///
/// # Errors
///
/// The graph breaks one of those rules.
pub fn validate_graph(nodes: &[PhysicalNode], root: &PortRef) -> Result<(), String> {
    let mut by_id: BTreeMap<&str, &PhysicalNode> = BTreeMap::new();
    for node in nodes {
        if by_id.insert(node.id(), node).is_some() {
            return Err(format!("duplicate physical node id {:?}", node.id()));
        }
    }
    let outputs: BTreeMap<(String, String), OutputPort> = nodes
        .iter()
        .flat_map(|node| {
            node.output_ports()
                .into_iter()
                .map(|port| ((node.id().to_string(), port.name.clone()), port))
        })
        .collect();
    for node in nodes {
        for input in node.inputs() {
            let key = (input.source.node_id.clone(), input.source.port.clone());
            let Some(actual) = outputs.get(&key) else {
                return Err(format!(
                    "{:?} reads missing output {:?}.{:?}",
                    node.id(),
                    key.0,
                    key.1
                ));
            };
            let expected = input.value_type;
            if expected != ValueType::Any
                && actual.value_type != ValueType::Any
                && expected != actual.value_type
            {
                return Err(format!(
                    "{:?} input {:?} expects {}, got {}",
                    node.id(),
                    input.name,
                    value_type_name(expected),
                    value_type_name(actual.value_type)
                ));
            }
            if !input.schema.is_empty()
                && !actual.schema.is_empty()
                && input.schema != actual.schema
            {
                return Err(format!(
                    "{:?} input {:?} has a different schema from its source",
                    node.id(),
                    input.name
                ));
            }
        }
    }
    let root_key = (root.node_id.clone(), root.port.clone());
    if !outputs.contains_key(&root_key) {
        return Err(format!(
            "root output {:?}.{:?} does not exist",
            root_key.0, root_key.1
        ));
    }
    // depth-first, each node after its inputs
    fn visit<'a>(
        id: &'a str,
        by_id: &BTreeMap<&'a str, &'a PhysicalNode>,
        visiting: &mut BTreeSet<&'a str>,
        visited: &mut BTreeSet<&'a str>,
    ) -> Result<(), String> {
        if visiting.contains(id) {
            return Err("physical graph contains a cycle".into());
        }
        if visited.contains(id) {
            return Ok(());
        }
        visiting.insert(id);
        if let Some(node) = by_id.get(id) {
            for input in node.inputs() {
                visit(input.source.node_id.as_str(), by_id, visiting, visited)?;
            }
        }
        visiting.remove(id);
        visited.insert(id);
        Ok(())
    }
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for id in by_id.keys() {
        visit(id, &by_id, &mut visiting, &mut visited)?;
    }
    Ok(())
}
