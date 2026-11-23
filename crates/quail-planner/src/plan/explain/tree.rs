//! The physical plan tree of `explain()`: each node, its fields and
//! estimates, and the stages.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::physical::{PhysicalNode, PortRef};
use serde::Deserialize;
use serde_json::Value;

use crate::expressions::asked_call;
use crate::expressions::{column_name, condition_text, repr_str};
use crate::operators::effective_selectivity;
use crate::operators::table_name;
use crate::operators::{Operators, SemanticJoin};
use crate::plan::stages::Stage;
use crate::pricing::statistics::{OrTerm, RowFlow};

use super::values::{
    Shown, commas, commas_int, default_note, field_lines, field_lines_as, format_g, number, prefix,
    py_str, seconds, selectivity, short_prompt, width,
};
use super::{ExtensionNode, PhysicalRequest, RunRequest};

/// The table's columns in display order: (cell key, header).
const COLUMNS: [(&str, &str); 7] = [
    ("est_rows", "est. rows"),
    ("rows", "rows"),
    ("est_pass", "est. pass"),
    ("pass", "pass"),
    ("est_time", "est. time"),
    ("time", "time"),
    ("tokens", "fresh tokens"),
];

// ---------------------------------------------------------------- physical

/// One node of an explained graph: one of Quail's, or an extension's.
struct Node {
    id: String,
    /// Each input as (port name, source port).
    inputs: Vec<(String, PortRef)>,
    known: Option<PhysicalNode>,
    extension: Option<ExtensionNode>,
}

impl Node {
    fn outputs(&self) -> Vec<String> {
        match (&self.known, &self.extension) {
            (Some(node), _) => node.outputs().into_iter().map(|(name, _)| name).collect(),
            (None, Some(extension)) => extension.outputs.clone(),
            (None, None) => Vec::new(),
        }
    }

    fn type_name(&self) -> String {
        match (&self.known, &self.extension) {
            (Some(node), _) => node.type_name().to_string(),
            (None, Some(extension)) => extension.type_name.clone(),
            (None, None) => String::new(),
        }
    }

    /// The node's explain fields, as its Python class's `explain_fields`,
    /// and whether its arrays are tuples.
    fn fields(&self) -> (Vec<(String, Shown)>, bool) {
        match (&self.known, &self.extension) {
            (Some(node), _) => known_fields(node),
            (None, Some(extension)) => (extension.fields.clone(), false),
            (None, None) => (Vec::new(), false),
        }
    }

    /// The node's explain fields as `key=value` lines.
    fn field_lines(&self, extra: Vec<(String, Shown)>, depth: usize) -> Vec<String> {
        let (fields, tuples) = self.fields();
        let mut all = extra;
        all.extend(fields);
        field_lines_as(&all, depth, tuples)
    }
}

/// A node explain shows by its fields alone: an extension's node, with
/// its attributes, or a request backend's, with each operator it
/// submits; None for any other node.
fn shown_alone(node: &PhysicalNode) -> Option<ExtensionNode> {
    let inputs = node
        .inputs()
        .iter()
        .map(|input| {
            (
                input.name.clone(),
                input.source.node_id.clone(),
                input.source.port.clone(),
            )
        })
        .collect();
    let outputs = node.outputs().into_iter().map(|(name, _)| name).collect();
    let fields = match node {
        PhysicalNode::Extension(extension) => extension
            .attributes
            .iter()
            .map(|(key, value)| (key.clone(), Shown::Text(py_str(value))))
            .collect(),
        PhysicalNode::RequestExecution(request) => request_fields(&request.attributes),
        _ => return None,
    };
    Some(ExtensionNode {
        type_name: node.type_name().to_string(),
        inputs,
        outputs,
        fields,
    })
}

/// Text without runs of whitespace, cut to 60 characters.
fn short_text(text: &str) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= 60 {
        text
    } else {
        format!("{}...", prefix(&text, 57))
    }
}

/// A request backend's node fields: its backend and tables, then each
/// filter's questions, join, classification, extraction, and map.
fn request_fields(request: &crate::physical::RequestExecutionAttributes) -> Vec<(String, Shown)> {
    let text = |value: Value| Shown::Text(py_str(&value));
    let mut fields = vec![
        (
            "backend".to_string(),
            text(Value::from(request.backend_name.clone())),
        ),
        (
            "aliases".to_string(),
            text(Value::from(request.aliases.clone())),
        ),
    ];
    for spec in &request.filters {
        let texts: Vec<String> = if spec.question_texts.is_empty() {
            spec.written_positions
                .iter()
                .map(|position| format!("predicate {}", position + 1))
                .collect()
        } else {
            spec.question_texts.clone()
        };
        let shown: Vec<String> = texts.iter().map(|found| short_text(found)).collect();
        fields.push((format!("filter {}", spec.alias), text(Value::from(shown))));
    }
    for spec in &request.joins {
        fields.push((
            format!("join {}", spec.written_position + 1),
            Shown::Text(format!(
                "{}: {}",
                spec.semantics,
                short_text(&spec.tail_text)
            )),
        ));
    }
    for spec in &request.classifies {
        fields.push((
            format!("classify {}", spec.output),
            text(Value::from(spec.labels.clone())),
        ));
    }
    for spec in &request.extracts {
        fields.push((
            format!("extract {}", spec.output),
            Shown::Text(format!("question={}", repr_str(&spec.question))),
        ));
    }
    for spec in &request.maps {
        fields.push((
            format!("map {}", spec.output),
            Shown::Text(short_text(&spec.tail_text)),
        ));
    }
    fields
}

#[derive(Deserialize)]
struct WrittenNode {
    attributes: Shown,
}

/// A node's attributes in declaration order.
pub(super) fn attributes(node: &PhysicalNode) -> Vec<(String, Shown)> {
    let written = serde_json::to_string(node)
        .ok()
        .and_then(|text| serde_json::from_str::<WrittenNode>(&text).ok());
    match written.map(|written| written.attributes) {
        Some(Shown::Map(pairs)) => pairs,
        _ => Vec::new(),
    }
}

/// A node's explain fields, as its Python class's `explain_fields`, and
/// whether its arrays are tuples: the default lists every attribute.
fn known_fields(node: &PhysicalNode) -> (Vec<(String, Shown)>, bool) {
    let attributes = attributes(node);
    let get = |name: &str| {
        attributes
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
            .unwrap_or(Shown::Null)
    };
    let pick = |names: &[&str]| -> Vec<(String, Shown)> {
        names
            .iter()
            .map(|name| ((*name).to_string(), get(name)))
            .collect()
    };
    match node {
        PhysicalNode::Scan(scan) => {
            let scan = &scan.attributes;
            let mut fields = pick(&["alias", "input_id", "num_documents", "total_tokens"]);
            fields.push((
                "predicates".to_string(),
                Shown::List(
                    scan.predicates
                        .iter()
                        .map(|predicate| Shown::Text(condition_text(predicate)))
                        .collect(),
                ),
            ));
            fields.push(("expected_rows".to_string(), get("expected_rows")));
            (fields, false)
        }
        PhysicalNode::Filter(filter) => {
            let predicate = filter
                .attributes
                .predicate
                .as_ref()
                .map_or_else(|| "None".to_string(), condition_text);
            let mut fields = vec![("predicate".to_string(), Shown::Text(predicate))];
            fields.extend(pick(&["selectivity", "aliases"]));
            (fields, false)
        }
        PhysicalNode::AiFilter(_) => (
            attributes
                .into_iter()
                .filter(|(key, _)| key != "question_token_ids" && key != "document_head")
                .collect(),
            true,
        ),
        PhysicalNode::AiJoin(_) => (
            pick(&[
                "written_position",
                "exec_index",
                "anchor",
                "partners",
                "semantics",
                "selectivity",
                "expected_rows",
                "anchor_frame_tokens",
                "pair_tail_tokens",
                "anchor_resident",
                "tuple_tokens",
                "pairs_from",
                "keep_kv",
                "attention",
                "share_prefixes",
            ]),
            false,
        ),
        PhysicalNode::AiScore(_)
        | PhysicalNode::AiClassify(_)
        | PhysicalNode::AiExtract(_)
        | PhysicalNode::AiMap(_)
        | PhysicalNode::AiSaeFeatures(_) => (inference_fields(node, &get("spec"), &get), false),
        _ => (attributes, true),
    }
}

/// The explain fields every model call node shares, then its own.
fn inference_fields(
    node: &PhysicalNode,
    spec: &Shown,
    get: &dyn Fn(&str) -> Shown,
) -> Vec<(String, Shown)> {
    let of = |field: &str, missing: Shown| -> Shown {
        match spec {
            Shown::Map(pairs) => pairs
                .iter()
                .find(|(key, _)| key == field)
                .map(|(_, value)| value.clone())
                .unwrap_or(missing),
            _ => missing,
        }
    };
    let text = |value: &str| Shown::Text(value.to_string());
    let mut fields = vec![
        ("backend".to_string(), get("backend_name")),
        ("keep_kv".to_string(), get("keep_kv")),
        ("model".to_string(), get("model")),
        ("batching".to_string(), text("token_based_admission")),
        ("output".to_string(), of("name", Shown::Null)),
        (
            "aliases".to_string(),
            of("aliases", Shown::List(Vec::new())),
        ),
        (
            "expected_rows".to_string(),
            of("expected_rows", Shown::Number(0.into())),
        ),
        (
            "estimated_seconds".to_string(),
            of("estimated_seconds", Shown::Number(0.into())),
        ),
        (
            "share_prefixes".to_string(),
            of("share_prefixes", Shown::Bool(false)),
        ),
    ];
    match node {
        PhysicalNode::AiClassify(_) => {
            fields.push(("labels".to_string(), of("labels", Shown::List(Vec::new()))));
            fields.push(("scoring".to_string(), of("scoring", Shown::Null)));
            fields.push(("draws".to_string(), of("draws", Shown::Number(1.into()))));
            fields.push((
                "probabilities".to_string(),
                of("probabilities", Shown::Bool(false)),
            ));
        }
        PhysicalNode::AiExtract(_) => {
            fields.push(("question".to_string(), of("question", Shown::Null)));
        }
        PhysicalNode::AiMap(_) => {
            fields.push((
                "max_answer_tokens".to_string(),
                of("max_answer_tokens", Shown::Null),
            ));
            fields.push(("attention".to_string(), get("attention")));
        }
        PhysicalNode::AiSaeFeatures(_) => {
            fields.push(("pooling".to_string(), of("pooling", Shown::Null)));
        }
        _ => {}
    }
    fields
}

/// A physical graph to explain: its nodes in written order and its root.
pub(super) struct Graph {
    nodes: Vec<Node>,
    root: PortRef,
    by_id: HashMap<String, usize>,
}

impl Graph {
    /// The documents the graph's scans read.
    pub(super) fn num_documents(&self) -> i64 {
        self.nodes
            .iter()
            .filter_map(|node| match &node.known {
                Some(PhysicalNode::Scan(scan)) => Some(scan.attributes.num_documents),
                _ => None,
            })
            .sum()
    }

    pub(super) fn read(physical: &PhysicalRequest) -> Result<Self, String> {
        let mut nodes = Vec::new();
        for value in physical.graph["nodes"].as_array().into_iter().flatten() {
            let id = value["id"].as_str().unwrap_or_default().to_string();
            let node: PhysicalNode = serde_json::from_value(value.clone())
                .or_else(|_| serde_json::from_value(numeric_tokens(value.clone(), false)))
                .map_err(|error| format!("the node {id} does not read: {error}"))?;
            let extension = shown_alone(&node);
            nodes.push(Node {
                id,
                inputs: node
                    .inputs()
                    .iter()
                    .map(|input| (input.name.clone(), input.source.clone()))
                    .collect(),
                known: extension.is_none().then_some(node),
                extension,
            });
        }
        let root: PortRef = serde_json::from_value(physical.graph["root"].clone())
            .map_err(|error| format!("the graph's root does not read: {error}"))?;
        let by_id = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id.clone(), index))
            .collect();
        Ok(Self { nodes, root, by_id })
    }

    fn node(&self, id: &str) -> &Node {
        &self.nodes[self.by_id[id]]
    }

    /// Node positions, each after the nodes it reads.
    fn topological(&self) -> Vec<usize> {
        fn visit(graph: &Graph, index: usize, visited: &mut Vec<bool>, ordered: &mut Vec<usize>) {
            if visited[index] {
                return;
            }
            for (_, source) in &graph.nodes[index].inputs {
                visit(graph, graph.by_id[&source.node_id], visited, ordered);
            }
            visited[index] = true;
            ordered.push(index);
        }
        let mut visited = vec![false; self.nodes.len()];
        let mut ordered = Vec::with_capacity(self.nodes.len());
        for index in 0..self.nodes.len() {
            visit(self, index, &mut visited, &mut ordered);
        }
        ordered
    }

    /// The planner's estimated rows on each output port.
    fn estimated_rows(&self) -> HashMap<(String, String), Option<f64>> {
        let mut rows: HashMap<(String, String), Option<f64>> = HashMap::new();
        // by table, the rows its AI filters have passed so far
        let mut flows: HashMap<String, RowFlow> = HashMap::new();
        for index in self.topological() {
            let node = &self.nodes[index];
            let inputs: Vec<Option<f64>> = node
                .inputs
                .iter()
                .map(|(_, source)| {
                    rows.get(&(source.node_id.clone(), source.port.clone()))
                        .copied()
                        .flatten()
                })
                .collect();
            let single = (inputs.len() == 1).then(|| inputs[0]).flatten();
            let value = match &node.known {
                Some(PhysicalNode::Scan(scan)) => Some(
                    scan.attributes
                        .expected_rows
                        .unwrap_or(scan.attributes.num_documents as f64),
                ),
                Some(PhysicalNode::AiFilter(filter)) => {
                    inputs.first().copied().flatten().map(|value| {
                        let filter = &filter.attributes;
                        let term = filter.disjunction.as_ref().map(|name| OrTerm {
                            disjunction: name.clone(),
                            short_circuit: filter.short_circuit,
                        });
                        let passing = match filter.answer {
                            Some(_) => 1.0,
                            None => effective_selectivity(filter.selectivity),
                        };
                        let mut flow = flows
                            .remove(&filter.alias)
                            .unwrap_or_else(|| RowFlow::new(value))
                            .with_rows(value);
                        flow.pass(passing, term.as_ref(), filter.closes_disjunction);
                        let passed = flow.rows();
                        flows.insert(filter.alias.clone(), flow);
                        passed
                    })
                }
                Some(PhysicalNode::Limit(limit)) => {
                    single.map(|value| value.min(limit.attributes.count as f64))
                }
                _ => None,
            };
            for output in node.outputs() {
                if !output.starts_with("filter_answers:") {
                    rows.insert((node.id.clone(), output), value);
                }
            }
        }
        rows
    }
}

/// `value` with each list of words under a token field read as zeros:
/// a tokenizer that splits words gives words for token ids, and explain
/// reads only how many there are.
fn numeric_tokens(value: Value, in_tokens: bool) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| {
                    let tokens = in_tokens || key.contains("token") || key == "document_head";
                    (key, numeric_tokens(value, tokens))
                })
                .collect(),
        ),
        Value::Array(items)
            if in_tokens && !items.is_empty() && items.iter().all(Value::is_string) =>
        {
            Value::Array(items.iter().map(|_| Value::from(0)).collect())
        }
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| numeric_tokens(item, in_tokens))
                .collect(),
        ),
        other => other,
    }
}

/// A run's stage measurements, by the predicate they ran.
#[derive(Default)]
struct Stages {
    filters: HashMap<(String, i64), Value>,
    joins: HashMap<i64, Value>,
}

impl Stages {
    fn read(report: Option<&Value>) -> Self {
        let mut stages = Self::default();
        let Some(report) = report else {
            return stages;
        };
        for stage in report["stages"].as_array().into_iter().flatten() {
            let Some(position) = stage.get("written_position").and_then(Value::as_i64) else {
                continue;
            };
            match stage["op"].as_str() {
                Some("filter") => {
                    let alias = py_str(&stage["alias"]);
                    stages.filters.insert((alias, position), stage.clone());
                }
                Some("join") => {
                    stages.joins.insert(position, stage.clone());
                }
                _ => {}
            }
        }
        stages
    }
}

type Cells = BTreeMap<&'static str, String>;

/// The rows of a physical tree: each node's line, details, and
/// predicate rows, with their cells.
struct TreeRows<'a> {
    graph: &'a Graph,
    verbose: bool,
    metrics: Option<&'a HashMap<String, Vec<(String, Value)>>>,
    estimates: HashMap<&'a str, &'a Value>,
    stages: Stages,
    scans: HashMap<String, crate::operators::Scan>,
    filters: HashMap<String, Vec<crate::physical::FilterPredicate>>,
    joins: Vec<SemanticJoin>,
    rows_estimate: HashMap<(String, String), Option<f64>>,
    shared: HashMap<String, usize>,
    visited: HashSet<String>,
    rows: Vec<(String, Cells)>,
    used_default: bool,
}

impl TreeRows<'_> {
    fn stage_cells(
        &mut self,
        selectivity_value: Option<f64>,
        expected: f64,
        measured: Option<Value>,
        evaluated_key: &str,
    ) -> Cells {
        if selectivity_value.is_none() {
            self.used_default = true;
        }
        let mut cells = Cells::new();
        cells.insert("est_rows", number(expected));
        cells.insert("est_pass", selectivity(selectivity_value, true));
        if let Some(measured) = measured {
            cells.insert("rows", commas(&measured[evaluated_key]));
            cells.insert(
                "pass",
                selectivity(measured["observed_selectivity"].as_f64(), false),
            );
        }
        cells
    }

    /// The node's title, detail lines, and predicate rows.
    fn describe(&mut self, node: &Node) -> (String, Vec<String>, Vec<(String, Cells)>) {
        let mut details: Vec<String> = Vec::new();
        let mut predicates: Vec<(String, Cells)> = Vec::new();
        let Some(known) = &node.known else {
            let title = node.type_name();
            if !self.verbose {
                details.extend(node.field_lines(Vec::new(), 0));
            }
            return (title, details, predicates);
        };
        if let Some(guard) = known.guard() {
            details.push(format!(
                "asks only where {}; every other row's answer is NULL",
                condition_text(guard)
            ));
        }
        let (stops, when_true) = known.answer_stops();
        let tests: Vec<String> = stops.iter().map(condition_text).collect();
        if !tests.is_empty() {
            let also = if when_true {
                ", or once every one is TRUE"
            } else {
                ""
            };
            details.push(format!(
                "stops an answer once {} is not TRUE on its tokens so far{also}",
                tests.join(" or ")
            ));
        }
        if let Some(alternative) = known.alternative() {
            details.push(format!(
                "passes each row where {}, whatever the model answers",
                condition_text(alternative)
            ));
        }
        if let PhysicalNode::AiFilter(filter) = known
            && filter.attributes.disjunction.is_some()
        {
            if filter.attributes.short_circuit {
                details.push("asks only the rows no earlier term of its OR passed".to_string());
            }
            details.push(if filter.attributes.closes_disjunction {
                "returns the rows a term of its OR passed".to_string()
            } else {
                "keeps every row for the later terms of its OR".to_string()
            });
        }
        if matches!(known, PhysicalNode::AiFilter(filter) if filter.attributes.negated) {
            details.push("passes each row where the model answers no".to_string());
        }
        if let Some(answer) = known.answer() {
            details.push(format!(
                "keeps every row and returns its answer as {answer}"
            ));
        }
        let title = match known {
            PhysicalNode::Scan(scan) => {
                let scan = &scan.attributes;
                let source = self.scans.get(&scan.alias);
                let title = match source {
                    Some(source) => {
                        format!("Scan {} as {}", table_name(&source.provider), scan.alias)
                    }
                    None => format!("Scan: {}", scan.alias),
                };
                let mean = if scan.num_documents != 0 {
                    scan.total_tokens as f64 / scan.num_documents as f64
                } else {
                    0.0
                };
                details.push(format!(
                    "tokens={}, mean_document_tokens={}",
                    commas_int(scan.total_tokens),
                    number(mean)
                ));
                if let Some(source) = source
                    && !source.predicates.is_empty()
                {
                    let predicates: Vec<String> =
                        source.predicates.iter().map(condition_text).collect();
                    details.push(format!("where {}", predicates.join(" and ")));
                }
                title
            }
            PhysicalNode::Limit(limit) => format!("Limit: {}", commas_int(limit.attributes.count)),
            PhysicalNode::AiFilter(filter) => {
                let filter = &filter.attributes;
                let mut kv = Vec::new();
                if filter.arena_writes {
                    kv.push("KV rewind=on");
                }
                if filter.keep_kv {
                    kv.push("retain KV for later joins");
                }
                details.push(if !kv.is_empty() {
                    kv.join(", ")
                } else if filter.arena_writes {
                    "KV: stored".to_string()
                } else {
                    "KV: not stored".to_string()
                });
                if !filter.stop_key.is_empty() && filter.num_key_survivors > 1 {
                    details.push(format!(
                        "stop per key: {} (the rest skipped once {} survive)",
                        filter.stop_key.join(", "),
                        filter.num_key_survivors
                    ));
                } else if !filter.stop_key.is_empty() {
                    details.push(format!(
                        "stop per key: {} (one document at a time, the rest skipped once one \
                         survives)",
                        filter.stop_key.join(", ")
                    ));
                }
                let mut label = format!("predicate {}", filter.written_position + 1);
                if let Some(written) = self.filters.get(&filter.alias)
                    && let Some(predicate) = usize::try_from(filter.written_position)
                        .ok()
                        .and_then(|position| written.get(position))
                {
                    label += &format!(
                        "  {}",
                        short_prompt(&asked_call(&predicate.expression).prompt)
                    );
                }
                let measured = self
                    .stages
                    .filters
                    .get(&(filter.alias.clone(), filter.written_position))
                    .cloned();
                let cells = self.stage_cells(
                    filter.selectivity,
                    filter.expected_rows,
                    measured,
                    "evaluated",
                );
                predicates.push((label, cells));
                format!("AiFilter: {}", filter.alias)
            }
            PhysicalNode::AiClassify(classify) => match &classify.attributes.spec {
                Some(spec) => {
                    let tokens: usize = spec.label_token_ids.iter().map(Vec::len).sum();
                    let draws = if spec.draws > 1 {
                        format!(", up to {} noise draws", spec.draws)
                    } else {
                        String::new()
                    };
                    details.push(format!(
                        "rule={}, labels={}, label tokens={tokens}{draws}",
                        spec.scoring,
                        spec.labels.len()
                    ));
                    format!(
                        "AiClassify: {} over {}",
                        spec.name,
                        spec.aliases.first().cloned().unwrap_or_default()
                    )
                }
                None => "AiClassify".to_string(),
            },
            PhysicalNode::AiExtract(extract) => match &extract.attributes.spec {
                Some(spec) => {
                    details.push(format!(
                        "question={}, up to {} answer tokens",
                        repr_str(&spec.question),
                        spec.max_answer_tokens
                    ));
                    format!(
                        "AiExtract: {} over {}",
                        spec.name,
                        spec.aliases.first().cloned().unwrap_or_default()
                    )
                }
                None => "AiExtract".to_string(),
            },
            PhysicalNode::Filter(filter) => format!(
                "Filter: {}",
                filter
                    .attributes
                    .predicate
                    .as_ref()
                    .map(condition_text)
                    .unwrap_or_else(|| "None".to_string())
            ),
            PhysicalNode::AiJoin(join) => {
                let join = &join.attributes;
                let source = match join.anchor_resident.as_str() {
                    "none" => "not resident",
                    "filter" => "from filters",
                    "kept" => "from an earlier join",
                    other => other,
                };
                let keep = if join.keep_kv { "yes" } else { "no" };
                let size = self
                    .estimates
                    .get(node.id.as_str())
                    .and_then(|estimate| estimate.get("pass_questions"))
                    .filter(|size| !size.is_null() && size.as_f64() != Some(0.0));
                details.push(match size {
                    Some(size) => format!(
                        "set passes: each anchor document, then up to {} partner rows as \
                         questions",
                        py_str(size)
                    ),
                    None => format!("KV: anchor={source}, retain after join={keep}"),
                });
                let mut label = format!(
                    "join {} {} ({}, {})",
                    join.written_position + 1,
                    join.semantics,
                    join.anchor,
                    join.partners.join(", ")
                );
                if !join.pairs_from.is_empty() {
                    label += &format!(" over pairs from {}", join.pairs_from);
                }
                if let Some(written) = usize::try_from(join.written_position)
                    .ok()
                    .and_then(|position| self.joins.get(position))
                {
                    label += &format!("  {}", short_prompt(&asked_call(&written.predicate).prompt));
                }
                let measured = self.stages.joins.get(&join.written_position).cloned();
                let cells =
                    self.stage_cells(join.selectivity, join.expected_rows, measured, "tuples");
                predicates.push((label, cells));
                format!("AiJoin: anchor={}", join.anchor)
            }
            PhysicalNode::HashJoin(join) => {
                let join = &join.attributes;
                let on: Vec<String> = join
                    .on
                    .iter()
                    .map(|(left, right)| {
                        format!("{} = {}", column_name(left, true), column_name(right, true))
                    })
                    .collect();
                if let Some(filter) = &join.filter {
                    details.push(format!("filter: {}", condition_text(filter)));
                }
                details.push(format!(
                    "pairs kept: {} of the cross product",
                    format_g(join.pair_fraction, 4)
                ));
                if on.is_empty() {
                    format!("HashJoin: {} x {}", join.left, join.right)
                } else {
                    format!("HashJoin: {}", on.join(" and "))
                }
            }
            PhysicalNode::Barrier(barrier) => {
                format!("Barrier: next_anchor={}", barrier.attributes.next_anchor)
            }
            _ => {
                if !self.verbose {
                    details.extend(node.field_lines(Vec::new(), 0));
                }
                node.type_name()
            }
        };
        (title, details, predicates)
    }

    fn visit(&mut self, index: usize, depth: usize) {
        let graph = self.graph;
        let node = &graph.nodes[index];
        let pad = "  ".repeat(depth);
        let (mut title, mut details, predicates) = self.describe(node);
        let reference = self.shared.get(&node.id).copied();
        if self.visited.contains(&node.id) {
            let reference = reference.map(|found| found.to_string()).unwrap_or_default();
            self.rows
                .push((format!("{pad}Reuse [{reference}]"), Cells::new()));
            return;
        }
        self.visited.insert(node.id.clone());
        if let Some(reference) = reference {
            title += &format!(" [{reference}]");
        }
        let values: Vec<Option<f64>> = node
            .outputs()
            .into_iter()
            .filter(|output| !output.starts_with("filter_answers:"))
            .map(|output| {
                self.rows_estimate
                    .get(&(node.id.clone(), output))
                    .copied()
                    .flatten()
            })
            .collect();
        let value = if values.len() == 1 { values[0] } else { None };
        let estimate = self.estimates.get(node.id.as_str()).copied();
        let mut cells = Cells::new();
        if let Some(value) = value {
            cells.insert("est_rows", number(value));
        }
        if let Some(estimate) = estimate
            && let Some(found) = estimate.get("seconds")
        {
            cells.insert("est_time", seconds(found.as_f64()));
        }
        if let Some(metrics) = self.metrics {
            match metrics.get(&node.id) {
                Some(measured) => {
                    let field = |name: &str| {
                        measured
                            .iter()
                            .find(|(key, _)| key == name)
                            .map(|(_, value)| value.clone())
                            .unwrap_or(Value::Null)
                    };
                    cells.insert("rows", commas(&field("output_rows")));
                    cells.insert("time", seconds(field("wall_s").as_f64()));
                    let fresh = field("fresh_tokens");
                    if fresh.as_f64().is_some_and(|tokens| tokens != 0.0) {
                        cells.insert("tokens", commas(&fresh));
                    }
                    if self.verbose {
                        let shown: Vec<(String, Shown)> = measured
                            .iter()
                            .map(|(key, value)| (key.clone(), Shown::from(value)))
                            .collect();
                        details.extend(field_lines(&shown, 0));
                    }
                }
                None => title += " (metrics unavailable)",
            }
        }
        let members: Vec<String> = estimate
            .and_then(|estimate| estimate.get("pass_members"))
            .and_then(Value::as_array)
            .map(|members| members.iter().map(py_str).collect())
            .unwrap_or_default();
        if !members.is_empty() {
            let others: Vec<&str> = members
                .iter()
                .filter(|member| **member != node.id)
                .map(String::as_str)
                .collect();
            if members[0] == node.id {
                let mut text = format!(
                    "set pass: {} question{} per document pass",
                    members.len(),
                    if members.len() > 1 { "s" } else { "" }
                );
                if !others.is_empty() {
                    text += &format!(", with {}", others.join(", "));
                }
                details.push(text);
            } else {
                details.push(format!(
                    "set pass: asked in {}'s pass, priced there",
                    members[0]
                ));
            }
        }
        if let Some(estimate) = estimate
            && let Some(tokens) = estimate.get("release_recompute_tokens")
        {
            details.push(format!(
                "expected recompute at the join if the sequence's KV is released: {} tokens, \
                 {:.3} s",
                commas(tokens),
                estimate["release_recompute_seconds"]
                    .as_f64()
                    .unwrap_or_default()
            ));
        }
        self.rows.push((format!("{pad}{title}"), cells));
        self.rows.extend(
            details
                .into_iter()
                .map(|detail| (format!("{pad}  {detail}"), Cells::new())),
        );
        self.rows.extend(
            predicates
                .into_iter()
                .map(|(label, cells)| (format!("{pad}  {label}"), cells)),
        );
        if self.verbose {
            let named = vec![
                ("node_id".to_string(), Shown::Text(node.id.clone())),
                ("type".to_string(), Shown::Text(node.type_name())),
            ];
            self.rows.extend(
                node.field_lines(named, depth + 1)
                    .into_iter()
                    .map(|line| (line, Cells::new())),
            );
            self.rows.extend(node.inputs.iter().map(|(name, source)| {
                (
                    format!("{pad}  {name} <- {}[{}]", source.node_id, source.port),
                    Cells::new(),
                )
            }));
        }
        let mut seen = HashSet::new();
        let sources: Vec<String> = node
            .inputs
            .iter()
            .map(|(_, source)| source.node_id.clone())
            .filter(|id| seen.insert(id.clone()))
            .collect();
        for source in sources {
            let child = graph.by_id[&source];
            self.visit(child, depth + 1);
        }
    }
}

/// Lay out (text, cells) rows as a tree column and aligned columns.
fn table(rows: &[(String, Cells)]) -> Vec<String> {
    let columns: Vec<(&str, &str)> = COLUMNS
        .iter()
        .copied()
        .filter(|(key, _)| {
            rows.iter()
                .any(|(_, cells)| cells.get(key).is_some_and(|cell| !cell.is_empty()))
        })
        .collect();
    let text_width = rows.iter().map(|(text, _)| width(text)).max().unwrap_or(0);
    let widths: HashMap<&str, usize> = columns
        .iter()
        .map(|(key, label)| {
            let cells = rows
                .iter()
                .map(|(_, cells)| cells.get(key).map_or(0, |cell| width(cell)));
            (*key, cells.fold(width(label), usize::max))
        })
        .collect();
    let right =
        |text: &str, size: usize| format!("{}{text}", " ".repeat(size.saturating_sub(width(text))));
    let mut lines = Vec::new();
    if !columns.is_empty() {
        let mut header = " ".repeat(text_width);
        for (key, label) in &columns {
            header += &format!("   {}", right(label, widths[key]));
        }
        lines.push(header);
    }
    for (text, cells) in rows {
        let mut line = format!(
            "{text}{}",
            " ".repeat(text_width.saturating_sub(width(text)))
        );
        for (key, _) in &columns {
            let cell = cells.get(key).map_or("", String::as_str);
            line += &format!("   {}", right(cell, widths[key]));
        }
        lines.push(line.trim_end().to_string());
    }
    lines
}

/// A physical tree with estimated and measured columns.
pub(super) fn physical_tree(
    graph: &Graph,
    found: Option<&Operators>,
    verbose: bool,
    run: Option<&RunRequest>,
    estimates: &[(String, Value)],
) -> String {
    let (scans, filters, joins) = match found {
        Some(found) => (
            found
                .scans
                .iter()
                .map(|scan| (scan.alias.clone(), scan.clone()))
                .collect(),
            found.filters.iter().cloned().collect(),
            found.joins.clone(),
        ),
        None => (HashMap::new(), HashMap::new(), Vec::new()),
    };
    let topological = graph.topological();
    let mut uses: HashMap<&str, usize> = HashMap::new();
    for node in &graph.nodes {
        let sources: HashSet<&str> = node
            .inputs
            .iter()
            .map(|(_, source)| source.node_id.as_str())
            .collect();
        for source in sources {
            *uses.entry(source).or_default() += 1;
        }
    }
    let shared: HashMap<String, usize> = topological
        .iter()
        .map(|index| &graph.nodes[*index])
        .filter(|node| uses.get(node.id.as_str()).copied().unwrap_or(0) > 1)
        .enumerate()
        .map(|(position, node)| (node.id.clone(), position + 1))
        .collect();
    let mut tree = TreeRows {
        graph,
        verbose,
        metrics: run.map(|run| &run.metrics),
        estimates: estimates
            .iter()
            .map(|(id, estimate)| (id.as_str(), estimate))
            .collect(),
        stages: Stages::read(run.map(|run| &run.report)),
        scans,
        filters,
        joins,
        rows_estimate: graph.estimated_rows(),
        shared,
        visited: HashSet::new(),
        rows: Vec::new(),
        used_default: false,
    };
    // each node no other node reads heads one tree: the root first, then
    // the latest first
    let root = graph.by_id[&graph.root.node_id];
    let sinks = topological
        .iter()
        .rev()
        .filter(|index| **index != root && !uses.contains_key(graph.nodes[**index].id.as_str()));
    for index in std::iter::once(&root).chain(sinks) {
        if !tree.visited.contains(&graph.nodes[*index].id) {
            tree.visit(*index, 0);
        }
    }
    let mut lines = table(&tree.rows);
    let mut passes: Vec<Vec<String>> = Vec::new();
    for (_, estimate) in estimates {
        let Some(members) = estimate.get("pass_members").and_then(Value::as_array) else {
            continue;
        };
        if members.is_empty() {
            continue;
        }
        let members: Vec<String> = members.iter().map(py_str).collect();
        if !passes.contains(&members) {
            passes.push(members);
        }
    }
    let first_alias = |id: &str| -> String {
        graph
            .node(id)
            .known
            .as_ref()
            .and_then(|node| node.document_aliases().into_iter().next())
            .unwrap_or_default()
    };
    if !passes.is_empty() {
        let mut aliases: Vec<String> = Vec::new();
        for members in &passes {
            let alias = first_alias(&members[0]);
            if !aliases.contains(&alias) {
                aliases.push(alias);
            }
        }
        for alias in aliases {
            let own: Vec<String> = passes
                .iter()
                .filter(|members| first_alias(&members[0]) == alias)
                .map(|members| format!("[{}]", members.join(", ")))
                .collect();
            lines.push(format!(
                "Passes on {alias}: {} (each pass reads the document again, then its \
                 questions; nothing is kept between passes)",
                own.join(" -> ")
            ));
        }
        if tree.used_default {
            lines.push(default_note());
        }
        return lines.join("\n");
    }
    let known: Option<Vec<PhysicalNode>> =
        graph.nodes.iter().map(|node| node.known.clone()).collect();
    if let Some(known) = known {
        let pipelines = crate::topology::build_pipelines(&known, &[]);
        for members in &pipelines.members {
            if members.len() < 2 {
                continue;
            }
            let alias = known[members[0]]
                .document_aliases()
                .into_iter()
                .next()
                .unwrap_or_default();
            let ids: Vec<&str> = members.iter().map(|member| known[*member].id()).collect();
            lines.push(format!(
                "Pipeline on {alias}: {} (each document runs every operator with its KV \
                 resident)",
                ids.join(" -> ")
            ));
        }
    }
    if tree.used_default {
        lines.push(default_note());
    }
    lines.join("\n")
}

/// One line per stage of a plan on several GPUs: its nodes, its inputs
/// split across the GPUs by its key's ids, its inputs collected once and
/// sent whole to every GPU, and the stages it waits for.
pub(super) fn stage_lines(stages: &[Stage]) -> Vec<String> {
    stages
        .iter()
        .enumerate()
        .map(|(index, stage)| {
            let named = |hashed: bool| -> Vec<String> {
                stage
                    .inputs
                    .iter()
                    .filter(|input| input.hashed == hashed)
                    .map(|input| match input.stage {
                        None => input.node_id.clone(),
                        Some(_) => format!("{} {}", input.node_id, input.port),
                    })
                    .collect()
            };
            let mut parts = vec![format!("Stage {}: {}", index + 1, stage.nodes.join(" -> "))];
            let (split, whole) = (named(true), named(false));
            if !split.is_empty() {
                parts.push(format!("split by {} id: {}", stage.key, split.join(", ")));
            }
            if !whole.is_empty() {
                parts.push(format!("sent whole to every GPU: {}", whole.join(", ")));
            }
            let waits: BTreeSet<usize> = stage
                .inputs
                .iter()
                .filter_map(|input| input.stage)
                .collect();
            if !waits.is_empty() {
                let waits: Vec<String> = waits.iter().map(|from| (from + 1).to_string()).collect();
                let noun = if waits.len() == 1 { "stage" } else { "stages" };
                parts.push(format!("waits for {noun} {}", waits.join(", ")));
            }
            parts.join("; ")
        })
        .collect()
}
