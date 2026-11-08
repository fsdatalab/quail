//! The text of `explain()`: a region's logical plan and physical plan,
//! with the planner's estimates and, after a run, what was measured.
//!
//! The logical plan is DataFusion's plan of the region, as its
//! `display_indent` writes it. The request is JSON from Python: that
//! text, the region's operators, the physical plan
//! as its envelope with its per node estimates in the plan's order, or a
//! refusal, and optionally the run's report and per node metrics. A node
//! whose type is not one of Quail's comes with its inputs, outputs, and
//! fields, which its own Python class reports.

use std::collections::{BTreeMap, HashMap, HashSet};

use quail_physical::{PhysicalNode, PortRef};
use serde::Deserialize;
use serde_json::Value;

use crate::expressions::asked_call;
use crate::expressions::{column_name, condition_text, repr_float, repr_str};
use crate::operators::table_name;
use crate::operators::{Operators, SemanticJoin};
use crate::statistics::{DEFAULT_SELECTIVITY, effective_selectivity};

/// The mark beside a selectivity the planner assumed.
const DEFAULT_MARK: &str = "*";
/// The characters of a prompt template a predicate row shows.
const PROMPT_CHARS: usize = 32;
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

/// What to explain, as Python sends it.
#[derive(Debug, Deserialize)]
pub struct ExplainRequest {
    /// DataFusion's plan of the region, one operator per line.
    pub logical: Option<String>,
    /// The operators the physical plan lowers.
    pub operators: Option<Operators>,
    pub physical: Option<PhysicalRequest>,
    /// The refusal's fields, when no configuration can run the region.
    pub refusal: Option<Value>,
    #[serde(default)]
    pub verbose: bool,
    /// The registered rules in order, each (name, kind).
    pub rules: Option<Vec<(String, String)>>,
    pub usd_per_hour: Option<f64>,
    /// The registered spec of the plan's model, without type tags.
    pub model_spec: Option<Value>,
    pub run: Option<RunRequest>,
    /// Only the physical tree, as `QueryResult.explain` shows it.
    #[serde(default)]
    pub tree_only: bool,
}

/// A physical plan: its envelope fields, with the settings and the per
/// node estimates as (key, value) pairs in the plan's order.
#[derive(Debug, Deserialize)]
pub struct PhysicalRequest {
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub device: String,
    #[serde(default)]
    pub gpus: i64,
    #[serde(default)]
    pub settings: Vec<(String, Value)>,
    /// The settings as Python's `str` shows each value, for the verbose
    /// settings section: a mapping, or a list of mappings, nests.
    #[serde(default)]
    pub shown_settings: Vec<(String, Shown)>,
    /// The encoded graph: `nodes` and `root`.
    pub graph: Value,
    #[serde(default)]
    pub estimates: Vec<(String, Value)>,
}

/// A node that explain shows by its fields alone: an extension's, or a
/// request backend's.
#[derive(Debug)]
pub struct ExtensionNode {
    pub type_name: String,
    /// Each input as (port name, source node id, source port).
    pub inputs: Vec<(String, String, String)>,
    pub outputs: Vec<String>,
    pub fields: Vec<(String, Shown)>,
}

/// What a run measured.
#[derive(Debug, Deserialize)]
pub struct RunRequest {
    pub report: Value,
    /// node id -> its metrics as (field, value) pairs.
    #[serde(default)]
    pub metrics: HashMap<String, Vec<(String, Value)>>,
}

/// Format the request as `explain()` text.
///
/// # Errors
///
/// A graph node does not read as a physical node.
pub fn explain_text(request: &ExplainRequest) -> Result<String, String> {
    if request.tree_only {
        let Some(physical) = &request.physical else {
            return Ok("no physical plan was executed".to_string());
        };
        let graph = Graph::read(physical)?;
        return Ok(physical_tree(
            &graph,
            None,
            request.verbose,
            request.run.as_ref(),
            &[],
        ));
    }
    let mut lines = vec!["logical:".to_string()];
    if let Some(logical) = &request.logical {
        lines.extend(logical.lines().map(|line| format!("  {line}")));
    }
    if let Some(refusal) = &request.refusal {
        lines.push(format!(
            "refusal: {}: needed {} {}, available {}",
            py_str(&refusal["constraint"]),
            py_str(&refusal["needed"]),
            py_str(&refusal["unit"]),
            py_str(&refusal["available"]),
        ));
        if let Some(reasons) = refusal["reasons"].as_array() {
            lines.extend(reasons.iter().map(|reason| format!("  {}", py_str(reason))));
        }
        return Ok(lines.join("\n"));
    }
    let Some(physical) = &request.physical else {
        return Ok(lines.join("\n"));
    };
    let graph = Graph::read(physical)?;
    let settings: BTreeMap<&str, &Value> = physical
        .settings
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect();
    lines.push(String::new());
    lines.push(format!(
        "physical: backend={}, model={}, gpus={}",
        physical.backend, physical.model, physical.gpus
    ));
    if physical.backend == "quail" {
        let mut budgets = vec!["KV=bf16".to_string()];
        if let Some(chunk) = settings
            .get("chunk_tokens")
            .filter(|value| !value.is_null())
        {
            budgets.push(format!("chunk budget={} tokens", commas(chunk)));
        }
        if let Some(admission) = settings
            .get("admission_tokens")
            .filter(|value| !value.is_null())
            && reuses_kv(request.model_spec.as_ref())
        {
            budgets.push(format!("admission budget={} tokens", commas(admission)));
        }
        lines.push(format!("  {}", budgets.join(", ")));
        if let Some(Value::Object(deduplicate)) = settings.get("deduplicate") {
            for (alias, columns) in deduplicate {
                let mut grouping = vec!["text".to_string()];
                grouping.extend(columns.as_array().into_iter().flatten().map(py_str));
                lines.push(format!(
                    "  deduplicate: {alias} (by {})",
                    grouping.join(" and ")
                ));
            }
        }
        lines.extend(
            architecture_lines(request.model_spec.as_ref())
                .into_iter()
                .map(|line| format!("  {line}")),
        );
    }
    lines.push(String::new());
    let tree = physical_tree(
        &graph,
        request.operators.as_ref(),
        request.verbose,
        request.run.as_ref(),
        &physical.estimates,
    );
    lines.extend(tree.lines().map(|line| format!("  {line}")));
    let passes = physical
        .estimates
        .iter()
        .any(|(_, estimate)| estimate.get("pass_members").is_some());
    if passes {
        lines.push(
            "  est. time is each pass group's work, on its first operator; the other \
             operators in the group are asked in its passes"
                .to_string(),
        );
    } else if !physical.estimates.is_empty() {
        lines.push(
            "  est. time is each node's work alone; node times do not add up to the plan \
             estimate"
                .to_string(),
        );
        lines.push("  because chunk packing shares forward passes across nodes".to_string());
    }
    if let Some(run) = &request.run {
        lines.push(String::new());
        lines.push("run:".to_string());
        lines.extend(
            run_summary(&run.report, &graph, physical.gpus, request.usd_per_hour)
                .into_iter()
                .map(|line| format!("  {line}")),
        );
    }
    if request.verbose {
        lines.push(String::new());
        lines.push("settings:".to_string());
        let mut fields = vec![
            ("model".to_string(), Shown::Text(physical.model.clone())),
            ("device".to_string(), Shown::Text(physical.device.clone())),
        ];
        fields.extend(physical.shown_settings.iter().cloned());
        lines.extend(field_lines(&fields, 1));
        if physical.backend == "quail" {
            lines.push("  KV dtype=bf16".to_string());
        }
        if let Some(rules) = &request.rules {
            lines.push(String::new());
            lines.extend(rule_lines(rules));
        }
    }
    Ok(lines.join("\n"))
}

// ---------------------------------------------------------------- numbers

/// Python's `format(value, f".{precision}g")`.
pub fn format_g(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let precision = precision.max(1);
    let scientific = format!("{:.*e}", precision - 1, value);
    let (mantissa, exponent) = scientific.split_once('e').expect("an exponent");
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    if exponent < -4 || exponent >= precision as i32 {
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!(
            "{}e{sign}{:02}",
            strip_zeros(mantissa),
            exponent.unsigned_abs()
        );
    }
    let decimals = (precision as i32 - 1 - exponent).max(0) as usize;
    strip_zeros(&format!("{value:.decimals$}")).to_string()
}

fn strip_zeros(text: &str) -> &str {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text
    }
}

/// `text`, a decimal number, with commas between thousands.
fn group_thousands(text: &str) -> String {
    let (sign, unsigned) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", text),
    };
    let (whole, fraction) = match unsigned.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (unsigned, None),
    };
    let digits: Vec<char> = whole.chars().collect();
    let mut grouped = String::new();
    for (index, digit) in digits.iter().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(*digit);
    }
    match fraction {
        Some(fraction) => format!("{sign}{grouped}.{fraction}"),
        None => format!("{sign}{grouped}"),
    }
}

/// Python's `f"{value:,}"`.
fn commas(value: &Value) -> String {
    match value {
        Value::Number(number) if number.is_f64() => {
            group_thousands(&repr_float(number.as_f64().unwrap_or_default()))
        }
        Value::Number(number) => group_thousands(&number.to_string()),
        other => py_str(other),
    }
}

/// Python's `f"{value:,}"` of an integer.
fn commas_int(value: i64) -> String {
    group_thousands(&value.to_string())
}

/// A row count or rate: three significant digits below one, else one
/// decimal with commas, without a trailing `.0`.
fn number(value: f64) -> String {
    if value != 0.0 && value.abs() < 1.0 {
        return format_g(value, 3);
    }
    let text = group_thousands(&format!("{value:.1}"));
    text.strip_suffix(".0").map(str::to_string).unwrap_or(text)
}

fn seconds(value: Option<f64>) -> String {
    match value {
        None => String::new(),
        Some(value) if value < 0.001 => "<1 ms".to_string(),
        Some(value) if value < 1.0 => format!("{} ms", format_g(value * 1000.0, 3)),
        Some(value) => format!("{} s", format_g(value, 3)),
    }
}

fn selectivity(value: Option<f64>, mark: bool) -> String {
    match value {
        None => {
            let default = format!("{}%", format_g(DEFAULT_SELECTIVITY * 100.0, 6));
            if mark {
                default + DEFAULT_MARK
            } else {
                default + " (default)"
            }
        }
        Some(value) => format!("{}%", format_g(value * 100.0, 3)),
    }
}

fn default_note() -> String {
    format!(
        "{DEFAULT_MARK} no selectivity was given; the planner assumes {}%",
        format_g(DEFAULT_SELECTIVITY * 100.0, 6)
    )
}

// ---------------------------------------------------------------- values

/// Python's `repr` of a JSON value, reading each array as a tuple when
/// `tuples` holds, as a frozen dataclass's fields are.
fn repr_as(value: &Value, tuples: bool) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) if number.is_f64() => repr_float(number.as_f64().unwrap_or_default()),
        Value::Number(number) => number.to_string(),
        Value::String(text) => repr_str(text),
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(|item| repr_as(item, tuples)).collect();
            match (tuples, items.len()) {
                (true, 1) => format!("({},)", items[0]),
                (true, _) => format!("({})", items.join(", ")),
                (false, _) => format!("[{}]", items.join(", ")),
            }
        }
        Value::Object(fields) => {
            let fields: Vec<String> = fields
                .iter()
                .map(|(key, value)| format!("{}: {}", repr_str(key), repr_as(value, tuples)))
                .collect();
            format!("{{{}}}", fields.join(", "))
        }
    }
}

/// Python's `str` of a JSON value.
fn py_str(value: &Value) -> String {
    str_as(value, false)
}

fn str_as(value: &Value, tuples: bool) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => repr_as(other, tuples),
    }
}

/// `key=value` lines, nesting a mapping and a list of mappings.
fn field_lines(fields: &[(String, Shown)], depth: usize) -> Vec<String> {
    field_lines_as(fields, depth, false)
}

/// `key=value` lines, each array shown as a tuple when `tuples` holds.
fn field_lines_as(fields: &[(String, Shown)], depth: usize, tuples: bool) -> Vec<String> {
    let pad = "  ".repeat(depth);
    let mut lines = Vec::new();
    for (key, value) in fields {
        let tagged = |pairs: &[(String, Shown)]| pairs.iter().any(|(key, _)| key == "type");
        match value {
            Shown::Map(nested) if !tagged(nested) => {
                lines.push(format!("{pad}{key}:"));
                lines.extend(field_lines_as(nested, depth + 1, tuples));
            }
            Shown::List(items) if matches!(items.first(), Some(Shown::Map(first)) if !tagged(first)) => {
                for (index, item) in items.iter().enumerate() {
                    lines.push(format!("{pad}{key} {}:", index + 1));
                    if let Shown::Map(nested) = item {
                        lines.extend(field_lines_as(nested, depth + 1, tuples));
                    }
                }
            }
            other => lines.push(format!("{pad}{key}={}", other.text(tuples))),
        }
    }
    lines
}

/// The first `count` characters of `text`.
fn prefix(text: &str, count: usize) -> String {
    text.chars().take(count).collect()
}

fn width(text: &str) -> usize {
    text.chars().count()
}

// ---------------------------------------------------------------- prompts

fn short_prompt(prompt: &quail_physical::Prompt) -> String {
    let mut template = prompt.template.clone();
    if width(&template) > PROMPT_CHARS {
        template = prefix(&template, PROMPT_CHARS) + "…";
    }
    format!("PROMPT({})", repr_str(&template))
}

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
fn request_fields(request: &quail_physical::RequestExecutionAttributes) -> Vec<(String, Shown)> {
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

/// A JSON value with each object's keys in written order, as explain's
/// `key=value` lines show it.
#[derive(Debug, Clone, PartialEq)]
pub enum Shown {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    Text(String),
    List(Vec<Shown>),
    Map(Vec<(String, Shown)>),
}

impl<'de> Deserialize<'de> for Shown {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Shown;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a JSON value")
            }
            fn visit_unit<E>(self) -> Result<Shown, E> {
                Ok(Shown::Null)
            }
            fn visit_none<E>(self) -> Result<Shown, E> {
                Ok(Shown::Null)
            }
            fn visit_bool<E>(self, value: bool) -> Result<Shown, E> {
                Ok(Shown::Bool(value))
            }
            fn visit_i64<E>(self, value: i64) -> Result<Shown, E> {
                Ok(Shown::Number(value.into()))
            }
            fn visit_u64<E>(self, value: u64) -> Result<Shown, E> {
                Ok(Shown::Number(value.into()))
            }
            fn visit_f64<E>(self, value: f64) -> Result<Shown, E> {
                Ok(serde_json::Number::from_f64(value).map_or(Shown::Null, Shown::Number))
            }
            fn visit_str<E>(self, value: &str) -> Result<Shown, E> {
                Ok(Shown::Text(value.to_string()))
            }
            fn visit_string<E>(self, value: String) -> Result<Shown, E> {
                Ok(Shown::Text(value))
            }
            fn visit_seq<S: serde::de::SeqAccess<'de>>(
                self,
                mut seq: S,
            ) -> Result<Shown, S::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element::<Shown>()? {
                    items.push(item);
                }
                Ok(Shown::List(items))
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Shown, M::Error> {
                let mut pairs = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, Shown>()? {
                    pairs.push((key, value));
                }
                Ok(Shown::Map(pairs))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

impl From<&Value> for Shown {
    fn from(value: &Value) -> Self {
        match value {
            Value::Null => Shown::Null,
            Value::Bool(found) => Shown::Bool(*found),
            Value::Number(number) => Shown::Number(number.clone()),
            Value::String(text) => Shown::Text(text.clone()),
            Value::Array(items) => Shown::List(items.iter().map(Shown::from).collect()),
            Value::Object(fields) => Shown::Map(
                fields
                    .iter()
                    .map(|(key, value)| (key.clone(), Shown::from(value)))
                    .collect(),
            ),
        }
    }
}

impl Shown {
    /// Python's `repr`, each array a tuple when `tuples` holds and each
    /// object with a class tag the dataclass it encodes.
    fn repr(&self, tuples: bool) -> String {
        match self {
            Shown::Null => "None".to_string(),
            Shown::Bool(true) => "True".to_string(),
            Shown::Bool(false) => "False".to_string(),
            Shown::Number(number) if number.is_f64() => {
                repr_float(number.as_f64().unwrap_or_default())
            }
            Shown::Number(number) => number.to_string(),
            Shown::Text(text) => repr_str(text),
            Shown::List(items) => {
                let items: Vec<String> = items.iter().map(|item| item.repr(tuples)).collect();
                match (tuples, items.len()) {
                    (true, 1) => format!("({},)", items[0]),
                    (true, _) => format!("({})", items.join(", ")),
                    (false, _) => format!("[{}]", items.join(", ")),
                }
            }
            Shown::Map(fields) => {
                let class = fields
                    .iter()
                    .find_map(|(key, value)| match (key.as_str(), value) {
                        ("type", Shown::Text(name)) => name.rsplit('.').next().map(str::to_string),
                        _ => None,
                    });
                match class {
                    Some(class) => {
                        let fields: Vec<String> = fields
                            .iter()
                            .filter(|(key, _)| key != "type")
                            .map(|(key, value)| format!("{key}={}", value.repr(tuples)))
                            .collect();
                        format!("{class}({})", fields.join(", "))
                    }
                    None => {
                        let fields: Vec<String> = fields
                            .iter()
                            .map(|(key, value)| {
                                format!("{}: {}", repr_str(key), value.repr(tuples))
                            })
                            .collect();
                        format!("{{{}}}", fields.join(", "))
                    }
                }
            }
        }
    }

    /// Python's `str`.
    fn text(&self, tuples: bool) -> String {
        match self {
            Shown::Text(text) => text.clone(),
            other => other.repr(tuples),
        }
    }
}

#[derive(Deserialize)]
struct WrittenNode {
    attributes: Shown,
}

/// A node's attributes in declaration order.
fn attributes(node: &PhysicalNode) -> Vec<(String, Shown)> {
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
            let mut fields = pick(&[
                "alias",
                "input_id",
                "num_documents",
                "total_tokens",
                "shard_token_loads",
            ]);
            fields.push((
                "shard_docs".to_string(),
                Shown::List(
                    scan.shard_ranges
                        .iter()
                        .map(|(start, stop)| Shown::Number((stop - start).into()))
                        .collect(),
                ),
            ));
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
struct Graph {
    nodes: Vec<Node>,
    root: PortRef,
    by_id: HashMap<String, usize>,
}

impl Graph {
    fn read(physical: &PhysicalRequest) -> Result<Self, String> {
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
                Some(PhysicalNode::AiFilter(filter)) => inputs
                    .first()
                    .copied()
                    .flatten()
                    .map(|value| value * effective_selectivity(filter.attributes.selectivity)),
                Some(PhysicalNode::Exchange(_)) => single,
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
    filters: HashMap<String, Vec<quail_physical::FilterPredicate>>,
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
        if let Some(alternative) = known.alternative() {
            details.push(format!(
                "passes each row where {}, whatever the model answers",
                condition_text(alternative)
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
                details.push(format!(
                    "pairs kept: {} of the cross product",
                    format_g(join.pair_fraction, 4)
                ));
                format!("HashJoin: {}", on.join(" and "))
            }
            PhysicalNode::Exchange(exchange) => format!(
                "Exchange: {} to the GPU holding its KV",
                exchange.attributes.anchor
            ),
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
fn physical_tree(
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
        let pipelines = quail_physical::build_pipelines(&known, &[]);
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

// ---------------------------------------------------------------- footers

/// The measured totals of one run as labeled lines.
fn run_summary(report: &Value, graph: &Graph, gpus: i64, usd_per_hour: Option<f64>) -> Vec<String> {
    let Some(wall) = report.get("wall_s").and_then(Value::as_f64) else {
        return vec!["query time   not measured".to_string()];
    };
    let mut items = vec![(
        "query time",
        format!("{} (model startup excluded)", seconds(Some(wall))),
    )];
    if let Some(boot) = report.get("boot_s").filter(|boot| !boot.is_null()) {
        let kind = report
            .get("boot_kind")
            .filter(|kind| kind.as_str().is_some_and(|kind| !kind.is_empty()))
            .map(|kind| format!(" ({})", py_str(kind)))
            .unwrap_or_default();
        items.push(("startup", seconds(boot.as_f64()) + &kind));
    }
    let mut pairs: f64 = report["stages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|stage| stage["op"].as_str() == Some("join"))
        .map(|stage| stage.get("tuples").and_then(Value::as_f64).unwrap_or(0.0))
        .sum();
    if pairs == 0.0 {
        pairs = report
            .get("evaluated_document_pairs")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
    }
    let documents: i64 = graph
        .nodes
        .iter()
        .filter_map(|node| match &node.known {
            Some(PhysicalNode::Scan(scan)) => Some(scan.attributes.num_documents),
            _ => None,
        })
        .sum();
    if wall > 0.0 && pairs != 0.0 {
        items.push((
            "throughput",
            format!(
                "{} document pairs/second over {} evaluated pairs",
                number(pairs / wall),
                commas_int(pairs as i64)
            ),
        ));
    } else if wall > 0.0 {
        items.push((
            "throughput",
            format!(
                "{} documents/second over {} input documents",
                number(documents as f64 / wall),
                commas_int(documents)
            ),
        ));
    }
    let mut tokens = format!("{} fresh", commas(&report["fresh_tokens"]));
    if let Some(cached) = report
        .get("cached_tokens")
        .filter(|cached| !cached.is_null())
    {
        tokens += &format!(", {} read from KV", commas(cached));
    }
    items.push(("tokens", tokens));
    if let Some(price) = usd_per_hour {
        let cost = wall / 3600.0 * gpus as f64 * price;
        items.push((
            "GPU cost",
            format!("${cost:.4} per query ({gpus} GPU at ${price:.4}/hour, startup excluded)"),
        ));
    }
    let label_width = items
        .iter()
        .map(|(label, _)| width(label))
        .max()
        .unwrap_or(0);
    items
        .into_iter()
        .map(|(label, value)| format!("{label:<label_width$}   {value}"))
        .collect()
}

/// The rule categories and the order used during search.
fn rule_lines(rules: &[(String, String)]) -> Vec<String> {
    let names = |kind: &str| {
        let names: Vec<&str> = rules
            .iter()
            .filter(|(_, found)| found == kind)
            .map(|(name, _)| name.as_str())
            .collect();
        if names.is_empty() {
            "none".to_string()
        } else {
            names.join(", ")
        }
    };
    vec![
        "optimizer rules:".to_string(),
        format!("  logical: {}", names("logical")),
        format!("  physical: {}", names("physical")),
        "  search: rewrite logical candidates, then lower each alternative".to_string(),
        format!("  refine each graph in order: {}", names("physical")),
        "  selection: lowest estimated cost after graph refinement".to_string(),
    ]
}

/// Whether a model keeps documents' KV for later questions: every model
/// but one whose attention is set attention.
fn reuses_kv(spec: Option<&Value>) -> bool {
    spec.is_none_or(|spec| spec["attention_mask"].as_str() != Some("set"))
}

/// The architecture, weight precision, and kernel path lines of a model,
/// or none without its spec.
///
/// The gated delta net recurrence kernel is read from QUAIL_GDN_PREFILL
/// in this process.
fn architecture_lines(spec: Option<&Value>) -> Vec<String> {
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut kinds: Vec<(String, usize)> = Vec::new();
    for kind in spec["layer_kinds"].as_array().into_iter().flatten() {
        let kind = py_str(kind);
        match kinds.iter_mut().find(|(found, _)| *found == kind) {
            Some((_, count)) => *count += 1,
            None => kinds.push((kind, 1)),
        }
    }
    let has = |name: &str| kinds.iter().any(|(kind, _)| kind == name);
    let layers = if kinds.is_empty() {
        format!("{} layers", py_str(&spec["layers"]))
    } else {
        kinds
            .iter()
            .map(|(kind, count)| format!("{count} {kind}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let precision = py_str(&spec["weight_precision"]);
    let mut gemm = if precision == "fp8" {
        "fp8 GEMM (DeepGEMM)".to_string()
    } else {
        "bf16 GEMM (cuBLAS)".to_string()
    };
    let arch = py_str(&spec["arch"]);
    if arch == "qwen3_5" {
        gemm = "vLLM per-layer GEMM".to_string();
        if let Some(recurrent) = spec
            .get("gdn_weight_precision")
            .filter(|value| !value.is_null())
        {
            gemm += &format!(" (recurrent projections {})", py_str(recurrent));
        }
    }
    let mut paths = vec![format!("linear layers: {gemm}")];
    if spec["attention_mask"].as_str() == Some("set") {
        paths.push(
            "attention: FlashAttention, set attention on the full layers as three calls \
             merged by log-sum-exp, no KV kept"
                .to_string(),
        );
    } else if has("full_attention") || has("sliding_attention") {
        paths.push("attention: FlashAttention over paged KV".to_string());
    }
    if has("gated_delta_net") {
        let requested = std::env::var("QUAIL_GDN_PREFILL").unwrap_or_else(|_| "auto".into());
        paths.push(format!(
            "gated_delta_net: Triton convolution, recurrence={requested} (QUAIL_GDN_PREFILL)"
        ));
    }
    if has("short_conv") {
        paths.push("short_conv: Triton convolution".to_string());
    }
    vec![
        format!("architecture={arch} ({layers}), weights={precision}"),
        format!("kernels: {}", paths.join("; ")),
    ]
}

#[cfg(test)]
mod tests {
    use super::{format_g, group_thousands, number, seconds};

    #[test]
    fn numbers_format_as_python_formats_them() {
        assert_eq!(format_g(0.123456, 3), "0.123");
        assert_eq!(format_g(12345.0, 3), "1.23e+04");
        assert_eq!(format_g(0.00001234, 3), "1.23e-05");
        assert_eq!(format_g(20.0, 6), "20");
        assert_eq!(format_g(100.0, 3), "100");
        assert_eq!(format_g(0.5, 4), "0.5");
        assert_eq!(group_thousands("1234567.5"), "1,234,567.5");
        assert_eq!(group_thousands("-1234"), "-1,234");
        assert_eq!(number(12.5), "12.5");
        assert_eq!(number(10.0), "10");
        assert_eq!(number(1234.0), "1,234");
        assert_eq!(number(0.25), "0.25");
        assert_eq!(seconds(Some(0.0005)), "<1 ms");
        assert_eq!(seconds(Some(0.0123)), "12.3 ms");
        assert_eq!(seconds(Some(2.5)), "2.5 s");
    }
}

#[cfg(test)]
mod repr_tests {
    use crate::expressions::repr_str;

    #[test]
    fn a_repr_escapes_what_python_does_not_print() {
        assert_eq!(repr_str("a\u{a0}b"), "'a\\xa0b'");
        assert_eq!(repr_str("\u{200b}"), "'\\u200b'");
        assert_eq!(repr_str("café — ok"), "'café — ok'");
        assert_eq!(repr_str("it's"), "\"it's\"");
    }
}
