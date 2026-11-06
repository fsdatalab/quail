//! Quail's logical plan, as `quail/physical/codec.py` writes it.
//!
//! A node is written `{"type": <its Python class>, ...fields}`, with
//! its inputs nested in its fields.

use quail_physical::{
    Asked, Condition, Equality, ExtractStruct, FilterPredicate, Projected, SortKey,
    TaggedModelCall, ValueExpression, tagged_struct,
};
use serde::{Deserialize, Serialize};

tagged_struct! {
    /// An Arrow schema, as IPC bytes in base64.
    Schema(SchemaFields, SchemaTagged) = "pyarrow.Schema" {
        ipc: String,
    }
}

/// The document column of a scan: an expression, or empty before a
/// prompt names one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScanColumn {
    Unnamed(String),
    Expression(ValueExpression),
}

/// What separates a table's name from its alias in the name of a model
/// region's input that holds some of the table's rows, as in
/// `reviews__r`.
pub const INPUT_SEPARATOR: &str = "__";

/// The registered table a scanned provider reads: the provider itself,
/// or the table a region's input `<table>__<alias>` holds rows of.
pub fn table_name(provider: &str) -> &str {
    provider.split(INPUT_SEPARATOR).next().unwrap_or(provider)
}

/// One logical operator and its inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LogicalNode {
    /// Reads one table, applying its regular predicates.
    #[serde(rename = "quail.logical.nodes.scan.Scan")]
    Scan(Scan),
    /// `AI.IF` predicates on one table, in written order.
    #[serde(rename = "quail.logical.nodes.filter.SemanticFilter")]
    SemanticFilter(SemanticFilter),
    /// A regular condition, over columns operators below compute.
    #[serde(rename = "quail.logical.nodes.filter.Filter")]
    Filter(Filter),
    /// The cross product of two inputs, narrowed by equalities.
    #[serde(rename = "quail.logical.nodes.join.Join")]
    Join(Join),
    /// A model call tested on the pairs of a join.
    #[serde(rename = "quail.logical.nodes.join.SemanticJoin")]
    SemanticJoin(SemanticJoin),
    /// The query's returned columns.
    #[serde(rename = "quail.logical.nodes.relational.Project")]
    Project(Project),
    /// `ORDER BY`, `DISTINCT`, `OFFSET`, and `LIMIT`.
    #[serde(rename = "quail.logical.nodes.relational.Result")]
    Result(ResultNode),
    /// `AI.CLASSIFY`.
    #[serde(rename = "quail.logical.nodes.classify.SemanticClassify")]
    SemanticClassify(SemanticColumn),
    /// `AI.EXTRACT`.
    #[serde(rename = "quail.logical.nodes.extract.SemanticExtract")]
    SemanticExtract(SemanticColumn),
    /// `AI.MAP`.
    #[serde(rename = "quail.logical.nodes.map.SemanticMap")]
    SemanticMap(SemanticColumn),
    /// `AI.SAE_FEATURES`.
    #[serde(rename = "quail.logical.nodes.sae_features.SemanticSaeFeatures")]
    SemanticSaeFeatures(SemanticColumn),
    /// One row per item of each listed list column.
    #[serde(rename = "quail.logical.nodes.unnest.Unnest")]
    Unnest(Unnest),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scan {
    pub provider: String,
    pub alias: String,
    pub column: ScanColumn,
    pub columns: Vec<String>,
    pub predicates: Vec<Condition>,
    pub schema: Option<Schema>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticFilter {
    pub input: Box<LogicalNode>,
    pub predicates: Vec<FilterPredicate>,
    pub stop_key: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Filter {
    pub input: Box<LogicalNode>,
    pub condition: Condition,
    pub selectivity: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Join {
    pub left: Box<LogicalNode>,
    pub right: Box<LogicalNode>,
    pub on: Vec<Equality>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticJoin {
    pub input: Box<LogicalNode>,
    pub predicate: Asked,
    /// "full", "exists", or "anti".
    pub semantics: String,
    pub selectivity: Option<f64>,
    /// The table alias whose KV is kept.
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Project {
    pub input: Box<LogicalNode>,
    /// Value expressions, Aliases, and the names of columns an unnest adds.
    pub columns: Vec<Projected>,
    pub structs: Vec<ExtractStruct>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultNode {
    pub input: Box<LogicalNode>,
    pub limit: Option<i64>,
    pub order: Vec<SortKey>,
    pub offset: i64,
    pub distinct: bool,
}

/// One model call over its input rows, adding its answer column `name`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticColumn {
    pub input: Box<LogicalNode>,
    pub call: TaggedModelCall,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Unnest {
    pub input: Box<LogicalNode>,
    pub columns: Vec<String>,
    pub outer: bool,
}

tagged_struct! {
    /// A query's logical plan.
    LogicalPlan(LogicalPlanFields, LogicalPlanTagged) = "quail.logical.plan.LogicalPlan" {
        root: LogicalNode,
    }
}

/// Read a logical plan from JSON.
pub fn read_logical_plan(text: &str) -> serde_json::Result<LogicalPlan> {
    serde_json::from_str(text)
}

/// Write a logical plan as JSON.
pub fn write_logical_plan(plan: &LogicalPlan) -> serde_json::Result<String> {
    serde_json::to_string(plan)
}

impl Schema {
    /// The schema's column names.
    ///
    /// # Panics
    ///
    /// The IPC bytes do not hold a schema.
    pub fn names(&self) -> Vec<String> {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&self.ipc)
            .expect("base64 schema bytes");
        let schema =
            arrow_ipc::convert::try_schema_from_ipc_buffer(&bytes).expect("an IPC schema message");
        schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }
}
