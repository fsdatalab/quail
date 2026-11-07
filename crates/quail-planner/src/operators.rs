//! A region's operators as the planner reads them.
//!
//! The physical planner reads a region only through `Operators`, which
//! `quail_datafusion::derive` reads from the region's DataFusion plan.
//! They cross into Python as JSON. Each value and condition the
//! executor computes is an `Expression`, which DataFusion evaluates.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use quail_physical::{
    Asked, Equality, ExtractStruct, FilterPredicate, ModelCall, Projected, ValueExpression,
    tagged_struct,
};
use serde_json::Value;

use crate::expressions::{PROBABILITIES_SUFFIX, SPAN_SUFFIX, call_aliases, column_references};

tagged_struct! {
    /// An Arrow schema, as IPC bytes in base64.
    Schema(SchemaFields, SchemaTagged) = "pyarrow.Schema" {
        ipc: String,
    }
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

/// The names of the source columns a document column reads.
pub fn source_columns(column: &ScanColumn) -> Vec<String> {
    match column {
        ScanColumn::Unnamed(name) => vec![name.clone()],
        ScanColumn::Expression(expression) => {
            let mut found: Vec<String> = Vec::new();
            for reference in column_references(expression) {
                if !found.contains(&reference.column) {
                    found.push(reference.column);
                }
            }
            found
        }
    }
}

/// One table a region reads, with its regular predicates and the
/// columns its operators read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scan {
    pub provider: String,
    pub alias: String,
    pub column: ScanColumn,
    pub columns: Vec<String>,
    /// Conditions on the table's own columns, each an `Expression`.
    pub predicates: Vec<ValueExpression>,
    pub schema: Option<Schema>,
}

/// A Filter on the columns that column operators compute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnTest {
    /// The column operator the Filter follows, the last one it reads.
    pub name: String,
    pub call: ModelCall,
    /// The condition, an `Expression` over the table's columns and the
    /// answer columns.
    pub condition: ValueExpression,
    /// The labels a test of a classification's label accepts, when it is
    /// `label IN (...)`.
    pub labels: Option<Value>,
    pub selectivity: Option<f64>,
    /// Its position among the table's filters, after the table's AI.IF
    /// predicates, the label tests first.
    pub position: usize,
    /// Whether it runs before any join.
    pub before_joins: bool,
}

/// One model call over its input rows, adding its answer column `name`:
/// `AI.CLASSIFY`, `AI.EXTRACT`, `AI.SAE_FEATURES`, or `AI.MAP`, as the
/// call's kind says.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticColumn {
    pub call: ModelCall,
    pub name: String,
    /// Whether it runs above a join.
    pub above_joins: bool,
    /// The condition, an `Expression` over the table's columns, that
    /// says which rows the model is asked about; every other row's
    /// answer is NULL. None asks about every row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard: Option<ValueExpression>,
}

impl SemanticColumn {
    /// The table alias its answer columns belong to.
    pub fn alias(&self) -> String {
        call_aliases(&self.call)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// The columns it adds, its answer first.
    pub fn added_columns(&self) -> Vec<String> {
        match self.call.kind.as_str() {
            "label" if self.call.probabilities => vec![
                self.name.clone(),
                format!("{}{PROBABILITIES_SUFFIX}", self.name),
            ],
            "extract" => vec![self.name.clone(), format!("{}{SPAN_SUFFIX}", self.name)],
            _ => vec![self.name.clone()],
        }
    }
}

/// A model call tested on the pairs of a join.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticJoin {
    pub predicate: Asked,
    /// "full", "exists", or "anti".
    pub semantics: String,
    pub selectivity: Option<f64>,
    /// The table alias whose KV is kept.
    pub anchor: Option<String>,
    /// The equality conditions of the joins below it, in written order.
    pub on: Vec<Equality>,
    /// The aliases of the rows it extends: everything below its new tables.
    pub outer_aliases: Vec<String>,
}

/// The columns a region returns.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    /// Value expressions and Aliases.
    pub columns: Vec<Projected>,
    pub structs: Vec<ExtractStruct>,
}

/// The operators of one plan, in the order the plan was written.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Operators {
    pub scans: Vec<Scan>,
    /// alias -> its AI.IF predicates, by alias first seen.
    pub filters: Vec<(String, Vec<FilterPredicate>)>,
    /// alias -> whether each of its AI.IF predicates runs before any
    /// join, in the order of `filters`.
    pub filters_before_joins: Vec<(String, Vec<bool>)>,
    pub joins: Vec<SemanticJoin>,
    /// The named model calls SELECT returns, in SELECT order.
    pub projections: Vec<ValueExpression>,
    pub classifies: Vec<SemanticColumn>,
    pub extracts: Vec<SemanticColumn>,
    pub sae_features: Vec<SemanticColumn>,
    pub maps: Vec<SemanticColumn>,
    /// alias -> its label tests, then its answer tests.
    pub column_filters: Vec<(String, Vec<ColumnTest>)>,
    /// The columns the region returns: the operators above it in
    /// DataFusion read them.
    pub projection: Projection,
    /// The most rows the region returns: the fetch of its last AI
    /// operator, or of a LIMIT above the region, which a LIMIT and its
    /// OFFSET need. The executor stops asking the model once that many
    /// rows pass.
    pub fetch: Option<i64>,
    /// alias -> the DISTINCT stop key of its AI.IF predicates.
    pub stop_keys: BTreeMap<String, Vec<String>>,
}

impl Operators {
    /// These operators returning at most `fetch` rows, when DataFusion
    /// passes a LIMIT above the region down to it, so the model can stop
    /// once enough rows pass.
    pub fn limited(mut self, fetch: i64) -> Self {
        self.fetch = Some(self.fetch.map_or(fetch, |found| found.min(fetch)));
        self
    }

    /// The column operators of every kind: classifications,
    /// extractions, SAE features, then maps.
    pub fn columns(&self) -> impl Iterator<Item = &SemanticColumn> {
        self.classifies
            .iter()
            .chain(&self.extracts)
            .chain(&self.sae_features)
            .chain(&self.maps)
    }
}
