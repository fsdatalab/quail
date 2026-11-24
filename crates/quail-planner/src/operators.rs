use crate::constants::{DEFAULT_SELECTIVITY, PROBABILITIES_SUFFIX, SPAN_SUFFIX};
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::physical::{
    Asked, Equality, ExtractStruct, FilterPredicate, ModelCall, Projected, ValueExpression,
};
use crate::tagged_struct;
use serde_json::Value;

use crate::expressions::{call_aliases, column_references};

tagged_struct! {
    Schema(SchemaFields, SchemaTagged) = "pyarrow.Schema" {
        ipc: String,
    }
}

impl Schema {
    pub fn names(&self) -> Result<Vec<String>, crate::error::CompileError> {
        use crate::error::CompileError;
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&self.ipc)
            .map_err(|error| CompileError::new(format!("invalid schema base64: {error}")))?;
        let schema = arrow_ipc::convert::try_schema_from_ipc_buffer(&bytes)
            .map_err(|error| CompileError::new(format!("invalid IPC schema: {error}")))?;
        Ok(schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScanColumn {
    Unnamed(String),
    Expression(ValueExpression),
}

pub const INPUT_SEPARATOR: &str = "@";

pub fn table_name(provider: &str) -> &str {
    provider.split(INPUT_SEPARATOR).next().unwrap_or(provider)
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scan {
    pub provider: String,
    pub alias: String,
    pub column: ScanColumn,
    pub columns: Vec<String>,
    pub predicates: Vec<ValueExpression>,
    pub schema: Option<Schema>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnTest {
    pub name: String,
    pub call: ModelCall,
    pub condition: ValueExpression,
    pub labels: Option<Value>,
    pub selectivity: Option<f64>,
    pub position: usize,
    pub before_joins: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticColumn {
    pub call: ModelCall,
    pub name: String,
    pub above_joins: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precondition: Option<ValueExpression>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precondition_selectivity: Option<f64>,
}

impl SemanticColumn {
    pub fn asked_fraction(&self) -> f64 {
        precondition_fraction(self.precondition.as_ref(), self.precondition_selectivity)
    }

    pub fn alias(&self) -> String {
        call_aliases(&self.call)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticJoin {
    pub predicate: Asked,
    pub semantics: String,
    pub selectivity: Option<f64>,
    pub anchor: Option<String>,
    pub on: Vec<Equality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<ValueExpression>,
    pub outer_aliases: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    pub columns: Vec<Projected>,
    pub structs: Vec<ExtractStruct>,
    #[serde(default)]
    pub spans: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Operators {
    pub scans: Vec<Scan>,
    pub filters: Vec<(String, Vec<FilterPredicate>)>,
    pub filters_before_joins: Vec<(String, Vec<bool>)>,
    pub joins: Vec<SemanticJoin>,
    pub projections: Vec<ValueExpression>,
    pub classifies: Vec<SemanticColumn>,
    pub extracts: Vec<SemanticColumn>,
    pub sae_features: Vec<SemanticColumn>,
    pub maps: Vec<SemanticColumn>,
    pub column_filters: Vec<(String, Vec<ColumnTest>)>,
    pub projection: Projection,
    pub fetch: Option<i64>,
    pub stop_keys: BTreeMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub num_key_survivors: BTreeMap<String, i64>,
}

impl Operators {
    pub fn writes_answers(&self) -> bool {
        !(self.maps.is_empty() && self.extracts.is_empty())
    }

    pub fn limited(mut self, fetch: i64) -> Self {
        self.fetch = Some(self.fetch.map_or(fetch, |found| found.min(fetch)));
        self
    }

    pub fn filter_answers(&self) -> BTreeSet<(&str, &str)> {
        self.filters
            .iter()
            .flat_map(|(alias, predicates)| {
                predicates.iter().filter_map(move |predicate| {
                    Some((alias.as_str(), predicate.answer.as_deref()?))
                })
            })
            .collect()
    }

    pub fn columns(&self) -> impl Iterator<Item = &SemanticColumn> {
        self.classifies
            .iter()
            .chain(&self.extracts)
            .chain(&self.sae_features)
            .chain(&self.maps)
    }
}

pub fn effective_selectivity(selectivity: Option<f64>) -> f64 {
    selectivity.unwrap_or(DEFAULT_SELECTIVITY)
}

pub fn precondition_fraction(
    precondition: Option<&crate::physical::ValueExpression>,
    selectivity: Option<f64>,
) -> f64 {
    match precondition {
        Some(_) => effective_selectivity(selectivity),
        None => 1.0,
    }
}

pub fn predicate_asked_fraction(predicate: &crate::physical::FilterPredicate) -> f64 {
    precondition_fraction(
        predicate.precondition.as_ref(),
        predicate.precondition_selectivity,
    )
}
