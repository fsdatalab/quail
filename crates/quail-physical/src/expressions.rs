//! The expressions physical nodes carry, from `quail.common.expressions`.
//!
//! Each variant is tagged with its Python class, and also accepts the
//! tag the class had before it moved from `quail.logical.expressions`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tagged_struct;

/// A value computed from one row's columns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ValueExpression {
    #[serde(
        rename = "quail.common.expressions.ColumnRef",
        alias = "quail.logical.expressions.ColumnRef"
    )]
    ColumnRef(ColumnRef),
    #[serde(
        rename = "quail.common.expressions.GetField",
        alias = "quail.logical.expressions.GetField"
    )]
    GetField(GetField),
    #[serde(
        rename = "quail.common.expressions.ScalarFunction",
        alias = "quail.logical.expressions.ScalarFunction"
    )]
    ScalarFunction(ScalarFunction),
    #[serde(
        rename = "quail.common.expressions.Literal",
        alias = "quail.logical.expressions.Literal"
    )]
    Literal(Literal),
    #[serde(
        rename = "quail.common.expressions.Alias",
        alias = "quail.logical.expressions.Alias"
    )]
    Alias(Alias),
    #[serde(
        rename = "quail.common.expressions.ModelCall",
        alias = "quail.logical.expressions.ModelCall"
    )]
    ModelCall(ModelCall),
}

/// One column of one table in the query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnRef {
    pub alias: String,
    pub provider: String,
    pub column: String,
}

/// One field of a struct value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GetField {
    pub input: Box<ValueExpression>,
    pub field: String,
}

/// An Arrow compute function applied to values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarFunction {
    pub function: String,
    pub args: Vec<ValueExpression>,
}

/// A constant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Literal {
    pub value: Value,
}

/// A value under the name a query gives it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Alias {
    pub expression: Box<ValueExpression>,
    pub name: String,
}

tagged_struct! {
    /// The text of one model call, laid out for a model's prompt format.
    Prompt(PromptFields, PromptTagged) = "quail.common.expressions.Prompt" | "quail.logical.expressions.Prompt" {
        template: String,
        /// Column references in placeholder order.
        args: Vec<ValueExpression>,
        preamble: String,
        tail: String,
        frame: String,
        frame_text: String,
        kind: String,
        layout: String,
        tail_segments: Vec<String>,
        /// Per placeholder: alias, its label text, its frame text.
        label_texts: Vec<(String, String, String)>,
        label_prefix: String,
        categories: Vec<String>,
        descriptions: Vec<String>,
        task_description: String,
        turn: (String, String),
        source: String,
        letters: Vec<String>,
    }
}

/// One prompt asked of every row; the model answers with a value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCall {
    pub prompt: Box<Prompt>,
    pub kind: String,
    pub labels: Value,
    pub descriptions: Value,
    pub probabilities: bool,
    pub question: String,
    pub pooling: String,
    pub max_answer_tokens: i64,
}

/// A condition on a row: a comparison, or a NOT, AND, or OR of them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Condition {
    #[serde(
        rename = "quail.common.expressions.RegularPredicate",
        alias = "quail.logical.expressions.RegularPredicate"
    )]
    Compare(Box<RegularPredicate>),
    #[serde(
        rename = "quail.common.expressions.Not",
        alias = "quail.logical.expressions.Not"
    )]
    Not(Not),
    #[serde(
        rename = "quail.common.expressions.And",
        alias = "quail.logical.expressions.And"
    )]
    And(Connective),
    #[serde(
        rename = "quail.common.expressions.Or",
        alias = "quail.logical.expressions.Or"
    )]
    Or(Connective),
}

/// One comparison of a value with a constant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegularPredicate {
    pub expression: ValueExpression,
    pub comparison: String,
    pub value: Value,
}

/// The negation of a condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Not {
    pub operand: Box<Condition>,
}

/// The operands of an AND or an OR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Connective {
    pub operands: Vec<Condition>,
}

tagged_struct! {
    /// One window function call.
    WindowCall(WindowCallFields, WindowCallTagged) = "quail.common.expressions.WindowCall" | "quail.logical.expressions.WindowCall" {
        function: String,
        argument: Option<String>,
        partition: Vec<String>,
        /// Sort keys: column, descending, nulls first.
        order: Vec<(String, bool, bool)>,
        name: String,
    }
}

/// A value expression or a column name, as a projection lists them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Projected {
    Name(String),
    Expression(ValueExpression),
}

/// A model call, written with its class tag where a field holds one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TaggedModelCall {
    #[serde(
        rename = "quail.common.expressions.ModelCall",
        alias = "quail.logical.expressions.ModelCall"
    )]
    Call(ModelCall),
}

/// What a semantic filter or join asks: a yes-or-no model call, or a
/// comparison of an `AI.SCORE` call with a constant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Asked {
    #[serde(
        rename = "quail.common.expressions.ModelCall",
        alias = "quail.logical.expressions.ModelCall"
    )]
    Call(ModelCall),
    #[serde(
        rename = "quail.common.expressions.RegularPredicate",
        alias = "quail.logical.expressions.RegularPredicate"
    )]
    Compare(RegularPredicate),
}

tagged_struct! {
    /// One `AI.IF` predicate of a semantic filter.
    FilterPredicate(FilterPredicateFields, FilterPredicateTagged) = "quail.common.expressions.FilterPredicate" | "quail.logical.expressions.FilterPredicate" {
        expression: Asked,
        /// The fraction of documents that pass, if the query gives one.
        selectivity: Option<f64>,
    }
}

tagged_struct! {
    /// An equality between a value of one table and a value of another.
    Equality(EqualityFields, EqualityTagged) = "quail.common.expressions.Equality" | "quail.logical.expressions.Equality" {
        left: ValueExpression,
        right: ValueExpression,
    }
}

tagged_struct! {
    /// One aggregate function call.
    AggregateCall(AggregateCallFields, AggregateCallTagged) = "quail.common.expressions.AggregateCall" | "quail.logical.expressions.AggregateCall" {
        function: String,
        argument: Option<String>,
        name: String,
    }
}

tagged_struct! {
    /// One `HAVING` comparison of an aggregate with a constant.
    HavingTest(HavingTestFields, HavingTestTagged) = "quail.common.expressions.HavingTest" | "quail.logical.expressions.HavingTest" {
        aggregate: AggregateCall,
        comparison: String,
        value: Value,
    }
}

/// What a sort key orders by.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SortExpression {
    Value(ValueExpression),
    Aggregate(AggregateCall),
}

tagged_struct! {
    /// One `ORDER BY` key.
    SortKey(SortKeyFields, SortKeyTagged) = "quail.common.expressions.SortKey" | "quail.logical.expressions.SortKey" {
        expression: SortExpression,
        descending: bool,
        nulls_first: bool,
    }
}

tagged_struct! {
    /// The struct column of an `AI.EXTRACT` call and its fields.
    ExtractStruct(ExtractStructFields, ExtractStructTagged) = "quail.common.expressions.ExtractStruct" | "quail.logical.expressions.ExtractStruct" {
        name: String,
        fields: Vec<String>,
    }
}
