//! The expressions physical nodes carry, from `quail.common.expressions`.
//!
//! A value or condition the executor computes is an [`Expression`],
//! which DataFusion evaluates; the other types describe the model calls
//! and the columns the planner reasons about.
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
    #[serde(rename = "quail.common.expressions.Expression")]
    Value(Expression),
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

impl ColumnRef {
    /// The column's name, with its table alias when `qualified` and it
    /// has one.
    pub fn name(&self, qualified: bool) -> String {
        if qualified && !self.alias.is_empty() {
            format!("{}.{}", self.alias, self.column)
        } else {
            self.column.clone()
        }
    }
}

/// One field of a struct value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GetField {
    pub input: Box<ValueExpression>,
    pub field: String,
}

/// A value or condition DataFusion computes from one row's columns.
///
/// `encoded` is DataFusion's logical expression as datafusion-proto
/// bytes in base64. It reads the column named `#i` where the rows hold
/// `columns[i]`, so a planner can rename the columns it reads without
/// DataFusion. `types` holds the Arrow type of each column as Arrow
/// writes it, such as `Int64` or `Utf8`. `text` is how `explain()`
/// writes the expression, with `{i}` for the name of `columns[i]` and
/// `{{` and `}}` for braces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expression {
    pub encoded: String,
    pub columns: Vec<ColumnRef>,
    /// Empty in an expression saved before types were written.
    #[serde(default)]
    pub types: Vec<String>,
    pub text: String,
}

impl Expression {
    /// The expression's text, each column named with its table alias
    /// when `qualified`.
    pub fn text_with(&self, qualified: bool) -> String {
        let mut out = String::new();
        let mut chars = self.text.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '{' if chars.peek() == Some(&'{') => {
                    chars.next();
                    out.push('{');
                }
                '}' if chars.peek() == Some(&'}') => {
                    chars.next();
                    out.push('}');
                }
                '{' => {
                    let mut digits = String::new();
                    for digit in chars.by_ref() {
                        if digit == '}' {
                            break;
                        }
                        digits.push(digit);
                    }
                    match digits
                        .parse::<usize>()
                        .ok()
                        .and_then(|at| self.columns.get(at))
                    {
                        Some(column) => out.push_str(&column.name(qualified)),
                        None => out.push_str(&format!("{{{digits}}}")),
                    }
                }
                other => out.push(other),
            }
        }
        out
    }
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

/// An `AI.SCORE` call compared with a constant. `test` is an
/// `Expression` that reads the score as its one column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreTest {
    pub call: ModelCall,
    pub test: ValueExpression,
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
    #[serde(rename = "quail.common.expressions.ScoreTest")]
    Compare(ScoreTest),
}

tagged_struct! {
    /// One `AI.IF` predicate of a semantic filter.
    FilterPredicate(FilterPredicateFields, FilterPredicateTagged) = "quail.common.expressions.FilterPredicate" | "quail.logical.expressions.FilterPredicate" {
        expression: Asked,
        /// The fraction of documents that pass, if the query gives one.
        selectivity: Option<f64>,
        /// The condition that says which documents the model is asked
        /// about; None asks about every document.
        guard: Option<ValueExpression>,
        /// The other terms of an OR the predicate is one term of: a
        /// document where it is TRUE passes whatever the model answers,
        /// and a document the guard leaves out passes only then. None
        /// for a predicate that is not in an OR.
        alternative: Option<ValueExpression>,
        /// The column the predicate adds with its answer, when it keeps
        /// every row it reads instead of dropping the rows it rejects:
        /// TRUE or FALSE for a document it asks about, NULL for one its
        /// guard leaves out. None for a predicate that drops rows.
        answer: Option<String>,
        /// The estimated fraction of documents where the guard is TRUE;
        /// None without a guard, or for the default selectivity.
        guard_selectivity: Option<f64>,
        /// Whether the predicate is `NOT AI.IF(...)`: a document passes
        /// when the model answers no, and `selectivity` is the fraction
        /// of documents that do.
        negated: bool = default,
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
    /// The struct column of an `AI.EXTRACT` call and its fields.
    ExtractStruct(ExtractStructFields, ExtractStructTagged) = "quail.common.expressions.ExtractStruct" | "quail.logical.expressions.ExtractStruct" {
        name: String,
        fields: Vec<String>,
    }
}
