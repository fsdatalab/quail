//! Reading and checking the expressions of a region's operators: the
//! columns they read, their names and text, and the model calls they
//! hold. DataFusion evaluates the values and conditions.

use std::fmt;

use quail_physical::{
    Alias, Asked, ColumnRef, Equality, ExtractStruct, ModelCall, Projected, ValueExpression,
};
use serde_json::Value;

/// The phrase every refusal of a query shape the binder does not read
/// ends with: DataFusion runs such operators itself, around the model
/// regions it claims.
pub const UNSUPPORTED: &str = "is outside what Quail binds from DataFusion";

/// A query that is malformed or outside the language, or a plan the
/// configuration cannot run, with its refusal.
#[derive(Debug, Clone, PartialEq)]
pub struct CompileError(pub String, pub Option<Box<Refusal>>);

impl CompileError {
    /// An error with the message and no refusal.
    pub fn new(message: impl Into<String>) -> Self {
        CompileError(message.into(), None)
    }

    /// An error carrying a refusal; its message is the first reason.
    pub fn refused(refusal: Refusal) -> Self {
        CompileError(refusal.reasons[0].clone(), Some(Box::new(refusal)))
    }
}

/// A plan the configuration cannot run within its limits, as `Refusal`.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub reasons: Vec<String>,
    pub constraint: String,
    pub needed: f64,
    pub available: f64,
    pub unit: String,
}

impl Refusal {
    /// A refusal with one reason.
    pub fn new(
        reason: impl Into<String>,
        constraint: &str,
        needed: f64,
        available: f64,
        unit: &str,
    ) -> Self {
        Refusal {
            reasons: vec![reason.into()],
            constraint: constraint.to_string(),
            needed,
            available,
            unit: unit.to_string(),
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CompileError {}

/// Return a `CompileError` with the message.
pub fn compile_error<T>(message: impl Into<String>) -> Result<T, CompileError> {
    Err(CompileError::new(message))
}

pub const SPAN_SUFFIX: &str = "_span";
pub const PROBABILITIES_SUFFIX: &str = "_probabilities";
pub const DESCRIPTION_WORDS: usize = 25;
pub const SCORE_COMPARISONS: [&str; 4] = ["<", "<=", ">", ">="];
pub const MODEL_CALL_KINDS: [&str; 6] = ["boolean", "score", "label", "extract", "features", "map"];
pub const SAE_POOLINGS: [&str; 2] = ["last", "max"];
// ---------------------------------------------------------------- Python text

/// Python's `repr` of a string.
/// Whether Python's `str.isprintable` holds for `c`: not a control,
/// format, private use, line or paragraph separator, or space separator
/// other than the ASCII space. Unassigned code points count as printable.
fn python_printable(c: char) -> bool {
    let code = c as u32;
    let space_separator = matches!(
        code,
        0xa0 | 0x1680 | 0x2000..=0x200a | 0x202f | 0x205f | 0x3000
    );
    let format = matches!(
        code,
        0xad | 0x600..=0x605
            | 0x61c
            | 0x6dd
            | 0x70f
            | 0x890..=0x891
            | 0x8e2
            | 0x180e
            | 0x200b..=0x200f
            | 0x202a..=0x202e
            | 0x2060..=0x2064
            | 0x2066..=0x206f
            | 0xfeff
            | 0xfff9..=0xfffb
            | 0x110bd
            | 0x110cd
            | 0x13430..=0x1343f
            | 0x1bca0..=0x1bca3
            | 0x1d173..=0x1d17a
            | 0xe0001
            | 0xe0020..=0xe007f
    );
    let private_use = matches!(
        code,
        0xe000..=0xf8ff | 0xf0000..=0xffffd | 0x100000..=0x10fffd
    );
    let separator = matches!(code, 0x2028 | 0x2029);
    !(c.is_control() || space_separator || format || private_use || separator)
}

pub fn repr_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !python_printable(c) => {
                let code = c as u32;
                if code < 0x100 {
                    out.push_str(&format!("\\x{code:02x}"));
                } else if code < 0x10000 {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    out.push_str(&format!("\\U{code:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `repr` of a float.
pub fn repr_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let exponential = format!("{value:e}");
    let (mantissa, exponent) = exponential.split_once('e').expect("an exponent");
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if negative { "-" } else { "" };
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let text = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{text}")
    } else {
        let body = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let exponent_sign = if exponent < 0 { "-" } else { "+" };
        format!("{sign}{body}e{exponent_sign}{:02}", exponent.abs())
    }
}

/// Python's `repr` of a literal: a JSON array stands for a tuple.
pub fn repr_value(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => match number.as_i64() {
            Some(integer) => integer.to_string(),
            None => match number.as_u64() {
                Some(integer) => integer.to_string(),
                None => repr_float(number.as_f64().expect("a float")),
            },
        },
        Value::String(text) => repr_str(text),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(repr_value).collect();
            if parts.len() == 1 {
                format!("({},)", parts[0])
            } else {
                format!("({})", parts.join(", "))
            }
        }
        Value::Object(_) => value.to_string(),
    }
}

/// Python's `repr` of a list of strings.
pub fn repr_list(items: &[String]) -> String {
    let parts: Vec<String> = items.iter().map(|item| repr_str(item)).collect();
    format!("[{}]", parts.join(", "))
}

// ---------------------------------------------------------------- columns

/// The source columns a value expression reads, in first use order.
pub fn column_references(expression: &ValueExpression) -> Vec<ColumnRef> {
    let mut found = Vec::new();
    collect_references(expression, &mut found);
    found
}

fn collect_references(expression: &ValueExpression, found: &mut Vec<ColumnRef>) {
    match expression {
        ValueExpression::ColumnRef(reference) => {
            if !found.contains(reference) {
                found.push(reference.clone());
            }
        }
        ValueExpression::Alias(alias) => collect_references(&alias.expression, found),
        ValueExpression::GetField(field) => collect_references(&field.input, found),
        ValueExpression::Value(value) => {
            for reference in &value.columns {
                if !found.contains(reference) {
                    found.push(reference.clone());
                }
            }
        }
        ValueExpression::ModelCall(_) => {}
    }
}

/// The name of a column reference, with its table alias when it has one.
pub fn reference_name(reference: &ColumnRef, qualified: bool) -> String {
    reference.name(qualified)
}

/// The name of a source, computed, or projected column, as `column_name`.
///
/// # Panics
///
/// The expression is a bare model call, which has no name.
pub fn column_name(expression: &ValueExpression, qualified: bool) -> String {
    match expression {
        ValueExpression::ColumnRef(reference) => reference_name(reference, qualified),
        ValueExpression::GetField(field) => {
            format!("{}.{}", column_name(&field.input, qualified), field.field)
        }
        ValueExpression::Value(value) => value.text_with(qualified),
        ValueExpression::Alias(alias) => alias.name.clone(),
        ValueExpression::ModelCall(_) => panic!("a model call has no column name"),
    }
}

/// The name of a projected column.
pub fn projected_name(column: &Projected) -> String {
    match column {
        Projected::Name(name) => name.clone(),
        Projected::Expression(expression) => column_name(expression, true),
    }
}

/// The model call an Alias projects, if any and of one of `kinds`.
pub fn aliased_call<'a>(expression: &'a ValueExpression, kinds: &[&str]) -> Option<&'a ModelCall> {
    let ValueExpression::Alias(alias) = expression else {
        return None;
    };
    let ValueExpression::ModelCall(call) = alias.expression.as_ref() else {
        return None;
    };
    (kinds.is_empty() || kinds.contains(&call.kind.as_str())).then_some(call)
}

/// The aliased call of a projected column.
pub fn projected_call<'a>(column: &'a Projected, kinds: &[&str]) -> Option<&'a ModelCall> {
    match column {
        Projected::Expression(expression) => aliased_call(expression, kinds),
        Projected::Name(_) => None,
    }
}

/// The table a prompt argument reads, as the argument's `alias` property.
pub fn argument_alias(argument: &ValueExpression) -> String {
    match argument {
        ValueExpression::ColumnRef(reference) => reference.alias.clone(),
        other => column_references(other)
            .into_iter()
            .next()
            .map(|reference| reference.alias)
            .unwrap_or_default(),
    }
}

/// The table aliases a model call's prompt reads, in placeholder order.
pub fn call_aliases(call: &ModelCall) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for argument in &call.prompt.args {
        let alias = argument_alias(argument);
        if !found.contains(&alias) {
            found.push(alias);
        }
    }
    found
}

/// The struct fields of one feature list, as `FEATURE_LIST` holds them.
pub const FEATURE_FIELDS: [&str; 2] = ["feature", "strength"];

/// The predicate of an AI filter or AI join as a model call.
pub fn asked_call(asked: &Asked) -> &ModelCall {
    match asked {
        Asked::Call(call) => call,
        Asked::Compare(test) => &test.call,
    }
}

/// Every struct column of a SELECT list and its member columns, answers
/// and spans.
pub fn packing(structs: &[ExtractStruct]) -> Vec<(String, Vec<(String, String)>)> {
    let mut found = Vec::new();
    for found_struct in structs {
        for suffix in ["", SPAN_SUFFIX] {
            found.push((
                format!("{}{suffix}", found_struct.name),
                found_struct
                    .fields
                    .iter()
                    .map(|field| {
                        (
                            field.clone(),
                            format!("{}.{field}{suffix}", found_struct.name),
                        )
                    })
                    .collect(),
            ));
        }
    }
    found
}

/// The one table alias a value expression of an equality reads.
pub fn column_alias_of(expression: &ValueExpression) -> String {
    column_references(expression)
        .into_iter()
        .next()
        .map(|found| found.alias)
        .unwrap_or_default()
}

/// The text of an equality, as `str(Equality)`.
pub fn equality_text(equality: &Equality) -> String {
    format!(
        "{} = {}",
        column_name(&equality.left, true),
        column_name(&equality.right, true)
    )
}

fn push_unique(found: &mut Vec<ColumnRef>, references: Vec<ColumnRef>) {
    for reference in references {
        if !found.contains(&reference) {
            found.push(reference);
        }
    }
}

/// The columns a value expression reads, through a model call's prompt,
/// in first use order.
pub fn value_refs(expression: &ValueExpression) -> Vec<ColumnRef> {
    match expression {
        ValueExpression::ModelCall(call) => call_refs(call),
        ValueExpression::Alias(alias) => match alias.expression.as_ref() {
            ValueExpression::ModelCall(call) => call_refs(call),
            other => column_references(other),
        },
        other => column_references(other),
    }
}

/// The columns a model call's prompt reads, in first use order.
pub fn call_refs(call: &ModelCall) -> Vec<ColumnRef> {
    let mut found = Vec::new();
    for argument in &call.prompt.args {
        push_unique(&mut found, value_refs(argument));
    }
    found
}

/// The columns an AI predicate reads, in first use order.
pub fn asked_refs(asked: &Asked) -> Vec<ColumnRef> {
    match asked {
        Asked::Call(call) => call_refs(call),
        Asked::Compare(test) => call_refs(&test.call),
    }
}

/// The columns a join equality reads, in first use order.
pub fn equality_refs(equality: &Equality) -> Vec<ColumnRef> {
    let mut found = value_refs(&equality.left);
    push_unique(&mut found, value_refs(&equality.right));
    found
}

// ---------------------------------------------------------------- conditions

/// The unqualified names of the columns a condition reads, in first
/// use order.
pub fn condition_columns(condition: &ValueExpression) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for reference in column_references(condition) {
        if !found.contains(&reference.column) {
            found.push(reference.column);
        }
    }
    found
}

/// The text of a condition, each column named with its table alias.
pub fn condition_text(condition: &ValueExpression) -> String {
    column_name(condition, true)
}

// ---------------------------------------------------------------- text

/// The text of an expression in a message, as Python's `_explain`.
pub fn explain(expression: &ValueExpression) -> String {
    match expression {
        ValueExpression::Alias(alias) => match aliased_call(expression, &[]) {
            None => format!("{} AS {}", column_name(&alias.expression, true), alias.name),
            Some(call) if ["label", "extract", "map"].contains(&call.kind.as_str()) => {
                alias.name.clone()
            }
            Some(call) => format!("{} AS {}", explain_call(call), alias.name),
        },
        ValueExpression::ModelCall(call) => explain_call(call),
        other => column_name(other, true),
    }
}

/// The text of a model call in a message.
pub fn explain_call(call: &ModelCall) -> String {
    let template = repr_str(&call.prompt.template);
    match call.kind.as_str() {
        "label" => format!("AI.CLASSIFY({template}, {})", repr_json_list(&call.labels)),
        "extract" => format!("AI.EXTRACT({template}, {})", repr_str(&call.question)),
        "features" => format!(
            "AI.SAE_FEATURES({template}, pooling={})",
            repr_str(&call.pooling)
        ),
        "map" => format!(
            "AI.MAP({template}, max_answer_tokens={})",
            call.max_answer_tokens
        ),
        "score" => format!("AI.SCORE({template})"),
        _ => format!("AI.IF({template})"),
    }
}

fn repr_json_list(value: &Value) -> String {
    match value {
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(repr_value).collect();
            format!("[{}]", parts.join(", "))
        }
        other => repr_value(other),
    }
}

// ---------------------------------------------------------------- checks

/// Validate an Alias: its model call or value expression, and its name.
pub fn validate_alias(alias: &Alias) -> Result<(), CompileError> {
    if let ValueExpression::ModelCall(call) = alias.expression.as_ref() {
        validate_call(call)?;
    }
    if alias.name.is_empty() {
        return compile_error("a projected expression needs a name");
    }
    Ok(())
}

fn strings(value: &Value) -> Vec<Value> {
    match value {
        Value::Array(items) => items.clone(),
        _ => Vec::new(),
    }
}

fn non_empty(value: &Value) -> bool {
    !strings(value).is_empty()
}

/// Check an AI.CLASSIFY label list and its descriptions.
pub fn validate_labels(labels: &Value, descriptions: &Value) -> Result<(), CompileError> {
    let labels = strings(labels);
    let descriptions = strings(descriptions);
    if labels.len() < 2 {
        return compile_error("AI.CLASSIFY needs at least two labels");
    }
    let texts: Vec<&str> = labels.iter().filter_map(Value::as_str).collect();
    if texts.len() != labels.len() || texts.iter().any(|label| label.trim().is_empty()) {
        return compile_error("every AI.CLASSIFY label is non-empty text");
    }
    let mut folded: Vec<String> = texts.iter().map(|label| label.to_lowercase()).collect();
    folded.sort();
    folded.dedup();
    if folded.len() != texts.len() {
        let written: Vec<String> = texts.iter().map(|label| label.to_string()).collect();
        return compile_error(format!(
            "AI.CLASSIFY labels must differ ignoring case, got {}",
            repr_list(&written)
        ));
    }
    if !descriptions.is_empty() && descriptions.len() != labels.len() {
        return compile_error("AI.CLASSIFY needs one description per label, empty for none");
    }
    for description in &descriptions {
        let Some(text) = description.as_str() else {
            return compile_error("an AI.CLASSIFY description is text or empty");
        };
        let words = text.split_whitespace().count();
        if words > DESCRIPTION_WORDS {
            return compile_error(format!(
                "an AI.CLASSIFY description has at most {DESCRIPTION_WORDS} words, got {words}"
            ));
        }
    }
    Ok(())
}

/// Check a model call's kind and the fields that kind takes.
pub fn validate_call(call: &ModelCall) -> Result<(), CompileError> {
    let kind = call.kind.as_str();
    if !MODEL_CALL_KINDS.contains(&kind) {
        let kinds: Vec<String> = MODEL_CALL_KINDS.iter().map(|k| repr_str(k)).collect();
        return compile_error(format!(
            "model call kind must be one of ({}), got {}",
            kinds.join(", "),
            repr_str(kind)
        ));
    }
    if kind != "label"
        && (non_empty(&call.labels) || non_empty(&call.descriptions) || call.probabilities)
    {
        return compile_error("only an AI.CLASSIFY call has labels");
    }
    if kind != "extract" && !call.question.is_empty() {
        return compile_error("only an AI.EXTRACT call has a question");
    }
    if kind != "features" && !call.pooling.is_empty() {
        return compile_error("only an AI.SAE_FEATURES call has a pooling");
    }
    if kind == "map" && call.max_answer_tokens < 1 {
        return compile_error("an AI.MAP answer limit is a positive integer");
    }
    if kind != "map" && call.max_answer_tokens != 0 {
        return compile_error("only an AI.MAP call has an answer limit");
    }
    match kind {
        "label" => validate_labels(&call.labels, &call.descriptions),
        "extract" if call.question.trim().is_empty() => {
            compile_error("AI.EXTRACT needs a question with at least one word")
        }
        "features" if !SAE_POOLINGS.contains(&call.pooling.as_str()) => compile_error(format!(
            "AI.SAE_FEATURES pooling must be one of ('last', 'max'), got {}",
            repr_str(&call.pooling)
        )),
        _ => Ok(()),
    }
}

/// Check an `AI.SCORE` comparison: its call, its comparison, and its
/// threshold, which lies between 0 and 1.
pub fn validate_score_test(
    call: &ModelCall,
    comparison: &str,
    threshold: Option<f64>,
) -> Result<(), CompileError> {
    validate_call(call)?;
    if call.kind != "score" {
        return compile_error("only an AI.SCORE call compares with a threshold");
    }
    if !SCORE_COMPARISONS.contains(&comparison) {
        return compile_error(format!(
            "unsupported AI.SCORE comparison {}",
            repr_str(comparison)
        ));
    }
    if !threshold.is_some_and(|value| (0.0..=1.0).contains(&value)) {
        return compile_error("AI.SCORE threshold must be between 0 and 1");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_prints_them() {
        for (value, text) in [
            (1.0, "1.0"),
            (0.25, "0.25"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (123456789.5, "123456789.5"),
            (-2.5e-7, "-2.5e-07"),
        ] {
            assert_eq!(repr_float(value), text);
        }
    }

    #[test]
    fn strings_print_with_python_quotes() {
        assert_eq!(repr_str("lease"), "'lease'");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("a\nb"), "'a\\nb'");
    }
}
