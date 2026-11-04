//! Reading and checking expressions and conditions, as
//! `quail/common/expressions.py` does.
//!
//! Error messages and the text of conditions match Python's, so a
//! query refused by the Rust planner reads as it did.

use std::fmt;

use quail_physical::{
    AggregateCall, Alias, ColumnRef, Condition, Connective, ModelCall, Not, Projected,
    RegularPredicate, SortExpression, ValueExpression,
};
use serde_json::Value;

use crate::arrow_functions::ARROW_FUNCTIONS;

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
pub const VALUE_COMPARISONS: [&str; 6] = ["=", "<>", "<", "<=", ">", ">="];
pub const TEXT_COMPARISONS: [&str; 7] = [
    "like",
    "ilike",
    "contains",
    "icontains",
    "starts with",
    "ends with",
    "regex",
];
pub const SCORE_COMPARISONS: [&str; 4] = ["<", "<=", ">", ">="];
pub const MODEL_CALL_KINDS: [&str; 6] = ["boolean", "score", "label", "extract", "features", "map"];
pub const SAE_POOLINGS: [&str; 2] = ["last", "max"];
/// The SQL names of scalar functions whose Arrow compute function has another name.
const FUNCTION_ALIASES: [(&str, &str); 4] = [
    ("length", "utf8_length"),
    ("lower", "utf8_lower"),
    ("upper", "utf8_upper"),
    ("trim", "utf8_trim_whitespace"),
];

fn column_comparisons() -> Vec<&'static str> {
    let mut found: Vec<&str> = VALUE_COMPARISONS.to_vec();
    found.extend(["in", "is null", "is not null", "is true"]);
    found.extend(TEXT_COMPARISONS);
    found
}

// ---------------------------------------------------------------- Python text

/// Python's `repr` of a string.
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
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
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

/// A literal as Python compares it for equality: True equals 1, 1 equals 1.0.
fn value_key(value: &Value) -> String {
    match value {
        Value::Bool(flag) => repr_float(if *flag { 1.0 } else { 0.0 }),
        Value::Number(number) => repr_float(number.as_f64().unwrap_or(f64::NAN)),
        other => other.to_string(),
    }
}

fn is_number(value: &Value) -> bool {
    matches!(value, Value::Number(_) | Value::Bool(_))
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
        ValueExpression::ScalarFunction(function) => {
            for argument in &function.args {
                collect_references(argument, found);
            }
        }
        ValueExpression::Literal(_) | ValueExpression::ModelCall(_) => {}
    }
}

/// The name of a column reference, with its table alias when it has one.
pub fn reference_name(reference: &ColumnRef, qualified: bool) -> String {
    if qualified && !reference.alias.is_empty() {
        format!("{}.{}", reference.alias, reference.column)
    } else {
        reference.column.clone()
    }
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
        ValueExpression::ScalarFunction(function) => {
            let arguments: Vec<String> = function
                .args
                .iter()
                .map(|argument| column_name(argument, qualified))
                .collect();
            format!("{}({})", function.function, arguments.join(", "))
        }
        ValueExpression::Literal(literal) => repr_value(&literal.value),
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

/// The name a sort key sorts on.
pub fn sort_name(expression: &SortExpression) -> String {
    match expression {
        SortExpression::Value(value) => column_name(value, true),
        SortExpression::Aggregate(aggregate) => aggregate.name.clone(),
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

// ---------------------------------------------------------------- conditions

/// The column tests under a condition, in written order.
pub fn tests(condition: &Condition) -> Vec<&RegularPredicate> {
    match condition {
        Condition::Compare(test) => vec![test.as_ref()],
        Condition::Not(not) => tests(&not.operand),
        Condition::And(connective) | Condition::Or(connective) => {
            connective.operands.iter().flat_map(tests).collect()
        }
    }
}

/// The tables a test reads.
pub fn test_aliases(test: &RegularPredicate) -> Vec<String> {
    if let ValueExpression::ModelCall(call) = &test.expression {
        return call_aliases(call);
    }
    let mut found: Vec<String> = Vec::new();
    for reference in column_references(&test.expression) {
        if !found.contains(&reference.alias) {
            found.push(reference.alias);
        }
    }
    found
}

/// The tables a condition's tests read.
pub fn condition_aliases(condition: &Condition) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for test in tests(condition) {
        for alias in test_aliases(test) {
            if !found.contains(&alias) {
                found.push(alias);
            }
        }
    }
    found
}

/// The unqualified names of the columns a condition's tests read.
pub fn condition_columns(condition: &Condition) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for test in tests(condition) {
        for reference in column_references(&test.expression) {
            if !found.contains(&reference.column) {
                found.push(reference.column);
            }
        }
    }
    found
}

/// The source columns a condition reads, in first use order.
pub fn condition_references(condition: &Condition) -> Vec<ColumnRef> {
    tests(condition)
        .into_iter()
        .flat_map(|test| column_references(&test.expression))
        .collect()
}

fn not(operand: Condition) -> Condition {
    Condition::Not(Not {
        operand: Box::new(operand),
    })
}

/// The condition that accepts the rows the condition rejects.
pub fn negate(condition: Condition) -> Condition {
    match condition {
        Condition::Not(not) => *not.operand,
        Condition::And(connective) => Condition::Or(Connective {
            operands: connective.operands.into_iter().map(negate).collect(),
        }),
        Condition::Or(connective) => Condition::And(Connective {
            operands: connective.operands.into_iter().map(negate).collect(),
        }),
        Condition::Compare(test) => {
            let negated = match test.comparison.as_str() {
                "is null" => Some("is not null"),
                "is not null" => Some("is null"),
                _ => None,
            };
            match negated {
                Some(comparison) => Condition::Compare(Box::new(RegularPredicate {
                    expression: test.expression,
                    comparison: comparison.to_string(),
                    value: Value::Null,
                })),
                None => not(Condition::Compare(test)),
            }
        }
    }
}

/// The condition with NOT on column tests only and AND and OR flat.
pub fn normalize(condition: Condition) -> Condition {
    match condition {
        Condition::Not(not) => {
            let negated = negate(normalize(*not.operand));
            if matches!(negated, Condition::Not(_)) {
                negated
            } else {
                normalize(negated)
            }
        }
        Condition::And(connective) => Condition::And(Connective {
            operands: flatten(connective, true),
        }),
        Condition::Or(connective) => Condition::Or(Connective {
            operands: flatten(connective, false),
        }),
        test => test,
    }
}

fn flatten(connective: Connective, and: bool) -> Vec<Condition> {
    let mut operands = Vec::new();
    for operand in connective.operands.into_iter().map(normalize) {
        match (operand, and) {
            (Condition::And(inner), true) | (Condition::Or(inner), false) => {
                operands.extend(inner.operands)
            }
            (operand, _) => operands.push(operand),
        }
    }
    operands
}

/// The conditions an AND combines, or the condition alone.
pub fn conjuncts(condition: &Condition) -> Vec<Condition> {
    match condition {
        Condition::And(connective) => connective.operands.clone(),
        other => vec![other.clone()],
    }
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

/// The text of a test, as `str(RegularPredicate)`.
pub fn test_text(test: &RegularPredicate) -> String {
    let name = explain(&test.expression);
    match test.comparison.as_str() {
        "in" => {
            let values: Vec<String> = match &test.value {
                Value::Array(items) => items.iter().map(repr_value).collect(),
                other => vec![repr_value(other)],
            };
            format!("{name} IN ({})", values.join(", "))
        }
        "is true" => name,
        comparison if comparison.starts_with("is") => {
            format!("{name} {}", comparison.to_uppercase())
        }
        comparison => format!("{name} {comparison} {}", repr_value(&test.value)),
    }
}

/// The text of a condition, as `str()` of a Python condition.
pub fn condition_text(condition: &Condition) -> String {
    match condition {
        Condition::Compare(test) => test_text(test),
        Condition::Not(not) => format!("NOT {}", operand_text(&not.operand, true)),
        Condition::And(connective) => join_operands(connective, " AND "),
        Condition::Or(connective) => join_operands(connective, " OR "),
    }
}

fn join_operands(connective: &Connective, separator: &str) -> String {
    connective
        .operands
        .iter()
        .map(|operand| operand_text(operand, false))
        .collect::<Vec<_>>()
        .join(separator)
}

fn operand_text(operand: &Condition, always: bool) -> String {
    let text = condition_text(operand);
    if always || matches!(operand, Condition::And(_) | Condition::Or(_)) {
        format!("({text})")
    } else {
        text
    }
}

// ---------------------------------------------------------------- checks

/// Check that a scalar function name calls a scalar function of Arrow compute.
pub fn check_arrow_function(name: &str) -> Result<(), CompileError> {
    let arrow = FUNCTION_ALIASES
        .iter()
        .find(|(sql, _)| *sql == name)
        .map(|(_, arrow)| *arrow)
        .unwrap_or(name);
    let Some((_, kind, attribute)) = ARROW_FUNCTIONS.iter().find(|(found, _, _)| *found == arrow)
    else {
        return compile_error(format!(
            "unknown function {}; a scalar function of Arrow compute is called by its \
             name, such as abs or utf8_reverse",
            repr_str(name)
        ));
    };
    if !["scalar", "meta"].contains(kind) || !attribute || (*kind == "meta" && arrow != "cast") {
        return compile_error(format!(
            "{name} is an Arrow {kind} function; an expression calls a scalar function, \
             which reads one row at a time"
        ));
    }
    Ok(())
}

fn type_name(expression: &ValueExpression) -> &'static str {
    match expression {
        ValueExpression::ColumnRef(_) => "ColumnRef",
        ValueExpression::GetField(_) => "GetField",
        ValueExpression::ScalarFunction(_) => "ScalarFunction",
        ValueExpression::Literal(_) => "Literal",
        ValueExpression::Alias(_) => "Alias",
        ValueExpression::ModelCall(_) => "ModelCall",
    }
}

/// Validate one value expression, whatever its node type.
pub fn validate_tree(expression: &ValueExpression) -> Result<(), CompileError> {
    if matches!(expression, ValueExpression::ModelCall(_)) {
        return compile_error(format!(
            "a value expression is a column, a field, a function, a literal, or an \
             alias, got {}",
            type_name(expression)
        ));
    }
    validate_expression(expression)
}

/// Validate a value expression node and the nodes below it.
pub fn validate_expression(expression: &ValueExpression) -> Result<(), CompileError> {
    match expression {
        ValueExpression::ColumnRef(reference) => {
            if reference.alias.is_empty() || reference.column.is_empty() {
                return compile_error("a column reference names a table alias and a column");
            }
            Ok(())
        }
        ValueExpression::GetField(field) => {
            validate_tree(&field.input)?;
            if field.field.is_empty() {
                return compile_error("a field access names a struct field");
            }
            Ok(())
        }
        ValueExpression::Literal(literal) => {
            let values = match &literal.value {
                Value::Array(items) => items.clone(),
                other => vec![other.clone()],
            };
            if values
                .iter()
                .any(|value| matches!(value, Value::Array(_) | Value::Object(_)))
            {
                return compile_error(format!(
                    "a literal is a string, number, boolean, NULL, or a list of them, got {}",
                    repr_value(&literal.value)
                ));
            }
            Ok(())
        }
        ValueExpression::ScalarFunction(function) => {
            check_arrow_function(&function.function)?;
            for argument in &function.args {
                validate_tree(argument)?;
            }
            Ok(())
        }
        ValueExpression::Alias(alias) => validate_alias(alias),
        ValueExpression::ModelCall(call) => validate_call(call),
    }
}

/// Validate an Alias: its model call or value expression, and its name.
pub fn validate_alias(alias: &Alias) -> Result<(), CompileError> {
    match alias.expression.as_ref() {
        ValueExpression::ModelCall(call) => validate_call(call)?,
        other => validate_tree(other)?,
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

/// Check a test: a model score against a threshold, or columns against literals.
pub fn validate_test(test: &RegularPredicate) -> Result<(), CompileError> {
    if let ValueExpression::ModelCall(call) = &test.expression {
        validate_call(call)?;
        if call.kind != "score" {
            return compile_error("only an AI.SCORE call compares with a threshold");
        }
        if !SCORE_COMPARISONS.contains(&test.comparison.as_str()) {
            return compile_error(format!(
                "unsupported AI.SCORE comparison {}",
                repr_str(&test.comparison)
            ));
        }
        let threshold = match &test.value {
            Value::Number(number) => number.as_f64(),
            Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
            _ => None,
        };
        if !threshold.is_some_and(|value| (0.0..=1.0).contains(&value)) {
            return compile_error("AI.SCORE threshold must be between 0 and 1");
        }
        return Ok(());
    }
    if column_references(&test.expression).is_empty() {
        return compile_error("a column predicate tests a column");
    }
    validate_expression(&test.expression)?;
    let comparisons = column_comparisons();
    let comparison = test.comparison.as_str();
    if !comparisons.contains(&comparison) {
        return compile_error(format!(
            "unsupported column comparison {}; supported: {}",
            repr_str(comparison),
            comparisons.join(", ")
        ));
    }
    if comparison == "in" {
        let Value::Array(values) = &test.value else {
            return compile_error("IN needs a non-empty list of literals");
        };
        if values.is_empty() {
            return compile_error("IN needs a non-empty list of literals");
        }
        let mut keys: Vec<String> = values.iter().map(value_key).collect();
        keys.sort();
        keys.dedup();
        if keys.len() != values.len() {
            return compile_error(format!(
                "{} IN lists a value twice",
                explain(&test.expression)
            ));
        }
    } else if comparison.starts_with("is") {
        if !test.value.is_null() {
            return compile_error("a null test takes no value");
        }
    } else if TEXT_COMPARISONS.contains(&comparison) {
        if !test.value.is_string() {
            return compile_error(format!("{comparison} compares with a string"));
        }
    } else if test.value.is_null() || test.value.is_array() {
        return compile_error(format!("{comparison} compares with one literal"));
    }
    Ok(())
}

fn connective_name(condition: &Condition) -> &'static str {
    match condition {
        Condition::Not(_) => "NOT",
        Condition::And(_) => "AND",
        Condition::Or(_) => "OR",
        Condition::Compare(_) => "",
    }
}

/// Check a condition and every condition under it.
pub fn validate_condition(condition: &Condition) -> Result<(), CompileError> {
    let (operands, minimum): (Vec<&Condition>, usize) = match condition {
        Condition::Compare(test) => return validate_test(test),
        Condition::Not(not) => (vec![not.operand.as_ref()], 1),
        Condition::And(connective) | Condition::Or(connective) => {
            (connective.operands.iter().collect(), 2)
        }
    };
    let name = connective_name(condition);
    if operands.len() < minimum {
        return compile_error(format!("{name} takes at least {minimum} conditions"));
    }
    for operand in operands {
        if let Condition::Compare(test) = operand
            && matches!(test.expression, ValueExpression::ModelCall(_))
        {
            return compile_error(format!(
                "{name} combines column tests, not AI calls; put the logic inside one \
                 prompt's text"
            ));
        }
        validate_condition(operand)?;
    }
    Ok(())
}

/// Check an aggregate call.
pub fn validate_aggregate(aggregate: &AggregateCall) -> Result<(), CompileError> {
    const FUNCTIONS: [&str; 6] = ["count", "count_distinct", "sum", "avg", "min", "max"];
    if !FUNCTIONS.contains(&aggregate.function.as_str()) {
        return compile_error(format!(
            "unsupported aggregate {}; supported: {}",
            repr_str(&aggregate.function),
            FUNCTIONS.join(", ")
        ));
    }
    if aggregate.argument.is_none() && aggregate.function != "count" {
        return compile_error(format!("{}(*) is not an aggregate", aggregate.function));
    }
    if aggregate.name.is_empty() {
        return compile_error("an aggregate needs a name");
    }
    Ok(())
}

/// The text of an aggregate call, as `str(AggregateCall)`.
pub fn aggregate_text(aggregate: &AggregateCall) -> String {
    let function = if aggregate.function == "count_distinct" {
        "count(distinct ".to_string()
    } else {
        format!("{}(", aggregate.function)
    };
    format!(
        "{} = {}{})",
        aggregate.name,
        function,
        aggregate.argument.as_deref().unwrap_or("*")
    )
}

/// Whether a literal is a number, as Python's `isinstance(value, (int, float))`.
pub fn is_numeric(value: &Value) -> bool {
    is_number(value)
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
