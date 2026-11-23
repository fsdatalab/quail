//! How `explain()` writes numbers, values, and prompts.

use crate::constants::DEFAULT_SELECTIVITY;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::physical::{PhysicalNode, PortRef};
use serde::Deserialize;
use serde_json::Value;

use crate::expressions::asked_call;
use crate::expressions::{column_name, condition_text, repr_float, repr_str};
use crate::operators::table_name;
use crate::operators::{Operators, SemanticJoin};
use crate::plan::stages::Stage;
use crate::operators::effective_selectivity;
use crate::pricing::statistics::{OrTerm, RowFlow};


/// The mark beside a selectivity the planner assumed.
const DEFAULT_MARK: &str = "*";
/// The characters of a prompt template a predicate row shows.
const PROMPT_CHARS: usize = 32;

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
pub(super) fn commas(value: &Value) -> String {
    match value {
        Value::Number(number) if number.is_f64() => {
            group_thousands(&repr_float(number.as_f64().unwrap_or_default()))
        }
        Value::Number(number) => group_thousands(&number.to_string()),
        other => py_str(other),
    }
}

/// Python's `f"{value:,}"` of an integer.
pub(super) fn commas_int(value: i64) -> String {
    group_thousands(&value.to_string())
}

/// A row count or rate: three significant digits below one, else one
/// decimal with commas, without a trailing `.0`.
pub(super) fn number(value: f64) -> String {
    if value != 0.0 && value.abs() < 1.0 {
        return format_g(value, 3);
    }
    let text = group_thousands(&format!("{value:.1}"));
    text.strip_suffix(".0").map(str::to_string).unwrap_or(text)
}

pub(super) fn seconds(value: Option<f64>) -> String {
    match value {
        None => String::new(),
        Some(value) if value < 0.001 => "<1 ms".to_string(),
        Some(value) if value < 1.0 => format!("{} ms", format_g(value * 1000.0, 3)),
        Some(value) => format!("{} s", format_g(value, 3)),
    }
}

pub(super) fn selectivity(value: Option<f64>, mark: bool) -> String {
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

pub(super) fn default_note() -> String {
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
pub(super) fn py_str(value: &Value) -> String {
    str_as(value, false)
}

fn str_as(value: &Value, tuples: bool) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => repr_as(other, tuples),
    }
}

/// `key=value` lines, nesting a mapping and a list of mappings.
pub(super) fn field_lines(fields: &[(String, Shown)], depth: usize) -> Vec<String> {
    field_lines_as(fields, depth, false)
}

/// `key=value` lines, each array shown as a tuple when `tuples` holds.
pub(super) fn field_lines_as(fields: &[(String, Shown)], depth: usize, tuples: bool) -> Vec<String> {
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
pub(super) fn prefix(text: &str, count: usize) -> String {
    text.chars().take(count).collect()
}

pub(super) fn width(text: &str) -> usize {
    text.chars().count()
}

// ---------------------------------------------------------------- prompts

pub(super) fn short_prompt(prompt: &crate::physical::Prompt) -> String {
    let mut template = prompt.template.clone();
    if width(&template) > PROMPT_CHARS {
        template = prefix(&template, PROMPT_CHARS) + "…";
    }
    format!("PROMPT({})", repr_str(&template))
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

