//! Quail's AI functions as DataFusion scalar UDFs.
//!
//! The UDFs let DataFusion parse, bind, and type a query that calls
//! the AI functions. They never run: `ClaimModelRegions` moves every
//! call into a model region, which Quail plans and executes.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, FieldRef, Fields};
use datafusion::common::{Result, exec_err, plan_err};
use datafusion::logical_expr::{
    ColumnarValue, Expr, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};

/// The AI functions, by their internal names.
///
/// `AI.IF` reaches DataFusion as `ai_filter`, and every other
/// `AI.<name>` as `ai_<name>`; see [`normalize_ai_calls`].
/// `ai_extract_span` is the answer positions of an `ai_extract`; see
/// [`crate::add_extract_spans`].
pub const AI_FUNCTION_NAMES: [&str; 7] = [
    "ai_filter",
    "ai_score",
    "ai_classify",
    "ai_extract",
    "ai_extract_span",
    "ai_map",
    "ai_sae_features",
];

/// The kind of each Quail function, which decides its result type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FunctionKind {
    If,
    Score,
    Classify,
    Extract,
    ExtractSpan,
    Map,
    SaeFeatures,
    Prompt,
    Hint,
    /// The `WHERE` test recording an `UNNEST`; see [`crate::take_unnests`].
    Unnest,
    /// A field of an unnested list's element.
    UnnestField,
}

/// One Quail function: an AI function, `PROMPT`, or `HINT`.
#[derive(Debug, PartialEq, Eq, Hash)]
struct QuailFunction {
    name: &'static str,
    kind: FunctionKind,
    signature: Signature,
}

impl QuailFunction {
    fn new(name: &'static str, kind: FunctionKind) -> Self {
        Self {
            name,
            kind,
            signature: Signature::variadic_any(Volatility::Volatile),
        }
    }
}

impl ScalarUDFImpl for QuailFunction {
    fn name(&self) -> &str {
        self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        match self.kind {
            FunctionKind::If | FunctionKind::Hint | FunctionKind::Unnest => Ok(DataType::Boolean),
            FunctionKind::Score => Ok(DataType::Float64),
            FunctionKind::Classify | FunctionKind::Map | FunctionKind::Prompt => Ok(DataType::Utf8),
            FunctionKind::SaeFeatures => Ok(sae_features_type()),
            FunctionKind::Extract | FunctionKind::ExtractSpan => {
                plan_err!("{}'s type depends on its questions argument", self.name)
            }
            FunctionKind::UnnestField => {
                plan_err!("{}'s type depends on its list and field", self.name)
            }
        }
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let data_type = match self.kind {
            FunctionKind::Extract => extract_type(args.arg_fields, DataType::Utf8)?,
            FunctionKind::ExtractSpan => extract_type(args.arg_fields, span_type())?,
            FunctionKind::UnnestField => unnest_field_type(&args)?,
            _ => self.return_type(&[])?,
        };
        Ok(Arc::new(Field::new(self.name, data_type, true)))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        exec_err!(
            "{} runs only inside a model region; ClaimModelRegions did not claim it",
            self.name
        )
    }
}

/// The type of one field of an unnested list's element: the list, the
/// list's name, and the field's name are the arguments.
fn unnest_field_type(args: &ReturnFieldArgs) -> Result<DataType> {
    let field = match args.scalar_arguments.get(2) {
        Some(Some(datafusion::common::ScalarValue::Utf8(Some(name)))) => name.clone(),
        _ => return plan_err!("an unnested field is named by a string literal"),
    };
    match args.arg_fields.first().map(|found| found.data_type()) {
        Some(DataType::List(item)) => match item.data_type() {
            DataType::Struct(fields) => match fields.iter().find(|found| found.name() == &field) {
                Some(found) => Ok(found.data_type().clone()),
                None => plan_err!("the unnested list has no field {field}"),
            },
            other => Ok(other.clone()),
        },
        _ => plan_err!("UNNEST reads a list"),
    }
}

/// The type of `AI.SAE_FEATURES`: one struct per nonzero feature, as
/// Quail's `FEATURE_LIST`.
fn sae_features_type() -> DataType {
    let feature = Fields::from(vec![
        Field::new("feature", DataType::Int32, true),
        Field::new("strength", DataType::Float32, true),
    ]);
    DataType::List(Arc::new(Field::new(
        "item",
        DataType::Struct(feature),
        true,
    )))
}

/// The type of `AI.EXTRACT`, or of its answer positions: one nullable
/// field of type `answer` per question.
///
/// The questions argument is a struct literal such as
/// `{'end_date': 'When does the lease end?'}`, so the field names are
/// in its type when the query is bound.
fn extract_type(arg_fields: &[FieldRef], answer: DataType) -> Result<DataType> {
    let Some(DataType::Struct(questions)) = arg_fields.get(1).map(|field| field.data_type()) else {
        return plan_err!("ai_extract takes a document column and a struct of questions");
    };
    let answers: Vec<Field> = questions
        .iter()
        .map(|question| Field::new(question.name(), answer.clone(), true))
        .collect();
    Ok(DataType::Struct(Fields::from(answers)))
}

/// The position of one answer: its first and past-the-end characters.
fn span_type() -> DataType {
    DataType::Struct(Fields::from(vec![
        Field::new("start", DataType::Int32, false),
        Field::new("end", DataType::Int32, false),
    ]))
}

/// Return every Quail function as a DataFusion UDF.
pub fn quail_functions() -> Vec<Arc<ScalarUDF>> {
    [
        ("ai_filter", FunctionKind::If),
        ("ai_score", FunctionKind::Score),
        ("ai_classify", FunctionKind::Classify),
        ("ai_extract", FunctionKind::Extract),
        ("ai_extract_span", FunctionKind::ExtractSpan),
        ("ai_map", FunctionKind::Map),
        ("ai_sae_features", FunctionKind::SaeFeatures),
        ("prompt", FunctionKind::Prompt),
        ("hint", FunctionKind::Hint),
        (crate::unnest::UNNEST_TEST, FunctionKind::Unnest),
        (crate::unnest::UNNEST_FIELD, FunctionKind::UnnestField),
    ]
    .into_iter()
    .map(|(name, kind)| Arc::new(ScalarUDF::new_from_impl(QuailFunction::new(name, kind))))
    .collect()
}

/// A scalar function of Arrow compute that DataFusion does not have,
/// such as `utf8_is_upper`.
///
/// Quail's language calls any scalar function of Arrow compute by its
/// name; Quail evaluates it, so it runs only inside a model region.
/// DataFusion needs its type to plan the query: a function named
/// `is_*` or `utf8_is_*` is a test, another `utf8_*` function returns
/// text, and any other returns its first argument's type.
#[derive(Debug, PartialEq, Eq, Hash)]
struct ArrowFunction {
    name: String,
    signature: Signature,
}

impl ScalarUDFImpl for ArrowFunction {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if self.name.starts_with("is_") || self.name.starts_with("utf8_is_") {
            return Ok(DataType::Boolean);
        }
        if self.name.starts_with("utf8_") {
            return Ok(DataType::Utf8);
        }
        Ok(arg_types.first().cloned().unwrap_or(DataType::Null))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        exec_err!(
            "{} is an Arrow compute function Quail evaluates; it runs only inside a model region",
            self.name
        )
    }
}

/// A UDF for each scalar function of Arrow compute whose name is not in
/// `taken`, the names DataFusion already has.
pub fn arrow_functions(taken: &dyn Fn(&str) -> bool) -> Vec<Arc<ScalarUDF>> {
    quail_planner::arrow_functions::ARROW_FUNCTIONS
        .iter()
        .filter(|(name, kind, attribute)| *kind == "scalar" && *attribute && !taken(name))
        .map(|(name, _, _)| {
            Arc::new(ScalarUDF::new_from_impl(ArrowFunction {
                name: name.to_string(),
                signature: Signature::variadic_any(Volatility::Immutable),
            }))
        })
        .collect()
}

/// Whether an expression is a call to an AI function.
pub fn is_ai_call(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(call) if AI_FUNCTION_NAMES.contains(&call.name()))
}

/// Rewrite dotted AI calls to internal names: `AI.IF(` to
/// `ai_filter(` and `AI.<name>(` to `ai_<name>(`.
///
/// DataFusion reads `AI.IF(...)` as a call to a function named `IF`
/// in schema `AI`, so the dot must go before parsing. Text inside
/// string literals and quoted identifiers is left alone.
pub fn normalize_ai_calls(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len());
    let mut chars = sql.char_indices().peekable();
    let mut quote: Option<char> = None;
    while let Some((position, c)) = chars.next() {
        if let Some(open) = quote {
            out.push(c);
            if c == '\\' && open == '\'' {
                // Snowflake escapes a quote in a string as \'
                if let Some((_, escaped)) = chars.next() {
                    out.push(escaped);
                }
            } else if c == open {
                quote = None;
            }
            continue;
        }
        if c == '\'' || c == '"' || c == '`' {
            quote = Some(c);
            out.push(c);
            continue;
        }
        let rest = &sql[position..];
        let starts_word = out
            .chars()
            .last()
            .is_none_or(|previous| !(previous.is_alphanumeric() || previous == '_'));
        if starts_word && is_dotted_ai_call(rest) {
            let is_if = rest.len() > 5
                && rest[3..5].eq_ignore_ascii_case("if")
                && !rest[5..].starts_with(|c: char| c.is_alphanumeric() || c == '_');
            let (replacement, skipped) = if is_if { ("ai_filter", 4) } else { ("ai_", 2) };
            out.push_str(replacement);
            for _ in 0..skipped {
                chars.next();
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Whether text starts with `AI.<name>(`, allowing spaces before `(`.
fn is_dotted_ai_call(text: &str) -> bool {
    if text.len() < 4 || !text[..3].eq_ignore_ascii_case("ai.") {
        return false;
    }
    let after_dot = &text[3..];
    let name_length = after_dot
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(after_dot.len());
    name_length > 0 && after_dot[name_length..].trim_start().starts_with('(')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotted_ai_calls_become_underscored_outside_strings() {
        let sql = "SELECT AI.IF(PROMPT('Is AI.IF here?', r.body)) FROM r WHERE ai.score(x) > 0";
        assert_eq!(
            normalize_ai_calls(sql),
            "SELECT ai_filter(PROMPT('Is AI.IF here?', r.body)) FROM r WHERE ai_score(x) > 0"
        );
    }

    #[test]
    fn a_column_ending_in_ai_is_not_rewritten() {
        assert_eq!(normalize_ai_calls("SELECT t.thai.x"), "SELECT t.thai.x");
    }

    #[test]
    fn a_table_aliased_ai_keeps_its_columns() {
        let sql = "SELECT ai.body FROM reviews ai";
        assert_eq!(normalize_ai_calls(sql), sql);
    }
}
