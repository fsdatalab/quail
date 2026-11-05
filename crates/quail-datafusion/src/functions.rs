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
/// SQL may also write `AI.IF` for `ai_filter`, and `AI.<name>` or
/// `ai.<name>` for every other `ai_<name>`; see [`ai_function_name`].
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

/// The internal name of the AI function SQL writes as `name`, such as
/// `ai_filter` for `AI.IF` or `ai.if`, or None for any other function.
pub fn ai_function_name(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    let internal = match lower.strip_prefix("ai.") {
        Some("if") => "ai_filter".to_string(),
        Some(rest) => format!("ai_{rest}"),
        None => lower,
    };
    AI_FUNCTION_NAMES
        .into_iter()
        .find(|known| *known == internal)
}

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
    /// The names SQL writes the function by, such as `AI.IF` for
    /// `ai_filter`: DataFusion looks a dotted name up as written.
    aliases: Vec<String>,
}

impl QuailFunction {
    fn new(name: &'static str, kind: FunctionKind) -> Self {
        let dotted = match name {
            "ai_filter" => Some("if".to_string()),
            _ => name.strip_prefix("ai_").map(str::to_string),
        };
        let aliases = dotted
            .map(|rest| vec![format!("AI.{}", rest.to_uppercase()), format!("ai.{rest}")])
            .unwrap_or_default();
        Self {
            name,
            kind,
            signature: Signature::variadic_any(Volatility::Volatile),
            aliases,
        }
    }
}

impl ScalarUDFImpl for QuailFunction {
    fn name(&self) -> &str {
        self.name
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
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
        Field::new("start", DataType::Int32, true),
        Field::new("end", DataType::Int32, true),
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

/// Write each `{'key': value, ...}` literal outside quotes as
/// `named_struct('key', value, ...)`.
///
/// DataFusion reads the brace literal in the Snowflake dialect only;
/// the front end also takes it in BigQuery SQL, as the questions of an
/// `AI.EXTRACT`.
pub fn brace_structs(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len());
    // per open brace, the parentheses opened inside it
    let mut braces: Vec<usize> = Vec::new();
    let mut quote: Option<char> = None;
    let mut chars = sql.chars().peekable();
    while let Some(c) = chars.next() {
        if let Some(open) = quote {
            out.push(c);
            if c == open {
                if chars.peek() == Some(&open) {
                    out.push(chars.next().expect("a doubled quote"));
                } else {
                    quote = None;
                }
            }
            continue;
        }
        match c {
            '\'' | '"' | '`' => {
                quote = Some(c);
                out.push(c);
            }
            '{' => {
                braces.push(0);
                out.push_str("named_struct(");
            }
            '}' if !braces.is_empty() => {
                braces.pop();
                out.push(')');
            }
            '(' => {
                if let Some(depth) = braces.last_mut() {
                    *depth += 1;
                }
                out.push(c);
            }
            ')' => {
                if let Some(depth) = braces.last_mut() {
                    *depth = depth.saturating_sub(1);
                }
                out.push(c);
            }
            ':' if braces.last() == Some(&0) => out.push(','),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod brace_tests {
    use super::brace_structs;

    #[test]
    fn a_brace_literal_becomes_a_named_struct_outside_quotes() {
        assert_eq!(
            brace_structs("AI.EXTRACT(c.text, {'terms': 'Why: {0}?', 'n': f(a, b)})"),
            "AI.EXTRACT(c.text, named_struct('terms', 'Why: {0}?', 'n', f(a, b)))"
        );
        assert_eq!(brace_structs("SELECT 'it''s {x}'"), "SELECT 'it''s {x}'");
    }
}
