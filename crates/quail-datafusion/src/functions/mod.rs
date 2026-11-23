pub(crate) mod field_access;
pub(crate) mod image;
pub(crate) mod options;
pub(crate) mod parse_document;
pub(crate) mod pdf_pages;
mod quiet;
pub(crate) mod regexps;

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, FieldRef, Fields};
use datafusion::common::{Result, exec_err, plan_err};
use datafusion::logical_expr::{
    ColumnarValue, Expr, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};

pub const AI_FUNCTION_NAMES: [&str; 7] = [
    "ai_filter",
    "ai_score",
    "ai_classify",
    "ai_extract",
    "ai_extract_span",
    "ai_map",
    "ai_sae_features",
];

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
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct QuailFunction {
    name: &'static str,
    kind: FunctionKind,
    signature: Signature,
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
            FunctionKind::If | FunctionKind::Hint => Ok(DataType::Boolean),
            FunctionKind::Score => Ok(DataType::Float64),
            FunctionKind::Classify | FunctionKind::Map | FunctionKind::Prompt => Ok(DataType::Utf8),
            FunctionKind::SaeFeatures => Ok(sae_features_type()),
            FunctionKind::Extract | FunctionKind::ExtractSpan => {
                plan_err!("{}'s type depends on its questions argument", self.name)
            }
        }
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let data_type = match self.kind {
            FunctionKind::Extract => extract_type(args.arg_fields, DataType::Utf8)?,
            FunctionKind::ExtractSpan => extract_type(args.arg_fields, span_type())?,
            _ => self.return_type(&[])?,
        };
        Ok(Arc::new(Field::new(self.name, data_type, true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if self.kind == FunctionKind::Hint
            && let Some(test) = args.args.into_iter().next()
        {
            return Ok(test);
        }
        exec_err!(
            "{} runs only inside a model region; claim_model_regions did not claim it",
            self.name
        )
    }
}

pub(crate) fn sae_features_type() -> DataType {
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

pub(crate) fn span_type() -> DataType {
    DataType::Struct(Fields::from(vec![
        Field::new("start", DataType::Int32, true),
        Field::new("end", DataType::Int32, true),
    ]))
}

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
    ]
    .into_iter()
    .map(|(name, kind)| Arc::new(ScalarUDF::new_from_impl(QuailFunction::new(name, kind))))
    .collect()
}

pub(crate) fn quail_function(name: &str) -> Arc<ScalarUDF> {
    quail_functions()
        .into_iter()
        .find(|function| function.name() == name)
        .expect("a Quail function")
}

pub fn is_quail_call(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(call)
        if call.func.inner().downcast_ref::<QuailFunction>().is_some())
}

pub fn is_ai_call(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(call) if AI_FUNCTION_NAMES.contains(&call.name()))
}

pub(crate) fn function_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::ScalarFunction(call) => Some(call.name()),
        _ => None,
    }
}

pub(crate) fn column_call_kind(expr: &Expr) -> Option<&'static str> {
    let Expr::ScalarFunction(call) = expr else {
        return None;
    };
    match call.name() {
        "ai_classify" => Some("label"),
        "ai_extract" => Some("extract"),
        "ai_map" => Some("map"),
        "ai_sae_features" => Some("features"),
        _ => None,
    }
}
