//! Bind AI calls and their prompts.
//! Read literal arguments, options, and classification categories.

use super::{Binder, Options};
use crate::functions::function_name;
use datafusion::arrow::datatypes::DataType;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, ScalarValue};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use quail_planner::error::{CompileError, compile_error};
use quail_planner::expressions::{column_alias_of, column_name, explain_call, repr_str};
use quail_planner::physical::{
    Alias, ColumnRef, ExtractStruct, ModelCall, ScoreTest, ValueExpression,
};
use quail_planner::prompts;
use serde_json::Value;

/// The answer limit of an AI.MAP call written without one.
const DEFAULT_MAP_ANSWER_TOKENS: i64 = 256;

pub(super) fn arguments(expr: &Expr) -> &[Expr] {
    match expr {
        Expr::ScalarFunction(ScalarFunction { args, .. }) => args,
        _ => &[],
    }
}

pub(super) fn is_call(expr: &Expr, name: &str) -> bool {
    function_name(expr) == Some(name)
}

pub(crate) fn string_literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(value, _) => value.try_as_str().flatten().map(str::to_string),
        _ => None,
    }
}

/// A numeric SQL literal cast to `to`, or None.
fn numeric_literal(expr: &Expr, to: &DataType) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) if value.data_type().is_numeric() => value.cast_to(to).ok(),
        Expr::Negative(inner) => numeric_literal(inner, to)?.arithmetic_negate().ok(),
        _ => None,
    }
}

/// An integer SQL literal, or None.
fn integer_of(expr: &Expr) -> Option<i64> {
    let Expr::Literal(value, _) = expr else {
        return None;
    };
    match value
        .data_type()
        .is_integer()
        .then(|| value.cast_to(&DataType::Int64))?
    {
        Ok(ScalarValue::Int64(value)) => value,
        _ => None,
    }
}

/// The (key, value) pairs of an option object, `named_struct(k, v, ...)`.
fn object_fields(expr: &Expr) -> Option<Vec<(String, &Expr)>> {
    if !is_call(expr, "named_struct") {
        return None;
    }
    arguments(expr)
        .chunks(2)
        .map(|pair| Some((string_literal(&pair[0])?, &pair[1])))
        .collect()
}

fn sorted_keys(keys: &[&str]) -> String {
    let mut keys: Vec<String> = keys.iter().map(|key| repr_str(key)).collect();
    keys.sort();
    format!("[{}]", keys.join(", "))
}

fn number_of(expr: &Expr) -> Option<f64> {
    match numeric_literal(expr, &DataType::Float64)? {
        ScalarValue::Float64(value) => value,
        _ => None,
    }
}

pub(super) fn parse_options(
    node: Option<&Expr>,
    allowed: &[&str],
) -> Result<Options, CompileError> {
    let Some(node) = node else {
        return Ok(Options::default());
    };
    let Some(fields) = object_fields(node) else {
        return compile_error("AI options must be an object such as {'selectivity': 0.3}.");
    };
    let mut options = Options::default();
    for (key, value) in fields {
        if !allowed.contains(&key.as_str()) {
            return compile_error(format!(
                "Unknown option {} must be replaced with one of {}.",
                repr_str(&key),
                sorted_keys(allowed)
            ));
        }
        if key == "selectivity" {
            match (string_literal(value), number_of(value)) {
                (None, Some(number)) => options.selectivity = Some(number),
                _ => return compile_error("The selectivity option must be a number."),
            }
        } else {
            match string_literal(value) {
                Some(text) => options.anchor = Some(text),
                None => return compile_error("The anchor option must be a table alias string."),
            }
        }
    }
    Ok(options)
}

impl Binder {
    fn note_document_column(&mut self, column: &ValueExpression) -> Result<(), CompileError> {
        let alias = column_alias_of(column);
        match self
            .document_columns
            .iter()
            .find(|(found, _)| *found == alias)
        {
            Some((_, seen)) if seen != column => super::unsupported(&format!(
                "Model calls reading {}.{} and {} must run in separate regions.",
                alias,
                column_name(seen, false),
                column_name(column, false)
            )),
            Some(_) => Ok(()),
            None => {
                self.document_columns.push((alias, column.clone()));
                Ok(())
            }
        }
    }

    fn bind_document_value(&mut self, argument: &Expr) -> Result<ValueExpression, CompileError> {
        let value = self.bind_value(argument)?;
        if matches!(value, ValueExpression::Value(_)) {
            return super::unsupported(
                "DataFusion must compute this document expression before the model reads it.",
            );
        }
        Ok(value)
    }

    fn prompt_arguments(
        &mut self,
        prompt: &Expr,
        function: &str,
    ) -> Result<(String, Vec<ValueExpression>), CompileError> {
        if !is_call(prompt, "prompt") {
            return compile_error(format!(
                "{} needs a PROMPT as its first argument.",
                if function == "ai_filter" {
                    "AI.IF".to_string()
                } else {
                    function.replacen("ai_", "AI.", 1).to_uppercase()
                }
            ));
        }
        let args = arguments(prompt);
        let Some(template) = args.first().and_then(string_literal) else {
            return compile_error("PROMPT needs a string template as its first argument.");
        };
        let mut references = Vec::new();
        for argument in &args[1..] {
            references.push(self.bind_document_value(argument)?);
        }
        Ok((template, references))
    }

    /// Bind an AI_FILTER or AI.SCORE prompt: (prompt, aliases).
    #[allow(clippy::too_many_arguments)]
    fn bind_ai_call(
        &mut self,
        prompt_expr: &Expr,
        function: &str,
        join: Option<bool>,
        classify: Option<(&[String], &[String], &str)>,
    ) -> Result<(quail_planner::physical::Prompt, Vec<String>), CompileError> {
        let (template, references) = self.prompt_arguments(prompt_expr, function)?;
        self.bind_prompt(template, references, function, join, classify)
    }

    /// Bind a prompt's template and references: (prompt, aliases).
    fn bind_prompt(
        &mut self,
        template: String,
        references: Vec<ValueExpression>,
        function: &str,
        join: Option<bool>,
        classify: Option<(&[String], &[String], &str)>,
    ) -> Result<(quail_planner::physical::Prompt, Vec<String>), CompileError> {
        let mut aliases: Vec<String> = Vec::new();
        for reference in &references {
            let alias = quail_planner::expressions::argument_alias(reference);
            if !aliases.contains(&alias) {
                aliases.push(alias);
            }
        }
        let join = join.unwrap_or(aliases.len() > 1);
        if function == "ai_score" && join && template.matches("{0}").count() != 1 {
            return super::unsupported(
                "This pair score must run over a materialized input relation.",
            );
        }
        let prompt = match classify {
            Some((labels, descriptions, task)) => prompts::bind(
                "label",
                &template,
                &references,
                labels,
                descriptions,
                task,
                &self.turn,
                &self.layout,
            )?,
            None => {
                let kind = match function {
                    "ai_score" => "score",
                    "ai_sae_features" => "features",
                    _ if join => "join",
                    _ => "filter",
                };
                prompts::bind(
                    kind,
                    &template,
                    &references,
                    &[],
                    &[],
                    "",
                    &self.turn,
                    &self.layout,
                )?
            }
        };
        for reference in &references {
            self.note_document_column(reference)?;
        }
        Ok((prompt, aliases))
    }

    /// Bind an `AI_FILTER(PROMPT(...), options)` call: its model call
    /// and options.
    pub(crate) fn bind_ai_filter_call(
        &mut self,
        term: &Expr,
        allowed: &[&str],
        join: Option<bool>,
    ) -> Result<(ModelCall, Options), CompileError> {
        if !is_call(term, "ai_filter") {
            return super::unsupported(
                "DataFusion must evaluate this expression of a model result.",
            );
        }
        let args = arguments(term);
        if args.is_empty() {
            return compile_error("AI.IF needs a PROMPT as its first argument.");
        }
        if args.len() > 2 {
            return compile_error("AI.IF takes a PROMPT and at most one options object.");
        }
        let options = parse_options(args.get(1), allowed)?;
        let (prompt, _) = self.bind_ai_call(&args[0], "ai_filter", join, None)?;
        Ok((model_call(prompt, "boolean"), options))
    }

    /// Bind a comparison of an `AI.SCORE` call with a threshold: the
    /// test and the call's options.
    pub(crate) fn bind_ai_score_test(
        &mut self,
        term: &Expr,
        allowed: &[&str],
        join: Option<bool>,
    ) -> Result<(ScoreTest, Options), CompileError> {
        let Some((comparison, call, threshold)) = score_comparison(term) else {
            return super::unsupported("DataFusion must evaluate this comparison of a score.");
        };
        let Some(value) = number_of(threshold).filter(|_| string_literal(threshold).is_none())
        else {
            if !matches!(threshold, Expr::Literal(value, _) if !value.is_null()) {
                return super::unsupported(
                    "DataFusion must compare this score with the computed threshold.",
                );
            }
            return compile_error("The AI.SCORE threshold must be a number.");
        };
        if !(0.0..=1.0).contains(&value) {
            return compile_error("The AI.SCORE threshold must be between 0 and 1.");
        }
        let args = arguments(call);
        if args.len() > 2 {
            return compile_error("AI.SCORE takes a PROMPT and at most one options object.");
        }
        let options = parse_options(args.get(1), allowed)?;
        let Some(prompt) = args.first() else {
            return compile_error("AI.SCORE needs a PROMPT as its first argument.");
        };
        let (prompt, _) = self.bind_ai_call(prompt, "ai_score", join, None)?;
        let call = model_call(prompt, "score");
        quail_planner::expressions::validate_score_test(&call, comparison.0, Some(value))?;
        // the planner names the score's column; the test reads it as its
        // one column, shown as the call until then
        let score = ColumnRef {
            alias: String::new(),
            provider: String::new(),
            column: explain_call(&call),
        };
        let compared = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(Expr::Column(Column::new_unqualified("score"))),
            comparison.1,
            Box::new(Expr::Literal(ScalarValue::Float64(Some(value)), None)),
        ));
        let test = crate::derive::expressions::expression(&compared, |_| {
            Ok((score.clone(), DataType::Float64))
        })?;
        let test = ValueExpression::Value(test);
        Ok((ScoreTest { call, test }, options))
    }

    /// The selectivity option of the `AI.CLASSIFY` call a test of its
    /// label makes, as the query wrote the test; None without one.
    pub(crate) fn label_selectivity(&self, written: &Expr) -> Result<Option<f64>, CompileError> {
        let mut options = None;
        let _ = written.apply(|node| {
            if is_call(node, "ai_classify") {
                options = arguments(node).get(2).cloned();
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        });
        for (key, value) in options.as_ref().and_then(object_fields).unwrap_or_default() {
            if key.eq_ignore_ascii_case("selectivity") {
                return match (string_literal(value), number_of(value)) {
                    (None, Some(number)) => Ok(Some(number)),
                    _ => compile_error("The selectivity option must be a number."),
                };
            }
        }
        Ok(None)
    }

    fn bind_ai_classify(&mut self, node: &Expr) -> Result<(ModelCall, Options), CompileError> {
        let args = arguments(node);
        if args.len() < 2 {
            return compile_error(
                "AI.CLASSIFY needs a prompt or document column and its categories.",
            );
        }
        let categories = match object_fields(&args[1]) {
            Some(fields) if fields.len() == 1 && fields[0].0 == "categories" => fields[0].1,
            _ => &args[1],
        };
        let (labels, descriptions) = parse_categories(categories)?;
        let mut options = Options::default();
        let (mut task, mut probabilities) = (String::new(), false);
        if let Some(config) = args.get(2) {
            let Some(fields) = object_fields(config) else {
                return compile_error("AI.CLASSIFY options must be an object.");
            };
            for (key, value) in fields {
                match key.to_lowercase().as_str() {
                    "selectivity" => match (string_literal(value), number_of(value)) {
                        (None, Some(number)) => options.selectivity = Some(number),
                        _ => return compile_error("The selectivity option must be a number."),
                    },
                    "task_description" => match string_literal(value) {
                        Some(text) => task = text,
                        None => {
                            return compile_error("The task_description option must be a string.");
                        }
                    },
                    "probabilities" => match value {
                        Expr::Literal(ScalarValue::Boolean(Some(flag)), _) => probabilities = *flag,
                        _ => {
                            return compile_error(
                                "The probabilities option must be true or false.",
                            );
                        }
                    },
                    other => {
                        return compile_error(format!(
                            "AI.CLASSIFY option {} must be selectivity, task_description, or probabilities.",
                            repr_str(other)
                        ));
                    }
                }
            }
        }
        let classify = Some((labels.as_slice(), descriptions.as_slice(), task.as_str()));
        let (prompt, aliases) = if is_call(&args[0], "prompt") {
            self.bind_ai_call(&args[0], "ai_classify", Some(false), classify)?
        } else {
            // a bare document column uses the default instruction
            let reference = self.bind_document_value(&args[0])?;
            self.bind_prompt(
                "{0}".to_string(),
                vec![reference],
                "ai_classify",
                Some(false),
                classify,
            )?
        };
        if !(1..=2).contains(&aliases.len()) {
            return compile_error(
                "AI.CLASSIFY needs one document column or one from each side of a join.",
            );
        }
        let mut call = model_call(prompt, "label");
        call.labels = Value::Array(labels.into_iter().map(Value::String).collect());
        call.descriptions = Value::Array(descriptions.into_iter().map(Value::String).collect());
        call.probabilities = probabilities;
        quail_planner::expressions::validate_call(&call)?;
        Ok((call, options))
    }

    pub(crate) fn bind_ai_extract(
        &mut self,
        node: &Expr,
        name: &str,
    ) -> Result<(ExtractStruct, Vec<ValueExpression>), CompileError> {
        let args = arguments(node);
        if args.len() != 2 {
            return compile_error(
                "AI.EXTRACT needs a document column and an object of field questions.",
            );
        }
        let reference = self.bind_document_value(&args[0])?;
        let Some(fields) = object_fields(&args[1]) else {
            return compile_error(
                "AI.EXTRACT needs an object of field questions as its second argument.",
            );
        };
        let mut questions: Vec<(String, String)> = Vec::new();
        for (field, question) in fields {
            if questions.iter().any(|(found, _)| *found == field) {
                return compile_error(format!(
                    "AI.EXTRACT {} must not name field {} twice.",
                    repr_str(name),
                    repr_str(&field)
                ));
            }
            let Some(question) = string_literal(question) else {
                return compile_error(format!(
                    "AI.EXTRACT field {} needs a question string.",
                    repr_str(&field)
                ));
            };
            questions.push((field, question));
        }
        let found = ExtractStruct {
            name: name.to_string(),
            fields: questions.iter().map(|(field, _)| field.clone()).collect(),
        };
        let mut members = Vec::new();
        for (field, question) in &questions {
            let prompt = prompts::bind(
                "extract",
                question,
                std::slice::from_ref(&reference),
                &[],
                &[],
                "",
                &self.turn,
                &self.layout,
            )?;
            self.note_document_column(&reference)?;
            let mut call = model_call(prompt, "extract");
            call.question = question.clone();
            quail_planner::expressions::validate_call(&call)?;
            members.push(ValueExpression::Alias(Alias {
                expression: Box::new(ValueExpression::ModelCall(call)),
                name: format!("{name}.{field}"),
            }));
        }
        Ok((found, members))
    }

    pub(super) fn bind_ai_map(&mut self, node: &Expr) -> Result<ModelCall, CompileError> {
        let args = arguments(node);
        if !(2..=3).contains(&args.len()) {
            return compile_error(
                "AI.MAP takes a document column, an instruction string, and an optional token limit.",
            );
        }
        let Some(instruction) = string_literal(&args[1]) else {
            return compile_error("AI.MAP needs an instruction string as its second argument.");
        };
        let limit = match args.get(2) {
            None => DEFAULT_MAP_ANSWER_TOKENS,
            Some(found) => match integer_of(found) {
                Some(limit) if limit >= 0 => limit,
                _ => {
                    return compile_error("The AI.MAP token limit must be a positive integer.");
                }
            },
        };
        let reference = self.bind_document_value(&args[0])?;
        let prompt = prompts::bind(
            "map",
            &instruction,
            std::slice::from_ref(&reference),
            &[],
            &[],
            "",
            &self.turn,
            &self.layout,
        )?;
        self.note_document_column(&reference)?;
        let mut call = model_call(prompt, "map");
        call.max_answer_tokens = limit;
        quail_planner::expressions::validate_call(&call)?;
        Ok(call)
    }

    pub(crate) fn bind_ai_sae_features(&mut self, node: &Expr) -> Result<ModelCall, CompileError> {
        let args = arguments(node);
        let pooling = args.get(1).and_then(object_fields).and_then(|fields| {
            fields
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case("pooling"))
                .and_then(|(_, value)| string_literal(value))
        });
        let Some(pooling) = pooling.filter(|_| args.len() == 2) else {
            return compile_error(
                "AI.SAE_FEATURES needs a PROMPT and pooling set to 'last' or 'max'.",
            );
        };
        let (prompt, aliases) =
            self.bind_ai_call(&args[0], "ai_sae_features", Some(false), None)?;
        if aliases.len() != 1 {
            return compile_error("AI.SAE_FEATURES needs one document column.");
        }
        let mut call = model_call(prompt, "features");
        call.pooling = pooling;
        quail_planner::expressions::validate_call(&call)?;
        Ok(call)
    }

    pub(crate) fn bind_column_call(
        &mut self,
        node: &Expr,
        kind: &str,
    ) -> Result<ModelCall, CompileError> {
        match kind {
            "label" => Ok(self.bind_ai_classify(node)?.0),
            "map" => self.bind_ai_map(node),
            _ => self.bind_ai_sae_features(node),
        }
    }

    /// Bind a score or classification to its model call.
    pub(crate) fn bind_ai_value(&mut self, node: &Expr) -> Result<ModelCall, CompileError> {
        if is_call(node, "ai_classify") {
            return Ok(self.bind_ai_classify(node)?.0);
        }
        let args = arguments(node);
        if args.len() > 2 {
            return compile_error("AI.SCORE takes a PROMPT and at most one options object.");
        }
        parse_options(args.get(1), &["selectivity"])?;
        let Some(prompt) = args.first() else {
            return compile_error("AI.SCORE needs a PROMPT as its first argument.");
        };
        let (prompt, _) = self.bind_ai_call(prompt, "ai_score", None, None)?;
        Ok(model_call(prompt, "score"))
    }
}

/// The comparison a score test makes, as SQL writes it.
fn comparison(op: Operator) -> Option<(&'static str, Operator)> {
    Some(match op {
        Operator::Lt => ("<", op),
        Operator::LtEq => ("<=", op),
        Operator::Gt => (">", op),
        Operator::GtEq => (">=", op),
        _ => return None,
    })
}

/// (comparison, AI.SCORE call, threshold) of a compared score, or None.
pub(super) fn score_comparison(expr: &Expr) -> Option<((&'static str, Operator), &Expr, &Expr)> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = expr else {
        return None;
    };
    let forward = comparison(*op)?;
    let backward = comparison(op.swap()?)?;
    if is_call(left, "ai_score") {
        return Some((forward, left, right));
    }
    if is_call(right, "ai_score") {
        return Some((backward, right, left));
    }
    None
}

fn model_call(prompt: quail_planner::physical::Prompt, kind: &str) -> ModelCall {
    ModelCall {
        prompt: Box::new(prompt),
        kind: kind.to_string(),
        labels: Value::Array(Vec::new()),
        descriptions: Value::Array(Vec::new()),
        probabilities: false,
        question: String::new(),
        pooling: String::new(),
        max_answer_tokens: 0,
    }
}

fn parse_categories(node: &Expr) -> Result<(Vec<String>, Vec<String>), CompileError> {
    if !is_call(node, "make_array") {
        return compile_error(
            "AI.CLASSIFY categories must be an array of labels, label-description pairs, or \
             category objects, or a registered label table name.",
        );
    }
    let (mut labels, mut descriptions) = (Vec::new(), Vec::new());
    for item in arguments(node) {
        if let Some(label) = string_literal(item) {
            labels.push(label);
            descriptions.push(String::new());
            continue;
        }
        let (label, description) = if is_call(item, "struct") && arguments(item).len() == 2 {
            (&arguments(item)[0], Some(&arguments(item)[1]))
        } else if let Some(fields) = object_fields(item) {
            let label = fields
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case("label"));
            let description = fields
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case("description"))
                .map(|(_, value)| *value);
            let unknown = fields.iter().any(|(key, _)| {
                !key.eq_ignore_ascii_case("label") && !key.eq_ignore_ascii_case("description")
            });
            match label {
                Some((_, label)) if !unknown => (*label, description),
                _ => {
                    return compile_error(
                        "A category object must contain a label and may contain a description.",
                    );
                }
            }
        } else {
            return compile_error(format!(
                "AI.CLASSIFY category {item} must be a label, a label-description pair, or a \
                 category object."
            ));
        };
        let Some(label) = string_literal(label) else {
            return compile_error("AI.CLASSIFY labels must be string literals.");
        };
        let description = match description {
            None => String::new(),
            Some(Expr::Literal(value, _)) if value.is_null() => String::new(),
            Some(found) => match string_literal(found) {
                Some(text) => text,
                None => {
                    return compile_error(
                        "An AI.CLASSIFY description must be a string literal or NULL.",
                    );
                }
            },
        };
        labels.push(label);
        descriptions.push(description);
    }
    if descriptions.iter().all(String::is_empty) {
        descriptions.clear();
    }
    Ok((labels, descriptions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::{col, lit};
    use quail_planner::error::ErrorKind;

    fn binder() -> Binder {
        Binder::new(&(String::new(), String::new()), "default")
    }

    fn call(name: &str, args: Vec<Expr>) -> Expr {
        Expr::ScalarFunction(ScalarFunction::new_udf(
            crate::rules::lower::quail_function(name),
            args,
        ))
    }

    #[test]
    fn malformed_options_are_invalid_but_document_storage_limits_are_unsupported() {
        let error = parse_options(Some(&lit(0.5)), &["selectivity"])
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Invalid);
        let mut binder = binder();
        let document = |name: &str| {
            ValueExpression::ColumnRef(ColumnRef {
                alias: "r".into(),
                provider: "reviews".into(),
                column: name.into(),
            })
        };
        binder.note_document_column(&document("body")).unwrap();
        let error = binder.note_document_column(&document("title")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn score_thresholds_outside_zero_to_one_are_invalid() {
        let prompt = call("prompt", vec![lit("Is {0} relevant?"), col("r.body")]);
        let score = call("ai_score", vec![prompt]);
        for threshold in [-0.1, 1.1] {
            let test = score.clone().gt(lit(threshold));
            let error = binder()
                .bind_ai_score_test(&test, &["selectivity"], None)
                .err()
                .unwrap();
            assert_eq!(error.kind(), ErrorKind::Invalid);
            assert!(error.message().contains("between 0 and 1"));
        }
    }

    #[test]
    fn projected_score_validates_options_and_arity() {
        let prompt = call("prompt", vec![lit("Is {0} relevant?"), col("r.body")]);
        let bad_options = call("ai_score", vec![prompt.clone(), lit(0.5)]);
        let error = binder().bind_ai_value(&bad_options).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Invalid);
        assert!(error.message().contains("options"));
        let too_many = call("ai_score", vec![prompt, lit(0.5), lit(0.5)]);
        let error = binder().bind_ai_value(&too_many).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Invalid);
        assert!(error.message().contains("at most one"));
    }
}
