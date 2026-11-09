//! Quail's expressions from the DataFusion expressions of a model
//! region's plan.
//!
//! [`crate::derive`] hands the [`Binder`] each table, each ON and WHERE
//! term, AI call, and SELECT list item as the query wrote it, in that
//! order. The binder binds each AI call's prompt in the session model's
//! chat turn and layout, each column as a `ColumnRef`, each other value
//! and each condition as an `Expression` that DataFusion evaluates, and
//! the SELECT list as the region's projection, and refuses SQL outside
//! Quail's language. A condition reads each answer it tests as the
//! answer's column.

use std::collections::BTreeSet;

use crate::functions::function_name;
use crate::functions::{ai_function_name, column_call_kind, sae_features_type, span_type};
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, ScalarValue};
use datafusion::logical_expr::expr::{InList, ScalarFunction};
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use quail_physical::{
    Alias, ColumnRef, Equality, ExtractStruct, GetField, ModelCall, ScoreTest, ValueExpression,
};
use quail_planner::expressions::{
    CompileError, SPAN_SUFFIX, aliased_call, call_aliases, column_alias_of, column_name,
    compile_error, explain_call, repr_list, repr_str,
};
use quail_planner::operators::{Scan, ScanColumn, Schema};
use quail_planner::prompts::{self, Turn};
use serde_json::Value;

/// A condition and the selectivity its HINT gives, if any.
pub(crate) type Hinted = (ValueExpression, Option<f64>);

/// The answer limit of an AI.MAP call written without one.
const DEFAULT_MAP_ANSWER_TOKENS: i64 = 256;

pub(crate) fn unsupported<T>(what: &str) -> Result<T, CompileError> {
    Err(CompileError::unsupported(what))
}

/// Each output column name of a SELECT list and its expression, but a
/// column the list passes on under its own name.
pub(crate) fn select_outputs(expressions: &[Expr]) -> Vec<(String, Expr)> {
    let mut outputs: Vec<(String, Expr)> = Vec::new();
    for item in expressions {
        let (inner, _) = unalias(item);
        let name = output_name(item);
        let self_reference = matches!(inner, Expr::Column(column) if column.name == name);
        if !self_reference && !outputs.iter().any(|(found, _)| *found == name) {
            outputs.push((name, inner.clone()));
        }
    }
    outputs
}

/// Each column-operator call a SELECT list names with AS, by its name.
pub(crate) fn selected_calls(expressions: &[Expr]) -> Vec<(String, Expr)> {
    expressions
        .iter()
        .filter_map(|expr| match expr {
            Expr::Alias(alias) if column_call_kind(&alias.expr).is_some() => {
                Some((alias.name.clone(), (*alias.expr).clone()))
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn output_name(expr: &Expr) -> String {
    match expr {
        Expr::Alias(alias) => alias.name.clone(),
        Expr::Column(column) => column.name.clone(),
        other => other.schema_name().to_string(),
    }
}

/// An expression without the AS names on it, and the name the query
/// wrote for it.
///
/// DataFusion names `COUNT(*)` itself, as `count(Int64(1)) AS count(*)`,
/// and a call written by a dotted name, as `ai_map(...) AS AI.MAP(...)`;
/// those names are not ones the query wrote.
fn unalias(expr: &Expr) -> (&Expr, Option<&str>) {
    let mut names = Vec::new();
    let mut node = expr;
    while let Expr::Alias(alias) = node {
        names.push(alias.name.as_str());
        node = &alias.expr;
    }
    let automatic = names.last().is_some_and(|name| {
        *name == "count(*)"
            || name.split_once('(').is_some_and(|(function, _)| {
                function.contains('.') && ai_function_name(function).is_some()
            })
    });
    let written = if automatic {
        names.len() > 1
    } else {
        !names.is_empty()
    };
    (
        node,
        if written {
            names.first().copied()
        } else {
            None
        },
    )
}

// ---------------------------------------------------------------- calls

fn arguments(expr: &Expr) -> &[Expr] {
    match expr {
        Expr::ScalarFunction(ScalarFunction { args, .. }) => args,
        _ => &[],
    }
}

fn is_call(expr: &Expr, name: &str) -> bool {
    function_name(expr) == Some(name)
}

/// Whether `expr` or an expression inside it is one `found` accepts.
fn contains(expr: &Expr, found: impl Fn(&Expr) -> bool) -> bool {
    expr.exists(|inner| Ok(found(inner))).unwrap_or(false)
}

/// Whether a term asks the model, counting AI.EXTRACT only bare.
fn has_ai_call(expr: &Expr) -> bool {
    has_ai_call_under(expr, false)
}

fn has_ai_call_under(expr: &Expr, in_field: bool) -> bool {
    match expr {
        Expr::Exists(_) => true,
        Expr::ScalarFunction(call) => {
            let name = call.name();
            if name == "ai_filter" || name == "ai_score" {
                return true;
            }
            if (name == "ai_extract" || name == "ai_extract_span") && !in_field {
                return true;
            }
            let field = name == "get_field";
            call.args
                .iter()
                .enumerate()
                .any(|(index, arg)| has_ai_call_under(arg, field && index == 0))
        }
        other => {
            let mut any = false;
            let _ = other.apply_children(|child| {
                any = has_ai_call_under(child, false);
                Ok(if any {
                    TreeNodeRecursion::Stop
                } else {
                    TreeNodeRecursion::Continue
                })
            });
            any
        }
    }
}

/// `term`, negated when `negate`, with each NOT moved down to the tests
/// below AND and OR by De Morgan's laws, so the parts of an AND, such as
/// `NOT (a OR b)`, are placed apart.
fn negated_inward(term: Expr, negate: bool) -> Expr {
    match term {
        Expr::Not(inner) => negated_inward(*inner, !negate),
        Expr::BinaryExpr(BinaryExpr { left, op, right })
            if matches!(op, Operator::And | Operator::Or) =>
        {
            let op = match (op, negate) {
                (Operator::And, true) => Operator::Or,
                (Operator::Or, true) => Operator::And,
                (op, _) => op,
            };
            Expr::BinaryExpr(BinaryExpr::new(
                Box::new(negated_inward(*left, negate)),
                op,
                Box::new(negated_inward(*right, negate)),
            ))
        }
        Expr::IsNull(inner) if negate => Expr::IsNotNull(inner),
        Expr::IsNotNull(inner) if negate => Expr::IsNull(inner),
        other if negate => Expr::Not(Box::new(other)),
        other => other,
    }
}

pub(crate) fn string_literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(value, _) => value.try_as_str().flatten().map(str::to_string),
        _ => None,
    }
}

/// A SQL literal as a JSON value, or None.
fn literal_value(expr: &Expr) -> Option<Value> {
    match expr {
        Expr::Literal(value, _) => scalar_value(value),
        Expr::Negative(inner) => match literal_value(inner)? {
            Value::Number(number) => Some(match number.as_i64() {
                Some(integer) => Value::from(-integer),
                None => Value::from(-number.as_f64()?),
            }),
            _ => None,
        },
        _ => None,
    }
}

fn scalar_value(value: &ScalarValue) -> Option<Value> {
    Some(match value {
        ScalarValue::Null => Value::Null,
        ScalarValue::Boolean(Some(flag)) => Value::Bool(*flag),
        ScalarValue::Int8(Some(v)) => Value::from(*v),
        ScalarValue::Int16(Some(v)) => Value::from(*v),
        ScalarValue::Int32(Some(v)) => Value::from(*v),
        ScalarValue::Int64(Some(v)) => Value::from(*v),
        ScalarValue::UInt8(Some(v)) => Value::from(*v),
        ScalarValue::UInt16(Some(v)) => Value::from(*v),
        ScalarValue::UInt32(Some(v)) => Value::from(*v),
        ScalarValue::UInt64(Some(v)) => Value::from(*v),
        ScalarValue::Float32(Some(v)) => Value::from(f64::from(*v)),
        ScalarValue::Float64(Some(v)) => Value::from(*v),
        ScalarValue::Decimal128(Some(v), _, scale) => {
            Value::from(*v as f64 / 10f64.powi(i32::from(*scale)))
        }
        other if other.is_null() => Value::Null,
        other => Value::String(other.try_as_str().flatten()?.to_string()),
    })
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

/// An AI predicate's options: selectivity and anchor.
#[derive(Default, Clone)]
pub(crate) struct Options {
    pub(crate) selectivity: Option<f64>,
    pub(crate) anchor: Option<String>,
}

fn number_of(expr: &Expr) -> Option<f64> {
    literal_value(expr)?.as_f64()
}

fn parse_options(node: Option<&Expr>, allowed: &[&str]) -> Result<Options, CompileError> {
    let Some(node) = node else {
        return Ok(Options::default());
    };
    let Some(fields) = object_fields(node) else {
        return compile_error(format!(
            "the second AI operator argument must be an option object like \
             {{'selectivity': 0.3}}, got {node}"
        ));
    };
    let mut options = Options::default();
    for (key, value) in fields {
        if !allowed.contains(&key.as_str()) {
            return compile_error(format!(
                "unknown option key {}; allowed: {}",
                repr_str(&key),
                sorted_keys(allowed)
            ));
        }
        if key == "selectivity" {
            match (string_literal(value), number_of(value)) {
                (None, Some(number)) => options.selectivity = Some(number),
                _ => return compile_error("selectivity must be a number"),
            }
        } else {
            match string_literal(value) {
                Some(text) => options.anchor = Some(text),
                None => return compile_error("anchor must be a table alias string"),
            }
        }
    }
    Ok(options)
}

// ---------------------------------------------------------------- the binder

pub(crate) struct Binder {
    turn: Turn,
    layout: String,
    /// (alias, provider, schema) of each table in scope, in the order
    /// they were added; an unnested list has no provider.
    tables: Vec<(String, String, SchemaRef)>,
    /// alias -> the document expression its prompts read.
    document_columns: Vec<(String, ValueExpression)>,
    /// A column operator a WHERE test reads other than as a label test
    /// -> its name.
    pub(crate) tested: Vec<(ModelCall, String)>,
    /// AS name -> the column-operator call SELECT gives it.
    selected: Vec<(String, Expr)>,
    /// Output column name -> the group, aggregate, or window expression.
    outputs: Vec<(String, Expr)>,
    /// (alias, column, type) of each answer column a test reads.
    answer_types: Vec<(String, String, DataType)>,
}

/// A bound SELECT list.
pub(crate) struct Selected {
    pub(crate) columns: Vec<ValueExpression>,
    pub(crate) structs: Vec<ExtractStruct>,
    /// The structs whose spans the SELECT list returns.
    pub(crate) spans: Vec<String>,
    /// The output column names, in SELECT order.
    pub(crate) output: Vec<String>,
}

impl Binder {
    /// A binder with no table in scope yet, for prompts rendered with
    /// `turn` and `layout`, and the SELECT list `select`.
    pub(crate) fn new(turn: &Turn, layout: &str, select: &[Expr]) -> Self {
        Self {
            turn: turn.clone(),
            layout: layout.to_string(),
            tables: Vec::new(),
            document_columns: Vec::new(),
            tested: Vec::new(),
            selected: selected_calls(select),
            outputs: select_outputs(select),
            answer_types: Vec::new(),
        }
    }

    fn provider(&self, alias: &str) -> Option<&str> {
        self.tables
            .iter()
            .find(|(found, _, _)| found == alias)
            .map(|(_, provider, _)| provider.as_str())
    }

    pub(crate) fn add_table(
        &mut self,
        alias: &str,
        provider: &str,
        schema: SchemaRef,
    ) -> Result<(), CompileError> {
        if self.provider(alias).is_some() {
            return compile_error(format!("duplicate table alias {}", repr_str(alias)));
        }
        self.tables
            .push((alias.to_string(), provider.to_string(), schema));
        Ok(())
    }

    /// Bind `left = right`, an ON equality of the join of `joined` and a
    /// table already in the query.
    pub(crate) fn bind_equality(
        &mut self,
        term: &Expr,
        joined: &str,
    ) -> Result<Equality, CompileError> {
        let Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        }) = term
        else {
            return compile_error(format!(
                "ON supports column = column and AI predicates, got {term}"
            ));
        };
        let equality = Equality {
            left: self.bind_value(left)?,
            right: self.bind_value(right)?,
        };
        let sides = [
            column_alias_of(&equality.left),
            column_alias_of(&equality.right),
        ];
        if !sides.contains(&joined.to_string()) || sides[0] == sides[1] {
            return compile_error(format!(
                "JOIN {joined} ON {term} must relate {} to a table already in the query",
                repr_str(joined)
            ));
        }
        Ok(equality)
    }

    // ------------------------------------------------------------ WHERE

    /// Bind a WHERE term that asks no model: its conditions, one per
    /// part of an AND unless a HINT estimates the whole term, and the
    /// selectivity the HINT gives; or None for a term that asks one.
    pub(crate) fn bind_plain_where_term(
        &mut self,
        term: &Expr,
    ) -> Result<Option<Vec<Hinted>>, CompileError> {
        let (term, hint) = self.unwrap_hint(term)?;
        if has_ai_call(&term) || label_test(&term).is_some() {
            return Ok(None);
        }
        let term = negated_inward(term, false);
        let parts = match hint {
            Some(_) => vec![&term],
            None => split_conjunction(&term),
        };
        let mut bound = Vec::new();
        for part in parts {
            bound.push((self.bind_condition(part)?, hint));
        }
        Ok(Some(bound))
    }

    /// Refuse a WHERE term that is no plain condition and no test of a
    /// label, with the message that names what is wrong with it.
    ///
    /// # Errors
    ///
    /// The term is outside Quail's language, or its AI filter or score
    /// does not bind.
    pub(crate) fn refuse_where_term(&mut self, term: &Expr) -> Result<(), CompileError> {
        let (term, _) = self.unwrap_hint(term)?;
        if matches!(
            &term,
            Expr::BinaryExpr(BinaryExpr {
                op: Operator::Or,
                ..
            })
        ) {
            // an OR of AI.IF calls and plain tests is an AiNode's filter
            return unsupported(&format!(
                "{term}, an OR of a model call other than AI.IF and other tests,"
            ));
        }
        let (node, anti) = match &term {
            Expr::Not(inner) => (inner.as_ref(), true),
            other => (other, false),
        };
        if matches!(node, Expr::Exists(_)) {
            return Ok(());
        }
        if anti {
            return compile_error(format!(
                "NOT is only supported as NOT EXISTS, got NOT {node}"
            ));
        }
        if contains(&term, |inner| is_call(inner, "ai_classify")) {
            return compile_error(
                "AI.CLASSIFY in WHERE is tested with = 'label' or IN ('label', ...)",
            );
        }
        if contains(&term, |inner| is_call(inner, "ai_extract")) {
            return compile_error(
                "an AI.EXTRACT call is not a predicate; test one of its fields, such as \
                 AI.EXTRACT(...).end_date IS NOT NULL, or its AS name from SELECT",
            );
        }
        if score_comparison(&term).is_some() {
            self.bind_ai_score_test(&term, &["selectivity"], None)?;
        } else {
            if contains(&term, |inner| is_call(inner, "ai_score")) {
                return compile_error("AI.SCORE must be compared with <, <=, >, or >=");
            }
            self.bind_ai_filter_call(&term, &["selectivity", "anchor"], None)?;
        }
        Ok(())
    }

    pub(crate) fn unwrap_hint(&mut self, term: &Expr) -> Result<(Expr, Option<f64>), CompileError> {
        if !is_call(term, "hint") {
            return Ok((term.clone(), None));
        }
        let args = arguments(term);
        if args.len() != 2 {
            return compile_error(
                "HINT takes a test and an option object like {'selectivity': 0.4}",
            );
        }
        let options = parse_options(Some(&args[1]), &["selectivity"])?;
        Ok((args[0].clone(), options.selectivity))
    }

    // ------------------------------------------------------------ conditions

    /// Bind a term that asks no model as a condition, each answer it
    /// tests read as the answer's column.
    pub(crate) fn bind_condition(&mut self, term: &Expr) -> Result<ValueExpression, CompileError> {
        let read = self.answer_columns(term)?;
        self.computed(&read)
    }

    /// `term` with each column-operator call it tests as the column of
    /// the call's answer: an AI.CLASSIFY, AI.MAP, or AI.SAE_FEATURES
    /// call, or a field of an AI.EXTRACT call or its span.
    fn answer_columns(&mut self, term: &Expr) -> Result<Expr, CompileError> {
        // each label a test compares a classification with
        let mut compared: Vec<(Expr, Value)> = Vec::new();
        let _ = term.apply(|node| {
            match node {
                Expr::BinaryExpr(BinaryExpr {
                    left,
                    op: Operator::Eq,
                    right,
                }) => {
                    for (call, other) in [(left, right), (right, left)] {
                        if column_call_kind(call) == Some("label")
                            && let Some(text) = string_literal(other)
                        {
                            compared.push((call.as_ref().clone(), Value::from(text)));
                        }
                    }
                }
                Expr::InList(InList { expr, list, .. })
                    if column_call_kind(expr) == Some("label") =>
                {
                    let labels = list.iter().filter_map(string_literal).map(Value::from);
                    compared.push((expr.as_ref().clone(), Value::Array(labels.collect())));
                }
                _ => {}
            }
            Ok(TreeNodeRecursion::Continue)
        });
        for (node, labels) in &compared {
            let call = self.bind_column_call(node, "label")?;
            let name = self.answer_name(node, &call)?;
            check_labels(&name, labels, &call)?;
        }
        let mut failed: Option<CompileError> = None;
        let read = term
            .clone()
            .transform_down(|node| {
                if failed.is_some() {
                    return Ok(Transformed::no(node));
                }
                let found = match &node {
                    // AI.SAE_FEATURES returns a list, which a test reads only here
                    Expr::IsNull(inner) | Expr::IsNotNull(inner)
                        if column_call_kind(inner) == Some("features") =>
                    {
                        self.answer_column(inner, "features").map(|column| {
                            let column = Box::new(column);
                            Some(match &node {
                                Expr::IsNull(_) => Expr::IsNull(column),
                                _ => Expr::IsNotNull(column),
                            })
                        })
                    }
                    _ => match column_call_kind(&node) {
                        Some("extract") => {
                            compile_error("test a field of AI.EXTRACT, such as lease.end_date")
                        }
                        Some("features") => compile_error(
                            "AI.SAE_FEATURES returns a list; test IS NULL or IS NOT NULL, or \
                             UNNEST it to test its fields",
                        ),
                        Some(kind) => self.answer_column(&node, kind).map(Some),
                        None => self.extract_column(&node),
                    },
                };
                match found {
                    Ok(Some(column)) => Ok(Transformed::new(column, true, TreeNodeRecursion::Jump)),
                    Ok(None) => Ok(Transformed::no(node)),
                    Err(error) => {
                        failed = Some(error);
                        Ok(Transformed::no(node))
                    }
                }
            })
            .map_err(|error| CompileError::new(error.to_string()))?
            .data;
        match failed {
            Some(error) => Err(error),
            None => Ok(read),
        }
    }

    /// The answer column of a column-operator call a test reads: the
    /// SELECT list's name for it, or a name of its own.
    fn answer_column(&mut self, node: &Expr, kind: &str) -> Result<Expr, CompileError> {
        let call = self.bind_column_call(node, kind)?;
        let name = self.answer_name(node, &call)?;
        let alias = call_aliases(&call)[0].clone();
        let data_type = match kind {
            "features" => sae_features_type(),
            _ => DataType::Utf8,
        };
        self.answer_types
            .push((alias.clone(), name.clone(), data_type));
        Ok(Expr::Column(Column::new(Some(alias), name)))
    }

    /// Record that a condition may read `alias.column`, an answer of
    /// type `data_type` that no table holds.
    pub(crate) fn note_answer_type(&mut self, alias: &str, column: &str, data_type: DataType) {
        let entry = (alias.to_string(), column.to_string(), data_type);
        if !self.answer_types.contains(&entry) {
            self.answer_types.push(entry);
        }
    }

    /// The name of the answer column of `call`, the call `node` makes.
    fn answer_name(&mut self, node: &Expr, call: &ModelCall) -> Result<String, CompileError> {
        let aliases = call_aliases(call);
        if aliases.len() != 1 {
            return compile_error("a filter on a label tests a one-document classification");
        }
        let alias = aliases[0].clone();
        let name = self
            .selected
            .iter()
            .find(|(_, selected)| selected == node)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| format!("__column_{alias}_{}", self.tested.len()));
        Ok(match self.tested.iter().find(|(found, _)| found == call) {
            Some((_, existing)) => existing.clone(),
            None => {
                self.tested.push((call.clone(), name.clone()));
                name
            }
        })
    }

    /// The column of the AI.EXTRACT answer, or the field of its span, a
    /// field path reads: `get_field(ai_extract(...), field)`, or
    /// `get_field(ai_extract_span(...), field, start|end)`; None for
    /// another expression.
    fn extract_column(&mut self, node: &Expr) -> Result<Option<Expr>, CompileError> {
        let mut path = Vec::new();
        let mut inner = node;
        while is_get_field(inner) {
            let args = arguments(inner);
            let mut fields: Vec<String> = args[1..]
                .iter()
                .map(|field| string_literal(field).unwrap_or_default())
                .collect();
            fields.extend(path);
            path = fields;
            inner = &args[0];
        }
        let span = is_call(inner, "ai_extract_span");
        if !(span || is_call(inner, "ai_extract")) || path.is_empty() {
            return Ok(None);
        }
        let Some((name, call_node)) = self
            .selected
            .iter()
            .find(|(_, selected)| {
                is_call(selected, "ai_extract") && arguments(selected) == arguments(inner)
            })
            .map(|(name, selected)| (name.clone(), selected.clone()))
        else {
            return compile_error(format!(
                "{inner} in WHERE is not returned by SELECT; name it there and test it by its \
                 AS name"
            ));
        };
        let field = path[0].clone();
        let rest = &path[1..];
        if span && !(rest == ["start"] || rest == ["end"]) {
            return compile_error(format!(
                "{name}{SPAN_SUFFIX}.{field} reads a span; test its start or end, such as \
                 {name}{SPAN_SUFFIX}.{field}.start"
            ));
        }
        if !span && !rest.is_empty() {
            return compile_error(format!("{name}.{field} reads a field under an answer"));
        }
        let (found, members) = self.bind_ai_extract(&call_node, &name)?;
        if !found.fields.contains(&field) {
            return compile_error(format!(
                "AI.EXTRACT {} has no field {}; its fields are {}",
                repr_str(&name),
                repr_str(&field),
                repr_list(&found.fields)
            ));
        }
        let member = members
            .iter()
            .find(|member| column_name(member, true) == format!("{name}.{field}"))
            .expect("a member per field");
        let alias = call_aliases(aliased_call(member, &[]).expect("an extraction"))[0].clone();
        let column = format!("{name}.{field}{}", if span { SPAN_SUFFIX } else { "" });
        let data_type = if span { span_type() } else { DataType::Utf8 };
        self.answer_types
            .push((alias.clone(), column.clone(), data_type));
        let mut read = Expr::Column(Column::new(Some(alias), column));
        for part in rest {
            read = datafusion::functions::core::expr_fn::get_field(read, part.as_str());
        }
        Ok(Some(read))
    }

    /// The condition `alias.column IN (labels)` on a label's column.
    pub(crate) fn label_condition(
        &mut self,
        alias: &str,
        column: &str,
        labels: &Value,
    ) -> Result<ValueExpression, CompileError> {
        let list = labels
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|label| Expr::Literal(ScalarValue::Utf8(Some(label.to_string())), None))
            .collect();
        self.answer_types
            .push((alias.to_string(), column.to_string(), DataType::Utf8));
        let test = Expr::InList(InList::new(
            Box::new(Expr::Column(Column::new(Some(alias), column))),
            list,
            false,
        ));
        self.computed(&test)
    }

    // ------------------------------------------------------------ values

    fn resolve_column(&self, column: &Column) -> Result<ColumnRef, CompileError> {
        let alias = match &column.relation {
            Some(relation) => relation.table().to_string(),
            None => {
                let owners: Vec<&(String, String, SchemaRef)> = self
                    .tables
                    .iter()
                    .filter(|(_, _, schema)| schema.field_with_name(&column.name).is_ok())
                    .collect();
                if owners.len() != 1 {
                    return compile_error(format!(
                        "column {} is {}; qualify it with a table alias",
                        repr_str(&column.name),
                        if owners.is_empty() {
                            "unknown"
                        } else {
                            "ambiguous"
                        }
                    ));
                }
                owners[0].0.clone()
            }
        };
        let Some(provider) = self.provider(&alias) else {
            return compile_error(format!("unknown table alias {}", repr_str(&alias)));
        };
        Ok(ColumnRef {
            alias,
            provider: provider.to_string(),
            column: column.name.clone(),
        })
    }

    /// Bind a value: a column or a field of one as itself, and any other
    /// value as an `Expression`.
    fn bind_value(&mut self, expr: &Expr) -> Result<ValueExpression, CompileError> {
        match expr {
            Expr::Column(column) => {
                if column.relation.is_none()
                    && let Some((_, output)) =
                        self.outputs.iter().find(|(name, _)| *name == column.name)
                {
                    let output = output.clone();
                    return self.bind_value(&output);
                }
                return self.resolve_column(column).map(ValueExpression::ColumnRef);
            }
            Expr::OuterReferenceColumn(_, column) => {
                return self.resolve_column(column).map(ValueExpression::ColumnRef);
            }
            Expr::Alias(alias) => return self.bind_value(&alias.expr),
            Expr::ScalarFunction(call)
                if call.name() == "get_field"
                    && call.args[1..]
                        .iter()
                        .all(|field| string_literal(field).is_some()) =>
            {
                let input = self.bind_value(&call.args[0])?;
                if matches!(
                    input,
                    ValueExpression::ColumnRef(_) | ValueExpression::GetField(_)
                ) {
                    let mut value = input;
                    for field in &call.args[1..] {
                        value = ValueExpression::GetField(GetField {
                            input: Box::new(value),
                            field: string_literal(field).expect("a field name"),
                        });
                    }
                    return Ok(value);
                }
            }
            _ => {}
        }
        if contains(expr, |inner| {
            function_name(inner).is_some_and(|name| name.starts_with("ai_") || name == "prompt")
        }) {
            return compile_error(format!("{expr} is not a value expression"));
        }
        self.computed(expr)
    }

    /// A value or condition DataFusion computes, as an `Expression`
    /// over the columns it reads.
    fn computed(&self, expr: &Expr) -> Result<ValueExpression, CompileError> {
        let expr = expr
            .clone()
            .transform_up(|node| match &node {
                Expr::Column(column) if column.relation.is_none() => Ok(
                    match self.outputs.iter().find(|(name, _)| *name == column.name) {
                        Some((_, output)) => Transformed::yes(output.clone()),
                        None => Transformed::no(node),
                    },
                ),
                _ => Ok(Transformed::no(node)),
            })
            .map_err(|error| CompileError::new(error.to_string()))?
            .data;
        if contains(&expr, |inner| {
            matches!(
                inner,
                Expr::AggregateFunction(_)
                    | Expr::WindowFunction(_)
                    | Expr::ScalarSubquery(_)
                    | Expr::InSubquery(_)
                    | Expr::Exists(_)
                    | Expr::Placeholder(_)
            )
        }) {
            return compile_error(format!("{expr} is not a value expression"));
        }
        crate::expressions::expression(&expr, |column| {
            let reference = self.resolve_column(column)?;
            let data_type = self.column_type(&reference)?;
            Ok((reference, data_type))
        })
        .map(ValueExpression::Value)
    }

    /// The Arrow type of a column a computed value reads: an answer's,
    /// or a table's column's.
    fn column_type(&self, reference: &ColumnRef) -> Result<DataType, CompileError> {
        if let Some((_, _, data_type)) = self
            .answer_types
            .iter()
            .find(|(alias, column, _)| *alias == reference.alias && *column == reference.column)
        {
            return Ok(data_type.clone());
        }
        self.tables
            .iter()
            .find(|(alias, _, _)| *alias == reference.alias)
            .and_then(|(_, _, schema)| schema.field_with_name(&reference.column).ok())
            .map(|field| field.data_type().clone())
            .ok_or_else(|| {
                CompileError::new(format!(
                    "column {} has no type the planner knows",
                    repr_str(&reference.name(true))
                ))
            })
    }

    // ------------------------------------------------------------ AI calls

    fn note_document_column(&mut self, column: &ValueExpression) -> Result<(), CompileError> {
        let alias = column_alias_of(column);
        match self
            .document_columns
            .iter()
            .find(|(found, _)| *found == alias)
        {
            Some((_, seen)) if seen != column => compile_error(format!(
                "alias {} is referenced through two columns ({}, {}); predicates over one \
                 table must share one document column so its KV is computed once",
                repr_str(&alias),
                repr_str(&column_name(seen, false)),
                repr_str(&column_name(column, false))
            )),
            Some(_) => Ok(()),
            None => {
                self.document_columns.push((alias, column.clone()));
                Ok(())
            }
        }
    }

    fn prompt_arguments(
        &mut self,
        prompt: &Expr,
        function: &str,
    ) -> Result<(String, Vec<ValueExpression>), CompileError> {
        if !is_call(prompt, "prompt") {
            return compile_error(format!(
                "{}'s first argument must be PROMPT('template', columns...)",
                function.to_uppercase()
            ));
        }
        let args = arguments(prompt);
        let Some(template) = args.first().and_then(string_literal) else {
            return compile_error("PROMPT's first argument must be a string template");
        };
        let mut references = Vec::new();
        for argument in &args[1..] {
            references.push(self.bind_value(argument)?);
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
    ) -> Result<(quail_physical::Prompt, Vec<String>), CompileError> {
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
    ) -> Result<(quail_physical::Prompt, Vec<String>), CompileError> {
        let mut aliases: Vec<String> = Vec::new();
        for reference in &references {
            let alias = quail_planner::expressions::argument_alias(reference);
            if !aliases.contains(&alias) {
                aliases.push(alias);
            }
        }
        let join = join.unwrap_or(aliases.len() > 1);
        if function == "ai_score" && join && template.matches("{0}").count() != 1 {
            return compile_error(
                "an AI.SCORE pair prompt mentions {0} exactly once, as the place its document \
                 is inserted; refer to it again in words",
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
            return compile_error(format!(
                "only AI_FILTER(PROMPT(...)) predicates are supported here, got: {term}"
            ));
        }
        let args = arguments(term);
        if args.is_empty() {
            return compile_error(
                "AI_FILTER's first argument must be PROMPT('template', columns...)",
            );
        }
        if args.len() > 2 {
            return compile_error("AI_FILTER takes PROMPT and at most one option object");
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
            return compile_error("AI.SCORE must be compared with <, <=, >, or >=");
        };
        let Some(value) = number_of(threshold).filter(|_| string_literal(threshold).is_none())
        else {
            return compile_error("AI.SCORE threshold must be a number");
        };
        if !(0.0..=1.0).contains(&value) {
            return compile_error("AI.SCORE threshold must be between 0 and 1");
        }
        let args = arguments(call);
        if args.len() > 2 {
            return compile_error("AI_SCORE takes PROMPT and at most one option object");
        }
        let options = parse_options(args.get(1), allowed)?;
        let Some(prompt) = args.first() else {
            return compile_error(
                "AI_SCORE's first argument must be PROMPT('template', columns...)",
            );
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
        let test =
            crate::expressions::expression(&compared, |_| Ok((score.clone(), DataType::Float64)))?;
        let test = ValueExpression::Value(test);
        Ok((ScoreTest { call, test }, options))
    }

    /// Bind a test of an `AI.CLASSIFY` call's label, `= 'label'` or
    /// `IN ('label', ...)`: the classification, the labels it accepts,
    /// and its selectivity, the HINT's or the call's own.
    pub(crate) fn bind_label_test(
        &mut self,
        classify: &Expr,
        items: &[&Expr],
        hint: Option<f64>,
    ) -> Result<(ModelCall, Value, Option<f64>), CompileError> {
        let (call, options) = self.bind_ai_classify(classify)?;
        let mut accepted = Vec::new();
        for item in items {
            match string_literal(item) {
                Some(text) => accepted.push(Value::String(text)),
                None => {
                    return compile_error("AI.CLASSIFY is compared with string literals");
                }
            }
        }
        let mut unique = accepted.clone();
        unique.sort_by_key(|value| value.to_string());
        unique.dedup();
        if unique.len() != accepted.len() {
            return compile_error("a filter on a label lists a label twice");
        }
        Ok((call, Value::Array(accepted), hint.or(options.selectivity)))
    }

    fn bind_ai_classify(&mut self, node: &Expr) -> Result<(ModelCall, Options), CompileError> {
        let args = arguments(node);
        if args.len() < 2 {
            return compile_error("AI.CLASSIFY takes a prompt and its categories");
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
                return compile_error(format!(
                    "AI.CLASSIFY's config is an object like {{'task_description': '...'}}, got \
                     {config}"
                ));
            };
            for (key, value) in fields {
                match key.to_lowercase().as_str() {
                    "selectivity" => match (string_literal(value), number_of(value)) {
                        (None, Some(number)) => options.selectivity = Some(number),
                        _ => return compile_error("selectivity must be a number"),
                    },
                    "task_description" => match string_literal(value) {
                        Some(text) => task = text,
                        None => return compile_error("task_description is a string"),
                    },
                    "probabilities" => match value {
                        Expr::Literal(ScalarValue::Boolean(Some(flag)), _) => probabilities = *flag,
                        _ => return compile_error("probabilities is true or false"),
                    },
                    other => {
                        return compile_error(format!(
                            "unknown AI.CLASSIFY option {}; the options are selectivity, \
                             task_description, and probabilities",
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
            let reference = self.bind_value(&args[0])?;
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
                "AI.CLASSIFY reads one document column, or one from each side of a join",
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
                "AI.EXTRACT takes a document column and an object of field questions",
            );
        }
        let reference = self.bind_value(&args[0])?;
        let Some(fields) = object_fields(&args[1]) else {
            return compile_error(format!(
                "AI.EXTRACT's second argument is an object of field questions such as \
                 {{'ends': 'When does the lease end?'}}, got {}",
                args[1]
            ));
        };
        let mut questions: Vec<(String, String)> = Vec::new();
        for (field, question) in fields {
            if questions.iter().any(|(found, _)| *found == field) {
                return compile_error(format!(
                    "AI.EXTRACT {} names {} twice",
                    repr_str(name),
                    repr_str(&field)
                ));
            }
            let Some(question) = string_literal(question) else {
                return compile_error(format!(
                    "AI.EXTRACT field {} is a question string, got {question}",
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

    fn bind_ai_map(&mut self, node: &Expr) -> Result<ModelCall, CompileError> {
        let args = arguments(node);
        if !(2..=3).contains(&args.len()) {
            return compile_error(
                "AI.MAP takes a document column, an instruction string, and an optional answer \
                 limit in tokens",
            );
        }
        let Some(instruction) = string_literal(&args[1]) else {
            return compile_error(format!(
                "AI.MAP's second argument is an instruction string, got {}",
                args[1]
            ));
        };
        let limit = match args.get(2) {
            None => DEFAULT_MAP_ANSWER_TOKENS,
            Some(found) => match literal_value(found).and_then(|value| value.as_i64()) {
                Some(limit) if limit >= 0 => limit,
                _ => {
                    return compile_error(format!(
                        "AI.MAP's answer limit is a whole number of tokens, got {found}"
                    ));
                }
            },
        };
        let reference = self.bind_value(&args[0])?;
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
                "AI.SAE_FEATURES takes PROMPT(...) and pooling => 'last' or 'max'",
            );
        };
        let (prompt, aliases) =
            self.bind_ai_call(&args[0], "ai_sae_features", Some(false), None)?;
        if aliases.len() != 1 {
            return compile_error("AI.SAE_FEATURES reads one document column");
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
    fn bind_ai_value(&mut self, node: &Expr) -> Result<ModelCall, CompileError> {
        if is_call(node, "ai_classify") {
            return Ok(self.bind_ai_classify(node)?.0);
        }
        let args = arguments(node);
        let Some(prompt) = args.first() else {
            return compile_error(
                "AI_SCORE's first argument must be PROMPT('template', columns...)",
            );
        };
        let (prompt, _) = self.bind_ai_call(prompt, "ai_score", None, None)?;
        Ok(model_call(prompt, "score"))
    }

    // ------------------------------------------------------------ checks

    pub(crate) fn validate_query(
        &self,
        columns: &[ValueExpression],
        structs: &[ExtractStruct],
    ) -> Result<(), CompileError> {
        // an unnested list has no provider; its name is not a table's
        let is_alias = |name: &str| self.provider(name).is_some_and(|found| !found.is_empty());
        for found in structs {
            if is_alias(&found.name) {
                return compile_error(format!(
                    "output name {} is also a table alias; pick another AS name",
                    repr_str(&found.name)
                ));
            }
        }
        for column in columns {
            let name = column_name(column, true);
            if let ValueExpression::Alias(_) = column {
                if is_alias(&name) {
                    return compile_error(format!(
                        "output name {} is also a table alias; pick another AS name",
                        repr_str(&name)
                    ));
                }
                if self
                    .tables
                    .iter()
                    .any(|(_, _, schema)| schema.field_with_name(&name).is_ok())
                {
                    return compile_error(format!(
                        "output name {} is also a column of a table in the query; pick another \
                         AS name",
                        repr_str(&name)
                    ));
                }
            }
        }
        let mut prompts_seen: Vec<(quail_physical::Prompt, String)> = Vec::new();
        for column in columns {
            let Some(call) = aliased_call(column, &[]) else {
                continue;
            };
            let name = column_name(column, true);
            if let Some((_, first)) = prompts_seen
                .iter()
                .find(|(prompt, _)| *prompt == *call.prompt)
            {
                return compile_error(format!(
                    "the same AI expression is projected as {} and {}; project it once",
                    repr_str(first),
                    repr_str(&name)
                ));
            }
            prompts_seen.push(((*call.prompt).clone(), name));
        }
        Ok(())
    }

    /// The Scan of a table in scope, reading its document column.
    ///
    /// # Panics
    ///
    /// No table in scope has the alias.
    pub(crate) fn scan(&self, alias: &str) -> Scan {
        let (_, provider, schema) = self
            .tables
            .iter()
            .find(|(found, _, _)| found == alias)
            .expect("a table in scope");
        Scan {
            column: match self
                .document_columns
                .iter()
                .find(|(found, _)| found == alias)
            {
                Some((_, column)) => ScanColumn::Expression(column.clone()),
                None => ScanColumn::Unnamed(String::new()),
            },
            schema: Some(schema_of(schema)),
            provider: provider.clone(),
            alias: alias.to_string(),
            columns: Vec::new(),
            predicates: Vec::new(),
        }
    }

    /// The unnested lists: the tables without a provider.
    pub(crate) fn unnested(&self) -> Vec<String> {
        self.tables
            .iter()
            .filter(|(_, provider, _)| provider.is_empty())
            .map(|(alias, _, _)| alias.clone())
            .collect()
    }

    // ------------------------------------------------------------ SELECT list

    fn output_of(&self, expr: &Expr) -> Option<Expr> {
        let Expr::Column(column) = expr else {
            return None;
        };
        self.outputs
            .iter()
            .find(|(name, _)| *name == column.name)
            .map(|(_, found)| found.clone())
    }

    /// Bind a SELECT list.
    ///
    /// # Errors
    ///
    /// An item does not bind, or two output columns share a name.
    pub(crate) fn bind_select(&mut self, projection: &[Expr]) -> Result<Selected, CompileError> {
        let mut selected = Selected {
            columns: Vec::new(),
            structs: Vec::new(),
            spans: Vec::new(),
            output: Vec::new(),
        };
        let mut extracts: Vec<(&str, Expr)> = Vec::new();
        let mut spans: Vec<(&str, Expr)> = Vec::new();
        for item in projection {
            let (node, alias) = unalias(item);
            let node = self.output_of(node).unwrap_or_else(|| node.clone());
            if is_call(&node, "ai_extract_span") {
                let Some(alias) = alias else {
                    return unsupported("AI.EXTRACT_SPAN without an AS name in SELECT");
                };
                spans.push((alias, node));
                continue;
            }
            if is_call(&node, "ai_extract")
                && let Some(alias) = alias
            {
                extracts.push((alias, node.clone()));
            }
            for column in self.bind_select_item(&node, alias, &mut selected.structs)? {
                selected.output.push(column_name(&column, true));
                if !selected.columns.contains(&column) {
                    selected.columns.push(column);
                }
            }
        }
        for (alias, span) in spans {
            let paired = extracts
                .iter()
                .find(|(_, extract)| arguments(extract) == arguments(&span));
            let Some((name, _)) = paired else {
                return compile_error(format!(
                    "AI.EXTRACT_SPAN(...) AS {alias} returns the answer positions of an \
                     AI.EXTRACT in SELECT with the same arguments"
                ));
            };
            selected.spans.push((*name).to_string());
        }
        let mut unique = selected.output.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != selected.output.len() {
            return compile_error(format!(
                "projection names must be unique, got {}",
                repr_list(&selected.output)
            ));
        }
        Ok(selected)
    }

    fn bind_select_item(
        &mut self,
        node: &Expr,
        alias: Option<&str>,
        structs: &mut Vec<ExtractStruct>,
    ) -> Result<Vec<ValueExpression>, CompileError> {
        let named = |expression: ValueExpression,
                     function: &str|
         -> Result<Vec<ValueExpression>, CompileError> {
            let Some(alias) = alias else {
                return unsupported(&format!("{function} without an AS name in SELECT"));
            };
            Ok(vec![ValueExpression::Alias(Alias {
                expression: Box::new(expression),
                name: alias.to_string(),
            })])
        };
        if is_call(node, "ai_score") {
            let call = self.bind_ai_value(node)?;
            if !(1..=2).contains(&call_aliases(&call).len()) {
                return compile_error("projected AI.SCORE must reference one or two relations");
            }
            return named(ValueExpression::ModelCall(call), "AI.SCORE");
        }
        if is_call(node, "ai_classify") {
            let call = self.bind_ai_value(node)?;
            return named(ValueExpression::ModelCall(call), "AI.CLASSIFY");
        }
        if is_call(node, "ai_extract") {
            let Some(alias) = alias else {
                return unsupported("AI.EXTRACT without an AS name in SELECT");
            };
            let (found, members) = self.bind_ai_extract(node, alias)?;
            structs.push(found);
            return Ok(members);
        }
        if is_call(node, "ai_sae_features") {
            let call = self.bind_ai_sae_features(node)?;
            return named(ValueExpression::ModelCall(call), "AI.SAE_FEATURES");
        }
        if is_call(node, "ai_map") {
            let call = self.bind_ai_map(node)?;
            return named(ValueExpression::ModelCall(call), "AI.MAP");
        }
        if has_ai_call(node) || contains(node, |inner| column_call_kind(inner).is_some()) {
            return unsupported("a model call inside a SELECT expression");
        }
        let value = self.bind_value(node)?;
        Ok(vec![match alias {
            Some(alias) => ValueExpression::Alias(Alias {
                expression: Box::new(value),
                name: alias.to_string(),
            }),
            None => value,
        }])
    }
}

// ---------------------------------------------------------------- helpers

fn is_get_field(expr: &Expr) -> bool {
    is_call(expr, "get_field")
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
fn score_comparison(expr: &Expr) -> Option<((&'static str, Operator), &Expr, &Expr)> {
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

/// The AI.CLASSIFY call and tested values of an = or IN test, or None.
pub(crate) fn label_test(expr: &Expr) -> Option<(&Expr, Vec<&Expr>)> {
    match expr {
        Expr::InList(InList {
            expr,
            list,
            negated: false,
        }) if is_call(expr, "ai_classify") => Some((expr, list.iter().collect())),
        // DataFusion writes a short IN list as an OR of equalities
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Or,
            right,
        }) => {
            let (call, mut labels) = label_test(left)?;
            let (other, more) = label_test(right)?;
            if call != other {
                return None;
            }
            labels.extend(more);
            Some((call, labels))
        }
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        }) => {
            if is_call(left, "ai_classify") {
                Some((left, vec![right]))
            } else if is_call(right, "ai_classify") {
                Some((right, vec![left]))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Refuse a test of the label `name` that accepts a label its
/// classification `call` lacks.
pub(crate) fn check_labels(
    name: &str,
    accepted: &Value,
    call: &ModelCall,
) -> Result<(), CompileError> {
    let values = match accepted {
        Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    };
    let labels: Vec<Value> = call.labels.as_array().cloned().unwrap_or_default();
    let unknown: BTreeSet<String> = values
        .iter()
        .filter(|value| !labels.contains(value))
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    let unknown: Vec<String> = unknown.into_iter().collect();
    let written: Vec<String> = labels
        .iter()
        .filter_map(|label| label.as_str().map(str::to_string))
        .collect();
    compile_error(format!(
        "the filter on {} lists {}, which are not among its labels {}",
        repr_str(name),
        repr_list(&unknown),
        repr_list(&written)
    ))
}

fn model_call(prompt: quail_physical::Prompt, kind: &str) -> ModelCall {
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
            "AI.CLASSIFY categories are an ARRAY of labels, of (label, description) pairs, or \
             of {'label': ..., 'description': ...} objects, or a registered label table's name",
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
                    return compile_error(format!(
                        "a category object has a label and an optional description, got {item}"
                    ));
                }
            }
        } else {
            return compile_error(format!(
                "AI.CLASSIFY category {item} is not a label, a (label, description) pair, or a \
                 category object"
            ));
        };
        let Some(label) = string_literal(label) else {
            return compile_error("AI.CLASSIFY labels are string literals");
        };
        let description = match description {
            None => String::new(),
            Some(Expr::Literal(value, _)) if value.is_null() => String::new(),
            Some(found) => match string_literal(found) {
                Some(text) => text,
                None => {
                    return compile_error("an AI.CLASSIFY description is a string literal or NULL");
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

/// An Arrow schema as the plan codec writes it: IPC bytes in base64.
fn schema_of(schema: &SchemaRef) -> Schema {
    use base64::Engine;
    use datafusion::arrow::ipc::writer::{
        DictionaryTracker, IpcDataGenerator, IpcWriteOptions, write_message,
    };
    let options = IpcWriteOptions::default();
    let mut tracker = DictionaryTracker::new(false);
    let encoded = IpcDataGenerator::default().schema_to_bytes_with_dictionary_tracker(
        schema,
        &mut tracker,
        &options,
    );
    let mut bytes = Vec::new();
    write_message(&mut bytes, encoded, &options).expect("an IPC schema message");
    Schema {
        ipc: base64::engine::general_purpose::STANDARD.encode(bytes),
    }
}
