//! Bind the terms of a model region's DataFusion plan into Quail's
//! expressions.
//!
//! [`crate::derive`] hands the [`Binder`] each table, ON and WHERE term,
//! AI call, and SELECT list item in the order the query wrote them. The
//! binder binds each AI call's prompt in the session model's chat turn
//! and layout, each condition as a [`Condition`], and the SELECT list as
//! the region's projection, and refuses SQL outside Quail's language.

use std::collections::BTreeMap;

use crate::functions::function_name;
use crate::functions::{ai_function_name, column_call_kind};
use datafusion::arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use datafusion::common::tree_node::TreeNode;
use datafusion::common::{Column, ScalarValue};
use datafusion::logical_expr::expr::{InList, Like, ScalarFunction};
use datafusion::logical_expr::{BinaryExpr, Distinct, Expr, LogicalPlan, Operator, SortExpr};
use quail_physical::{
    Alias, ColumnRef, Condition, Connective, Equality, ExtractStruct, GetField, Literal, ModelCall,
    Not, RegularPredicate, ScalarFunction as QuailFunction, SortExpression, SortKey,
    TaggedModelCall, ValueExpression,
};
use quail_planner::expressions::{
    CompileError, SPAN_SUFFIX, aliased_call, call_aliases, column_alias_of, column_name,
    column_references, compile_error, repr_str,
};
use quail_planner::operators::{Scan, ScanColumn, Schema};
use quail_planner::prompts::{self, Turn};
use serde_json::Value;

/// The answer limit of an AI.MAP call written without one.
const DEFAULT_MAP_ANSWER_TOKENS: i64 = 256;

// ---------------------------------------------------------------- the SELECT

pub use quail_planner::expressions::UNSUPPORTED;

pub(crate) fn unsupported<T>(what: &str) -> Result<T, CompileError> {
    compile_error(format!("{what} {UNSUPPORTED}"))
}

/// The LIMIT, OFFSET, ORDER BY, DISTINCT, and SELECT list of one SELECT,
/// and the relation below them.
pub(crate) struct Top<'a> {
    pub(crate) limit: Option<i64>,
    pub(crate) offset: i64,
    pub(crate) distinct: bool,
    pub(crate) order: Vec<SortExpr>,
    /// The SELECT list, with an ORDER BY column it does not return read
    /// through the outer SELECT list.
    pub(crate) projection: Vec<Expr>,
    /// The input of the SELECT list.
    pub(crate) relation: &'a LogicalPlan,
}

/// Read the operators of a SELECT above its relation.
///
/// # Errors
///
/// The LIMIT or OFFSET is not a literal, the SELECT has a DISTINCT ON,
/// or it has no SELECT list.
pub(crate) fn read_top(plan: &LogicalPlan) -> Result<Top<'_>, CompileError> {
    let mut node = plan;
    let (mut limit, mut offset, mut distinct, mut order) = (None, 0, false, Vec::new());
    let mut outer_projection: Option<&Vec<Expr>> = None;
    loop {
        match node {
            LogicalPlan::Limit(found) => {
                use datafusion::logical_expr::{FetchType, SkipType};
                match found.get_skip_type() {
                    Ok(SkipType::Literal(skip)) => offset = skip as i64,
                    _ => return compile_error("OFFSET must be a nonnegative integer"),
                }
                match found.get_fetch_type() {
                    Ok(FetchType::Literal(fetch)) => limit = fetch.map(|fetch| fetch as i64),
                    _ => return compile_error("LIMIT must be a positive integer"),
                }
                node = &found.input;
            }
            LogicalPlan::Sort(found) => {
                order = found.expr.clone();
                node = &found.input;
            }
            LogicalPlan::Distinct(Distinct::All(input)) => {
                distinct = true;
                node = input;
            }
            LogicalPlan::Distinct(Distinct::On(_)) => {
                return compile_error("DISTINCT ON is not supported; use DISTINCT");
            }
            LogicalPlan::Projection(found)
                if outer_projection.is_none()
                    && matches!(
                        *found.input,
                        LogicalPlan::Sort(_) | LogicalPlan::Distinct(_)
                    ) =>
            {
                outer_projection = Some(&found.expr);
                node = &found.input;
            }
            _ => break,
        }
    }
    let LogicalPlan::Projection(projection) = node else {
        return unsupported("a region without a SELECT list");
    };
    let mut expressions = projection.expr.clone();
    if let Some(outer) = outer_projection {
        // ORDER BY a column SELECT does not return: the inner
        // projection adds it, the outer one drops it
        expressions = outer
            .iter()
            .map(|expr| match expr {
                Expr::Column(column) => projection
                    .expr
                    .iter()
                    .find(|inner| output_name(inner) == column.name)
                    .cloned()
                    .unwrap_or_else(|| expr.clone()),
                other => other.clone(),
            })
            .collect();
    }
    Ok(Top {
        limit,
        offset,
        distinct,
        order,
        projection: expressions,
        relation: projection.input.as_ref(),
    })
}

/// Refuse a SELECT whose input is a GROUP BY or a window.
///
/// DataFusion runs a GROUP BY or window above the region, or as an input
/// it computes for the region.
///
/// # Errors
///
/// The input groups or windows rows.
pub(crate) fn refuse_grouped_input(relation: &LogicalPlan) -> Result<(), CompileError> {
    let groups =
        |plan: &LogicalPlan| matches!(plan, LogicalPlan::Aggregate(_) | LogicalPlan::Window(_));
    if groups(relation) || matches!(relation, LogicalPlan::Filter(filter) if groups(&filter.input))
    {
        return compile_error(
            "a model call that reads the result of a GROUP BY or a window is not \
             supported; read it from a CTE or FROM subquery instead",
        );
    }
    Ok(())
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

fn output_name(expr: &Expr) -> String {
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

/// The binder method a column-operator call takes, as `column_call`.
fn walk(expr: &Expr, found: &mut dyn FnMut(&Expr) -> bool) -> bool {
    if found(expr) {
        return true;
    }
    let mut any = false;
    let _ = expr.apply_children(|child| {
        if walk(child, found) {
            any = true;
            return Ok(datafusion::common::tree_node::TreeNodeRecursion::Stop);
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    });
    any
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
                if has_ai_call_under(child, false) {
                    any = true;
                    return Ok(datafusion::common::tree_node::TreeNodeRecursion::Stop);
                }
                Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
            });
            any
        }
    }
}

fn string_literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(text)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(text)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(text)), _) => Some(text.clone()),
        _ => None,
    }
}

/// A SQL literal as Python's `sql_literal` reads it, or None.
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
        other => Value::String(string_of(other)?),
    })
}

fn string_of(value: &ScalarValue) -> Option<String> {
    match value {
        ScalarValue::Utf8(Some(text))
        | ScalarValue::LargeUtf8(Some(text))
        | ScalarValue::Utf8View(Some(text)) => Some(text.clone()),
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
    /// (alias, provider) of each table in scope, in FROM and JOIN order.
    tables: Vec<(String, String)>,
    schemas: BTreeMap<String, SchemaRef>,
    /// alias -> the document expression its prompts read.
    document_columns: Vec<(String, ValueExpression)>,
    /// A column operator only a WHERE test reads -> its name.
    tested: Vec<(ModelCall, String)>,
    /// alias -> the ON equalities of its JOIN, in JOIN order.
    conditions: Vec<(String, Vec<Equality>)>,
    /// AS name -> the column-operator call SELECT gives it.
    pub(crate) selected: Vec<(String, Expr)>,
    /// Output column name -> the group, aggregate, or window expression.
    pub(crate) outputs: Vec<(String, Expr)>,
}

/// A bound ON term that is no AI predicate.
pub(crate) enum OnTerm {
    Condition(Condition),
    Equality(Box<Equality>),
}

impl Binder {
    /// A binder with no table in scope yet, for prompts rendered with
    /// `turn` and `layout`.
    pub(crate) fn new(turn: &Turn, layout: &str) -> Self {
        Self {
            turn: turn.clone(),
            layout: layout.to_string(),
            tables: Vec::new(),
            schemas: BTreeMap::new(),
            document_columns: Vec::new(),
            tested: Vec::new(),
            conditions: Vec::new(),
            selected: Vec::new(),
            outputs: Vec::new(),
        }
    }

    fn provider(&self, alias: &str) -> Option<&str> {
        self.tables
            .iter()
            .find(|(found, _)| found == alias)
            .map(|(_, provider)| provider.as_str())
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
        self.tables.push((alias.to_string(), provider.to_string()));
        self.schemas.insert(provider.to_string(), schema);
        Ok(())
    }

    /// Bind an ON term of the JOIN of `joined` that is no AI predicate:
    /// a condition, or an equality of the join's keys.
    pub(crate) fn bind_plain_on_term(
        &mut self,
        term: &Expr,
        joined: &str,
    ) -> Result<OnTerm, CompileError> {
        if walk(term, &mut |inner| is_call(inner, "ai_score")) {
            return compile_error("AI.SCORE must be compared with <, <=, >, or >=");
        }
        let condition = if has_ai_call(term) {
            None
        } else {
            self.bind_condition(term, true)?
        };
        Ok(match condition {
            Some(condition) => OnTerm::Condition(condition),
            None => OnTerm::Equality(Box::new(self.parse_equality(term, joined)?)),
        })
    }

    fn parse_equality(&mut self, term: &Expr, joined: &str) -> Result<Equality, CompileError> {
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

    /// Bind a WHERE term that asks no model: the condition and the
    /// selectivity its HINT gives, or None for a term that asks one.
    pub(crate) fn bind_plain_where_term(
        &mut self,
        term: &Expr,
    ) -> Result<Option<(Condition, Option<f64>)>, CompileError> {
        let (term, hint) = self.unwrap_hint(term)?;
        if !has_ai_call(&term)
            && label_test(&term).is_none()
            && let Some(condition) = self.bind_condition(&term, false)?
        {
            return Ok(Some((condition, hint)));
        }
        Ok(None)
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
            return unsupported(&format!("{term}, an OR of a model call and other tests,"));
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
        if walk(&term, &mut |inner| is_call(inner, "ai_classify")) {
            return compile_error(
                "AI.CLASSIFY in WHERE is tested with = 'label' or IN ('label', ...)",
            );
        }
        if walk(&term, &mut |inner| is_call(inner, "ai_extract")) {
            return compile_error(
                "an AI.EXTRACT call is not a predicate; test one of its fields, such as \
                 AI.EXTRACT(...).end_date IS NOT NULL, or its AS name from SELECT",
            );
        }
        if score_comparison(&term).is_some() {
            self.bind_ai_score_test(&term, &["selectivity"], None)?;
        } else {
            if walk(&term, &mut |inner| is_call(inner, "ai_score")) {
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

    /// Bind a term that asks no model as a condition; None for a join key.
    fn bind_condition(
        &mut self,
        term: &Expr,
        join_keys: bool,
    ) -> Result<Option<Condition>, CompileError> {
        match term {
            Expr::Not(inner) => Ok(Some(not(self.bind_operand(inner)?))),
            // x BETWEEN low AND high reads as x >= low AND x <= high
            Expr::Between(between) => {
                let low = Expr::BinaryExpr(BinaryExpr::new(
                    between.expr.clone(),
                    Operator::GtEq,
                    between.low.clone(),
                ));
                let high = Expr::BinaryExpr(BinaryExpr::new(
                    between.expr.clone(),
                    Operator::LtEq,
                    between.high.clone(),
                ));
                let both = low.and(high);
                let rewritten = if between.negated {
                    Expr::Not(Box::new(both))
                } else {
                    both
                };
                self.bind_condition(&rewritten, join_keys)
            }
            Expr::BinaryExpr(BinaryExpr {
                op: op @ (Operator::And | Operator::Or),
                ..
            }) => {
                let mut operands = Vec::new();
                for part in flatten(term, *op) {
                    operands.push(self.bind_operand(&part)?);
                }
                Ok(Some(if *op == Operator::And {
                    Condition::And(Connective { operands })
                } else {
                    Condition::Or(Connective { operands })
                }))
            }
            _ => {
                let test = self.parse_plain_test(term)?;
                if join_keys && self.join_condition(&test) {
                    return Ok(None);
                }
                let predicate = match self.column_test(&test)? {
                    Some(predicate) => predicate,
                    None => self.bind_plain_test(&test)?,
                };
                Ok(Some(if test.negated {
                    not(predicate)
                } else {
                    predicate
                }))
            }
        }
    }

    fn bind_operand(&mut self, term: &Expr) -> Result<Condition, CompileError> {
        Ok(self
            .bind_condition(term, false)?
            .expect("a condition outside ON is never a join key"))
    }

    fn join_condition(&mut self, test: &PlainTest) -> bool {
        let Expr::BinaryExpr(BinaryExpr { left, op, right }) = &test.operand else {
            return false;
        };
        if comparison_function(*op).is_none() || test.comparison != "is true" {
            return false;
        }
        let sides = [self.bind_value(left), self.bind_value(right)];
        let [Ok(left), Ok(right)] = sides else {
            return false;
        };
        let aliases = |side: &ValueExpression| -> Vec<String> {
            let mut found: Vec<String> = column_references(side)
                .into_iter()
                .map(|reference| reference.alias)
                .collect();
            found.sort();
            found.dedup();
            found
        };
        let (left, right) = (aliases(&left), aliases(&right));
        left.len() == 1 && right.len() == 1 && left != right
    }

    fn parse_plain_test(&mut self, term: &Expr) -> Result<PlainTest, CompileError> {
        match term {
            Expr::IsNull(inner) => {
                if let Some((operand, function)) = operand_of(inner) {
                    return Ok(PlainTest::new(operand, function, "is null", Value::Null));
                }
            }
            Expr::IsNotNull(inner) => {
                if let Some((operand, function)) = operand_of(inner) {
                    let mut test = PlainTest::new(operand, function, "is null", Value::Null);
                    test.negated = true;
                    return Ok(test);
                }
            }
            Expr::InList(InList {
                expr,
                list,
                negated,
            }) if !list.is_empty() => {
                if let Some((operand, function)) = operand_of(expr) {
                    let mut values = Vec::new();
                    for item in list {
                        values.push(sql_literal(item)?);
                    }
                    let mut test = PlainTest::new(operand, function, "in", Value::Array(values));
                    test.negated = *negated;
                    return Ok(test);
                }
            }
            Expr::Like(Like {
                negated,
                expr,
                pattern,
                case_insensitive,
                ..
            }) => {
                let Some((operand, function)) = operand_of(expr) else {
                    return compile_error(format!("{term} tests a column"));
                };
                let Some(pattern) = string_literal(pattern) else {
                    let name = if *case_insensitive { "ILIKE" } else { "LIKE" };
                    return compile_error(format!("{name} compares with a string"));
                };
                let comparison = if *case_insensitive { "ilike" } else { "like" };
                let mut test =
                    PlainTest::new(operand, function, comparison, Value::String(pattern));
                test.negated = *negated;
                return Ok(test);
            }
            Expr::ScalarFunction(call)
                if matches!(
                    call.name(),
                    "starts_with" | "ends_with" | "contains" | "regexp_like"
                ) && call.args.len() == 2 =>
            {
                let comparison = match call.name() {
                    "starts_with" => "starts with",
                    "ends_with" => "ends with",
                    "contains" => "contains",
                    _ => "regex",
                };
                let Some((operand, function)) = operand_of(&call.args[0]) else {
                    return compile_error(format!("{term} tests a column"));
                };
                let Some(mut pattern) = string_literal(&call.args[1]) else {
                    return compile_error(format!(
                        "{} compares with a string",
                        comparison.to_uppercase()
                    ));
                };
                if comparison == "regex" {
                    // Snowflake's REGEXP_LIKE matches the whole value
                    pattern = format!("^(?:{pattern})$");
                }
                return Ok(PlainTest::new(
                    operand,
                    function,
                    comparison,
                    Value::String(pattern),
                ));
            }
            Expr::BinaryExpr(BinaryExpr { left, op, right }) if plain_comparison(*op).is_some() => {
                let comparison = plain_comparison(*op).expect("a comparison");
                let (found_left, found_right) = (operand_of(left), operand_of(right));
                match (found_left, found_right) {
                    (Some((operand, function)), None) => {
                        return Ok(PlainTest::new(
                            operand,
                            function,
                            comparison,
                            sql_literal(right)?,
                        ));
                    }
                    (None, Some((operand, function))) => {
                        return Ok(PlainTest::new(
                            operand,
                            function,
                            flipped(comparison),
                            sql_literal(left)?,
                        ));
                    }
                    (Some(_), Some(_)) => {
                        return Ok(PlainTest::new(term.clone(), None, "is true", Value::Null));
                    }
                    (None, None) => {}
                }
            }
            _ => {}
        }
        if matches!(term, Expr::Column(_) | Expr::OuterReferenceColumn(..))
            || is_get_field(term)
            || computed_value(term)
        {
            return Ok(PlainTest::new(term.clone(), None, "is true", Value::Null));
        }
        compile_error(format!(
            "{term} is not supported; a regular predicate is =, <>, <, <=, >, >=, IN \
             (literals), IS NULL, IS NOT NULL, LIKE, ILIKE, CONTAINS, STARTSWITH, ENDSWITH, \
             REGEXP_LIKE, or a boolean column or scalar function of columns"
        ))
    }

    fn bind_plain_test(&mut self, test: &PlainTest) -> Result<Condition, CompileError> {
        let expression = self.operand_value(test)?;
        let predicate = RegularPredicate {
            expression,
            comparison: test.comparison.clone(),
            value: test.value.clone(),
        };
        quail_planner::expressions::validate_test(&predicate)?;
        Ok(Condition::Compare(Box::new(predicate)))
    }

    fn operand_value(&mut self, test: &PlainTest) -> Result<ValueExpression, CompileError> {
        let value = self.bind_value(&test.operand)?;
        Ok(match &test.function {
            Some(function) => ValueExpression::ScalarFunction(QuailFunction {
                function: function.clone(),
                args: vec![value],
            }),
            None => value,
        })
    }

    /// A test of a column operator's answer: an AI.CLASSIFY, AI.MAP, or
    /// AI.SAE_FEATURES call, or a field of an AI.EXTRACT call or its span.
    fn column_test(&mut self, test: &PlainTest) -> Result<Option<Condition>, CompileError> {
        if let Some(kind) = column_call_kind(&test.operand) {
            if kind == "extract" {
                return compile_error("test a field of AI.EXTRACT, such as lease.end_date");
            }
            let call = self.bind_column_call(&test.operand, kind)?;
            let aliases = call_aliases(&call);
            if aliases.len() != 1 {
                return compile_error("a filter on a label tests a one-document classification");
            }
            let alias = aliases[0].clone();
            let name = self
                .selected
                .iter()
                .find(|(_, node)| *node == test.operand)
                .map(|(name, _)| name.clone())
                .unwrap_or_else(|| format!("__column_{alias}_{}", self.tested.len()));
            let name = match self.tested.iter().find(|(found, _)| *found == call) {
                Some((_, existing)) => existing.clone(),
                None => {
                    self.tested.push((call.clone(), name.clone()));
                    name
                }
            };
            let reference = self.column_ref(&alias, &name);
            let expression = match &test.function {
                Some(function) => ValueExpression::ScalarFunction(QuailFunction {
                    function: function.clone(),
                    args: vec![reference],
                }),
                None => reference,
            };
            let predicate = RegularPredicate {
                expression,
                comparison: test.comparison.clone(),
                value: test.value.clone(),
            };
            if call.kind == "features"
                && (test.function.is_some()
                    || !["is null", "is not null"].contains(&test.comparison.as_str()))
            {
                return compile_error(
                    "AI.SAE_FEATURES returns a list; test IS NULL or IS NOT NULL, or UNNEST it \
                     to test its fields",
                );
            }
            quail_planner::expressions::validate_test(&predicate)?;
            return Ok(Some(Condition::Compare(Box::new(predicate))));
        }
        self.extract_test(test)
    }

    fn extract_test(&mut self, test: &PlainTest) -> Result<Option<Condition>, CompileError> {
        // get_field(ai_extract(...), field), or
        // get_field(get_field(ai_extract_span(...), field), start|end)
        let mut path = Vec::new();
        let mut node = &test.operand;
        while is_get_field(node) {
            let args = arguments(node);
            path.push(string_literal(&args[1]).unwrap_or_default());
            node = &args[0];
        }
        path.reverse();
        let span = is_call(node, "ai_extract_span");
        if !(span || is_call(node, "ai_extract")) || path.is_empty() {
            return Ok(None);
        }
        let Some((name, call_node)) = self
            .selected
            .iter()
            .find(|(_, selected)| {
                is_call(selected, "ai_extract") && arguments(selected) == arguments(node)
            })
            .map(|(name, selected)| (name.clone(), selected.clone()))
        else {
            return compile_error(format!(
                "{node} in WHERE is not returned by SELECT; name it there and test it by its \
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
                quail_planner::expressions::repr_list(&found.fields)
            ));
        }
        let member = members
            .iter()
            .find(|member| column_name(member, true) == format!("{name}.{field}"))
            .expect("a member per field");
        let alias = call_aliases(aliased_call(member, &[]).expect("an extraction"))[0].clone();
        let column = format!("{name}.{field}{}", if span { SPAN_SUFFIX } else { "" });
        let mut expression = self.column_ref(&alias, &column);
        for part in rest {
            expression = ValueExpression::GetField(GetField {
                input: Box::new(expression),
                field: part.clone(),
            });
        }
        if let Some(function) = &test.function {
            expression = ValueExpression::ScalarFunction(QuailFunction {
                function: function.clone(),
                args: vec![expression],
            });
        }
        let predicate = RegularPredicate {
            expression,
            comparison: test.comparison.clone(),
            value: test.value.clone(),
        };
        quail_planner::expressions::validate_test(&predicate)?;
        Ok(Some(Condition::Compare(Box::new(predicate))))
    }

    fn column_ref(&self, alias: &str, column: &str) -> ValueExpression {
        ValueExpression::ColumnRef(ColumnRef {
            alias: alias.to_string(),
            provider: self.provider(alias).unwrap_or_default().to_string(),
            column: column.to_string(),
        })
    }

    // ------------------------------------------------------------ values

    fn resolve_column(&self, column: &Column) -> Result<ValueExpression, CompileError> {
        let alias = match &column.relation {
            Some(relation) => relation.table().to_string(),
            None => {
                let owners: Vec<&(String, String)> = self
                    .tables
                    .iter()
                    .filter(|(_, provider)| {
                        self.schemas[provider].field_with_name(&column.name).is_ok()
                    })
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
        Ok(ValueExpression::ColumnRef(ColumnRef {
            alias,
            provider: provider.to_string(),
            column: column.name.clone(),
        }))
    }

    /// Bind a value expression: a column, a literal, or a function of them.
    fn bind_value(&mut self, expr: &Expr) -> Result<ValueExpression, CompileError> {
        self.bind_value_at(expr, true)
    }

    fn bind_value_at(&mut self, expr: &Expr, top: bool) -> Result<ValueExpression, CompileError> {
        let function = |name: &str, args: Vec<ValueExpression>| {
            ValueExpression::ScalarFunction(QuailFunction {
                function: name.to_string(),
                args,
            })
        };
        let literal = |value: Value| ValueExpression::Literal(Literal { value });
        Ok(match expr {
            Expr::Column(column) => {
                if column.relation.is_none()
                    && let Some((_, output)) =
                        self.outputs.iter().find(|(name, _)| *name == column.name)
                {
                    let output = output.clone();
                    return self.bind_value_at(&output, top);
                }
                self.resolve_column(column)?
            }
            Expr::OuterReferenceColumn(_, column) => self.resolve_column(column)?,
            Expr::Alias(alias) => self.bind_value_at(&alias.expr, top)?,
            Expr::Literal(..) | Expr::Negative(_) if literal_value(expr).is_some() => {
                literal(literal_value(expr).expect("a literal"))
            }
            Expr::Negative(inner) => function("negate", vec![self.bind_value_at(inner, false)?]),
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                let (left, right) = (
                    self.bind_value_at(left, false)?,
                    self.bind_value_at(right, false)?,
                );
                match op {
                    Operator::Plus => function("add", vec![left, right]),
                    Operator::Minus => function("subtract", vec![left, right]),
                    Operator::Multiply => function("multiply", vec![left, right]),
                    Operator::Divide => function(
                        "divide",
                        vec![
                            function("cast", vec![left, literal(Value::from("float64"))]),
                            right,
                        ],
                    ),
                    Operator::StringConcat => function(
                        "binary_join_element_wise",
                        vec![left, right, literal(Value::from(""))],
                    ),
                    Operator::Modulo => {
                        return compile_error(format!(
                            "{expr}: Arrow compute has no remainder or integer division function"
                        ));
                    }
                    Operator::And => function("and_kleene", vec![left, right]),
                    Operator::Or => function("or_kleene", vec![left, right]),
                    op => match comparison_function(*op) {
                        Some(name) => function(name, vec![left, right]),
                        None => return unsupported(&format!("the operator {op}")),
                    },
                }
            }
            Expr::Not(inner) => match inner.as_ref() {
                Expr::IsNull(value) => {
                    function("is_valid", vec![self.bind_value_at(value, false)?])
                }
                other => function("invert", vec![self.bind_value_at(other, false)?]),
            },
            Expr::IsNull(inner) => function("is_null", vec![self.bind_value_at(inner, false)?]),
            Expr::IsNotNull(inner) => function("is_valid", vec![self.bind_value_at(inner, false)?]),
            Expr::InList(InList {
                expr,
                list,
                negated,
            }) => {
                let value = self.bind_value_at(expr, false)?;
                let mut values = Vec::new();
                for item in list {
                    values.push(sql_literal(item)?);
                }
                let found = function(
                    "if_else",
                    vec![
                        function("is_null", vec![value.clone()]),
                        literal(Value::Null),
                        function("is_in", vec![value, literal(Value::Array(values))]),
                    ],
                );
                if *negated {
                    function("invert", vec![found])
                } else {
                    found
                }
            }
            Expr::Cast(cast) => {
                let Some(kind) = cast_type(cast.field.data_type()) else {
                    return compile_error(format!(
                        "CAST to {} is not supported",
                        cast.field.data_type()
                    ));
                };
                function(
                    "cast",
                    vec![
                        self.bind_value_at(&cast.expr, false)?,
                        literal(Value::from(kind)),
                    ],
                )
            }
            Expr::Case(case) => {
                let mut conditions = Vec::new();
                let mut values = Vec::new();
                for (when, then) in &case.when_then_expr {
                    let condition = match &case.expr {
                        Some(subject) => (**subject).clone().eq((**when).clone()),
                        None => (**when).clone(),
                    };
                    conditions.push(self.bind_value_at(&condition, false)?);
                    values.push(self.bind_value_at(then, false)?);
                }
                if let Some(default) = &case.else_expr {
                    values.push(self.bind_value_at(default, false)?);
                }
                let mut args = vec![function("make_struct", conditions)];
                args.extend(values);
                function("case_when", args)
            }
            Expr::Between(between) => {
                let value = self.bind_value_at(&between.expr, false)?;
                let found = function(
                    "and_kleene",
                    vec![
                        function(
                            "greater_equal",
                            vec![value.clone(), self.bind_value_at(&between.low, false)?],
                        ),
                        function(
                            "less_equal",
                            vec![value, self.bind_value_at(&between.high, false)?],
                        ),
                    ],
                );
                if between.negated {
                    function("invert", vec![found])
                } else {
                    found
                }
            }
            Expr::ScalarFunction(call)
                if translated(call.name())
                    && !(matches!(call.name(), "btrim" | "trim") && call.args.len() == 1) =>
            {
                let mut args = Vec::new();
                for arg in &call.args {
                    args.push(self.bind_value_at(arg, false)?);
                }
                let value = translate(call.name(), &call.args, args)?;
                quail_planner::expressions::validate_expression(&value)?;
                value
            }
            Expr::ScalarFunction(call) if call.name() == "get_field" => {
                let mut value = self.bind_value_at(&call.args[0], false)?;
                for field in &call.args[1..] {
                    let Some(field) = string_literal(field) else {
                        return unsupported("a field access by a computed name");
                    };
                    value = ValueExpression::GetField(GetField {
                        input: Box::new(value),
                        field,
                    });
                }
                value
            }
            Expr::ScalarFunction(call) => {
                let name = match (call.name(), top) {
                    ("character_length" | "length" | "char_length", _) => "length",
                    ("btrim" | "trim", true) if call.args.len() == 1 => "trim",
                    ("btrim" | "trim", false) if call.args.len() == 1 => "utf8_trim_whitespace",
                    (name, _) if name.starts_with("ai_") || name == "prompt" => {
                        return compile_error(format!("{expr} is not a value expression"));
                    }
                    (name, _) => name,
                };
                let mut args = Vec::new();
                for arg in &call.args {
                    args.push(self.bind_value_at(arg, false)?);
                }
                let value = function(name, args);
                quail_planner::expressions::validate_expression(&value)?;
                value
            }
            other => return compile_error(format!("{other} is not a value expression")),
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
        let call = ModelCall {
            prompt: Box::new(prompt),
            kind: "boolean".to_string(),
            labels: Value::Array(Vec::new()),
            descriptions: Value::Array(Vec::new()),
            probabilities: false,
            question: String::new(),
            pooling: String::new(),
            max_answer_tokens: 0,
        };
        Ok((call, options))
    }

    /// Bind a comparison of an `AI.SCORE` call with a threshold: the
    /// test and the call's options.
    pub(crate) fn bind_ai_score_test(
        &mut self,
        term: &Expr,
        allowed: &[&str],
        join: Option<bool>,
    ) -> Result<(RegularPredicate, Options), CompileError> {
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
        let predicate = RegularPredicate {
            expression: ValueExpression::ModelCall(model_call(prompt, "score")),
            comparison: comparison.to_string(),
            value: Value::from(value),
        };
        Ok((predicate, options))
    }

    /// Bind a test of an `AI.CLASSIFY` call's label, `= 'label'` or
    /// `IN ('label', ...)`: the test and its selectivity, the HINT's or
    /// the call's own.
    pub(crate) fn bind_label_test(
        &mut self,
        classify: &Expr,
        items: &[&Expr],
        hint: Option<f64>,
    ) -> Result<(RegularPredicate, Option<f64>), CompileError> {
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
        let test = RegularPredicate {
            expression: ValueExpression::ModelCall(call),
            comparison: "in".to_string(),
            value: Value::Array(accepted),
        };
        Ok((test, hint.or(options.selectivity)))
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

    fn bind_ai_extract(
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

    fn bind_column_call(&mut self, node: &Expr, kind: &str) -> Result<ModelCall, CompileError> {
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

    // ------------------------------------------------------------ ORDER BY

    pub(crate) fn bind_order(
        &mut self,
        order: &[SortExpr],
        columns: &[ValueExpression],
        structs: &[ExtractStruct],
    ) -> Result<Vec<SortKey>, CompileError> {
        let named: Vec<(String, SortExpression)> = columns
            .iter()
            .map(|column| {
                (
                    column_name(column, true),
                    SortExpression::Value(column.clone()),
                )
            })
            .collect();
        let mut keys = Vec::new();
        for sort in order {
            let target = &sort.expr;
            if let Expr::Column(column) = target
                && column.relation.is_none()
            {
                if structs.iter().any(|found| found.name == column.name) {
                    return compile_error(format!(
                        "ORDER BY does not compare the AI.EXTRACT struct column {}; extract the \
                         field as its own column",
                        repr_str(&column.name)
                    ));
                }
                if let Some((_, found)) = named.iter().rev().find(|(name, _)| *name == column.name)
                {
                    keys.push(SortKey {
                        expression: found.clone(),
                        descending: !sort.asc,
                        nulls_first: sort.nulls_first,
                    });
                    continue;
                }
            }
            let expression = self.bind_value(target)?;
            keys.push(SortKey {
                expression: SortExpression::Value(expression),
                descending: !sort.asc,
                nulls_first: sort.nulls_first,
            });
        }
        Ok(keys)
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
                    .any(|(_, provider)| self.schemas[provider].field_with_name(&name).is_ok())
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
        let provider = self.provider(alias).expect("a table in scope").to_string();
        Scan {
            column: match self
                .document_columns
                .iter()
                .find(|(found, _)| found == alias)
            {
                Some((_, column)) => ScanColumn::Expression(column.clone()),
                None => ScanColumn::Unnamed(String::new()),
            },
            schema: Some(schema_of(&self.schemas[&provider])),
            provider,
            alias: alias.to_string(),
            columns: Vec::new(),
            predicates: Vec::new(),
        }
    }

    /// alias -> the ON equalities of the JOIN of that table, last JOIN first.
    pub(crate) fn equalities(&self) -> Vec<(String, Vec<Equality>)> {
        let mut equalities = self.conditions.clone();
        equalities.reverse();
        equalities
    }

    /// Record the ON equalities of the JOIN of `alias`.
    pub(crate) fn add_equalities(&mut self, alias: &str, equalities: Vec<Equality>) {
        if !equalities.is_empty() {
            self.conditions.push((alias.to_string(), equalities));
        }
    }

    /// provider -> its schema, for each provider in scope.
    pub(crate) fn provider_schemas(&self) -> BTreeMap<String, Option<Schema>> {
        self.schemas
            .iter()
            .map(|(provider, schema)| (provider.clone(), Some(schema_of(schema))))
            .collect()
    }

    /// The unnested lists: the tables without a provider.
    pub(crate) fn unnested(&self) -> Vec<String> {
        self.tables
            .iter()
            .filter(|(_, provider)| provider.is_empty())
            .map(|(alias, _)| alias.clone())
            .collect()
    }

    /// (model call, column name) of each column operator only a test reads.
    pub(crate) fn tested(&self) -> Vec<(TaggedModelCall, String)> {
        self.tested
            .iter()
            .map(|(call, name)| (TaggedModelCall::Call(call.clone()), name.clone()))
            .collect()
    }
}

// ---------------------------------------------------------------- SELECT list

/// Binds the SELECT list, as `SelectBinder` does.
pub(crate) struct SelectBinder<'a> {
    binder: &'a mut Binder,
    pub(crate) columns: Vec<ValueExpression>,
    plain: Vec<String>,
    pub(crate) output: Vec<String>,
    pub(crate) structs: Vec<ExtractStruct>,
}

impl<'a> SelectBinder<'a> {
    pub(crate) fn new(binder: &'a mut Binder) -> Self {
        Self {
            binder,
            columns: Vec::new(),
            plain: Vec::new(),
            output: Vec::new(),
            structs: Vec::new(),
        }
    }

    fn output_of(&self, expr: &Expr) -> Option<Expr> {
        let Expr::Column(column) = expr else {
            return None;
        };
        self.binder
            .outputs
            .iter()
            .find(|(name, _)| *name == column.name)
            .map(|(_, found)| found.clone())
    }

    pub(crate) fn bind_projection(&mut self, projection: &[Expr]) -> Result<(), CompileError> {
        for item in projection {
            let (node, alias) = unalias(item);
            let node = self.output_of(node).unwrap_or_else(|| node.clone());
            if is_call(&node, "ai_extract_span") {
                continue;
            }
            let selected = self.compile_projection(&node, alias)?;
            for column in &selected {
                if !self.columns.contains(column) {
                    self.columns.push(column.clone());
                }
                self.plain.push(column_name(column, true));
                self.output.push(column_name(column, true));
            }
        }
        Ok(())
    }

    fn compile_projection(
        &mut self,
        node: &Expr,
        alias: Option<&str>,
    ) -> Result<Vec<ValueExpression>, CompileError> {
        let named = |expression: ValueExpression,
                     function: &str|
         -> Result<ValueExpression, CompileError> {
            let Some(alias) = alias else {
                return unsupported(&format!("{function} without an AS name in SELECT"));
            };
            Ok(ValueExpression::Alias(Alias {
                expression: Box::new(expression),
                name: alias.to_string(),
            }))
        };
        if is_call(node, "ai_score") {
            let call = self.binder.bind_ai_value(node)?;
            if !(1..=2).contains(&call_aliases(&call).len()) {
                return compile_error("projected AI.SCORE must reference one or two relations");
            }
            return Ok(vec![named(ValueExpression::ModelCall(call), "AI.SCORE")?]);
        }
        if is_call(node, "ai_classify") {
            let call = self.binder.bind_ai_value(node)?;
            return Ok(vec![named(
                ValueExpression::ModelCall(call),
                "AI.CLASSIFY",
            )?]);
        }
        if is_call(node, "ai_extract") {
            let Some(alias) = alias else {
                return unsupported("AI.EXTRACT without an AS name in SELECT");
            };
            let (found, members) = self.binder.bind_ai_extract(node, alias)?;
            self.structs.push(found);
            return Ok(members);
        }
        if is_call(node, "ai_sae_features") {
            let call = self.binder.bind_ai_sae_features(node)?;
            return Ok(vec![named(
                ValueExpression::ModelCall(call),
                "AI.SAE_FEATURES",
            )?]);
        }
        if is_call(node, "ai_map") {
            let call = self.binder.bind_ai_map(node)?;
            return Ok(vec![named(ValueExpression::ModelCall(call), "AI.MAP")?]);
        }
        if has_ai_call(node) || walk(node, &mut |inner| column_call_kind(inner).is_some()) {
            return unsupported("a model call inside a SELECT expression");
        }
        let value = self.binder.bind_value(node)?;
        Ok(vec![match alias {
            Some(alias) => ValueExpression::Alias(Alias {
                expression: Box::new(value),
                name: alias.to_string(),
            }),
            None => value,
        }])
    }

    /// Refuse a SELECT list that names two columns alike.
    pub(crate) fn check_output_names(&self) -> Result<(), CompileError> {
        let mut unique = self.output.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != self.output.len() {
            return compile_error(format!(
                "projection names must be unique, got {}",
                quail_planner::expressions::repr_list(&self.output)
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- helpers

/// One plain test before its operand is bound.
struct PlainTest {
    operand: Expr,
    function: Option<String>,
    comparison: String,
    value: Value,
    negated: bool,
}

impl PlainTest {
    fn new(operand: Expr, function: Option<String>, comparison: &str, value: Value) -> Self {
        Self {
            operand,
            function,
            comparison: comparison.to_string(),
            value,
            negated: false,
        }
    }
}

fn is_get_field(expr: &Expr) -> bool {
    is_call(expr, "get_field")
}

/// The SQL function on one column, as `unwrap_function`: (column, function).
fn unwrap_function(expr: &Expr) -> (&Expr, Option<String>) {
    if let Expr::ScalarFunction(call) = expr
        && call.args.len() == 1
    {
        let function = match call.name() {
            "character_length" | "length" | "char_length" => Some("length"),
            "lower" => Some("lower"),
            "upper" => Some("upper"),
            "btrim" | "trim" => Some("trim"),
            _ => None,
        };
        if let Some(function) = function {
            return (&call.args[0], Some(function.to_string()));
        }
    }
    (expr, None)
}

/// The column a test reads and the function around it, or None.
fn operand_of(expr: &Expr) -> Option<(Expr, Option<String>)> {
    let (inner, function) = unwrap_function(expr);
    if matches!(inner, Expr::Column(_) | Expr::OuterReferenceColumn(..))
        || is_get_field(inner)
        || column_call_kind(inner).is_some()
    {
        return Some((inner.clone(), function));
    }
    if computed_value(expr) {
        return Some((expr.clone(), None));
    }
    None
}

/// Whether a node is a function or arithmetic over columns and literals.
fn computed_value(expr: &Expr) -> bool {
    let shape = matches!(
        expr,
        Expr::ScalarFunction(_)
            | Expr::BinaryExpr(_)
            | Expr::Negative(_)
            | Expr::Cast(_)
            | Expr::Case(_)
            | Expr::InList(_)
            | Expr::Not(_)
            | Expr::IsNull(_)
            | Expr::IsNotNull(_)
    );
    shape
        && !walk(expr, &mut |inner| {
            function_name(inner).is_some_and(|name| name.starts_with("ai_"))
        })
        && walk(expr, &mut |inner| {
            matches!(inner, Expr::Column(_) | Expr::OuterReferenceColumn(..))
        })
}

fn sql_literal(expr: &Expr) -> Result<Value, CompileError> {
    literal_value(expr).ok_or_else(|| {
        CompileError::new(format!(
            "a column is compared with a number, string, or boolean literal, got {expr}"
        ))
    })
}

fn plain_comparison(op: Operator) -> Option<&'static str> {
    Some(match op {
        Operator::Eq => "=",
        Operator::NotEq => "<>",
        Operator::Lt => "<",
        Operator::LtEq => "<=",
        Operator::Gt => ">",
        Operator::GtEq => ">=",
        _ => return None,
    })
}

fn comparison_function(op: Operator) -> Option<&'static str> {
    Some(match op {
        Operator::Eq => "equal",
        Operator::NotEq => "not_equal",
        Operator::Lt => "less",
        Operator::LtEq => "less_equal",
        Operator::Gt => "greater",
        Operator::GtEq => "greater_equal",
        _ => return None,
    })
}

fn flipped(comparison: &str) -> &'static str {
    match comparison {
        "<" => ">",
        "<=" => ">=",
        ">" => "<",
        ">=" => "<=",
        "=" => "=",
        _ => "<>",
    }
}

/// (comparison, AI.SCORE call, threshold) of a compared score, or None.
fn score_comparison(expr: &Expr) -> Option<(&'static str, &Expr, &Expr)> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = expr else {
        return None;
    };
    let (forward, backward) = match op {
        Operator::Lt => ("<", ">"),
        Operator::LtEq => ("<=", ">="),
        Operator::Gt => (">", "<"),
        Operator::GtEq => (">=", "<="),
        _ => return None,
    };
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

pub(crate) fn conjuncts(expr: &Expr) -> Vec<Expr> {
    flatten(expr, Operator::And)
}

fn flatten(expr: &Expr, op: Operator) -> Vec<Expr> {
    match expr {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: found,
            right,
        }) if *found == op => {
            let mut terms = flatten(left, op);
            terms.extend(flatten(right, op));
            terms
        }
        other => vec![other.clone()],
    }
}

fn not(operand: Condition) -> Condition {
    Condition::Not(Not {
        operand: Box::new(operand),
    })
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

fn cast_type(data_type: &DataType) -> Option<&'static str> {
    Some(match data_type {
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::Float32 => "float32",
        DataType::Float64 | DataType::Decimal128(..) | DataType::Decimal256(..) => "float64",
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "string",
        DataType::Boolean => "bool",
        DataType::Date32 => "date32",
        DataType::Timestamp(TimeUnit::Microsecond, None) => "timestamp[us]",
        _ => return None,
    })
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

/// Whether a DataFusion function is one `sql_calls.translate` writes
/// as another Arrow call.
fn translated(name: &str) -> bool {
    matches!(
        name,
        "round"
            | "concat"
            | "greatest"
            | "least"
            | "strpos"
            | "instr"
            | "nullif"
            | "substr"
            | "substring"
            | "left"
            | "right"
            | "pow"
            | "power"
            | "reverse"
            | "initcap"
            | "replace"
            | "regexp_replace"
            | "lpad"
            | "rpad"
            | "ltrim"
            | "rtrim"
            | "btrim"
            | "trim"
            | "log10"
            | "log2"
            | "ln"
    )
}

fn integer_argument(expr: &Expr, what: &str) -> Result<i64, CompileError> {
    literal_value(expr)
        .and_then(|value| value.as_i64())
        .ok_or_else(|| CompileError::new(format!("{what} is an integer literal, got {expr}")))
}

/// The Arrow call a SQL function makes, as `sql_calls.translate`.
fn translate(
    name: &str,
    written: &[Expr],
    args: Vec<ValueExpression>,
) -> Result<ValueExpression, CompileError> {
    let function = |name: &str, args: Vec<ValueExpression>| {
        ValueExpression::ScalarFunction(QuailFunction {
            function: name.to_string(),
            args,
        })
    };
    let constant = |value: Value| ValueExpression::Literal(Literal { value });
    let mut args = args;
    Ok(match name {
        "round" => {
            let decimals = if args.len() > 1 {
                args.remove(1)
            } else {
                constant(Value::from(0))
            };
            function(
                "round",
                vec![
                    args.remove(0),
                    decimals,
                    constant(Value::from("half_towards_infinity")),
                ],
            )
        }
        "concat" => {
            args.push(constant(Value::from("")));
            function("binary_join_element_wise", args)
        }
        "greatest" | "least" => {
            let nulls: Vec<ValueExpression> = args
                .iter()
                .map(|argument| function("is_null", vec![argument.clone()]))
                .collect();
            let mut any_null = nulls[0].clone();
            for null in &nulls[1..] {
                any_null = function("or", vec![any_null, null.clone()]);
            }
            let extreme = if name == "greatest" {
                "max_element_wise"
            } else {
                "min_element_wise"
            };
            function(
                "if_else",
                vec![any_null, constant(Value::Null), function(extreme, args)],
            )
        }
        "strpos" | "instr" => function(
            "add",
            vec![function("find_substring", args), constant(Value::from(1))],
        ),
        "nullif" => function(
            "if_else",
            vec![
                function("equal", args.clone()),
                constant(Value::Null),
                args[0].clone(),
            ],
        ),
        "substr" | "substring" => {
            let start = integer_argument(&written[1], "SUBSTRING's start")?;
            if start < 1 {
                return compile_error("SUBSTRING's start counts from 1");
            }
            let mut call = vec![args.remove(0), constant(Value::from(start - 1))];
            if written.len() > 2 {
                let length = integer_argument(&written[2], "SUBSTRING's length")?;
                call.push(constant(Value::from(start - 1 + length)));
            }
            function("utf8_slice_codeunits", call)
        }
        "left" => function(
            "utf8_slice_codeunits",
            vec![
                args.remove(0),
                constant(Value::from(0)),
                constant(Value::from(integer_argument(&written[1], "LEFT's length")?)),
            ],
        ),
        "right" => function(
            "utf8_slice_codeunits",
            vec![
                args.remove(0),
                constant(Value::from(-integer_argument(
                    &written[1],
                    "RIGHT's length",
                )?)),
            ],
        ),
        "pow" | "power" => function("power", args),
        "reverse" => function("utf8_reverse", args),
        "initcap" => function("utf8_capitalize", args),
        "replace" => function("replace_substring", args),
        "regexp_replace" => function("replace_substring_regex", args),
        "lpad" => function("utf8_lpad", args),
        "rpad" => function("utf8_rpad", args),
        "ltrim" | "rtrim" | "btrim" | "trim" => {
            let side = match name {
                "ltrim" => "l",
                "rtrim" => "r",
                _ => "",
            };
            if args.len() == 1 {
                function(&format!("utf8_{side}trim_whitespace"), args)
            } else {
                function(&format!("utf8_{side}trim"), args)
            }
        }
        "log10" | "log2" | "ln" => function(name, args),
        other => function(other, args),
    })
}

/// Whether an expression calls an AI function anywhere.
pub(crate) fn contains_ai_call(expr: &Expr) -> bool {
    expr.exists(|node| Ok(crate::is_ai_call(node)))
        .unwrap_or(false)
}

/// One `UNNEST` of `FROM`: each list's name and call, and whether a row
/// with no elements stays as one row of nulls.
pub(crate) struct Unnested {
    pub(crate) lists: Vec<(String, Expr)>,
    pub(crate) outer: bool,
}

/// The `UNNEST` a WHERE term of [`crate::take_unnests`] stands for, if it
/// is one.
pub(crate) fn unnest_term(term: &Expr) -> Option<Unnested> {
    let Expr::ScalarFunction(call) = term else {
        return None;
    };
    if call.name() != crate::UNNEST_TEST {
        return None;
    }
    let outer = matches!(
        call.args.first(),
        Some(Expr::Literal(ScalarValue::Boolean(Some(true)), _))
    );
    let lists = call.args[1..]
        .chunks(2)
        .filter_map(|pair| string_literal(&pair[0]).map(|name| (name, pair[1].clone())))
        .collect();
    Some(Unnested { lists, outer })
}

/// `expr` reading each unnested field as a column of its list's
/// relation: `quail_unnest_field(list, 'f', 'feature')` as `f.feature`.
pub(crate) fn unnested_fields(expr: &Expr) -> Expr {
    expr.clone()
        .transform_up(|node| {
            if let Expr::ScalarFunction(call) = &node
                && call.name() == crate::UNNEST_FIELD
                && let (Some(list), Some(field)) =
                    (string_literal(&call.args[1]), string_literal(&call.args[2]))
            {
                return Ok(datafusion::common::tree_node::Transformed::yes(
                    Expr::Column(Column::new(Some(list), field)),
                ));
            }
            Ok(datafusion::common::tree_node::Transformed::no(node))
        })
        .map(|found| found.data)
        .unwrap_or_else(|_| expr.clone())
}

/// The schema of an `AI.SAE_FEATURES` element, as Quail's `FEATURE_LIST`.
pub(crate) fn feature_schema() -> SchemaRef {
    std::sync::Arc::new(datafusion::arrow::datatypes::Schema::new(vec![
        datafusion::arrow::datatypes::Field::new("feature", DataType::Int32, true),
        datafusion::arrow::datatypes::Field::new("strength", DataType::Float32, true),
    ]))
}

/// Whether a plan reads a CTE or a FROM subquery.
pub(crate) fn has_subqueries(plan: &LogicalPlan) -> bool {
    plan.exists(|node| {
        Ok(matches!(node, LogicalPlan::SubqueryAlias(alias)
            if !matches!(alias.input.as_ref(), LogicalPlan::TableScan(_))))
    })
    .unwrap_or(false)
}
