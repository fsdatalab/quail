//! The `LowerAiCalls` analyzer rule: AI calls as Quail's plan nodes.
//!
//! DataFusion binds a query's AI calls as scalar functions inside its
//! own operators. The rule rewrites each part of the query that Quail
//! plans so each call is an [`AiNode`] of its own, and DataFusion's
//! operators hold only the relational work:
//!
//! - A `Filter` keeps its other terms; each `AI.IF` term, or comparison
//!   of an `AI.SCORE` call, is an `AiNode` above it whose filter tests
//!   the answer. A term that reads two tables is an AI join.
//! - An inner `Join` keeps its other ON terms; each `AI.IF` term is an
//!   `AiNode` above it.
//! - Each such node records where the query wrote its term
//!   ([`crate::ai::Written`]).
//! - A `Filter`'s test of an `AI.CLASSIFY`, `AI.EXTRACT`, `AI.MAP`, or
//!   `AI.SAE_FEATURES` answer, such as `sentiment IN ('negative')`,
//!   reads the column an `AiNode` below it adds. Each node whose answer
//!   it reads keeps the test as written and its place in the WHERE
//!   ([`crate::ai::WrittenTest`]).
//! - A `Projection` of such a call `AS name` reads the column `name`,
//!   which an `AiNode` below it adds unless one already does. An
//!   extraction's column holds its answer, and `name` with the span
//!   suffix its spans.
//! - An `EXISTS` gate, `EXISTS (SELECT 1 FROM t s WHERE AI.IF(...))`,
//!   joins the table and keeps the rows of the outer one: an `AiNode`
//!   of semi semantics, or anti for `NOT EXISTS`.
//!
//! A column takes its SELECT alias as its name, or a name of its own when
//! only a test reads it. A call anywhere else stays as DataFusion bound
//! it.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, plan_err};
use datafusion::logical_expr::simplify::SimplifyContext;
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Expr, ExprSchemable, Filter, Join, JoinType, LogicalPlan, Operator, Projection,
};
use datafusion::optimizer::AnalyzerRule;
use datafusion::optimizer::analyzer::type_coercion::TypeCoercionRewriter;
use datafusion::optimizer::simplify_expressions::ExprSimplifier;

use crate::SPAN_SUFFIX;
use crate::ai::{AiNode, Semantics, Written, WrittenTest};
use crate::extension::{as_node, extension_node};
use crate::functions::column_call_kind;
use crate::functions::function_name;

/// Rewrite the AI calls of each part of a query that Quail plans into
/// Quail's plan nodes, as the module explains.
///
/// The rule runs first among the analyzer rules, on the plan as
/// DataFusion's SQL planner bound it; [`crate::claim`] explains which
/// parts Quail plans.
#[derive(Debug, Default)]
pub struct LowerAiCalls;

impl AnalyzerRule for LowerAiCalls {
    fn name(&self) -> &str {
        "lower_ai_calls"
    }

    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
        crate::claim::lower_regions(plan, config)
    }
}

/// The term inside a `HINT`, or the term itself.
fn hinted(expr: &Expr) -> &Expr {
    match expr {
        Expr::ScalarFunction(call) if call.name() == "hint" && !call.args.is_empty() => {
            hinted(&call.args[0])
        }
        other => other,
    }
}

/// The term inside a `HINT`, and the `HINT`'s options, if it has any.
fn unhint(expr: &Expr) -> (Expr, Option<Expr>) {
    match expr {
        Expr::ScalarFunction(call) if call.name() == "hint" && !call.args.is_empty() => {
            let (term, inner) = unhint(&call.args[0]);
            (term, call.args.get(1).cloned().or(inner))
        }
        other => (other.clone(), None),
    }
}

/// Whether a term is a predicate the model answers: an `AI.IF` call or a
/// comparison of an `AI.SCORE` call.
fn is_asked(term: &Expr) -> bool {
    match hinted(term) {
        Expr::BinaryExpr(binary)
            if matches!(
                binary.op,
                Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq
            ) =>
        {
            function_name(&binary.left) == Some("ai_score")
                || function_name(&binary.right) == Some("ai_score")
        }
        other => function_name(other) == Some("ai_filter"),
    }
}

/// Terms of one WHERE or ON, each with its position among them.
type Terms = Vec<(usize, Expr)>;

/// A predicate's asked terms and the rest, in written order, and how many
/// terms it has.
fn split_asked(predicate: &Expr) -> (Terms, Terms, usize) {
    let terms = split_conjunction(predicate);
    let num_terms = terms.len();
    let (mut asked, mut rest) = (Vec::new(), Vec::new());
    for (position, term) in terms.into_iter().enumerate() {
        if is_asked(term) {
            asked.push((position, term.clone()));
        } else {
            rest.push((position, term.clone()));
        }
    }
    (asked, rest, num_terms)
}

/// An `AiNode` over `input` asking `term`, an `AI.IF` call or a
/// comparison of an `AI.SCORE` call, whose filter tests the answer.
fn asked_node(
    input: LogicalPlan,
    term: &Expr,
    (answer, index): (String, usize),
    semantics: Semantics,
    written: Written,
) -> Result<LogicalPlan> {
    let (term, hint) = unhint(term);
    let read = Expr::Column(Column::from_name(answer.clone()));
    let (call, filter) = match term {
        Expr::BinaryExpr(mut binary) if function_name(&binary.left) == Some("ai_score") => {
            let call = std::mem::replace(binary.left.as_mut(), read);
            (call, Expr::BinaryExpr(binary))
        }
        Expr::BinaryExpr(mut binary) if function_name(&binary.right) == Some("ai_score") => {
            let call = std::mem::replace(binary.right.as_mut(), read);
            (call, Expr::BinaryExpr(binary))
        }
        call => (call, read),
    };
    let (_, field) = call.to_field(input.schema())?;
    let field = field.as_ref().clone().with_name(answer.clone());
    let mut node = AiNode::try_new(Arc::new(input), call, answer, vec![field])?;
    node.filter = Some(filter);
    node.hint = hint;
    node.projection = Some(Vec::new());
    node.semantics = semantics;
    node.written = Some(Written { index, ..written });
    Ok(extension_node(node.checked()?))
}

/// Stack one `AiNode` per asked term on `input`, in written order.
///
/// `on` is the alias of the table whose JOIN's ON holds the terms, or
/// None for a WHERE.
fn stack_asked(
    mut input: LogicalPlan,
    asked: Terms,
    (num_terms, clause): (usize, usize),
    on: Option<&str>,
    columns: &mut Columns,
) -> Result<LogicalPlan> {
    for (position, term) in asked {
        input = asked_node(
            input,
            &term,
            columns.answer_name(),
            Semantics::Rows,
            Written {
                position,
                num_terms,
                clause,
                index: 0,
                on: on.map(str::to_string),
            },
        )?;
    }
    Ok(input)
}

/// The column each call a filter tests or a SELECT list returns is
/// added under: a SELECT alias, or a name of its own.
#[derive(Debug, Default)]
struct Columns {
    /// (call, name) of each column call a SELECT list returns.
    selected: Vec<(Expr, String)>,
    /// How many columns took a name of their own.
    num_unnamed: usize,
    /// How many answers of tests took a name.
    num_answers: usize,
    /// How many WHEREs and ONs the lowering met.
    num_clauses: usize,
}

impl Columns {
    /// A name for the answer of an `AI.IF` or `AI.SCORE` test, and the
    /// test's number among the region's AI predicates.
    fn answer_name(&mut self) -> (String, usize) {
        self.num_answers += 1;
        (
            format!("__quail_answer_{}", self.num_answers),
            self.num_answers,
        )
    }

    /// The number of the next WHERE or ON.
    fn clause(&mut self) -> usize {
        self.num_clauses += 1;
        self.num_clauses
    }

    fn name_for(&mut self, call: &Expr) -> String {
        if let Some((_, name)) = self.selected.iter().find(|(found, _)| found == call) {
            return name.clone();
        }
        self.num_unnamed += 1;
        format!("__quail_column_{}", self.num_unnamed)
    }
}

/// The calls a SELECT list returns as columns, as `call AS name`.
fn selected_calls(plan: &LogicalPlan) -> Vec<(Expr, String)> {
    let mut found = Vec::new();
    let _ = plan.apply(|node| {
        if let LogicalPlan::Projection(projection) = node {
            for expr in &projection.expr {
                if let Expr::Alias(alias) = expr
                    && column_call_kind(&alias.expr).is_some()
                {
                    found.push((alias.expr.as_ref().clone(), alias.name.clone()));
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

pub(crate) fn quail_function(name: &str) -> Arc<datafusion::logical_expr::ScalarUDF> {
    crate::functions::quail_functions()
        .into_iter()
        .find(|function| function.name() == name)
        .expect("a Quail function")
}

/// Whether an argument is a `{'selectivity': ...}` hint and nothing else.
fn is_selectivity_hint(arg: &Expr) -> bool {
    match arg {
        Expr::ScalarFunction(call) if call.name() == "named_struct" => {
            call.args.len() == 2
                && matches!(&call.args[0], Expr::Literal(value, _)
                    if value.try_as_str().flatten() == Some("selectivity"))
        }
        _ => false,
    }
}

/// The call a column operator runs for `call`: an extraction for the
/// spans of one, and a test's call without its selectivity hint, which
/// the test keeps.
fn base_call(call: &Expr) -> Expr {
    let Expr::ScalarFunction(found) = call else {
        return call.clone();
    };
    let mut args = found.args.clone();
    if args.last().is_some_and(is_selectivity_hint) {
        args.pop();
    }
    let function = if found.name() == "ai_extract_span" {
        quail_function("ai_extract")
    } else {
        Arc::clone(&found.func)
    };
    Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
        function, args,
    ))
}

fn is_span_call(expr: &Expr) -> bool {
    function_name(expr) == Some("ai_extract_span")
}

/// The column calls a term reads, outermost first, without repeats.
fn column_calls(term: &Expr) -> Vec<Expr> {
    let mut found: Vec<Expr> = Vec::new();
    let _ = term.apply(|expr| {
        if column_call_kind(expr).is_some() || is_span_call(expr) {
            let call = base_call(expr);
            if !found.contains(&call) {
                found.push(call);
            }
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

/// The name of the column an `AiNode` below `plan` adds for `call`, if
/// one does.
fn added_column(plan: &LogicalPlan, call: &Expr) -> Option<String> {
    let mut name = None;
    let _ = plan.apply(|node| {
        if let Some(column) = as_node::<AiNode>(node)
            && !column.is_predicate()
            && column.call == *call
        {
            name = Some(column.answer.clone());
            return Ok(TreeNodeRecursion::Stop);
        }
        // a derived table's columns are named by its alias
        if matches!(node, LogicalPlan::SubqueryAlias(_)) {
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    name
}

/// `input` with an `AiNode` adding `call`'s answer as `name`,
/// and an extraction's spans as `name` with the span suffix.
fn add_column(input: LogicalPlan, call: Expr, name: &str) -> Result<LogicalPlan> {
    let schema = input.schema();
    let field = |expr: &Expr, name: String| -> Result<Field> {
        let (_, field) = expr.to_field(schema)?;
        Ok(field.as_ref().clone().with_name(name))
    };
    let mut fields = vec![field(&call, name.to_string())?];
    if function_name(&call) == Some("ai_extract") {
        let Expr::ScalarFunction(found) = &call else {
            unreachable!("a call")
        };
        let spans = Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
            quail_function("ai_extract_span"),
            found.args.clone(),
        ));
        fields.push(field(&spans, format!("{name}{SPAN_SUFFIX}"))?);
    }
    Ok(extension_node(AiNode::try_new(
        Arc::new(input),
        call,
        name.to_string(),
        fields,
    )?))
}

/// `term` reading each call's column instead of calling it.
fn read_columns(term: Expr, columns: &[(Expr, String)]) -> Result<Expr> {
    Ok(term
        .transform_down(|expr| {
            if !(column_call_kind(&expr).is_some() || is_span_call(&expr)) {
                return Ok(Transformed::no(expr));
            }
            let call = base_call(&expr);
            let Some((_, name)) = columns.iter().find(|(found, _)| *found == call) else {
                return Ok(Transformed::no(expr));
            };
            let name = if is_span_call(&expr) {
                format!("{name}{SPAN_SUFFIX}")
            } else {
                name.clone()
            };
            Ok(Transformed::new(
                Expr::Column(Column::from_name(name)),
                true,
                TreeNodeRecursion::Jump,
            ))
        })?
        .data)
}

/// An EXISTS gate the binder reads: `EXISTS (SELECT 1 FROM table alias
/// WHERE <one AI.IF over a row of each table>)`, as the table's plan and
/// the predicate with the outer row's columns as columns.
fn exists_gate(term: &Expr) -> Option<(LogicalPlan, Expr, bool)> {
    let Expr::Exists(exists) = hinted(term) else {
        return None;
    };
    let LogicalPlan::Projection(projection) = exists.subquery.subquery.as_ref() else {
        return None;
    };
    let LogicalPlan::Filter(filter) = projection.input.as_ref() else {
        return None;
    };
    let LogicalPlan::SubqueryAlias(alias) = filter.input.as_ref() else {
        return None;
    };
    if !matches!(alias.input.as_ref(), LogicalPlan::TableScan(_)) || !is_asked(&filter.predicate) {
        return None;
    }
    let predicate = filter
        .predicate
        .clone()
        .transform_up(|expr| match expr {
            Expr::OuterReferenceColumn(_, column) => Ok(Transformed::yes(Expr::Column(column))),
            other => Ok(Transformed::no(other)),
        })
        .ok()?
        .data;
    Some((filter.input.as_ref().clone(), predicate, exists.negated))
}

fn lower_filter(filter: &Filter, columns: &mut Columns) -> Result<Transformed<LogicalPlan>> {
    let (asked, rest, num_terms) = split_asked(&filter.predicate);
    let (gates, rest): (Terms, Terms) = rest
        .into_iter()
        .partition(|(_, term)| exists_gate(term).is_some());
    let (tested, plain): (Terms, Terms) = rest
        .into_iter()
        .partition(|(_, term)| !column_calls(term).is_empty());
    if asked.is_empty() && tested.is_empty() && gates.is_empty() {
        return Ok(Transformed::no(LogicalPlan::Filter(filter.clone())));
    }
    let mut input = match conjunction(plain.into_iter().map(|(_, term)| term)) {
        // a term DataFusion cannot type keeps its operator, for the binder to refuse
        Some(predicate) => match boolean_filter(predicate, &filter.input)? {
            Some(kept) => LogicalPlan::Filter(kept),
            None => return Ok(Transformed::no(LogicalPlan::Filter(filter.clone()))),
        },
        None => filter.input.as_ref().clone(),
    };
    let clause = columns.clause();
    input = stack_asked(input, asked, (num_terms, clause), None, columns)?;
    for (position, gate) in &gates {
        let (table, predicate, negated) = exists_gate(gate).expect("a gate");
        let joined = datafusion::logical_expr::LogicalPlanBuilder::from(input)
            .cross_join(table)?
            .build()?;
        input = asked_node(
            joined,
            &predicate,
            columns.answer_name(),
            if negated {
                Semantics::Anti
            } else {
                Semantics::Semi
            },
            Written {
                position: *position,
                num_terms,
                clause,
                index: 0,
                on: None,
            },
        )?;
    }
    if tested.is_empty() {
        return Ok(Transformed::yes(input));
    }
    let mut added: Vec<(Expr, String)> = Vec::new();
    for call in tested.iter().flat_map(|(_, term)| column_calls(term)) {
        if added.iter().any(|(found, _)| *found == call) {
            continue;
        }
        let name = match added_column(&input, &call) {
            Some(name) => name,
            None => {
                let name = columns.name_for(&call);
                input = add_column(input, call.clone(), &name)?;
                name
            }
        };
        added.push((call, name));
    }
    let mut tests = Vec::new();
    let mut written = Vec::new();
    for (position, term) in tested {
        let test = read_columns(term.clone(), &added)?;
        written.push(WrittenTest {
            test: simplified(test.clone(), &input),
            written: term,
            position,
            clause,
        });
        tests.push(test);
    }
    let names: Vec<String> = added.iter().map(|(_, name)| name.clone()).collect();
    input = with_tests(input, &names, &written)?;
    match boolean_filter(conjunction(tests).expect("a test"), &Arc::new(input))? {
        Some(test) => Ok(Transformed::yes(LogicalPlan::Filter(test))),
        None => Ok(Transformed::no(LogicalPlan::Filter(filter.clone()))),
    }
}

/// `test` as DataFusion's type coercion and `SimplifyExpressions`
/// rewrite it in a filter over `input`, or as it is when they fail: the
/// `AiNode`s whose answers it reads find the filter's term by this form.
fn simplified(test: Expr, input: &LogicalPlan) -> Expr {
    let Ok(coerced) = test
        .clone()
        .rewrite(&mut TypeCoercionRewriter::new(input.schema()))
    else {
        return test;
    };
    let context = SimplifyContext::builder()
        .with_schema(Arc::clone(input.schema()))
        .build();
    ExprSimplifier::new(context)
        .simplify(coerced.data)
        .unwrap_or(test)
}

/// `plan` with `tests` kept by each `AiNode` that adds a column of `names`.
fn with_tests(plan: LogicalPlan, names: &[String], tests: &[WrittenTest]) -> Result<LogicalPlan> {
    Ok(plan
        .transform_down(|node| {
            // a derived table's columns are named by its alias
            if matches!(node, LogicalPlan::SubqueryAlias(_)) {
                return Ok(Transformed::new(node, false, TreeNodeRecursion::Jump));
            }
            let Some(column) = as_node::<AiNode>(&node) else {
                return Ok(Transformed::no(node));
            };
            if column.is_predicate() || !names.contains(&column.answer) {
                return Ok(Transformed::no(node));
            }
            let mut column = column.clone();
            column.tests.extend(tests.iter().cloned());
            Ok(Transformed::yes(extension_node(column)))
        })?
        .data)
}

/// A filter over `input` that keeps the rows where `predicate` is true,
/// or `None` when DataFusion cannot type the predicate.
///
/// # Errors
///
/// The predicate is not true or false.
fn boolean_filter(predicate: Expr, input: &Arc<LogicalPlan>) -> Result<Option<Filter>> {
    let Ok(data_type) = predicate.get_type(input.schema()) else {
        return Ok(None);
    };
    if !matches!(data_type, DataType::Boolean | DataType::Null) {
        return plan_err!("a test is true or false, got {data_type}");
    }
    Ok(Filter::try_new(predicate, Arc::clone(input)).ok())
}

fn lower_join(join: &Join, columns: &mut Columns) -> Result<Transformed<LogicalPlan>> {
    let Some(predicate) = &join.filter else {
        return Ok(Transformed::no(LogicalPlan::Join(join.clone())));
    };
    let (asked, rest, num_terms) = split_asked(predicate);
    if asked.is_empty() || join.join_type != JoinType::Inner {
        return Ok(Transformed::no(LogicalPlan::Join(join.clone())));
    }
    // the joined table's alias, which names the ON
    let joined = join
        .right
        .schema()
        .iter()
        .find_map(|(qualifier, _)| qualifier.map(|found| found.table().to_string()))
        .unwrap_or_default();
    let clause = columns.clause();
    let mut kept = join.clone();
    kept.filter = conjunction(rest.into_iter().map(|(_, term)| term));
    Ok(Transformed::yes(stack_asked(
        LogicalPlan::Join(kept),
        asked,
        (num_terms, clause),
        Some(&joined),
        columns,
    )?))
}

fn lower_projection(
    projection: &Projection,
    columns: &mut Columns,
) -> Result<Transformed<LogicalPlan>> {
    let mut input = projection.input.as_ref().clone();
    let mut exprs = Vec::new();
    let mut changed = false;
    for expr in &projection.expr {
        let Expr::Alias(alias) = expr else {
            exprs.push(expr.clone());
            continue;
        };
        if !(column_call_kind(&alias.expr).is_some() || is_span_call(&alias.expr))
            || alias.relation.is_some()
            || alias.metadata.is_some()
        {
            exprs.push(expr.clone());
            continue;
        }
        let call = base_call(&alias.expr);
        let name = match added_column(&input, &call) {
            Some(name) => name,
            None => {
                let name = if is_span_call(&alias.expr) {
                    alias
                        .name
                        .strip_suffix(SPAN_SUFFIX)
                        .unwrap_or(&alias.name)
                        .to_string()
                } else {
                    columns.name_for(&call)
                };
                if input.schema().has_column_with_unqualified_name(&name) {
                    exprs.push(expr.clone());
                    continue;
                }
                input = add_column(input, call.clone(), &name)?;
                name
            }
        };
        let column = if is_span_call(&alias.expr) {
            format!("{name}{SPAN_SUFFIX}")
        } else {
            name
        };
        let read = Expr::Column(Column::from_name(column.clone()));
        exprs.push(if column == alias.name {
            read
        } else {
            read.alias(alias.name.clone())
        });
        changed = true;
    }
    if !changed {
        return Ok(Transformed::no(LogicalPlan::Projection(projection.clone())));
    }
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(exprs, Arc::new(input))?,
    )))
}

/// Rewrite the AI calls of one part of a query that Quail plans into
/// Quail's plan nodes, as the module explains.
///
/// # Errors
///
/// A rewritten operator's schema does not fit DataFusion's rules.
pub fn lower_ai_calls(plan: LogicalPlan) -> Result<LogicalPlan> {
    let mut columns = Columns {
        selected: selected_calls(&plan),
        num_unnamed: 0,
        num_answers: 0,
        num_clauses: 0,
    };
    Ok(plan
        .transform_up(|node| match &node {
            LogicalPlan::Filter(filter) => lower_filter(filter, &mut columns),
            LogicalPlan::Join(join) => lower_join(join, &mut columns),
            LogicalPlan::Projection(projection) => lower_projection(projection, &mut columns),
            _ => Ok(Transformed::no(node)),
        })?
        .data)
}
