//! The `LowerAiCalls` analyzer rule: AI calls as Quail's plan nodes.
//!
//! DataFusion binds a query's AI calls as scalar functions inside its
//! own operators. The rule rewrites each model region so the model's
//! work is in plan nodes of its own, and DataFusion's operators hold
//! only the relational work:
//!
//! - A `Filter` keeps its other terms; each `AI.IF` term, or comparison
//!   of an `AI.SCORE` call, goes into a [`SemanticFilter`] above it, or
//!   a [`SemanticJoin`] when the term reads two tables.
//! - An inner `Join` keeps its other ON terms; each `AI.IF` term goes
//!   into a [`SemanticJoin`] above it.
//! - A `Filter`'s test of an `AI.CLASSIFY`, `AI.EXTRACT`, `AI.MAP`, or
//!   `AI.SAE_FEATURES` answer, such as `sentiment IN ('negative')`,
//!   reads the column a [`SemanticColumn`] below it adds.
//! - A `Projection` of such a call `AS name` reads the column `name`,
//!   which a [`SemanticColumn`] below it adds unless one already does.
//!   An extraction's column holds its answer, and `name` with the span
//!   suffix its spans.
//!
//! A column takes its SELECT alias as its name, or a name of its own when
//! only a test reads it. An EXISTS subquery and a call anywhere else
//! stay as DataFusion bound them.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::arrow::datatypes::Field;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Expr, ExprSchemable, Extension, Filter, Join, JoinType, LogicalPlan, Operator, Projection,
    UserDefinedLogicalNode,
};
use datafusion::optimizer::AnalyzerRule;

use crate::SPAN_SUFFIX;
use crate::region::ModelRegion;
use crate::semantic::{SemanticColumn, SemanticFilter, SemanticJoin, Written};

/// Rewrite each model region's AI calls into Quail's plan nodes, as the
/// module explains.
#[derive(Debug, Default)]
pub struct LowerAiCalls;

impl AnalyzerRule for LowerAiCalls {
    fn name(&self) -> &str {
        "lower_ai_calls"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        Ok(plan
            .transform_up(|node| {
                if let LogicalPlan::Extension(extension) = &node
                    && let Some(region) = extension.node.as_any().downcast_ref::<ModelRegion>()
                {
                    let mut region = region.clone();
                    region.plan = lower_ai_calls(region.bound.clone())?;
                    return Ok(Transformed::yes(extension_node(region)));
                }
                Ok(Transformed::no(node))
            })?
            .data)
    }
}

fn extension_node(node: impl UserDefinedLogicalNode + 'static) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    })
}

fn function_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::ScalarFunction(call) => Some(call.name()),
        _ => None,
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

/// The tables a term reads.
fn tables(term: &Expr) -> BTreeSet<String> {
    term.column_refs()
        .into_iter()
        .filter_map(|column| column.relation.as_ref().map(ToString::to_string))
        .collect()
}

fn is_column_call(expr: &Expr) -> bool {
    matches!(
        function_name(expr),
        Some("ai_classify" | "ai_extract" | "ai_map" | "ai_sae_features")
    )
}

/// A predicate's asked terms with their positions, and the rest, each
/// in written order, and how many terms it has.
fn split_asked(predicate: &Expr) -> (Vec<(usize, Expr)>, Vec<Expr>, usize) {
    let terms = split_conjunction(predicate);
    let num_terms = terms.len();
    let (mut asked, mut rest) = (Vec::new(), Vec::new());
    for (position, term) in terms.into_iter().enumerate() {
        if is_asked(term) {
            asked.push((position, term.clone()));
        } else {
            rest.push(term.clone());
        }
    }
    (asked, rest, num_terms)
}

/// Stack one node per asked term on `input`, merging the one-table
/// terms of a run on the same table into one `SemanticFilter`.
fn stack_asked(
    mut input: LogicalPlan,
    asked: Vec<(usize, Expr)>,
    num_terms: usize,
    in_on: bool,
) -> LogicalPlan {
    let mut run: Vec<(usize, Expr)> = Vec::new();
    let mut run_table: Option<BTreeSet<String>> = None;
    let close = |input: LogicalPlan, run: &mut Vec<(usize, Expr)>| {
        if run.is_empty() {
            return input;
        }
        let (positions, predicates) = std::mem::take(run).into_iter().unzip();
        extension_node(SemanticFilter {
            input: Arc::new(input),
            predicates,
            stop_key: Vec::new(),
            written: Written {
                positions,
                num_terms,
            },
        })
    };
    for (position, term) in asked {
        let read = tables(&term);
        if !in_on && read.len() < 2 {
            if run_table.as_ref() != Some(&read) {
                input = close(input, &mut run);
                run_table = Some(read);
            }
            run.push((position, term));
            continue;
        }
        input = close(input, &mut run);
        run_table = None;
        input = extension_node(SemanticJoin {
            input: Arc::new(input),
            predicate: term,
            semantics: "full".to_string(),
            in_on,
            written: Written {
                positions: vec![position],
                num_terms,
            },
        });
    }
    close(input, &mut run)
}

/// The column each call a filter tests or a SELECT list returns is
/// added under: a SELECT alias, or a name of its own.
#[derive(Debug, Default)]
struct Columns {
    /// (call, name) of each column call a SELECT list returns.
    selected: Vec<(Expr, String)>,
    /// How many columns took a name of their own.
    num_unnamed: usize,
}

impl Columns {
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
                    && is_column_call(&alias.expr)
                {
                    found.push((alias.expr.as_ref().clone(), alias.name.clone()));
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

fn quail_function(name: &str) -> Arc<datafusion::logical_expr::ScalarUDF> {
    crate::functions::quail_functions()
        .into_iter()
        .find(|function| function.name() == name)
        .expect("a Quail function")
}

/// The call a column operator runs for `call`: an extraction for the
/// spans of one.
fn base_call(call: &Expr) -> Expr {
    match call {
        Expr::ScalarFunction(found) if found.name() == "ai_extract_span" => {
            Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                quail_function("ai_extract"),
                found.args.clone(),
            ))
        }
        other => other.clone(),
    }
}

fn is_span_call(expr: &Expr) -> bool {
    function_name(expr) == Some("ai_extract_span")
}

/// The column calls a term reads, outermost first, without repeats.
fn column_calls(term: &Expr) -> Vec<Expr> {
    let mut found: Vec<Expr> = Vec::new();
    let _ = term.apply(|expr| {
        if is_column_call(expr) || is_span_call(expr) {
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

/// The name of the column a `SemanticColumn` below `plan` adds for
/// `call`, if one does.
fn added_column(plan: &LogicalPlan, call: &Expr) -> Option<String> {
    let mut name = None;
    let _ = plan.apply(|node| {
        if let Some(column) = as_node::<SemanticColumn>(node)
            && column.call == *call
        {
            name = Some(column.name.clone());
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

/// `input` with a `SemanticColumn` adding `call`'s answer as `name`,
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
    Ok(extension_node(SemanticColumn::try_new(
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
            if !(is_column_call(&expr) || is_span_call(&expr)) {
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

fn lower_filter(filter: &Filter, columns: &mut Columns) -> Result<Transformed<LogicalPlan>> {
    let unchanged = || Ok(Transformed::no(LogicalPlan::Filter(filter.clone())));
    let (asked, rest, num_terms) = split_asked(&filter.predicate);
    let (tested, plain): (Vec<Expr>, Vec<Expr>) = rest
        .into_iter()
        .partition(|term| !column_calls(term).is_empty());
    if asked.is_empty() && tested.is_empty() {
        return unchanged();
    }
    let mut input = match conjunction(plain) {
        // a term DataFusion refuses keeps its operator, for DataFusion to refuse
        Some(predicate) => match Filter::try_new(predicate, Arc::clone(&filter.input)) {
            Ok(kept) => LogicalPlan::Filter(kept),
            Err(_) => return unchanged(),
        },
        None => filter.input.as_ref().clone(),
    };
    input = stack_asked(input, asked, num_terms, false);
    if tested.is_empty() {
        return Ok(Transformed::yes(input));
    }
    let mut added: Vec<(Expr, String)> = Vec::new();
    for call in tested.iter().flat_map(column_calls) {
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
    let tests = tested
        .into_iter()
        .map(|term| read_columns(term, &added))
        .collect::<Result<Vec<_>>>()?;
    match Filter::try_new(conjunction(tests).expect("a test"), Arc::new(input)) {
        Ok(filter) => Ok(Transformed::yes(LogicalPlan::Filter(filter))),
        Err(_) => unchanged(),
    }
}

fn lower_join(join: &Join) -> Result<Transformed<LogicalPlan>> {
    let Some(predicate) = &join.filter else {
        return Ok(Transformed::no(LogicalPlan::Join(join.clone())));
    };
    let (asked, rest, num_terms) = split_asked(predicate);
    if asked.is_empty() || join.join_type != JoinType::Inner {
        return Ok(Transformed::no(LogicalPlan::Join(join.clone())));
    }
    let mut kept = join.clone();
    kept.filter = conjunction(rest);
    Ok(Transformed::yes(stack_asked(
        LogicalPlan::Join(kept),
        asked,
        num_terms,
        true,
    )))
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
        if !(is_column_call(&alias.expr) || is_span_call(&alias.expr))
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

fn as_node<T: 'static>(plan: &LogicalPlan) -> Option<&T> {
    match plan {
        LogicalPlan::Extension(extension) => extension.node.as_any().downcast_ref::<T>(),
        _ => None,
    }
}

/// Rewrite a region's AI calls into Quail's plan nodes, as the module
/// explains.
///
/// # Errors
///
/// A rewritten operator's schema does not fit DataFusion's rules.
pub fn lower_ai_calls(plan: LogicalPlan) -> Result<LogicalPlan> {
    let mut columns = Columns {
        selected: selected_calls(&plan),
        num_unnamed: 0,
    };
    Ok(plan
        .transform_up(|node| match &node {
            LogicalPlan::Filter(filter) => lower_filter(filter, &mut columns),
            LogicalPlan::Join(join) => lower_join(join),
            LogicalPlan::Projection(projection) => lower_projection(projection, &mut columns),
            _ => Ok(Transformed::no(node)),
        })?
        .data)
}
