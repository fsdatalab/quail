//! A model region's operators, read from its optimized DataFusion plan.
//!
//! [`derive_region`] walks the region's plan once, children first and
//! the left input before the right, which is the order Quail numbers
//! operators in: each table's scan, the `AiNode`s on it, then the joins
//! and the AI joins above them. It then
//!
//! 1. binds each table, ON and WHERE term, AI call, and SELECT list item
//!    with the [`Binder`], in the order the query wrote them, reading
//!    each `AiNode`'s call and written place and the tests it keeps;
//! 2. names each column operator, by its SELECT alias or a name of its
//!    own, and places each condition: at a table's scan, after the
//!    operator that computes the columns it reads, or on the result rows;
//! 3. assembles [`Operators`] in the plan's order, each `AiNode` giving
//!    one AI filter, AI join, gate, or column operator.
//!
//! The engine plans each region from these operators, and a region whose
//! shape they cannot hold is refused with an unsupported
//! [`CompileError`].
//!
//! For example, `SELECT r.id, AI.CLASSIFY(...) AS sentiment FROM reviews
//! r WHERE AI.IF(...)` plans as an `AiNode` adding `sentiment` over an
//! `AiNode` filtering `r`, over the scan of `r`. The walk meets the scan,
//! the filter, then the classification, and the operators hold one
//! scan, one AI filter of `r`, and one classification named `sentiment`.

pub(crate) mod binding;
pub mod expressions;
pub mod partial;
pub mod similar;

mod assemble;
mod placement;
mod relation;
mod walk;

use datafusion::common::ScalarValue;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::{Expr, LogicalPlan};
use quail_planner::error::{CompileError, compile_error};
use quail_planner::expressions::{condition_text, projected_call};
use quail_planner::operators::{Operators, Projection};
use quail_planner::physical::Projected;
use quail_planner::prompts::Turn;

use self::assemble::{Assembly, push_down_columns};
use self::binding::{Binder, Selected, unsupported};
use self::relation::Relation;
use self::walk::{
    Walk, has_subqueries, read_fetch, read_top, refuse_grouped_input, with_select_list,
};
use crate::ai::AiNode;
use crate::region::ModelRegion;

/// A model region's operators and the plan after DataFusion's optimizer
/// they are read from.
///
/// `turn` and `layout` are the session model's chat-turn text and prompt
/// layout.
///
/// # Errors
///
/// The region uses SQL outside Quail's language; an unsupported error
/// ([`CompileError::is_unsupported`]) names a shape DataFusion can run
/// instead. Two calls that differ only in their texts' spelling are
/// refused.
pub fn derive_region(
    region: &ModelRegion,
    turn: &Turn,
    layout: &str,
) -> Result<(Operators, LogicalPlan), CompileError> {
    let plan = as_bound(region.plan.clone())
        .map_err(|error| CompileError::unsupported(error.to_string()))?;
    if let Some(found) = crate::derive::similar::similar_calls(&plan)
        .into_iter()
        .find(|found| found.likeness == crate::derive::similar::Likeness::Spelling)
    {
        return compile_error(found.message());
    }
    let mut operators = derive_operators(&plan, turn, layout)?;
    if let Some(fetch) = region.fetch {
        operators = operators.limited(fetch as i64);
    }
    let ored = operators
        .filters
        .iter()
        .flat_map(|(_, predicates)| predicates)
        .any(|predicate| predicate.alternative.is_some());
    if ored && operators.fetch.is_some() {
        // the executor counts the rows that pass among the rows it asks
        return unsupported("LIMIT over an OR filter must run in DataFusion.");
    }
    Ok((operators, plan))
}

/// `plan` with each struct or list literal in an `AiNode`'s expressions
/// written as the call it was folded from.
///
/// DataFusion's `SimplifyExpressions` runs over a region's operators and
/// folds a `named_struct`, `struct`, or `make_array` of literals into one
/// literal. The binder reads an AI call's options, categories, and
/// fields from the calls the query wrote.
fn as_bound(plan: LogicalPlan) -> datafusion::common::Result<LogicalPlan> {
    plan.transform_up(|node| {
        let Some(ai) = crate::extension::as_node::<AiNode>(&node) else {
            return Ok(Transformed::no(node));
        };
        let ai = ai.map_expressions(|expr| Ok(unfolded(expr)))?;
        Ok(Transformed::yes(crate::extension::extension_node(
            ai.checked()?,
        )))
    })
    .map(|done| done.data)
}

/// `expr` with each struct or list literal in it written as the call it
/// was folded from, as [`as_bound`] reads an `AiNode`'s expressions.
pub(crate) fn unfolded(expr: Expr) -> Expr {
    expr.transform_up(|inner| Ok(unfold(inner)))
        .map(|done| done.data)
        .expect("unfolding a literal does not fail")
}

/// `expr` as the `named_struct`, `struct`, or `make_array` call of
/// literals it was folded from, when it is a struct or list literal.
fn unfold(expr: Expr) -> Transformed<Expr> {
    use datafusion::arrow::array::Array;
    use datafusion::logical_expr::expr::ScalarFunction as Call;
    let Expr::Literal(value, _) = &expr else {
        return Transformed::no(expr);
    };
    let element = |array: &dyn Array, index: usize| {
        ScalarValue::try_from_array(array, index)
            .map(|value| unfold(Expr::Literal(value, None)).data)
    };
    match value {
        ScalarValue::Struct(array) if array.len() == 1 => {
            let fields = array.fields();
            let positional = fields
                .iter()
                .enumerate()
                .all(|(index, field)| *field.name() == format!("c{index}"));
            let mut args = Vec::new();
            for (index, field) in fields.iter().enumerate() {
                let Ok(value) = element(array.column(index).as_ref(), 0) else {
                    return Transformed::no(expr);
                };
                if !positional {
                    args.push(Expr::Literal(
                        ScalarValue::Utf8(Some(field.name().clone())),
                        None,
                    ));
                }
                args.push(value);
            }
            let function = if positional {
                datafusion::functions::core::r#struct()
            } else {
                datafusion::functions::core::named_struct()
            };
            Transformed::yes(Expr::ScalarFunction(Call::new_udf(function, args)))
        }
        ScalarValue::List(array) if array.len() == 1 => {
            let values = array.value(0);
            let mut args = Vec::new();
            for index in 0..values.len() {
                let Ok(value) = element(values.as_ref(), index) else {
                    return Transformed::no(expr);
                };
                args.push(value);
            }
            Transformed::yes(Expr::ScalarFunction(Call::new_udf(
                datafusion::functions_nested::make_array::make_array_udf(),
                args,
            )))
        }
        _ => Transformed::no(expr),
    }
}

/// A model region's operators, read from its plan after DataFusion's
/// rules as the module explains.
///
/// # Errors
///
/// As [`derive_region`].
fn derive_operators(
    plan: &LogicalPlan,
    turn: &Turn,
    layout: &str,
) -> Result<Operators, CompileError> {
    let listed = with_select_list(plan);
    let top = read_top(&listed)?;
    refuse_grouped_input(top.relation)?;
    let fetch = read_fetch(&top).ok_or_else(|| unplaced(plan))?;
    let mut walk = Walk::default();
    walk.visit(top.relation);
    if let Some(error) = walk.refusal.take() {
        return Err(error);
    }
    if walk.unknown {
        return Err(unplaced(plan));
    }
    let allow_relational = has_subqueries(plan);
    let answers = walk.answers();
    let select: Vec<Expr> = top
        .projection
        .iter()
        .map(|expr| walk.selected(expr, &answers))
        .collect();
    let mut binder = Binder::new(turn, layout, &select);
    let relation = Relation::bind(&walk, &mut binder, &answers)?;
    let select: Vec<Expr> = select
        .iter()
        .map(|expr| {
            let bound = walk.filter_answers(expr, &mut binder);
            if bound == *expr {
                bound
            } else {
                bound.alias(expr.schema_name().to_string())
            }
        })
        .collect();
    let Selected {
        columns,
        structs,
        spans,
        ..
    } = binder.bind_select(&select)?;
    binder.validate_query(&columns, &structs)?;
    let unplaced_term = relation.unplaced;
    let mut assembly = Assembly::new(&walk, &mut binder, relation, &columns, allow_relational)?;
    if assembly.place().is_none() {
        return Err(unplaced(plan));
    }
    let mut found = assembly.found.clone();
    if unplaced_term {
        return Err(unplaced(plan));
    }
    if columns.is_empty() {
        return unsupported("DataFusion must run a projection without output columns.");
    }
    if let Some(test) = assembly.conditions.rows.first() {
        // DataFusion runs a test of the result rows above the region
        return unsupported(&format!(
            "DataFusion must evaluate the result-row test {}.",
            condition_text(test)
        ));
    }
    found.projection = Projection {
        columns: columns.iter().cloned().map(Projected::Expression).collect(),
        structs,
        spans,
    };
    found.projections = found
        .projection
        .columns
        .iter()
        .filter(|column| projected_call(column, &[]).is_some())
        .filter_map(|column| match column {
            Projected::Expression(expression) => Some(expression.clone()),
            Projected::Name(_) => None,
        })
        .collect();
    found.fetch = fetch;
    if found.fetch.is_some_and(|limit| limit <= 0) {
        return unsupported("DataFusion must apply a non-positive LIMIT outside the model region.");
    }
    if allow_relational && quail_planner::plan_statistics::plan_prompts(&found).is_empty() {
        return unsupported("DataFusion must run this region because it has no model call.");
    }
    push_down_columns(&mut found, &walk, &assembly);
    Ok(found)
}

/// The refusal of a plan whose operators Quail's operators do not read.
fn unplaced(_plan: &LogicalPlan) -> CompileError {
    CompileError::unsupported("These operators must run outside the model region.")
}

/// Values kept per table alias, in the order the aliases first appear.
type ByAlias<T> = Vec<(String, Vec<T>)>;

/// Add `value` to `key`'s values, keeping keys in first use order.
fn push<V>(map: &mut ByAlias<V>, key: &str, value: V) {
    match map.iter_mut().find(|(found, _)| found == key) {
        Some((_, values)) => values.push(value),
        None => map.push((key.to_string(), vec![value])),
    }
}

fn lookup<'a, V>(map: &'a ByAlias<V>, key: &str) -> &'a [V] {
    map.iter()
        .find(|(found, _)| found == key)
        .map_or(&[], |(_, values)| values.as_slice())
}

fn note(aliases: &mut Vec<String>, alias: &str) {
    if !aliases.iter().any(|found| found == alias) {
        aliases.push(alias.to_string());
    }
}
