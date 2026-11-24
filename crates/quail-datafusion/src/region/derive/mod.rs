pub(crate) mod binding;
pub mod expressions;
pub mod partial;
pub mod similar;

mod assemble;
mod placement;
mod relation;
mod walk;

use std::collections::BTreeMap;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::{Expr, LogicalPlan};
use quail_planner::error::{CompileError, compile_error};
use quail_planner::expressions::projected_call;
use quail_planner::operators::{Operators, Projection};
use quail_planner::physical::Projected;
use quail_planner::prompts::Turn;

pub(crate) use self::relation::is_join_filter;

use self::assemble::{Assembly, set_scan_columns};
use self::binding::{Binder, Selected, output_name, unsupported};
use self::relation::Relation;
use self::walk::{Walk, read_fetch, read_top, with_select_list};
use super::ModelRegion;
use crate::ai::AiNode;
use crate::functions::options::unfolded;

pub fn derive_region(
    region: &ModelRegion,
    turn: &Turn,
    layout: &str,
) -> Result<(Operators, LogicalPlan), CompileError> {
    let plan = as_bound(region.plan.clone())
        .map_err(|error| CompileError::unsupported(error.to_string()))?;
    if let Some(found) = crate::region::derive::similar::similar_calls(&plan)
        .into_iter()
        .find(|found| found.likeness == crate::region::derive::similar::Likeness::Spelling)
    {
        return compile_error(found.message());
    }
    let operators = derive_operators(&plan, turn, layout)?;
    let ored = operators
        .filters
        .iter()
        .flat_map(|(_, predicates)| predicates)
        .any(|predicate| predicate.alternative.is_some());
    if ored && operators.fetch.is_some() {
        return Err(CompileError::new(
            "internal error: a model region has a fetch and an OR filter",
        ));
    }
    Ok((operators, plan))
}

fn as_bound(plan: LogicalPlan) -> datafusion::common::Result<LogicalPlan> {
    plan.transform_up(|node| {
        let Some(ai) = crate::plan::as_node::<AiNode>(&node) else {
            return Ok(Transformed::no(node));
        };
        let ai = ai.map_expressions(|expr| Ok(unfolded(expr)))?;
        Ok(Transformed::yes(crate::plan::extension_node(ai.checked()?)))
    })
    .map(|done| done.data)
}

fn derive_operators(
    plan: &LogicalPlan,
    turn: &Turn,
    layout: &str,
) -> Result<Operators, CompileError> {
    let listed = with_select_list(plan);
    let top = read_top(&listed)?;
    let fetch = read_fetch(&top).ok_or_else(|| unplaced(plan))?;
    let walk = Walk::of(top.relation)?;
    let select: Vec<Expr> = top
        .projection
        .iter()
        .map(|expr| walk.selected(expr))
        .collect();
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    for item in &select {
        let mut inner = item;
        while let Expr::Alias(alias) = inner {
            inner = &alias.expr;
        }
        if let Expr::Column(column) = inner
            && column.relation.is_none()
        {
            names
                .entry(column.name.clone())
                .or_insert_with(|| output_name(item));
        }
    }
    let mut binder = Binder::new(turn, layout);
    let relation = Relation::bind(&walk, &mut binder, &names)?;
    let returns_or_term = select.iter().any(|item| {
        item.column_refs()
            .iter()
            .any(|column| column.relation.is_none() && relation.in_or(&column.name))
    });
    if returns_or_term {
        return Err(CompileError::unsupported(
            "DataFusion must return this AI.IF answer, which an OR also tests.",
        ));
    }
    let Selected {
        columns,
        structs,
        spans,
        ..
    } = binder.bind_select(&select)?;
    binder.validate_query(&columns, &structs)?;
    let unplaced_term = relation.unplaced;
    let mut assembly = Assembly::new(&walk, &mut binder, relation, &columns)?;
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
    assembly.check_row_tests()?;
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
    set_scan_columns(&mut found);
    Ok(found)
}

fn unplaced(_plan: &LogicalPlan) -> CompileError {
    CompileError::unsupported("These operators must run outside the model region.")
}

type ByAlias<T> = Vec<(String, Vec<T>)>;

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
