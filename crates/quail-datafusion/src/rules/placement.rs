//! The `PlaceAiNodes` optimizer rule: Quail's `AiNode`s on the tables
//! and joins they read.
//!
//! [`crate::LowerAiCalls`] leaves each `AiNode` where the query wrote its
//! call: an `AI.IF` in WHERE above the joins, a classification below the
//! SELECT list. The rule moves each node as low as its inputs allow, as
//! `PushDownFilter` does for plain filters:
//!
//! - A node with a fetch stays where it is: below a filter or join it
//!   would limit different rows.
//! - An AI filter or AI column of one table, and a test of one table's
//!   answers, move through joins, AI joins, other filters, windows, and
//!   projections that only pass columns, down to the table. Through an
//!   outer, semi, or anti join they move only on the side whose rows
//!   the join returns ([`preserved_sides`]).
//! - An AI join or `EXISTS` gate, and an AI column of joined rows, move
//!   to the lowest join whose inputs hold every table they read; a
//!   classification of joined rows stays above the AI join of the same
//!   two tables.
//!
//! Then it orders the operators on each table as Quail's executor reads
//! them: the table's `AI.IF` and `AI.SCORE` filters in written order;
//! each test of a label right after the classification it reads; the
//! other classifications; then each extraction, SAE feature, and map
//! column, each followed by the tests its answers allow.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{Expr, Filter, JoinType, LogicalPlan, Projection};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};

/// Move each `AiNode` to the tables and joins it reads and order each
/// table's operators, as the module explains.
#[derive(Debug, Default)]
pub struct PlaceAiNodes;

impl OptimizerRule for PlaceAiNodes {
    fn name(&self) -> &str {
        "place_ai_nodes"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let mut plan = plan;
        let mut changed = false;
        // each step moves one node one operator down; a plan has fewer
        // operators than this many steps
        for _ in 0..256 {
            let step = plan.transform_down(push_down)?;
            plan = step.data;
            if !step.transformed {
                break;
            }
            changed = true;
        }
        let ordered = plan.transform_up(order_stack)?;
        Ok(Transformed::new_transformed(
            ordered.data,
            changed || ordered.transformed,
        ))
    }
}

/// The aliases of the tables a plan scans, not looking into derived tables.
fn plan_tables(plan: &LogicalPlan) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let _ = plan.apply(|node| {
        if let LogicalPlan::SubqueryAlias(alias) = node {
            found.insert(alias.alias.table().to_string());
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

/// The answer columns the `AiNode`s in a plan return, with the tables
/// each reads.
fn plan_answers(plan: &LogicalPlan) -> Vec<(String, BTreeSet<String>)> {
    let mut found = Vec::new();
    let _ = plan.apply(|node| {
        if let Some(column) = as_node::<AiNode>(node) {
            let read = expr_tables(&column.call, &[]);
            for field in &column.returned_fields() {
                found.push((field.name().clone(), read.clone()));
            }
        }
        if matches!(node, LogicalPlan::SubqueryAlias(_)) {
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

/// The tables an expression reads: its columns' tables, and for an
/// answer column, the tables its `AiNode` reads.
fn expr_tables(expr: &Expr, answers: &[(String, BTreeSet<String>)]) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for column in expr.column_refs() {
        match &column.relation {
            Some(table) => {
                found.insert(table.table().to_string());
            }
            None => {
                if let Some((_, read)) = answers.iter().find(|(name, _)| *name == column.name) {
                    found.extend(read.iter().cloned());
                }
            }
        }
    }
    found
}

/// The answer columns an expression reads.
fn expr_answers(expr: &Expr) -> BTreeSet<String> {
    expr.column_refs()
        .into_iter()
        .filter(|column| column.relation.is_none())
        .map(|column| column.name.clone())
        .collect()
}

/// A node that moves down, with the tables it reads and the columns it adds.
struct Mover {
    tables: BTreeSet<String>,
    added: BTreeSet<String>,
    /// Whether it pairs rows of two tables: an AI join or a
    /// classification of joined rows.
    pairs: bool,
}

/// What an `AiNode` does to the rows of the tables it reads.
#[derive(PartialEq)]
enum Role {
    /// Keeps the rows of one table that pass its test.
    TableFilter,
    /// Keeps the pairs of two tables that pass its test, or gates rows.
    Join,
    /// Adds an answer column.
    Column,
}

fn role(node: &AiNode) -> Role {
    if !node.is_predicate() {
        return Role::Column;
    }
    if node.semantics == Semantics::Rows && expr_tables(&node.call, &[]).len() < 2 {
        return Role::TableFilter;
    }
    Role::Join
}

/// The `AiNode` that is an AI join or gate, if `plan` is one.
fn ai_join(plan: &LogicalPlan) -> Option<&AiNode> {
    as_node::<AiNode>(plan).filter(|node| role(node) == Role::Join)
}

fn mover(node: &LogicalPlan) -> Option<Mover> {
    if let Some(ai) = as_node::<AiNode>(node) {
        // moving a node with a fetch below an operator moves the operator
        // above the fetch
        if ai.fetch.is_some() {
            return None;
        }
        let tables = expr_tables(&ai.call, &[]);
        return match role(ai) {
            Role::TableFilter => (tables.len() == 1).then_some(Mover {
                tables,
                added: BTreeSet::new(),
                pairs: false,
            }),
            Role::Column => Some(Mover {
                pairs: tables.len() > 1,
                tables,
                added: ai
                    .returned_fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect(),
            }),
            Role::Join => Some(Mover {
                tables,
                added: BTreeSet::new(),
                pairs: true,
            }),
        };
    }
    if let LogicalPlan::Filter(filter) = node {
        let answers = plan_answers(&filter.input);
        if expr_answers(&filter.predicate).is_empty() {
            // a plain filter is DataFusion's to push
            return None;
        }
        let tables = expr_tables(&filter.predicate, &answers);
        return (tables.len() == 1).then_some(Mover {
            tables,
            added: BTreeSet::new(),
            pairs: false,
        });
    }
    None
}

/// `node`, whose one input was `input`, over `below` instead.
fn with_input(node: &LogicalPlan, below: LogicalPlan) -> Result<LogicalPlan> {
    node.with_new_exprs(node.expressions(), vec![below])
}

/// The tables each input of a join holds.
fn join_sides(plan: &LogicalPlan) -> Option<(BTreeSet<String>, BTreeSet<String>)> {
    match plan {
        LogicalPlan::Join(join) => Some((plan_tables(&join.left), plan_tables(&join.right))),
        _ => None,
    }
}

/// Whether a join returns every row of its left and of its right input
/// that its condition does not drop for the other side's sake: the
/// sides a filter or column of that side alone moves below. It is
/// DataFusion's `lr_is_preserved` (`push_down_filter.rs`), which
/// `PushDownFilter` uses for a plain filter and does not export.
///
/// An outer join adds rows of NULLs for its other side, where a node
/// above the join would read a NULL document, so nothing moves there.
fn preserved_sides(join_type: JoinType) -> (bool, bool) {
    match join_type {
        JoinType::Inner => (true, true),
        JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
            (true, false)
        }
        JoinType::Right | JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark => {
            (false, true)
        }
        JoinType::Full => (false, false),
    }
}

/// Move `node` one operator down when its input does not need it above.
fn push_down(node: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let Some(moving) = mover(&node) else {
        return Ok(Transformed::no(node));
    };
    let inputs = node.inputs();
    let [input] = inputs.as_slice() else {
        return Ok(Transformed::no(node));
    };
    let input = (*input).clone();
    let reads_added = |plan: &LogicalPlan| {
        plan.expressions()
            .iter()
            .any(|expr| !expr_answers(expr).is_disjoint(&moving.added))
    };
    let moved =
        match &input {
            LogicalPlan::Join(join) => {
                let (left, right) = (plan_tables(&join.left), plan_tables(&join.right));
                let (keeps_left, keeps_right) = preserved_sides(join.join_type);
                if keeps_left && moving.tables.is_subset(&left) {
                    let below = with_input(&node, join.left.as_ref().clone())?;
                    Some(input.with_new_exprs(
                        input.expressions(),
                        vec![below, join.right.as_ref().clone()],
                    )?)
                } else if keeps_right && moving.tables.is_subset(&right) {
                    let below = with_input(&node, join.right.as_ref().clone())?;
                    Some(input.with_new_exprs(
                        input.expressions(),
                        vec![join.left.as_ref().clone(), below],
                    )?)
                } else {
                    None
                }
            }
            LogicalPlan::Extension(_) if ai_join(&input).is_some() => {
                let join = ai_join(&input).expect("an AI join");
                let sides = join_sides(&join.input);
                let fits = sides.is_some_and(|(left, right)| {
                    moving.tables.is_subset(&left) || moving.tables.is_subset(&right)
                });
                // a classification of joined rows reads the pairs the join keeps
                let same_pair = moving.pairs && expr_tables(&join.call, &[]) == moving.tables;
                if fits && !same_pair {
                    let below = with_input(&node, join.input.as_ref().clone())?;
                    Some(with_input(&input, below)?)
                } else {
                    None
                }
            }
            LogicalPlan::Filter(filter) => {
                let answers = plan_answers(&filter.input);
                let below_tables = expr_tables(&filter.predicate, &answers);
                // a test of the same table's answers belongs to the table's stack
                if reads_added(&input) || (!moving.pairs && below_tables == moving.tables) {
                    None
                } else {
                    let below = with_input(&node, filter.input.as_ref().clone())?;
                    Some(with_input(&input, below)?)
                }
            }
            LogicalPlan::Projection(projection)
                if projection
                    .expr
                    .iter()
                    .all(|expr| matches!(expr, Expr::Column(_))) =>
            {
                let below = with_input(&node, projection.input.as_ref().clone())?;
                let mut exprs = projection.expr.clone();
                for name in &moving.added {
                    exprs.push(Expr::Column(Column::from_name(name.clone())));
                }
                Some(LogicalPlan::Projection(Projection::try_new(
                    exprs,
                    Arc::new(below),
                )?))
            }
            _ => None,
        };
    Ok(match moved {
        Some(plan) => Transformed::new(plan, true, TreeNodeRecursion::Stop),
        None => Transformed::no(node),
    })
}

/// One operator of a table's stack.
enum Item {
    Filter(AiNode),
    Column(AiNode),
    Test(Expr),
}

/// Put the operators on one table in Quail's order, as the module explains.
fn order_stack(node: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    // a stack starts at its topmost operator: one whose parent is not on
    // the same table; reorder from each SubqueryAlias's parent chain
    let Some(top) = mover(&node) else {
        return Ok(Transformed::no(node));
    };
    if top.pairs {
        return Ok(Transformed::no(node));
    }
    let mut items: Vec<Item> = Vec::new();
    let mut below = node.clone();
    loop {
        match mover(&below) {
            Some(found) if !found.pairs && found.tables == top.tables => {}
            _ => break,
        }
        if let Some(ai) = as_node::<AiNode>(&below) {
            items.push(if role(ai) == Role::TableFilter {
                Item::Filter(ai.clone())
            } else {
                Item::Column(ai.clone())
            });
            below = ai.input.as_ref().clone();
        } else if let LogicalPlan::Filter(filter) = &below {
            for term in split_conjunction(&filter.predicate).into_iter().rev() {
                items.push(Item::Test(term.clone()));
            }
            below = filter.input.as_ref().clone();
        } else {
            break;
        }
    }
    if items.len() < 2 && !matches!(items.first(), Some(Item::Test(_))) {
        return Ok(Transformed::no(node));
    }
    items.reverse();
    let mut filters: Vec<AiNode> = Vec::new();
    let mut columns: Vec<AiNode> = Vec::new();
    let mut tests: Vec<Expr> = Vec::new();
    for item in items {
        match item {
            Item::Filter(filter) => filters.push(filter),
            Item::Column(column) => columns.push(column),
            Item::Test(term) => tests.push(term),
        }
    }
    let kind = |column: &AiNode| match column.function() {
        "ai_classify" => 0,
        "ai_extract" => 1,
        "ai_sae_features" => 2,
        _ => 3,
    };
    let names = |column: &AiNode| -> BTreeSet<String> {
        column
            .returned_fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    };
    let mut plan = below;
    for filter in &filters {
        plan = extension_node(filter.with_input(plan)?);
    }
    let mut placed: Vec<AiNode> = Vec::new();
    let add_column =
        |plan: LogicalPlan, column: &AiNode, placed: &mut Vec<AiNode>| -> Result<LogicalPlan> {
            placed.push(column.clone());
            Ok(extension_node(column.with_input(plan)?))
        };
    let mut done_tests = vec![false; tests.len()];
    // each test of a label after the classification it reads
    for (index, test) in tests.iter().enumerate() {
        let read = expr_answers(test);
        let reads: Vec<&AiNode> = columns
            .iter()
            .filter(|column| !names(column).is_disjoint(&read))
            .collect();
        if reads.is_empty() || reads.iter().any(|column| kind(column) != 0) {
            continue;
        }
        for column in reads {
            if !placed.iter().any(|found| found.answer == column.answer) {
                plan = add_column(plan, column, &mut placed)?;
            }
        }
        plan = LogicalPlan::Filter(Filter::try_new(test.clone(), Arc::new(plan))?);
        done_tests[index] = true;
    }
    let mut ordered: Vec<&AiNode> = columns.iter().collect();
    ordered.sort_by_key(|column| kind(column));
    for column in ordered {
        if placed.iter().any(|found| found.answer == column.answer) {
            continue;
        }
        plan = add_column(plan, column, &mut placed)?;
        let available: BTreeSet<String> = placed.iter().flat_map(&names).collect();
        for (index, test) in tests.iter().enumerate() {
            if !done_tests[index] && expr_answers(test).is_subset(&available) {
                plan = LogicalPlan::Filter(Filter::try_new(test.clone(), Arc::new(plan))?);
                done_tests[index] = true;
            }
        }
    }
    let rest: Vec<Expr> = tests
        .iter()
        .zip(&done_tests)
        .filter(|(_, done)| !**done)
        .map(|(test, _)| test.clone())
        .collect();
    if let Some(predicate) = conjunction(rest) {
        plan = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(plan))?);
    }
    let changed = plan != node;
    Ok(Transformed::new(plan, changed, TreeNodeRecursion::Jump))
}
