use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::logical_expr::{
    Expr, Filter, JoinType, LogicalPlan, Projection, Sort, SortExpr, UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::ai::{AiNode, Semantics};
use crate::plan::{as_node, extension_node};

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
        let lifted = plan.transform_down(lift_above_limit)?;
        let mut plan = lifted.data;
        let mut changed = lifted.transformed;
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

fn plan_tables(plan: &LogicalPlan) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let _ = plan.apply(|node| {
        if let LogicalPlan::SubqueryAlias(alias) = node {
            found.insert(alias.alias.table().to_string());
            return Ok(TreeNodeRecursion::Jump);
        }
        if let LogicalPlan::TableScan(scan) = node {
            found.insert(scan.table_name.table().to_string());
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

fn plan_answers(plan: &LogicalPlan) -> Vec<(String, BTreeSet<String>)> {
    let mut found = Vec::new();
    let _ = plan.apply(|node| {
        if let Some(column) = as_node::<AiNode>(node) {
            let read = expr_tables(&column.call(), &[]);
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

fn expr_answers(expr: &Expr) -> BTreeSet<String> {
    expr.column_refs()
        .into_iter()
        .filter(|column| column.relation.is_none())
        .map(|column| column.name.clone())
        .collect()
}

struct Mover {
    tables: BTreeSet<String>,
    added: BTreeSet<String>,
    pairs: bool,
}

#[derive(PartialEq)]
enum Role {
    TableFilter,
    Join,
    Column,
}

fn role(node: &AiNode) -> Role {
    if !node.is_predicate() {
        return Role::Column;
    }
    if node.semantics() == Semantics::Rows && expr_tables(&node.call(), &[]).len() < 2 {
        return Role::TableFilter;
    }
    Role::Join
}

fn ai_join(plan: &LogicalPlan) -> Option<&AiNode> {
    as_node::<AiNode>(plan).filter(|node| role(node) == Role::Join)
}

fn mover(node: &LogicalPlan) -> Option<Mover> {
    if let Some(ai) = as_node::<AiNode>(node) {
        if ai.has_fetch() {
            return None;
        }
        let answers = plan_answers(ai.inputs()[0]);
        let mut tables = expr_tables(&ai.call(), &[]);
        for expr in ai.guard().into_iter().chain(ai.filter()) {
            tables.extend(expr_tables(&expr, &answers));
        }
        let added = ai
            .returned_fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        return match role(ai) {
            Role::TableFilter => (tables.len() == 1).then_some(Mover {
                tables,
                added,
                pairs: false,
            }),
            Role::Column => Some(Mover {
                pairs: tables.len() > 1,
                tables,
                added,
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
            return None;
        }
        let unknown = filter.predicate.column_refs().iter().any(|column| {
            column.relation.is_none() && !answers.iter().any(|(name, _)| *name == column.name)
        });
        if unknown {
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

fn top_rows(plan: &LogicalPlan) -> Option<&Sort> {
    match plan {
        LogicalPlan::Limit(limit) => match limit.input.as_ref() {
            LogicalPlan::Sort(sort) if sort.fetch.is_none() => Some(sort),
            _ => None,
        },
        LogicalPlan::Sort(sort) if sort.fetch.is_some() => Some(sort),
        _ => None,
    }
}

fn sorted(expr: Vec<SortExpr>, input: LogicalPlan, fetch: Option<usize>) -> LogicalPlan {
    LogicalPlan::Sort(Sort {
        expr,
        input: Arc::new(input),
        fetch,
    })
}

fn lift_above_limit(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let Some(sort) = top_rows(&plan) else {
        return Ok(Transformed::no(plan));
    };
    let (projection, below) = match sort.input.as_ref() {
        LogicalPlan::Projection(projection) => (Some(projection), projection.input.as_ref()),
        other => (None, other),
    };
    let Some(node) = as_node::<AiNode>(below) else {
        return Ok(Transformed::no(plan));
    };
    let shares = as_node::<AiNode>(node.inputs()[0]).is_some();
    if !node.asks_every_row() || shares {
        return Ok(Transformed::no(plan));
    }
    let mut keys_below = Vec::new();
    for key in &sort.expr {
        let expr = match projection {
            Some(projection) => crate::rules::repeated::through_projection(&key.expr, projection),
            None => Some(key.expr.clone()),
        };
        match expr {
            Some(expr) if expr_answers(&expr).iter().all(|name| !node.is_field(name)) => {
                keys_below.push(SortExpr::new(expr, key.asc, key.nulls_first))
            }
            _ => return Ok(Transformed::no(plan)),
        }
    }
    let rows = node.inputs()[0].clone();
    let kept = match &plan {
        LogicalPlan::Limit(_) => with_input(&plan, sorted(keys_below, rows, None))?,
        _ => sorted(keys_below, rows, sort.fetch),
    };
    let mut lifted = extension_node(node.with_input(kept)?);
    if let Some(projection) = projection {
        lifted = LogicalPlan::Projection(Projection::try_new(
            projection.expr.clone(),
            Arc::new(lifted),
        )?);
    }
    Ok(Transformed::yes(lifted))
}

fn with_input(node: &LogicalPlan, below: LogicalPlan) -> Result<LogicalPlan> {
    crate::plan::with_inputs(node, vec![below])
}

fn join_sides(plan: &LogicalPlan) -> Option<(BTreeSet<String>, BTreeSet<String>)> {
    match plan {
        LogicalPlan::Join(join) => Some((plan_tables(&join.left), plan_tables(&join.right))),
        _ => None,
    }
}

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
    let moved = match &input {
        LogicalPlan::Join(join) => {
            let (left, right) = (plan_tables(&join.left), plan_tables(&join.right));
            let (keeps_left, keeps_right) = preserved_sides(join.join_type);
            if keeps_left && moving.tables.is_subset(&left) {
                let below = with_input(&node, join.left.as_ref().clone())?;
                Some(crate::plan::with_inputs(
                    &input,
                    vec![below, join.right.as_ref().clone()],
                )?)
            } else if keeps_right && moving.tables.is_subset(&right) {
                let below = with_input(&node, join.right.as_ref().clone())?;
                Some(crate::plan::with_inputs(
                    &input,
                    vec![join.left.as_ref().clone(), below],
                )?)
            } else {
                None
            }
        }
        LogicalPlan::Extension(_) if let Some(join) = ai_join(&input) => {
            let sides = join_sides(join.inputs()[0]);
            let fits = sides.is_some_and(|(left, right)| {
                moving.tables.is_subset(&left) || moving.tables.is_subset(&right)
            });
            let same_pair = moving.pairs && expr_tables(&join.call(), &[]) == moving.tables;
            if fits && !same_pair {
                let below = with_input(&node, join.inputs()[0].clone())?;
                Some(with_input(&input, below)?)
            } else {
                None
            }
        }
        LogicalPlan::Filter(filter) => {
            let answers = plan_answers(&filter.input);
            let below_tables = expr_tables(&filter.predicate, &answers);
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
                .all(|expr| matches!(expr, Expr::Column(_)))
                || as_node::<AiNode>(&node).is_some_and(|ai| {
                    ai.reads().iter().all(|read| {
                        projection
                            .expr
                            .iter()
                            .any(|passed| matches!(passed, Expr::Column(found) if found == read))
                    })
                }) =>
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

enum Item {
    Filter(AiNode),
    Column(AiNode),
    Test(Expr),
}

fn order_stack(node: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
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
            below = ai.inputs()[0].clone();
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
    let mut sequence: Vec<Item> = filters.iter().cloned().map(Item::Filter).collect();
    let mut placed: Vec<&AiNode> = Vec::new();
    let mut done_tests = vec![false; tests.len()];
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
            if !placed.iter().any(|found| found.answer() == column.answer()) {
                placed.push(column);
                sequence.push(Item::Column(column.clone()));
            }
        }
        sequence.push(Item::Test(test.clone()));
        done_tests[index] = true;
    }
    let mut ordered: Vec<&AiNode> = columns.iter().collect();
    ordered.sort_by_key(|column| kind(column));
    for column in ordered {
        if placed.iter().any(|found| found.answer() == column.answer()) {
            continue;
        }
        placed.push(column);
        sequence.push(Item::Column(column.clone()));
        let available: BTreeSet<String> = placed.iter().flat_map(|found| names(found)).collect();
        for (index, test) in tests.iter().enumerate() {
            if !done_tests[index] && expr_answers(test).is_subset(&available) {
                sequence.push(Item::Test(test.clone()));
                done_tests[index] = true;
            }
        }
    }
    for (test, _) in tests.iter().zip(&done_tests).filter(|(_, done)| !**done) {
        sequence.push(Item::Test(test.clone()));
    }
    let stack: BTreeSet<String> = filters
        .iter()
        .chain(&columns)
        .flat_map(AiNode::added_names)
        .collect();
    let needs = |item: &Item| -> BTreeSet<String> {
        let read: BTreeSet<String> = match item {
            Item::Filter(found) | Item::Column(found) => found
                .reads()
                .into_iter()
                .filter(|column| column.relation.is_none())
                .map(|column| column.name)
                .collect(),
            Item::Test(test) => expr_answers(test),
        };
        read.intersection(&stack).cloned().collect()
    };
    let mut plan = below;
    let mut available: BTreeSet<String> = BTreeSet::new();
    while !sequence.is_empty() {
        let Some(next) = sequence
            .iter()
            .position(|item| needs(item).is_subset(&available))
        else {
            return Ok(Transformed::no(node));
        };
        plan = match sequence.remove(next) {
            Item::Filter(found) | Item::Column(found) => {
                available.extend(found.added_names());
                extension_node(found.with_input(plan)?)
            }
            Item::Test(test) => LogicalPlan::Filter(Filter::try_new(test, Arc::new(plan))?),
        };
    }
    let changed = plan != node;
    Ok(Transformed::new(plan, changed, TreeNodeRecursion::Jump))
}
