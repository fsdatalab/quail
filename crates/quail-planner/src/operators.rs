//! A plan's operators as the planner reads them, as `LogicalPlan.operators`.

use quail_physical::{Condition, FilterPredicate, ModelCall, Projected, ValueExpression};

use crate::expressions::{call_aliases, condition_aliases, condition_columns, projected_call};
use crate::logical::{LogicalNode, LogicalPlan, Scan, SemanticColumn, SemanticJoin};
use crate::nodes::{added_columns, as_column, asked_call, column_call, input, subtree};

/// A Filter on the columns that column operators compute.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnTest {
    /// The column operator the Filter follows, the last one it reads.
    pub name: String,
    pub call: ModelCall,
    pub condition: Condition,
    pub selectivity: Option<f64>,
    /// Its position among the table's filters, after the table's AI.IF
    /// predicates, the label tests first.
    pub position: usize,
}

/// The operators of one plan, in the order the plan was written.
#[derive(Debug, Clone, Default)]
pub struct Operators {
    pub scans: Vec<Scan>,
    /// alias -> its AI.IF predicates, by alias first seen.
    pub filters: Vec<(String, Vec<FilterPredicate>)>,
    pub joins: Vec<SemanticJoin>,
    /// The named model calls SELECT returns, in SELECT order.
    pub projections: Vec<ValueExpression>,
    pub classifies: Vec<SemanticColumn>,
    pub extracts: Vec<SemanticColumn>,
    pub sae_features: Vec<SemanticColumn>,
    pub maps: Vec<SemanticColumn>,
    /// alias -> its label tests, then its answer tests.
    pub column_filters: Vec<(String, Vec<ColumnTest>)>,
}

/// The first projection over source columns and model answers.
///
/// # Panics
///
/// The plan has no Project.
pub fn projection(plan: &LogicalPlan) -> &LogicalNode {
    let root = match &plan.root {
        LogicalNode::WithCte(with) => with.input.as_ref(),
        other => other,
    };
    subtree(root)
        .into_iter()
        .find(|node| matches!(node, LogicalNode::Project(_)))
        .expect("a query plan needs a projection")
}

/// The last column operator below a Filter whose columns it reads.
fn producer(filter: &LogicalNode) -> &SemanticColumn {
    let LogicalNode::Filter(found) = filter else {
        panic!("a Filter")
    };
    let read = condition_columns(&found.condition);
    let mut node = found.input.as_ref();
    loop {
        if as_column(node).is_some() && added_columns(node).iter().any(|name| read.contains(name)) {
            return as_column(node).expect("a column operator");
        }
        node = input(node);
    }
}

fn push<T>(map: &mut Vec<(String, Vec<T>)>, alias: &str, value: T) {
    match map.iter_mut().find(|(found, _)| found == alias) {
        Some((_, values)) => values.push(value),
        None => map.push((alias.to_string(), vec![value])),
    }
}

/// The plan's operators as the planner reads them.
pub fn operators(plan: &LogicalPlan) -> Operators {
    let top = projection(plan);
    let nodes = subtree(top);
    let mut found = Operators::default();
    let mut tested: Vec<(String, Vec<&LogicalNode>)> = Vec::new();
    let mut aliases: Vec<String> = Vec::new();
    let note = |aliases: &mut Vec<String>, alias: &str| {
        if !aliases.iter().any(|found| found == alias) {
            aliases.push(alias.to_string());
        }
    };
    for node in &nodes {
        match node {
            LogicalNode::Scan(scan) => found.scans.push(scan.clone()),
            LogicalNode::SemanticFilter(filter) => {
                for predicate in &filter.predicates {
                    let alias = call_aliases(asked_call(&predicate.expression))[0].clone();
                    push(&mut found.filters, &alias, predicate.clone());
                    note(&mut aliases, &alias);
                }
            }
            LogicalNode::Filter(filter) => {
                let alias = condition_aliases(&filter.condition)[0].clone();
                push(&mut tested, &alias, *node);
                note(&mut aliases, &alias);
            }
            LogicalNode::SemanticJoin(join) => found.joins.push(join.clone()),
            LogicalNode::SemanticClassify(column) => found.classifies.push(column.clone()),
            LogicalNode::SemanticExtract(column) => found.extracts.push(column.clone()),
            LogicalNode::SemanticSaeFeatures(column) => found.sae_features.push(column.clone()),
            LogicalNode::SemanticMap(column) => found.maps.push(column.clone()),
            _ => {}
        }
    }
    for alias in &aliases {
        let Some((_, filters)) = tested.iter().find(|(found, _)| found == alias) else {
            continue;
        };
        let mut ordered: Vec<&LogicalNode> = filters.clone();
        ordered.sort_by_key(|filter| column_call(producer(filter)).kind != "label");
        let asks = found
            .filters
            .iter()
            .find(|(found, _)| found == alias)
            .map(|(_, predicates)| predicates.len())
            .unwrap_or(0);
        let column_tests = ordered
            .iter()
            .enumerate()
            .map(|(index, filter)| {
                let LogicalNode::Filter(test) = filter else {
                    unreachable!("a Filter")
                };
                let made_by = producer(filter);
                ColumnTest {
                    name: made_by.name.clone(),
                    call: column_call(made_by).clone(),
                    condition: test.condition.clone(),
                    selectivity: test.selectivity,
                    position: asks + index,
                }
            })
            .collect();
        found.column_filters.push((alias.clone(), column_tests));
    }
    if let LogicalNode::Project(project) = top {
        found.projections = project
            .columns
            .iter()
            .filter(|column| projected_call(column, &[]).is_some())
            .filter_map(|column| match column {
                Projected::Expression(expression) => Some(expression.clone()),
                Projected::Name(_) => None,
            })
            .collect();
    }
    found
}
