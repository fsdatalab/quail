//! A plan's operators as the planner reads them, as `LogicalPlan.operators`.
//!
//! The physical planner reads a plan only through `Operators`.

use std::collections::BTreeMap;

use quail_physical::{
    Asked, Condition, Equality, ExtractStruct, FilterPredicate, ModelCall, Projected, SortKey,
    ValueExpression,
};

use crate::expressions::{
    PROBABILITIES_SUFFIX, SPAN_SUFFIX, call_aliases, condition_aliases, condition_columns,
    projected_call,
};
use crate::logical::{LogicalNode, LogicalPlan, Scan};
use crate::nodes::{
    above_projection, added_columns, as_column, asked_call, column_call, input, output_schema,
    result_columns, subtree,
};

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
    /// Whether it runs before any join.
    pub before_joins: bool,
}

/// One model call over its input rows, adding its answer column `name`:
/// `AI.CLASSIFY`, `AI.EXTRACT`, `AI.SAE_FEATURES`, or `AI.MAP`, as the
/// call's kind says.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticColumn {
    pub call: ModelCall,
    pub name: String,
    /// Whether it runs above a join.
    pub above_joins: bool,
}

impl SemanticColumn {
    /// The table alias its answer columns belong to.
    pub fn alias(&self) -> String {
        call_aliases(&self.call)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// The columns it adds, its answer first.
    pub fn added_columns(&self) -> Vec<String> {
        match self.call.kind.as_str() {
            "label" if self.call.probabilities => vec![
                self.name.clone(),
                format!("{}{PROBABILITIES_SUFFIX}", self.name),
            ],
            "extract" => vec![self.name.clone(), format!("{}{SPAN_SUFFIX}", self.name)],
            _ => vec![self.name.clone()],
        }
    }
}

/// A model call tested on the pairs of a join.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticJoin {
    pub predicate: Asked,
    /// "full", "exists", or "anti".
    pub semantics: String,
    pub selectivity: Option<f64>,
    /// The table alias whose KV is kept.
    pub anchor: Option<String>,
    /// The equality conditions of the joins below it, in written order.
    pub on: Vec<Equality>,
    /// The aliases of the rows it extends: everything below its new tables.
    pub outer_aliases: Vec<String>,
}

/// The query's SELECT list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Projection {
    /// Value expressions, Aliases, and the names of columns an unnest adds.
    pub columns: Vec<Projected>,
    pub structs: Vec<ExtractStruct>,
}

/// `ORDER BY`, `DISTINCT`, `OFFSET`, and `LIMIT`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResultModifiers {
    pub limit: Option<i64>,
    pub order: Vec<SortKey>,
    pub offset: i64,
    pub distinct: bool,
}

/// An operator on result rows, above the SELECT list. Each holds the
/// names of the result columns it returns.
#[derive(Debug, Clone, PartialEq)]
pub enum RowOperator {
    /// One row per item of each listed list column.
    Unnest {
        lists: Vec<String>,
        outer: bool,
        columns: Vec<String>,
    },
    /// A regular condition on result rows.
    Filter {
        condition: Condition,
        columns: Vec<String>,
    },
    /// A SELECT list over result rows.
    Projection {
        expressions: Vec<Projected>,
        columns: Vec<String>,
    },
}

impl RowOperator {
    /// The names of the result columns it returns.
    pub fn columns(&self) -> &[String] {
        match self {
            RowOperator::Unnest { columns, .. }
            | RowOperator::Filter { columns, .. }
            | RowOperator::Projection { columns, .. } => columns,
        }
    }
}

/// The operators of one plan, in the order the plan was written.
#[derive(Debug, Clone, Default)]
pub struct Operators {
    pub scans: Vec<Scan>,
    /// alias -> its AI.IF predicates, by alias first seen.
    pub filters: Vec<(String, Vec<FilterPredicate>)>,
    /// alias -> whether each of its AI.IF predicates runs before any
    /// join, in the order of `filters`.
    pub filters_before_joins: Vec<(String, Vec<bool>)>,
    pub joins: Vec<SemanticJoin>,
    /// The named model calls SELECT returns, in SELECT order.
    pub projections: Vec<ValueExpression>,
    pub classifies: Vec<SemanticColumn>,
    pub extracts: Vec<SemanticColumn>,
    pub sae_features: Vec<SemanticColumn>,
    pub maps: Vec<SemanticColumn>,
    /// alias -> its label tests, then its answer tests.
    pub column_filters: Vec<(String, Vec<ColumnTest>)>,
    pub projection: Projection,
    pub result: ResultModifiers,
    /// The operators on result rows above the SELECT list, lowest first.
    pub after_projection: Vec<RowOperator>,
    /// alias -> the DISTINCT stop key of its AI.IF predicates.
    pub stop_keys: BTreeMap<String, Vec<String>>,
}

impl Operators {
    /// The column operators of every kind: classifications,
    /// extractions, SAE features, then maps.
    pub fn columns(&self) -> impl Iterator<Item = &SemanticColumn> {
        self.classifies
            .iter()
            .chain(&self.extracts)
            .chain(&self.sae_features)
            .chain(&self.maps)
    }
}

/// The first projection over source columns and model answers.
///
/// # Panics
///
/// The plan has no Project.
fn projection(plan: &LogicalPlan) -> &LogicalNode {
    let root = &plan.root;
    subtree(root)
        .into_iter()
        .find(|node| matches!(node, LogicalNode::Project(_)))
        .expect("a query plan needs a projection")
}

/// The last column operator below a Filter whose columns it reads.
fn producer(filter: &LogicalNode) -> &crate::logical::SemanticColumn {
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

fn joins_below(node: &LogicalNode) -> bool {
    subtree(node)
        .iter()
        .any(|below| matches!(below, LogicalNode::SemanticJoin(_)))
}

fn column_operator(column: &crate::logical::SemanticColumn) -> SemanticColumn {
    SemanticColumn {
        call: column_call(column).clone(),
        name: column.name.clone(),
        above_joins: joins_below(&column.input),
    }
}

fn join_operator(join: &crate::logical::SemanticJoin) -> SemanticJoin {
    fn visit(node: &LogicalNode, found: &mut Vec<Equality>) {
        if let LogicalNode::Join(join) = node {
            visit(&join.left, found);
            found.extend(join.on.iter().cloned());
        }
    }
    let mut on = Vec::new();
    visit(&join.input, &mut on);
    let mut outer = join.input.as_ref();
    while let LogicalNode::Join(inner) = outer {
        outer = &inner.left;
    }
    let mut outer_aliases: Vec<String> = Vec::new();
    for field in output_schema(outer) {
        if !outer_aliases.contains(&field.alias) {
            outer_aliases.push(field.alias);
        }
    }
    SemanticJoin {
        predicate: join.predicate.clone(),
        semantics: join.semantics.clone(),
        selectivity: join.selectivity,
        anchor: join.anchor.clone(),
        on,
        outer_aliases,
    }
}

/// The operators on result rows between the root and the first projection.
fn row_operators(root: &LogicalNode) -> Vec<RowOperator> {
    let mut chain: Vec<RowOperator> = Vec::new();
    let mut walked = match root {
        LogicalNode::Result(result) => result.input.as_ref(),
        other => other,
    };
    loop {
        let (operator, next) = match walked {
            LogicalNode::Unnest(unnest) => (
                RowOperator::Unnest {
                    lists: unnest.columns.clone(),
                    outer: unnest.outer,
                    columns: result_columns(walked),
                },
                &unnest.input,
            ),
            LogicalNode::Filter(filter) => (
                RowOperator::Filter {
                    condition: filter.condition.clone(),
                    columns: result_columns(walked),
                },
                &filter.input,
            ),
            LogicalNode::Project(project) => {
                if !above_projection(&project.input) {
                    break;
                }
                (
                    RowOperator::Projection {
                        expressions: project.columns.clone(),
                        columns: result_columns(walked),
                    },
                    &project.input,
                )
            }
            _ => break,
        };
        chain.insert(0, operator);
        walked = next;
    }
    chain
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
                let before_joins = !joins_below(&filter.input);
                for predicate in &filter.predicates {
                    let alias = call_aliases(asked_call(&predicate.expression))[0].clone();
                    push(&mut found.filters, &alias, predicate.clone());
                    push(&mut found.filters_before_joins, &alias, before_joins);
                    note(&mut aliases, &alias);
                }
            }
            LogicalNode::Filter(filter) => {
                let alias = condition_aliases(&filter.condition)[0].clone();
                push(&mut tested, &alias, *node);
                note(&mut aliases, &alias);
            }
            LogicalNode::SemanticJoin(join) => found.joins.push(join_operator(join)),
            LogicalNode::SemanticClassify(column) => {
                found.classifies.push(column_operator(column));
            }
            LogicalNode::SemanticExtract(column) => found.extracts.push(column_operator(column)),
            LogicalNode::SemanticSaeFeatures(column) => {
                found.sae_features.push(column_operator(column));
            }
            LogicalNode::SemanticMap(column) => found.maps.push(column_operator(column)),
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
                    before_joins: !joins_below(&test.input),
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
        found.projection = Projection {
            columns: project.columns.clone(),
            structs: project.structs.clone(),
        };
    }
    if let LogicalNode::Result(result) = &plan.root {
        found.result = ResultModifiers {
            limit: result.limit,
            order: result.order.clone(),
            offset: result.offset,
            distinct: result.distinct,
        };
    }
    found.after_projection = row_operators(&plan.root);
    for node in subtree(&plan.root) {
        if let LogicalNode::SemanticFilter(filter) = node
            && !filter.stop_key.is_empty()
        {
            let alias = call_aliases(asked_call(&filter.predicates[0].expression))[0].clone();
            found.stop_keys.insert(alias, filter.stop_key.clone());
        }
    }
    found
}
