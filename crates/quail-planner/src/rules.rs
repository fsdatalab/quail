//! The columns an operator's expressions read, and the projection of a
//! bound plan's scans to the column values the plan reads.

use std::collections::{BTreeMap, BTreeSet};

use quail_physical::{
    Asked, ColumnRef, Condition, Equality, ModelCall, Projected, SortExpression, SortKey,
    ValueExpression,
};

use crate::expressions::{column_references, tests};
use crate::logical::{LogicalNode, LogicalPlan, Scan};
use crate::nodes::{
    added_columns, as_column, children, column_alias, column_call, output_schema, source_columns,
    subtree,
};

// ---------------------------------------------------------------- tree editing

/// The node with its inputs replaced, as `with_children`.
///
/// # Panics
///
/// The number of children does not fit the node.
pub fn with_children(node: &LogicalNode, mut new: Vec<LogicalNode>) -> LogicalNode {
    let mut node = node.clone();
    match &mut node {
        LogicalNode::Scan(_) => assert!(new.is_empty(), "Scan has no children"),
        LogicalNode::Join(join) => {
            assert_eq!(new.len(), 2, "Join needs two children");
            *join.right = new.pop().expect("a right child");
            *join.left = new.pop().expect("a left child");
        }
        other => {
            assert_eq!(new.len(), 1, "the node needs one child");
            *input_mut(other) = new.pop().expect("an input");
        }
    }
    node
}

fn input_mut(node: &mut LogicalNode) -> &mut LogicalNode {
    match node {
        LogicalNode::SemanticFilter(found) => &mut found.input,
        LogicalNode::Filter(found) => &mut found.input,
        LogicalNode::SemanticJoin(found) => &mut found.input,
        LogicalNode::Project(found) => &mut found.input,
        LogicalNode::Result(found) => &mut found.input,
        LogicalNode::Unnest(found) => &mut found.input,
        LogicalNode::SemanticClassify(found)
        | LogicalNode::SemanticExtract(found)
        | LogicalNode::SemanticMap(found)
        | LogicalNode::SemanticSaeFeatures(found) => &mut found.input,
        LogicalNode::Scan(_) | LogicalNode::Join(_) => panic!("the node has no single input"),
    }
}

// ---------------------------------------------------------------- columns read

fn push_unique(found: &mut Vec<ColumnRef>, references: Vec<ColumnRef>) {
    for reference in references {
        if !found.contains(&reference) {
            found.push(reference);
        }
    }
}

/// The columns a value expression reads, through a model call's prompt,
/// in first use order.
pub fn value_refs(expression: &ValueExpression) -> Vec<ColumnRef> {
    match expression {
        ValueExpression::ModelCall(call) => call_refs(call),
        ValueExpression::Alias(alias) => match alias.expression.as_ref() {
            ValueExpression::ModelCall(call) => call_refs(call),
            other => column_references(other),
        },
        ValueExpression::Literal(_) => Vec::new(),
        other => column_references(other),
    }
}

/// The columns a model call's prompt reads, in first use order.
pub fn call_refs(call: &ModelCall) -> Vec<ColumnRef> {
    let mut found = Vec::new();
    for argument in &call.prompt.args {
        push_unique(&mut found, value_refs(argument));
    }
    found
}

/// The columns a condition reads, in first use order.
pub fn condition_refs(condition: &Condition) -> Vec<ColumnRef> {
    let mut found = Vec::new();
    for test in tests(condition) {
        push_unique(&mut found, value_refs(&test.expression));
    }
    found
}

/// The columns an AI predicate reads, in first use order.
pub fn asked_refs(asked: &Asked) -> Vec<ColumnRef> {
    match asked {
        Asked::Call(call) => call_refs(call),
        Asked::Compare(test) => value_refs(&test.expression),
    }
}

/// The columns a join equality reads, in first use order.
pub fn equality_refs(equality: &Equality) -> Vec<ColumnRef> {
    let mut found = value_refs(&equality.left);
    push_unique(&mut found, value_refs(&equality.right));
    found
}

/// The columns a sort key reads, in first use order.
pub fn sort_refs(key: &SortKey) -> Vec<ColumnRef> {
    match &key.expression {
        SortExpression::Value(value) => value_refs(value),
    }
}

/// The source columns a node's expressions read, in first use order.
fn node_refs(node: &LogicalNode) -> Vec<ColumnRef> {
    let mut found = Vec::new();
    match node {
        LogicalNode::Scan(scan) => {
            for predicate in &scan.predicates {
                push_unique(&mut found, condition_refs(predicate));
            }
        }
        LogicalNode::SemanticFilter(filter) => {
            for predicate in &filter.predicates {
                push_unique(&mut found, asked_refs(&predicate.expression));
            }
        }
        LogicalNode::Filter(filter) => found = condition_refs(&filter.condition),
        LogicalNode::Join(join) => {
            for equality in &join.on {
                push_unique(&mut found, equality_refs(equality));
            }
        }
        LogicalNode::SemanticJoin(join) => found = asked_refs(&join.predicate),
        LogicalNode::Project(project) => {
            for column in &project.columns {
                if let Projected::Expression(expression) = column {
                    push_unique(&mut found, value_refs(expression));
                }
            }
        }
        LogicalNode::Result(result) => {
            for key in &result.order {
                push_unique(&mut found, sort_refs(key));
            }
        }
        column if as_column(column).is_some() => {
            found = call_refs(column_call(as_column(column).expect("a column operator")));
        }
        _ => {}
    }
    found
}

// ---------------------------------------------------------------- projection

/// Rewrite every Scan to keep only the column values the plan reads.
pub fn push_down_projection(root: &LogicalNode) -> LogicalNode {
    let plan = LogicalPlan { root: root.clone() };
    let walk = subtree(root);
    let projection = walk
        .iter()
        .rev()
        .find(|node| matches!(node, LogicalNode::Project(_)));
    let mut valued: BTreeSet<(String, String)> = BTreeSet::new();
    if let Some(projection) = projection {
        for reference in output_schema(projection) {
            valued.insert((reference.alias, reference.column));
        }
        let LogicalNode::Project(found) = projection else {
            unreachable!("a Project")
        };
        for column in &found.columns {
            if let Projected::Expression(expression) = column {
                for reference in value_refs(expression) {
                    valued.insert((reference.alias, reference.column));
                }
            }
        }
    }
    let order: Vec<SortKey> = match &plan.root {
        LogicalNode::Result(result) => result.order.clone(),
        _ => Vec::new(),
    };
    for key in &order {
        for reference in sort_refs(key) {
            valued.insert((reference.alias, reference.column));
        }
    }
    for node in &walk {
        let references = match node {
            LogicalNode::Join(_) | LogicalNode::Scan(_) => node_refs(node),
            LogicalNode::SemanticExtract(column) | LogicalNode::SemanticMap(column) => {
                call_refs(column_call(column))
            }
            _ => continue,
        };
        for reference in references {
            valued.insert((reference.alias, reference.column));
        }
    }

    fn descend(
        node: &LogicalNode,
        needed: &BTreeMap<String, Vec<String>>,
        valued: &BTreeSet<(String, String)>,
    ) -> LogicalNode {
        let mut needed = needed.clone();
        for reference in node_refs(node) {
            let columns = needed.entry(reference.alias).or_default();
            if !columns.contains(&reference.column) {
                columns.push(reference.column);
            }
        }
        if let Some(column) = as_column(node) {
            let alias = column_alias(column);
            if let Some(columns) = needed.get_mut(&alias) {
                let added = added_columns(node);
                columns.retain(|name| !added.contains(name));
            }
        }
        if let LogicalNode::Scan(scan) = node {
            let sources = source_columns(&scan.column);
            let columns: Vec<String> = needed
                .get(&scan.alias)
                .map(|columns| {
                    columns
                        .iter()
                        .filter(|column| {
                            !sources.contains(column)
                                || valued.contains(&(scan.alias.clone(), (*column).clone()))
                        })
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            return if columns == scan.columns {
                node.clone()
            } else {
                LogicalNode::Scan(Scan {
                    columns,
                    ..scan.clone()
                })
            };
        }
        let old = children(node);
        let new: Vec<LogicalNode> = old
            .iter()
            .map(|child| descend(child, &needed, valued))
            .collect();
        if new.iter().zip(&old).all(|(a, b)| a == *b) {
            node.clone()
        } else {
            with_children(node, new)
        }
    }

    descend(root, &BTreeMap::new(), &valued)
}
