//! The built-in logical rules and the loop that applies them, as
//! `quail/planner/rules/logical.py`, `tree_transforms.py`, and
//! `quail/planner/optimizer/tree_rewrites.py` define them.

use std::collections::{BTreeMap, BTreeSet};

use quail_physical::{
    Asked, ColumnRef, Condition, Equality, ModelCall, Projected, SortExpression, SortKey,
    ValueExpression,
};

use crate::expressions::{CompileError, call_aliases, column_references, condition_aliases, tests};
use crate::logical::{Filter, LogicalNode, LogicalPlan, Scan};
use crate::nodes::{
    above_projection, added_columns, as_column, asked_call, children, column_alias, column_call,
    input, output_schema, source_columns, subtree, validate_plan,
};

/// Lifting stops after this many passes over the candidates.
pub const MAX_PASSES: usize = 4;

/// The name shown for a candidate whose column operators run above a join.
pub const LIFT: &str = "should_lift_column_operators";

/// A logical plan and the rewrites that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalAlternative {
    pub plan: LogicalPlan,
    pub rules: Vec<String>,
}

/// The placements of a plan's column operators that physical planning
/// prices: the plan as bound, then each plan with a classification,
/// extraction, or map moved above a join, each scan loading the columns
/// its plan reads.
///
/// # Errors
///
/// A moved plan is not valid.
pub fn column_placements(plan: LogicalPlan) -> Result<Vec<LogicalAlternative>, CompileError> {
    let mut candidates = vec![LogicalAlternative {
        plan,
        rules: Vec::new(),
    }];
    for _ in 0..MAX_PASSES {
        let mut added: Vec<LogicalAlternative> = Vec::new();
        for candidate in &candidates {
            let Some(lifted) = lift_column_operators(&candidate.plan.root) else {
                continue;
            };
            let root = push_down_projection(&lifted);
            if candidates
                .iter()
                .chain(&added)
                .any(|other| other.plan.root == root)
            {
                continue;
            }
            let plan = LogicalPlan { root };
            validate_plan(&plan)?;
            let mut rules = candidate.rules.clone();
            rules.push(LIFT.to_string());
            added.push(LogicalAlternative { plan, rules });
        }
        if added.is_empty() {
            break;
        }
        candidates.extend(added);
    }
    Ok(candidates)
}

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
        LogicalNode::WithCte(with) => {
            assert_eq!(
                new.len(),
                with.ctes.len() + 1,
                "WithCTE needs its definitions and its input"
            );
            *with.input = new.pop().expect("an input");
            for ((_, cte), child) in with.ctes.iter_mut().zip(new) {
                *cte = child;
            }
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
        LogicalNode::WithCte(found) => &mut found.input,
        LogicalNode::SemanticClassify(found)
        | LogicalNode::SemanticExtract(found)
        | LogicalNode::SemanticMap(found)
        | LogicalNode::SemanticSaeFeatures(found) => &mut found.input,
        LogicalNode::Scan(_) | LogicalNode::Join(_) => panic!("the node has no single input"),
    }
}

/// A Filter's one table, as `Filter.alias`.
fn filter_alias(filter: &Filter) -> String {
    let aliases = condition_aliases(&filter.condition);
    assert_eq!(
        aliases.len(),
        1,
        "a Filter below the Project reads one table"
    );
    aliases[0].clone()
}

// ---------------------------------------------------------------- columns read

fn push_unique(found: &mut Vec<ColumnRef>, references: Vec<ColumnRef>) {
    for reference in references {
        if !found.contains(&reference) {
            found.push(reference);
        }
    }
}

fn value_refs(expression: &ValueExpression) -> Vec<ColumnRef> {
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

fn call_refs(call: &ModelCall) -> Vec<ColumnRef> {
    let mut found = Vec::new();
    for argument in &call.prompt.args {
        push_unique(&mut found, value_refs(argument));
    }
    found
}

fn condition_refs(condition: &Condition) -> Vec<ColumnRef> {
    let mut found = Vec::new();
    for test in tests(condition) {
        push_unique(&mut found, value_refs(&test.expression));
    }
    found
}

fn asked_refs(asked: &Asked) -> Vec<ColumnRef> {
    match asked {
        Asked::Call(call) => call_refs(call),
        Asked::Compare(test) => value_refs(&test.expression),
    }
}

fn equality_refs(equality: &Equality) -> Vec<ColumnRef> {
    let mut found = value_refs(&equality.left);
    push_unique(&mut found, value_refs(&equality.right));
    found
}

fn sort_refs(key: &SortKey) -> Vec<ColumnRef> {
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
        LogicalNode::WithCte(with) => match with.input.as_ref() {
            LogicalNode::Result(result) => result.order.clone(),
            _ => Vec::new(),
        },
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

// ---------------------------------------------------------------- lifting

/// alias -> the column operators and Filters lifted from its sequence.
type Sequences = Vec<(String, Vec<LogicalNode>)>;

fn extend_sequence(sequences: &mut Sequences, alias: String, nodes: Vec<LogicalNode>) {
    match sequences.iter_mut().find(|(found, _)| *found == alias) {
        Some((_, found)) => found.extend(nodes),
        None => sequences.push((alias, nodes)),
    }
}

fn sequence_alias(node: &LogicalNode) -> String {
    match node {
        LogicalNode::Filter(filter) => filter_alias(filter),
        column => column_alias(as_column(column).expect("a column operator")),
    }
}

/// Split a table's sequence at its column operators: the node below the
/// lowest one, and the operators and Filters, lowest first.
fn column_sequence(node: &LogicalNode) -> (LogicalNode, Vec<LogicalNode>) {
    let mut lifted = Vec::new();
    let mut node = node;
    while matches!(node, LogicalNode::Filter(_)) || as_column(node).is_some() {
        lifted.push(node.clone());
        node = input(node);
    }
    lifted.reverse();
    (node.clone(), lifted)
}

/// Move a joined table's column operators above its joins; None when no
/// column operator moved.
pub fn lift_column_operators(root: &LogicalNode) -> Option<LogicalNode> {
    let is_project_over_rows =
        matches!(root, LogicalNode::Project(project) if above_projection(&project.input));
    if matches!(root, LogicalNode::Result(_)) || above_projection(root) || is_project_over_rows {
        let rewritten = lift_column_operators(input(root))?;
        return Some(with_children(root, vec![rewritten]));
    }
    let nodes = subtree(root);
    let mut fixed: BTreeSet<String> = BTreeSet::new();
    let mut joined: BTreeSet<String> = BTreeSet::new();
    for node in &nodes {
        if let LogicalNode::SemanticJoin(join) = node {
            for alias in call_aliases(asked_call(&join.predicate)) {
                if join.semantics != "full" && Some(&alias) != join.anchor.as_ref() {
                    fixed.insert(alias.clone());
                }
                joined.insert(alias);
            }
        }
    }
    let joined: BTreeSet<String> = joined.difference(&fixed).cloned().collect();
    let mut sequences: Sequences = Vec::new();
    let (top, lifted) = column_sequence(input(root));
    for node in lifted {
        let alias = sequence_alias(&node);
        extend_sequence(&mut sequences, alias, vec![node]);
    }

    fn cut(
        node: &LogicalNode,
        joined: &BTreeSet<String>,
        sequences: &mut Sequences,
    ) -> LogicalNode {
        let (base, sequence) = column_sequence(node);
        if let Some(first) = sequence.first() {
            let alias = sequence_alias(first);
            if joined.contains(&alias) {
                extend_sequence(sequences, alias, sequence);
                return base;
            }
        }
        node.clone()
    }

    fn visit(
        node: &LogicalNode,
        joined: &BTreeSet<String>,
        sequences: &mut Sequences,
    ) -> LogicalNode {
        match node {
            LogicalNode::Join(join) => {
                let left = visit(&join.left, joined, sequences);
                let right = cut(&join.right, joined, sequences);
                with_children(node, vec![left, right])
            }
            LogicalNode::SemanticJoin(_) => {
                let child = visit(input(node), joined, sequences);
                with_children(node, vec![child])
            }
            LogicalNode::SemanticClassify(column)
                if call_aliases(column_call(column)).len() == 2 =>
            {
                let child = visit(input(node), joined, sequences);
                with_children(node, vec![child])
            }
            _ => cut(node, joined, sequences),
        }
    }

    let mut spine = visit(&top, &joined, &mut sequences);
    if sequences.is_empty() {
        return None;
    }
    for scan in &nodes {
        if let LogicalNode::Scan(scan) = scan
            && let Some((_, found)) = sequences.iter().find(|(alias, _)| *alias == scan.alias)
        {
            for node in found {
                spine = with_children(node, vec![spine]);
            }
        }
    }
    let lifted = with_children(root, vec![spine]);
    (lifted != *root).then_some(lifted)
}
