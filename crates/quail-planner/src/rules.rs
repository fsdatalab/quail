//! The built-in logical rules and the loop that applies them, as
//! `quail/planner/rules/logical.py`, `tree_transforms.py`, and
//! `quail/planner/optimizer/tree_rewrites.py` define them.

use std::collections::{BTreeMap, BTreeSet};

use quail_physical::{
    Asked, ColumnRef, Condition, Equality, FilterPredicate, ModelCall, Projected, SortExpression,
    SortKey, ValueExpression,
};

use crate::expressions::{
    CompileError, aliased_call, call_aliases, column_references, condition_aliases, tests,
};
use crate::logical::{Filter, Join, LogicalNode, LogicalPlan, ResultNode, Scan};
use crate::nodes::{
    above_projection, added_columns, as_column, asked_call, children, column_alias, column_call,
    input, output_schema, reads_projection, source_columns, subtree, validate_plan,
};
use crate::operators::operators;

/// Rules that undo each other must not loop forever.
pub const MAX_PASSES: usize = 4;

/// The names of the built-in logical rules, in the order they run.
pub const LOGICAL_RULES: [&str; 5] = [
    "should_eliminate_distinct",
    "should_push_down_distinct",
    "should_push_down_projection",
    "should_push_down_filters",
    "should_lift_column_operators",
];

/// A logical plan and the rules that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalAlternative {
    pub plan: LogicalPlan,
    pub rules: Vec<String>,
}

/// What the logical rules read besides the plan: each provider's id column.
#[derive(Debug, Clone, Default)]
pub struct RuleContext {
    /// provider -> its id column; None when there is no catalog.
    pub id_columns: Option<BTreeMap<String, String>>,
}

/// Rewrite each candidate with each rule and deduplicate the trees in order.
///
/// # Errors
///
/// A rule's rewrite is not a valid plan, or a rule name is unknown.
pub fn logical_alternatives(
    plan: LogicalPlan,
    rules: &[String],
    context: &RuleContext,
) -> Result<Vec<LogicalAlternative>, CompileError> {
    let mut candidates = vec![LogicalAlternative {
        plan,
        rules: Vec::new(),
    }];
    for _ in 0..MAX_PASSES {
        let before: Vec<LogicalNode> = candidates.iter().map(|c| c.plan.root.clone()).collect();
        for rule in rules {
            let mut rewritten: Vec<LogicalAlternative> = Vec::new();
            for candidate in &candidates {
                let root = &candidate.plan.root;
                for replacement in rewrites(rule, root, context)? {
                    if rewritten.iter().any(|other| other.plan.root == replacement) {
                        continue;
                    }
                    if replacement == *root {
                        rewritten.push(candidate.clone());
                        continue;
                    }
                    let plan = LogicalPlan { root: replacement };
                    validate_plan(&plan)?;
                    let mut names = candidate.rules.clone();
                    names.push(rule.clone());
                    rewritten.push(LogicalAlternative { plan, rules: names });
                }
            }
            candidates = rewritten;
        }
        let after: Vec<LogicalNode> = candidates.iter().map(|c| c.plan.root.clone()).collect();
        if after == before {
            break;
        }
    }
    Ok(candidates)
}

/// The roots one rule rewrites a root into.
fn rewrites(
    rule: &str,
    root: &LogicalNode,
    context: &RuleContext,
) -> Result<Vec<LogicalNode>, CompileError> {
    Ok(match rule {
        "should_eliminate_distinct" => vec![eliminate_distinct(root, context)],
        "should_push_down_distinct" => vec![push_down_distinct(root)],
        "should_push_down_projection" => vec![push_down_projection(root)],
        "should_push_down_filters" => vec![push_down_filters(root).unwrap_or_else(|| root.clone())],
        "should_lift_column_operators" => match lift_column_operators(root) {
            Some(lifted) => vec![root.clone(), lifted],
            None => vec![root.clone()],
        },
        other => {
            return Err(CompileError(format!("unknown logical rule {other:?}")));
        }
    })
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
        LogicalNode::Aggregate(found) => &mut found.input,
        LogicalNode::Window(found) => &mut found.input,
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
        SortExpression::Aggregate(_) => Vec::new(),
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

// ---------------------------------------------------------------- distinct

fn eliminate_distinct(root: &LogicalNode, context: &RuleContext) -> LogicalNode {
    let LogicalNode::Result(result) = root else {
        return root.clone();
    };
    if !result.distinct {
        return root.clone();
    }
    let without = || {
        LogicalNode::Result(ResultNode {
            distinct: false,
            ..result.clone()
        })
    };
    if let LogicalNode::Aggregate(aggregate) = result.input.as_ref() {
        return if aggregate
            .keys
            .iter()
            .all(|key| aggregate.output.contains(key))
        {
            without()
        } else {
            root.clone()
        };
    }
    let (LogicalNode::Project(_), Some(id_columns)) = (result.input.as_ref(), &context.id_columns)
    else {
        return root.clone();
    };
    let returned: BTreeSet<(String, String)> = output_schema(&result.input)
        .into_iter()
        .map(|reference| (reference.alias, reference.column))
        .collect();
    let plan = LogicalPlan { root: root.clone() };
    for scan in operators(&plan).scans {
        let Some(id_column) = id_columns.get(&scan.provider) else {
            return root.clone();
        };
        if !returned.contains(&(scan.alias.clone(), id_column.clone())) {
            return root.clone();
        }
    }
    without()
}

fn push_down_distinct(root: &LogicalNode) -> LogicalNode {
    let LogicalNode::Result(result) = root else {
        return root.clone();
    };
    let LogicalNode::Project(project) = result.input.as_ref() else {
        return root.clone();
    };
    if !result.distinct {
        return root.clone();
    }
    let plan = LogicalPlan { root: root.clone() };
    let found = operators(&plan);
    if !found.joins.is_empty()
        || !found.classifies.is_empty()
        || !found.projections.is_empty()
        || !found.column_filters.is_empty()
        || found.filters.len() != 1
    {
        return root.clone();
    }
    let (alias, predicates) = &found.filters[0];
    if predicates
        .iter()
        .any(|predicate| asked_call(&predicate.expression).kind == "score")
    {
        return root.clone();
    }
    let mut key: Vec<String> = Vec::new();
    for column in &project.columns {
        let Projected::Expression(ValueExpression::ColumnRef(reference)) = column else {
            return root.clone();
        };
        if reference.alias != *alias {
            return root.clone();
        }
        if !key.contains(&reference.column) {
            key.push(reference.column.clone());
        }
    }
    if key.is_empty() {
        return root.clone();
    }
    fn mark(node: &LogicalNode, key: &[String]) -> LogicalNode {
        if let LogicalNode::SemanticFilter(filter) = node {
            let mut marked = filter.clone();
            marked.stop_key = key.to_vec();
            return LogicalNode::SemanticFilter(marked);
        }
        let new: Vec<LogicalNode> = children(node)
            .into_iter()
            .map(|child| mark(child, key))
            .collect();
        if new.iter().collect::<Vec<_>>() == children(node) {
            node.clone()
        } else {
            with_children(node, new)
        }
    }
    mark(root, &key)
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
            if let Projected::Expression(expression) = column
                && aliased_call(expression, &[]).is_some()
            {
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

// ---------------------------------------------------------------- filters

fn lets_through(node: &LogicalNode, alias: &str) -> bool {
    match node {
        LogicalNode::Join(_) => true,
        LogicalNode::SemanticJoin(join) => join.semantics == "full",
        LogicalNode::Filter(filter) => filter_alias(filter) != alias,
        LogicalNode::SemanticFilter(filter) => {
            filter.predicates.iter().all(|predicate: &FilterPredicate| {
                call_aliases(asked_call(&predicate.expression))[0] != alias
            })
        }
        column if as_column(column).is_some() => {
            let found = as_column(column).expect("a column operator");
            call_aliases(column_call(found)).len() == 2 || column_alias(found) != alias
        }
        _ => false,
    }
}

fn place_filter(filter: &LogicalNode, node: &LogicalNode, alias: &str) -> Option<LogicalNode> {
    if let LogicalNode::Join(join) = node {
        let left: BTreeSet<String> = output_schema(&join.left)
            .into_iter()
            .map(|field| field.alias)
            .collect();
        let on_left = left.contains(alias);
        let child = if on_left { &join.left } else { &join.right };
        let placed = place_filter(filter, child, alias)
            .unwrap_or_else(|| with_children(filter, vec![(**child).clone()]));
        let mut moved: Join = join.clone();
        if on_left {
            moved.left = Box::new(placed);
        } else {
            moved.right = Box::new(placed);
        }
        return Some(LogicalNode::Join(moved));
    }
    if lets_through(node, alias) {
        let child = input(node);
        let placed = place_filter(filter, child, alias)?;
        return Some(with_children(node, vec![placed]));
    }
    None
}

/// Move every Filter on one table below the joins above its table; None
/// when no filter moved.
pub fn push_down_filters(root: &LogicalNode) -> Option<LogicalNode> {
    fn visit(node: &LogicalNode, moved: &mut bool) -> LogicalNode {
        let old = children(node);
        let new: Vec<LogicalNode> = old.iter().map(|child| visit(child, moved)).collect();
        let node = if new.iter().zip(&old).all(|(a, b)| a == *b) {
            node.clone()
        } else {
            with_children(node, new)
        };
        let LogicalNode::Filter(filter) = &node else {
            return node;
        };
        if reads_projection(&filter.input) {
            return node;
        }
        let alias = filter_alias(filter);
        match place_filter(&node, &filter.input, &alias) {
            Some(placed) => {
                *moved = true;
                placed
            }
            None => node,
        }
    }
    let mut moved = false;
    let rewritten = visit(root, &mut moved);
    moved.then_some(rewritten)
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
    if matches!(
        root,
        LogicalNode::Result(_) | LogicalNode::Aggregate(_) | LogicalNode::Window(_)
    ) || above_projection(root)
        || is_project_over_rows
    {
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
