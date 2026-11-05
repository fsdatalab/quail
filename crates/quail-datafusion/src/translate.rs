//! A model region's relation, read node by node from DataFusion's plan.
//!
//! After [`crate::LowerAiCalls`] and DataFusion's rules, a region's plan
//! places each operator: a table's scan and plain tests, its
//! `SemanticFilter`, its `SemanticColumn` nodes and the tests of their
//! answers, the joins and their `SemanticJoin` nodes. [`translate_relation`]
//! writes that plan as Quail's logical nodes, one for each of DataFusion's,
//! taking each node's content (bound prompts, conditions, names) from the
//! binder's [`RelationContent`].
//!
//! The SELECT list, grouping, windows, ORDER BY, and LIMIT above the
//! relation are not read here: the builder's `project` adds them.

use std::collections::BTreeSet;

use datafusion::common::tree_node::TreeNode;
use datafusion::logical_expr::{Expr, LogicalPlan};
use quail_physical::FilterPredicate;
use quail_planner::builder::{RelationContent, TableContent, column_node};
use quail_planner::expressions::{call_aliases, condition_columns};
use quail_planner::logical::{Filter, Join, LogicalNode, Scan, SemanticFilter, SemanticJoin};
use quail_planner::nodes::added_columns;
use quail_planner::nodes::asked_call;

use crate::semantic::{SemanticColumn, SemanticFilter as DataFusionSemanticFilter};

fn as_node<T: 'static>(plan: &LogicalPlan) -> Option<&T> {
    match plan {
        LogicalPlan::Extension(extension) => extension.node.as_any().downcast_ref::<T>(),
        _ => None,
    }
}

/// The tables whose columns an expression reads.
fn tables(expr: &Expr) -> BTreeSet<String> {
    expr.column_refs()
        .into_iter()
        .filter_map(|column| {
            column
                .relation
                .as_ref()
                .map(|table| table.table().to_string())
        })
        .collect()
}

/// Whether an expression calls a model: a call the lowering did not take.
fn asks_model(expr: &Expr) -> bool {
    let mut asks = false;
    let _ = expr.apply(|found| {
        asks |= crate::is_ai_call(found);
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    });
    asks
}

/// The operator a column call's function runs as.
fn column_kind(call: &Expr) -> Option<&'static str> {
    match call {
        Expr::ScalarFunction(found) => match found.name() {
            "ai_classify" => Some("label"),
            "ai_extract" => Some("extract"),
            "ai_map" => Some("map"),
            "ai_sae_features" => Some("features"),
            _ => None,
        },
        _ => None,
    }
}

/// The relation below a SELECT: the operators under its LIMIT, ORDER BY,
/// DISTINCT, and SELECT list.
fn relation_root(plan: &LogicalPlan) -> &LogicalPlan {
    match plan {
        LogicalPlan::Limit(found) => relation_root(&found.input),
        LogicalPlan::Sort(found) => relation_root(&found.input),
        LogicalPlan::Distinct(found) => relation_root(found.input()),
        LogicalPlan::Projection(found) => relation_root(&found.input),
        other => other,
    }
}

/// The input of a node on one table's stack, if it is one.
fn stack_input(node: &mut LogicalNode) -> Option<&mut Box<LogicalNode>> {
    match node {
        LogicalNode::SemanticClassify(column)
        | LogicalNode::SemanticExtract(column)
        | LogicalNode::SemanticMap(column)
        | LogicalNode::SemanticSaeFeatures(column) => Some(&mut column.input),
        LogicalNode::Filter(filter) => Some(&mut filter.input),
        _ => None,
    }
}

/// The answer columns a stack's column operators add, at and below `node`.
fn stack_columns(node: &LogicalNode) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let mut node = node;
    loop {
        match node {
            LogicalNode::SemanticClassify(column)
            | LogicalNode::SemanticExtract(column)
            | LogicalNode::SemanticMap(column)
            | LogicalNode::SemanticSaeFeatures(column) => {
                found.extend(added_columns(node));
                node = &column.input;
            }
            LogicalNode::Filter(filter) => node = &filter.input,
            _ => return found,
        }
    }
}

/// Whether `node`, or an operator below it on the stack, is an
/// extraction, SAE feature, or map column.
fn has_answer_column(node: &LogicalNode) -> bool {
    match node {
        LogicalNode::SemanticExtract(_)
        | LogicalNode::SemanticMap(_)
        | LogicalNode::SemanticSaeFeatures(_) => true,
        LogicalNode::SemanticClassify(column) => has_answer_column(&column.input),
        LogicalNode::Filter(filter) => has_answer_column(&filter.input),
        _ => false,
    }
}

/// `node` with a test of answers right above the lowest operator after
/// which every answer column the test reads is added; a test that names
/// no answer column, such as one of an extraction's struct, goes right
/// above the first extraction, SAE feature, or map column.
fn insert_test(
    mut node: LogicalNode,
    condition: quail_physical::Condition,
    selectivity: Option<f64>,
    answers: &BTreeSet<String>,
) -> LogicalNode {
    let reads: BTreeSet<String> = condition_columns(&condition)
        .into_iter()
        .filter(|column| answers.contains(column))
        .collect();
    // a test stays above the tests placed before it
    let descend = |node: &mut LogicalNode| {
        if matches!(node, LogicalNode::Filter(_)) {
            return false;
        }
        stack_input(node).is_some_and(|input| {
            if reads.is_empty() {
                has_answer_column(input)
            } else {
                reads.is_subset(&stack_columns(input))
            }
        })
    };
    if !descend(&mut node) {
        return LogicalNode::Filter(Filter {
            input: Box::new(node),
            condition,
            selectivity,
        });
    }
    let input = stack_input(&mut node).expect("a stack operator");
    let below = std::mem::replace(
        input.as_mut(),
        LogicalNode::Scan(Scan {
            provider: String::new(),
            alias: String::new(),
            column: quail_planner::logical::ScanColumn::Unnamed(String::new()),
            columns: Vec::new(),
            predicates: Vec::new(),
            schema: None,
        }),
    );
    **input = insert_test(below, condition, selectivity, answers);
    node
}

/// The alias of the table at the bottom of a table's stack.
fn stack_alias(node: &LogicalNode) -> Option<String> {
    match node {
        LogicalNode::Scan(scan) => Some(scan.alias.clone()),
        LogicalNode::SemanticFilter(filter) => stack_alias(&filter.input),
        LogicalNode::SemanticClassify(column)
        | LogicalNode::SemanticExtract(column)
        | LogicalNode::SemanticMap(column)
        | LogicalNode::SemanticSaeFeatures(column) => stack_alias(&column.input),
        LogicalNode::Filter(filter) => stack_alias(&filter.input),
        _ => None,
    }
}

/// `node` with each table's tests of answers on its stack, in the
/// content's order.
fn place_tests(node: LogicalNode, content: &RelationContent) -> LogicalNode {
    if let Some(alias) = stack_alias(&node) {
        let Some(table) = content.tables.iter().find(|table| table.alias == alias) else {
            return node;
        };
        let answers = stack_columns(&node);
        let mut node = node;
        for (condition, selectivity) in &table.operators.computed_filters {
            node = insert_test(node, condition.clone(), *selectivity, &answers);
        }
        return node;
    }
    match node {
        LogicalNode::Join(mut join) => {
            join.left = Box::new(place_tests(*join.left, content));
            join.right = Box::new(place_tests(*join.right, content));
            LogicalNode::Join(join)
        }
        LogicalNode::SemanticJoin(mut join) => {
            join.input = Box::new(place_tests(*join.input, content));
            LogicalNode::SemanticJoin(join)
        }
        LogicalNode::SemanticClassify(mut column) => {
            column.input = Box::new(place_tests(*column.input, content));
            LogicalNode::SemanticClassify(column)
        }
        other => other,
    }
}

/// What a translation has taken from the content so far.
struct Translation<'a> {
    content: &'a RelationContent,
    /// (alias, Quail's name, DataFusion's name) of each column operator placed.
    columns: Vec<(String, String, String)>,
    /// Whether each join is placed.
    joins: Vec<bool>,
    /// (the reader's alias, the inner table's alias) of each derived
    /// table the binder merged into its reader.
    renames: Vec<(String, String)>,
}

impl<'a> Translation<'a> {
    /// The alias the binder gave a table: a merged derived table's.
    fn renamed(&self, alias: &str) -> String {
        self.renames
            .iter()
            .rev()
            .find(|(inner, _)| inner == alias)
            .map(|(_, outer)| outer.clone())
            .unwrap_or_else(|| alias.to_string())
    }

    /// The tables an expression reads, by the aliases the binder gave them.
    fn tables(&self, expr: &Expr) -> BTreeSet<String> {
        tables(expr)
            .iter()
            .map(|alias| self.renamed(alias))
            .collect()
    }

    fn table(&self, alias: &str) -> Option<&'a TableContent> {
        let alias = self.renamed(alias);
        self.content
            .tables
            .iter()
            .find(|table| table.alias == alias)
    }

    fn scan(&self, alias: &str) -> Option<LogicalNode> {
        let table = self.table(alias)?;
        Some(LogicalNode::Scan(Scan {
            provider: table.provider.clone(),
            alias: table.alias.clone(),
            column: table.column.clone(),
            columns: Vec::new(),
            predicates: table.operators.regular_predicates.clone(),
            schema: table.schema.clone(),
        }))
    }

    fn relation(&mut self, plan: &LogicalPlan) -> Option<LogicalNode> {
        if let Some(filter) = as_node::<DataFusionSemanticFilter>(plan) {
            return self.semantic_filter(filter);
        }
        if let Some(column) = as_node::<SemanticColumn>(plan) {
            return self.column(column);
        }
        if let Some(join) = as_node::<crate::semantic::SemanticJoin>(plan) {
            return self.semantic_join(join);
        }
        match plan {
            LogicalPlan::Projection(projection) => self.relation(&projection.input),
            LogicalPlan::SubqueryAlias(alias) => {
                let mut below = alias.input.as_ref();
                loop {
                    match below {
                        LogicalPlan::Filter(filter) if asks_model(&filter.predicate) => {
                            return None;
                        }
                        LogicalPlan::Filter(filter) => below = &filter.input,
                        LogicalPlan::Projection(projection) => below = &projection.input,
                        LogicalPlan::TableScan(_) => return self.scan(alias.alias.table()),
                        _ => return self.derived_table(alias),
                    }
                }
            }
            // a plain test is the scan's predicate or a row test; a test of
            // answers is placed with its table's other tests; a call left
            // in a filter is one the lowering did not take
            LogicalPlan::Filter(filter) if asks_model(&filter.predicate) => None,
            LogicalPlan::Filter(filter) => self.relation(&filter.input),
            LogicalPlan::Join(join) => {
                if join.join_type != datafusion::logical_expr::JoinType::Inner {
                    return None;
                }
                Some(LogicalNode::Join(Join {
                    left: Box::new(self.relation(&join.left)?),
                    right: Box::new(self.relation(&join.right)?),
                    on: Vec::new(),
                }))
            }
            _ => None,
        }
    }

    /// A derived table: a stage of its own, read as a scan of the stage,
    /// or one the binder merged into its reader, read through.
    fn derived_table(
        &mut self,
        alias: &datafusion::logical_expr::SubqueryAlias,
    ) -> Option<LogicalNode> {
        let outer = alias.alias.table().to_string();
        let mut inner = None;
        let _ = alias.input.apply(|node| {
            if let LogicalPlan::SubqueryAlias(found) = node
                && let LogicalPlan::TableScan(_) = found.input.as_ref()
            {
                inner = Some(found.alias.table().to_string());
                return Ok(datafusion::common::tree_node::TreeNodeRecursion::Stop);
            }
            Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
        });
        // a stage of its own: the content names it as a table
        if let Some(table) = self.table(&outer)
            && inner
                .as_ref()
                .is_none_or(|inner| self.table(inner).is_none())
        {
            return self.scan(&table.alias.clone());
        }
        // the binder merged the derived table into its reader, keeping
        // the inner table's alias; the reader names its columns by `outer`
        if let Some(inner) = inner {
            self.renames.push((outer, inner));
        }
        self.relation(&alias.input)
    }

    fn semantic_filter(&mut self, filter: &DataFusionSemanticFilter) -> Option<LogicalNode> {
        let read: BTreeSet<String> = filter
            .predicates
            .iter()
            .flat_map(|predicate| self.tables(predicate))
            .collect();
        let (alias,) = <[String; 1]>::try_from(read.into_iter().collect::<Vec<_>>())
            .ok()
            .map(|[alias]| (alias,))?;
        let predicates: Vec<FilterPredicate> = self.table(&alias)?.operators.predicates.clone();
        // the table's filters in one node, as Quail's operators read them
        if predicates.len() != filter.predicates.len() {
            return None;
        }
        Some(LogicalNode::SemanticFilter(SemanticFilter {
            input: Box::new(self.relation(&filter.input)?),
            predicates,
            stop_key: filter.stop_key.clone(),
        }))
    }

    fn column(&mut self, column: &SemanticColumn) -> Option<LogicalNode> {
        let kind = column_kind(&column.call)?;
        let read: Vec<String> = self.tables(&column.call).into_iter().collect();
        if read.len() == 2 {
            // a run of classifications of the same joined rows: the builder
            // puts each one right above the join, the first written on top
            let mut run = vec![column];
            let mut below = column.input.as_ref();
            while let Some(next) = as_node::<SemanticColumn>(below)
                && self.tables(&next.call) == self.tables(&column.call)
            {
                run.push(next);
                below = next.input.as_ref();
            }
            let mut node = self.relation(below)?;
            for found in run {
                node = self.joined_classification(found, node)?;
            }
            return Some(node);
        }
        let mut node = self.relation(&column.input)?;
        let [alias] = <[String; 1]>::try_from(read).ok()?;
        let operators = &self.table(&alias)?.operators;
        let entries = match kind {
            "label" => &operators.labels,
            "extract" => &operators.extracts,
            "map" => &operators.maps,
            _ => &operators.sae_features,
        };
        let placed = |name: &String| {
            self.columns
                .iter()
                .any(|(owner, found, _)| *owner == alias && found == name)
        };
        // an extraction of several questions is one Quail column per field
        let field_prefix = format!("{}.", column.name);
        let mut chosen: Vec<(quail_physical::ModelCall, String)> = entries
            .iter()
            .filter(|(_, name)| *name == column.name || name.starts_with(&field_prefix))
            .cloned()
            .collect();
        if chosen.is_empty() {
            // a column only a test reads: the next one not yet placed
            chosen.push(entries.iter().find(|(_, name)| !placed(name))?.clone());
        }
        for (call, name) in chosen {
            self.columns
                .push((alias.clone(), name.clone(), column.name.clone()));
            node = column_node(kind, node, &call, &name);
        }
        Some(node)
    }

    fn joined_classification(
        &mut self,
        column: &SemanticColumn,
        input: LogicalNode,
    ) -> Option<LogicalNode> {
        let placed = |name: &String| self.columns.iter().any(|(_, found, _)| found == name);
        let (call, name) = self
            .content
            .joined_classifications
            .iter()
            .find(|(_, name)| *name == column.name)
            .or_else(|| {
                self.content
                    .joined_classifications
                    .iter()
                    .find(|(_, name)| !placed(name))
            })?
            .clone();
        let read: Vec<String> = self.tables(&column.call).into_iter().collect();
        self.columns
            .push((read.join(","), name.clone(), column.name.clone()));
        Some(column_node("label", input, &call, &name))
    }

    fn semantic_join(&mut self, join: &crate::semantic::SemanticJoin) -> Option<LogicalNode> {
        // the lowest join first: the content lists them in written order
        let mut input = self.relation(&join.input)?;
        let read = self.tables(&join.predicate);
        let index = self
            .content
            .joins
            .iter()
            .enumerate()
            .position(|(index, spec)| {
                !self.joins.get(index).copied().unwrap_or(false)
                    && call_aliases(asked_call(&spec.predicate))
                        .into_iter()
                        .collect::<BTreeSet<String>>()
                        == read
            })?;
        self.joins[index] = true;
        let spec = self.content.joins[index].clone();
        if let LogicalNode::Join(found) = &mut input {
            found.on = spec.on.clone();
        }
        Some(LogicalNode::SemanticJoin(SemanticJoin {
            input: Box::new(input),
            predicate: spec.predicate,
            semantics: spec.semantics,
            selectivity: spec.selectivity,
            anchor: spec.anchor,
        }))
    }
}

/// Write a region's relation as Quail's logical nodes, as the module
/// explains; None when the region has an operator Quail's nodes do not
/// read here yet, such as a derived table or an EXISTS gate.
pub fn translate_relation(plan: &LogicalPlan, content: &RelationContent) -> Option<LogicalNode> {
    let mut translation = Translation {
        content,
        columns: Vec::new(),
        joins: vec![false; content.joins.len()],
        renames: Vec::new(),
    };
    let root = translation.relation(relation_root(plan))?;
    let root = place_tests(root, content);
    // every join placed, and every cross join
    if translation.joins.iter().any(|placed| !placed) {
        return None;
    }
    Some(root)
}
