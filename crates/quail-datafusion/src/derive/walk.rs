//! Read the plan in input order and track each operator's source tables.
//! Recover written expressions from answer columns and derived tables.

use super::binding::{Binder, unsupported};
use crate::SPAN_SUFFIX;
use crate::ai::{AiNode, Fetch, Semantics, WrittenTest};
use crate::extension::as_node;
use crate::rules::lower::quail_function;
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::common::Column;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::logical_expr::{Expr, Join, JoinType, LogicalPlan};
use quail_planner::error::{CompileError, compile_error};
use std::collections::BTreeSet;

/// `plan` with a SELECT list at its root: DataFusion removes one that
/// returns its input's columns as they are.
pub(super) fn with_select_list(plan: &LogicalPlan) -> LogicalPlan {
    if matches!(plan, LogicalPlan::Projection(_)) {
        return plan.clone();
    }
    let columns: Vec<Expr> = plan
        .schema()
        .iter()
        .map(|(qualifier, field)| {
            Expr::Column(Column::new(qualifier.cloned(), field.name().clone()))
        })
        .collect();
    datafusion::logical_expr::Projection::try_new(columns, std::sync::Arc::new(plan.clone()))
        .map(LogicalPlan::Projection)
        .unwrap_or_else(|_| plan.clone())
}

// ---------------------------------------------------------------- the SELECT

/// The SELECT list of a region and the relation below it.
pub(super) struct Top<'a> {
    pub(super) projection: Vec<Expr>,
    pub(super) relation: &'a LogicalPlan,
}

/// Read the SELECT list at a region's root.
///
/// # Errors
///
/// The region has no SELECT list at its root, such as a LIMIT, ORDER
/// BY, or DISTINCT, which DataFusion runs above the region.
pub(super) fn read_top(plan: &LogicalPlan) -> Result<Top<'_>, CompileError> {
    let LogicalPlan::Projection(projection) = plan else {
        return unsupported("a region without a SELECT list");
    };
    Ok(Top {
        projection: projection.expr.clone(),
        relation: projection.input.as_ref(),
    })
}

/// The global fetch of the region's last `AiNode`, which a LIMIT became
/// (`PushDownLimitIntoAiNodes`); None, in the outer Option, when another
/// node has one, as the region's operators cannot place a LIMIT there.
pub(super) fn read_fetch(top: &Top<'_>) -> Option<Option<i64>> {
    let mut last = top.relation;
    while let LogicalPlan::Projection(_) | LogicalPlan::SubqueryAlias(_) = last {
        last = last.inputs()[0];
    }
    let last = as_node::<AiNode>(last);
    let mut fetch = None;
    let mut elsewhere = false;
    let _ = top.relation.apply(|plan| {
        if let Some(node) = as_node::<AiNode>(plan)
            && let Some(Fetch::Global(rows)) = &node.fetch
        {
            match last {
                Some(found) if std::ptr::eq(found, node) => fetch = Some(*rows as i64),
                _ => elsewhere = true,
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    (!elsewhere).then_some(fetch)
}

/// Refuse a SELECT whose input is a GROUP BY or a window.
///
/// DataFusion runs a GROUP BY or window above the region, or as an input
/// it computes for the region.
///
/// # Errors
///
/// The input groups or windows rows.
pub(super) fn refuse_grouped_input(relation: &LogicalPlan) -> Result<(), CompileError> {
    let groups =
        |plan: &LogicalPlan| matches!(plan, LogicalPlan::Aggregate(_) | LogicalPlan::Window(_));
    if groups(relation) || matches!(relation, LogicalPlan::Filter(filter) if groups(&filter.input))
    {
        return compile_error(
            "a model call that reads the result of a GROUP BY or a window is not \
             supported; read it from a CTE or FROM subquery instead",
        );
    }
    Ok(())
}
/// Whether an expression calls an AI function anywhere.
pub(super) fn contains_ai_call(expr: &Expr) -> bool {
    expr.exists(|node| Ok(crate::is_ai_call(node)))
        .unwrap_or(false)
}

/// Whether a plan reads a CTE or a FROM subquery.
pub(super) fn has_subqueries(plan: &LogicalPlan) -> bool {
    plan.exists(|node| {
        Ok(matches!(node, LogicalPlan::SubqueryAlias(alias)
            if !matches!(alias.input.as_ref(), LogicalPlan::TableScan(_))))
    })
    .unwrap_or(false)
}
/// A table the region scans.
#[derive(Debug, Clone)]
pub(super) struct Table {
    pub(super) alias: String,
    pub(super) provider: String,
    pub(super) schema: SchemaRef,
    /// Whether an EXISTS gate reads it, rather than FROM or JOIN.
    pub(super) gate: bool,
}

/// What one operator of the plan is.
#[derive(Debug)]
pub(super) enum Kind<'a> {
    Scan(usize),
    Join(&'a Join),
    Filter(Expr),
    Ai(&'a AiNode),
    /// A projection, or a SubqueryAlias of a derived table.
    PassThrough,
}

/// One operator of the plan, as the walk meets it.
#[derive(Debug)]
pub(super) struct Visit<'a> {
    pub(super) kind: Kind<'a>,
    /// The ids of its inputs.
    pub(super) inputs: Vec<usize>,
    /// The id of the operator it is the input of.
    pub(super) parent: Option<usize>,
    /// The ids of the operators above it, outermost first.
    pub(super) ancestors: Vec<usize>,
    /// The aliases of the tables it reads, in scan order.
    pub(super) tables: Vec<String>,
    /// Whether an AI join or gate is at or below it.
    pub(super) ai_join: bool,
}

/// The region's operators, children first, and its tables.
#[derive(Debug, Default)]
pub(super) struct Walk<'a> {
    /// Each operator by its id.
    visits: Vec<Option<Visit<'a>>>,
    /// The ids, children first.
    order: Vec<usize>,
    pub(super) tables: Vec<Table>,
    /// The ids of the operators being visited, outermost first.
    stack: Vec<usize>,
    /// (alias, its columns' expressions) of each derived table the
    /// binder merges into the query that reads it.
    merged: Vec<(String, Vec<(String, Expr)>)>,
    /// The binder's refusal of the FROM and JOIN shape, if any.
    pub(super) refusal: Option<CompileError>,
    /// Whether the plan has an operator Quail's operators do not read.
    pub(super) unplaced: bool,
    /// Whether that operator is one no operator of Quail's is, such as a
    /// LIMIT below a filter.
    pub(super) unknown: bool,
}

impl<'a> Walk<'a> {
    pub(super) fn visit(&mut self, plan: &'a LogicalPlan) -> usize {
        let id = self.visits.len();
        self.visits.push(None);
        self.stack.push(id);
        let (kind, inputs) = match plan {
            LogicalPlan::Projection(projection) => {
                (Kind::PassThrough, vec![self.visit(&projection.input)])
            }
            LogicalPlan::Filter(filter) => {
                let predicate = filter.predicate.clone();
                (Kind::Filter(predicate), vec![self.visit(&filter.input)])
            }
            LogicalPlan::Join(join) => {
                if join.join_type != JoinType::Inner {
                    self.refuse(unsupported(&format!("a {} JOIN", join.join_type)));
                }
                let left = self.visit(&join.left);
                let right = self.visit(&join.right);
                (Kind::Join(join), vec![left, right])
            }
            LogicalPlan::Extension(extension) => {
                match extension.node.as_any().downcast_ref::<AiNode>() {
                    Some(node) => (Kind::Ai(node), vec![self.visit(&node.input)]),
                    None => {
                        self.unplaced = true;
                        self.unknown = true;
                        (Kind::PassThrough, Vec::new())
                    }
                }
            }
            LogicalPlan::SubqueryAlias(alias) => match self.table(plan) {
                Some(table) => (Kind::Scan(table), Vec::new()),
                None => {
                    let definitions = derived_columns(&alias.input);
                    let input = self.visit(&alias.input);
                    self.merged
                        .push((alias.alias.table().to_string(), definitions));
                    (Kind::PassThrough, vec![input])
                }
            },
            LogicalPlan::TableScan(_) => match self.table(plan) {
                Some(table) => (Kind::Scan(table), Vec::new()),
                None => unreachable!("a table scan reads a table"),
            },
            _ => {
                self.unplaced = true;
                self.unknown = true;
                (Kind::PassThrough, Vec::new())
            }
        };
        self.stack.pop();
        for input in &inputs {
            if let Some(visit) = self.visits[*input].as_mut() {
                visit.parent = Some(id);
            }
        }
        let mut tables: Vec<String> = Vec::new();
        let mut ai_join = false;
        for input in &inputs {
            let below = self.visits[*input].as_ref().expect("a visited input");
            tables.extend(below.tables.iter().cloned());
            ai_join |= below.ai_join;
        }
        if let Kind::Scan(table) = &kind {
            tables.push(self.tables[*table].alias.clone());
        }
        if let Kind::Ai(node) = &kind {
            ai_join |= node.semantics != Semantics::Rows
                || (node.is_predicate() && self.call_tables(&node.call).len() > 1);
            if node.semantics != Semantics::Rows
                && let Kind::Join(_) = &self.get(inputs[0]).kind
            {
                // the gate's table is the right input of its join
                let gated = self.get(self.get(inputs[0]).inputs[1]).tables.clone();
                for table in &mut self.tables {
                    if gated.contains(&table.alias) {
                        table.gate = true;
                    }
                }
            }
        }
        if let Kind::Join(_) = &kind
            && self.get(inputs[1]).tables.len() != 1
        {
            self.refuse(unsupported(
                "a JOIN of anything but one registered provider",
            ));
        }
        self.visits[id] = Some(Visit {
            kind,
            inputs,
            parent: None,
            ancestors: self.stack.clone(),
            tables,
            ai_join,
        });
        self.order.push(id);
        id
    }

    fn refuse(&mut self, refusal: Result<(), CompileError>) {
        if self.refusal.is_none() {
            self.refusal = refusal.err();
        }
    }

    /// Record the table a scan reads, with the filters DataFusion pushed
    /// between its alias and its scan; None for a derived table.
    fn table(&mut self, plan: &'a LogicalPlan) -> Option<usize> {
        let (alias, mut below) = match plan {
            LogicalPlan::SubqueryAlias(alias) => {
                (alias.alias.table().to_string(), alias.input.as_ref())
            }
            other => (String::new(), other),
        };
        let mut filters: Vec<&'a Expr> = Vec::new();
        let scan = loop {
            match below {
                LogicalPlan::Filter(filter) => {
                    filters.push(&filter.predicate);
                    below = &filter.input;
                }
                LogicalPlan::Projection(projection) => below = &projection.input,
                LogicalPlan::TableScan(scan) => break scan,
                _ => return None,
            }
        };
        let provider = scan.table_name.table().to_string();
        let alias = if alias.is_empty() {
            provider.clone()
        } else {
            alias
        };
        let index = self.tables.len();
        self.tables.push(Table {
            alias: alias.clone(),
            provider: provider.clone(),
            schema: scan.source.schema(),
            gate: false,
        });
        // a filter below the alias names the table by its provider
        let rename = |expr: &Expr| -> Expr {
            expr.clone()
                .transform_up(|node| match node {
                    Expr::Column(column)
                        if column
                            .relation
                            .as_ref()
                            .is_some_and(|found| found.table() == provider) =>
                    {
                        Ok(Transformed::yes(Expr::Column(Column::new(
                            Some(alias.clone()),
                            column.name,
                        ))))
                    }
                    other => Ok(Transformed::no(other)),
                })
                .map(|found| found.data)
                .unwrap_or_else(|_| expr.clone())
        };
        for predicate in filters.iter().rev().copied().chain(&scan.filters) {
            if contains_ai_call(predicate) {
                self.unplaced = true;
            }
            let id = self.visits.len();
            self.visits.push(Some(Visit {
                kind: Kind::Filter(rename(predicate)),
                inputs: Vec::new(),
                parent: None,
                ancestors: self.stack.clone(),
                tables: Vec::new(),
                ai_join: false,
            }));
            self.order.push(id);
        }
        Some(index)
    }

    /// Whether an EXISTS gate reads the table `alias`.
    pub(super) fn is_gate(&self, alias: &str) -> bool {
        self.tables
            .iter()
            .any(|table| table.alias == alias && table.gate)
    }

    pub(super) fn get(&self, id: usize) -> &Visit<'a> {
        self.visits[id].as_ref().expect("a visited operator")
    }

    /// The ids of the operators, children first.
    pub(super) fn ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.order.iter().copied()
    }

    /// The tables an expression reads, through the derived tables merged
    /// into the query.
    pub(super) fn call_tables(&self, expr: &Expr) -> BTreeSet<String> {
        self.written(expr, &[])
            .column_refs()
            .into_iter()
            .filter_map(|column| {
                column
                    .relation
                    .as_ref()
                    .map(|table| table.table().to_string())
            })
            .collect()
    }

    /// `expr` as the query wrote it: each merged derived table's column
    /// as its definition, and each answer column in `answers` as the call
    /// that computes it.
    pub(super) fn written(&self, expr: &Expr, answers: &[Answer]) -> Expr {
        expr.clone()
            .transform_up(|node| {
                let Expr::Column(column) = &node else {
                    return Ok(Transformed::no(node));
                };
                if let Some(relation) = &column.relation
                    && let Some((_, columns)) = self
                        .merged
                        .iter()
                        .rev()
                        .find(|(alias, _)| alias == relation.table())
                    && let Some((_, definition)) =
                        columns.iter().find(|(name, _)| *name == column.name)
                {
                    return Ok(Transformed::yes(self.written(definition, answers)));
                }
                if column.relation.is_none()
                    && let Some(answer) = answers.iter().find(|found| found.name == column.name)
                {
                    return Ok(Transformed::yes(answer.call.clone()));
                }
                Ok(Transformed::no(node))
            })
            .map(|found| found.data)
            .unwrap_or_else(|_| expr.clone())
    }

    /// `expr` as the query wrote it, as [`Walk::written`], with each
    /// answer of an AI filter that keeps every row read as a column of
    /// the filter's table, which `binder` then knows.
    pub(super) fn filter_answers(&self, expr: &Expr, binder: &mut Binder) -> Expr {
        let mut answers: Vec<(String, String)> = Vec::new();
        for id in self.ids() {
            let Kind::Ai(node) = &self.get(id).kind else {
                continue;
            };
            let tables = self.call_tables(&node.call);
            if node.function() == "ai_filter"
                && node.filter.is_none()
                && node.semantics == Semantics::Rows
                && let [alias] = tables.into_iter().collect::<Vec<_>>().as_slice()
            {
                answers.push((node.answer.clone(), alias.clone()));
            }
        }
        self.written(expr, &[])
            .transform_up(|found| match &found {
                Expr::Column(column) if column.relation.is_none() => {
                    match answers.iter().find(|(answer, _)| *answer == column.name) {
                        Some((answer, alias)) => {
                            binder.note_answer_type(alias, answer, DataType::Boolean);
                            Ok(Transformed::yes(Expr::Column(Column::new(
                                Some(alias.clone()),
                                answer.clone(),
                            ))))
                        }
                        None => Ok(Transformed::no(found)),
                    }
                }
                _ => Ok(Transformed::no(found)),
            })
            .map(|found| found.data)
            .unwrap_or_else(|_| self.written(expr, &[]))
    }

    /// The terms of the plan's filters, as the query wrote them, then its
    /// tests of answer columns, in written order.
    pub(super) fn where_terms(&self, answers: &[Answer]) -> Vec<Expr> {
        let mut tests: Vec<&WrittenTest> = Vec::new();
        for id in self.ids() {
            let Kind::Ai(node) = &self.get(id).kind else {
                continue;
            };
            for test in &node.tests {
                if !tests
                    .iter()
                    .any(|found| (found.clause, found.position) == (test.clause, test.position))
                {
                    tests.push(test);
                }
            }
        }
        let mut terms = Vec::new();
        for id in self.ids() {
            let found: Vec<&Expr> = match &self.get(id).kind {
                Kind::Filter(predicate) => split_conjunction(predicate),
                Kind::Ai(node) if !node.is_predicate() && node.semantics == Semantics::Rows => {
                    node.filter.iter().flat_map(split_conjunction).collect()
                }
                _ => continue,
            };
            // each test of answers is in written order below
            terms.extend(
                found
                    .into_iter()
                    .filter(|term| {
                        // DataFusion may write one test as several terms
                        !tests
                            .iter()
                            .any(|test| split_conjunction(&test.test).contains(term))
                    })
                    .map(|term| self.written(term, answers)),
            );
        }
        tests.sort_by_key(|test| (test.clause, test.position));
        terms.extend(
            tests
                .into_iter()
                .map(|test| self.written(&test.written, &[])),
        );
        terms
    }

    /// The answer columns the plan's AI columns add, with their calls.
    pub(super) fn answers(&self) -> Vec<Answer> {
        let mut found = Vec::new();
        for id in self.ids() {
            let Kind::Ai(node) = &self.get(id).kind else {
                continue;
            };
            if node.is_predicate() {
                continue;
            }
            found.push(Answer {
                name: node.answer.clone(),
                call: node.call.clone(),
            });
            if node.function() == "ai_extract"
                && let Expr::ScalarFunction(call) = &node.call
            {
                let spans = datafusion::logical_expr::expr::ScalarFunction::new_udf(
                    quail_function("ai_extract_span"),
                    call.args.clone(),
                );
                found.push(Answer {
                    name: format!("{}{SPAN_SUFFIX}", node.answer),
                    call: Expr::ScalarFunction(spans),
                });
            }
        }
        found
    }

    /// A SELECT list item as the query wrote it: an answer column as its
    /// call under the column's name.
    pub(super) fn selected(&self, expr: &Expr, answers: &[Answer]) -> Expr {
        let (inner, name) = match expr {
            Expr::Alias(alias) => (alias.expr.as_ref(), Some(alias.name.clone())),
            other => (other, None),
        };
        let merged = self.written(inner, &[]);
        if let Expr::Column(column) = &merged
            && column.relation.is_none()
            && let Some(answer) = answers.iter().find(|found| found.name == column.name)
        {
            let written = match inner {
                Expr::Column(column) => column.name.clone(),
                other => other.schema_name().to_string(),
            };
            return answer.call.clone().alias(name.unwrap_or(written));
        }
        // DataFusion merges a SELECT list over the region's into it; an
        // expression over an answer reads the answer's call, as written
        let merged = merged
            .clone()
            .transform_up(|found| match &found {
                Expr::Column(column) if column.relation.is_none() => Ok(
                    match answers.iter().find(|answer| answer.name == column.name) {
                        Some(answer) => Transformed::yes(answer.call.clone()),
                        None => Transformed::no(found),
                    },
                ),
                _ => Ok(Transformed::no(found)),
            })
            .map(|found| found.data)
            .unwrap_or(merged);
        if merged == *inner {
            return expr.clone();
        }
        let name = name.unwrap_or_else(|| match inner {
            Expr::Column(column) => column.name.clone(),
            other => other.schema_name().to_string(),
        });
        match &merged {
            Expr::Column(column) if column.name == name && !matches!(expr, Expr::Alias(_)) => {
                merged
            }
            _ => merged.alias(name),
        }
    }
}

/// An answer column an AI column adds, and the call that computes it.
#[derive(Debug, Clone)]
pub(super) struct Answer {
    pub(super) name: String,
    pub(super) call: Expr,
}

/// Each output column of a derived table and its expression: a SELECT
/// list's, or, where DataFusion removed one that passes its input on,
/// the input's columns.
fn derived_columns(plan: &LogicalPlan) -> Vec<(String, Expr)> {
    let mut node = plan;
    while let LogicalPlan::Filter(filter) = node {
        node = &filter.input;
    }
    if !matches!(node, LogicalPlan::Projection(_)) {
        return plan
            .schema()
            .iter()
            .map(|(qualifier, field)| {
                (
                    field.name().clone(),
                    Expr::Column(Column::new(qualifier.cloned(), field.name().clone())),
                )
            })
            .collect();
    }
    match node {
        LogicalPlan::Projection(projection) => projection
            .expr
            .iter()
            .map(|expr| match expr {
                Expr::Alias(alias) => (alias.name.clone(), alias.expr.as_ref().clone()),
                Expr::Column(column) => (column.name.clone(), expr.clone()),
                other => (other.schema_name().to_string(), other.clone()),
            })
            .collect(),
        _ => Vec::new(),
    }
}
