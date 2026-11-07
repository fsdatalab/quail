//! A model region's operators, read from its optimized DataFusion plan.
//!
//! [`derive_operators`] walks the region's plan once, children first and
//! the left input before the right, which is the order Quail numbers
//! operators in: each table's scan, the `AiNode`s on it, then the joins
//! and the AI joins above them. It binds each call's prompt and each
//! condition with the binder's [`Binder`], in the order the query wrote
//! them; names each table's operators and places its conditions with
//! [`RelationTerms`]; and writes [`Operators`] from the plan's shape.
//! The engine plans each region from these operators, and a region whose
//! shape they cannot hold is refused with [`UNSUPPORTED`] in the message.
//!
//! For example, `SELECT r.id, AI.CLASSIFY(...) AS sentiment FROM reviews
//! r WHERE AI.IF(...)` plans as an `AiNode` adding `sentiment` over an
//! `AiNode` filtering `r`, over the scan of `r`. The walk meets the scan,
//! the filter, then the classification, and the operators hold one
//! scan, one AI filter of `r`, and one classification named `sentiment`.

use std::collections::{BTreeMap, BTreeSet};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::Column;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::{Expr, Join, JoinType, LogicalPlan, SortExpr};
use quail_physical::{
    Asked, ColumnRef, Condition, ModelCall, Projected, SortExpression, ValueExpression,
};
use quail_planner::expressions::{
    CompileError, FEATURE_FIELDS, aliased_call, asked_call, asked_refs, call_aliases, call_refs,
    column_name, column_references, compile_error, condition_columns, condition_refs,
    condition_text, equality_refs, packed_columns, packing, projected_call, reference_name,
    repr_list, repr_str, sort_refs, value_refs,
};
use quail_planner::operators::{
    ColumnTest, Operators, Projection, ResultModifiers, RowOperator, Scan, SemanticColumn,
    SemanticJoin, source_columns,
};
use quail_planner::prompts::Turn;

use crate::SPAN_SUFFIX;
use crate::ai::{AiNode, Fetch, Semantics, WrittenTest};
use crate::binding::{
    Binder, OnTerm, SelectBinder, Unnested, conjuncts, contains_ai_call, feature_schema,
    has_subqueries, read_top, refuse_grouped_input, select_outputs, selected_calls, unnest_term,
    unnested_fields, unsupported,
};
use crate::functions::column_call_kind;
use crate::lower::quail_function;
use crate::region::ModelRegion;
use crate::terms::{
    RelationContent, RelationTerms, TableContent, check_feature_lists, push, repr_tuple,
    result_columns_read, result_row_condition, row_test_columns,
};

pub use crate::binding::UNSUPPORTED;

/// Derive a model region's operators from its plan after DataFusion's
/// rules, running [`crate::OptimizeModelRegions`]' default rules first on a
/// region that has not been through them.
///
/// `turn` and `layout` are the session model's chat-turn text and prompt
/// layout.
///
/// # Errors
///
/// The region uses SQL outside Quail's language; a message that
/// contains [`UNSUPPORTED`] names a shape DataFusion can run instead.
pub fn derive_region_operators(
    region: &ModelRegion,
    turn: &Turn,
    layout: &str,
) -> Result<Operators, CompileError> {
    derive_region(region, turn, layout).map(|(operators, _)| operators)
}

/// A model region's operators, as [`derive_region_operators`] derives
/// them, and the plan after DataFusion's rules they are read from.
///
/// # Errors
///
/// As [`derive_region_operators`].
pub fn derive_region(
    region: &ModelRegion,
    turn: &Turn,
    layout: &str,
) -> Result<(Operators, LogicalPlan), CompileError> {
    let plan = if region.optimized {
        region.plan.clone()
    } else {
        let rules: Vec<String> = crate::REGION_RULES
            .iter()
            .map(ToString::to_string)
            .collect();
        crate::lower_ai_calls(region.bound.clone())
            .and_then(|lowered| crate::optimize_region(&lowered, &rules))
            .map_err(|error| CompileError::new(error.to_string()))?
    };
    Ok((derive_operators(&plan, turn, layout)?, plan))
}

/// Derive a model region's operators from its plan after DataFusion's
/// rules, as the module explains.
///
/// # Errors
///
/// The region uses SQL outside Quail's language, with the message the
/// binder gives.
pub fn derive_operators(
    plan: &LogicalPlan,
    turn: &Turn,
    layout: &str,
) -> Result<Operators, CompileError> {
    let listed = with_select_list(plan);
    let top = read_top(&listed)?;
    refuse_grouped_input(top.relation)?;
    let mut walk = Walk::default();
    walk.visit(top.relation);
    if let Some(error) = walk.refusal.take() {
        return Err(error);
    }
    if walk.unknown {
        return Err(unplaced(plan));
    }
    let allow_relational = has_subqueries(plan);
    let answers = walk.answers();
    let unnests = walk.unnests(&answers);
    let select: Vec<Expr> = top
        .projection
        .iter()
        .map(|expr| unnested_fields(&walk.selected(expr, &answers)))
        .collect();
    let order: Vec<SortExpr> = top
        .order
        .iter()
        .map(|key| SortExpr {
            expr: unnested_fields(&walk.written(&key.expr, &[])),
            ..key.clone()
        })
        .collect();
    let mut binding = Binding {
        binder: Binder::new(turn, layout),
        bound: Vec::new(),
        unplaced: false,
    };
    binding.binder.outputs = select_outputs(&select);
    binding.binder.selected = selected_calls(&select);
    binding.bind_relation(&walk, &answers, &unnests)?;
    let mut selection = SelectBinder::new(&mut binding.binder);
    selection.bind_projection(&select)?;
    selection.check_output_names()?;
    let (mut columns, structs, output) = (selection.columns, selection.structs, selection.output);
    for unnest in &unnests {
        for (name, list) in &unnest.lists {
            let call = binding.binder.bind_ai_sae_features(list)?;
            columns.push(ValueExpression::Alias(quail_physical::Alias {
                expression: Box::new(ValueExpression::ModelCall(call)),
                name: name.clone(),
            }));
        }
    }
    let order = binding.binder.bind_order(&order, &columns, &structs)?;
    binding.binder.validate_query(&columns, &structs)?;
    let content = binding.content(&walk, &columns, allow_relational)?;
    let Some(placed) = Placement::new(&walk, &content).operators() else {
        return Err(unplaced(plan));
    };
    let mut found = placed.operators.clone();
    if binding.unplaced {
        return Err(unplaced(plan));
    }
    if columns.is_empty() {
        return compile_error("Project needs at least one column");
    }
    if !structs.is_empty() && top.distinct {
        return compile_error(
            "DISTINCT does not compare AI.EXTRACT struct columns; extract each field as \
             its own column",
        );
    }
    let rows = ResultRows {
        unnests: unnests
            .iter()
            .map(|unnest| unnest.lists.iter().map(|(name, _)| name.clone()).collect())
            .collect(),
        outer: unnests.iter().map(|unnest| unnest.outer).collect(),
        output,
        tests: content.rows.clone(),
    };
    let (projection, after_projection) =
        rows.operators(&columns, &structs, &order, top.distinct)?;
    found.projection = projection;
    found.after_projection = after_projection;
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
    found.result = ResultModifiers {
        limit: top.limit,
        order,
        offset: top.offset,
        distinct: top.distinct,
    };
    if found.result.limit.is_some_and(|limit| limit <= 0) {
        return compile_error("LIMIT must be a positive integer");
    }
    if found.result.offset < 0 {
        return compile_error("OFFSET must be a nonnegative integer");
    }
    if allow_relational && quail_planner::plan_statistics::plan_prompts(&found).is_empty() {
        return compile_error("the query has no AI predicate");
    }
    push_down_columns(&mut found, &walk, &placed, &content);
    Ok(found)
}

/// The refusal of a plan whose operators Quail's operators do not read.
fn unplaced(plan: &LogicalPlan) -> CompileError {
    CompileError::new(format!(
        "the placement of these operators {UNSUPPORTED}:\n{}",
        plan.display_indent()
    ))
}

/// `plan` with a SELECT list under its LIMIT, ORDER BY, and DISTINCT:
/// DataFusion removes one that returns its input's columns as they are.
fn with_select_list(plan: &LogicalPlan) -> LogicalPlan {
    let listed = |input: &LogicalPlan| -> LogicalPlan {
        if matches!(input, LogicalPlan::Projection(_)) {
            return input.clone();
        }
        let columns: Vec<Expr> = input
            .schema()
            .iter()
            .map(|(qualifier, field)| {
                Expr::Column(Column::new(qualifier.cloned(), field.name().clone()))
            })
            .collect();
        datafusion::logical_expr::Projection::try_new(columns, std::sync::Arc::new(input.clone()))
            .map(LogicalPlan::Projection)
            .unwrap_or_else(|_| input.clone())
    };
    match plan {
        LogicalPlan::Limit(_) | LogicalPlan::Sort(_) | LogicalPlan::Distinct(_) => {
            let inputs: Vec<LogicalPlan> =
                plan.inputs().into_iter().map(with_select_list).collect();
            plan.with_new_exprs(plan.expressions(), inputs)
                .unwrap_or_else(|_| plan.clone())
        }
        LogicalPlan::Projection(projection)
            if matches!(
                projection.input.as_ref(),
                LogicalPlan::Sort(_) | LogicalPlan::Distinct(_)
            ) =>
        {
            let input = with_select_list(&projection.input);
            plan.with_new_exprs(plan.expressions(), vec![input])
                .unwrap_or_else(|_| plan.clone())
        }
        other => listed(other),
    }
}

// ---------------------------------------------------------------- the walk

/// A table the region scans.
#[derive(Debug, Clone)]
struct Table {
    alias: String,
    provider: String,
    schema: SchemaRef,
    /// Whether an EXISTS gate reads it, rather than FROM or JOIN.
    gate: bool,
}

/// What one operator of the plan is.
#[derive(Debug)]
enum Kind<'a> {
    Scan(usize),
    Join(&'a Join),
    Filter(Expr),
    Ai(&'a AiNode),
    /// A projection, or a SubqueryAlias of a derived table.
    PassThrough,
}

/// One operator of the plan, as the walk meets it.
#[derive(Debug)]
struct Visit<'a> {
    kind: Kind<'a>,
    /// The ids of its inputs.
    inputs: Vec<usize>,
    /// The id of the operator it is the input of.
    parent: Option<usize>,
    /// The ids of the operators above it, outermost first.
    ancestors: Vec<usize>,
    /// The aliases of the tables it reads, in scan order.
    tables: Vec<String>,
    /// Whether an AI join or gate is at or below it.
    ai_join: bool,
}

/// The region's operators, children first, and its tables.
#[derive(Debug, Default)]
struct Walk<'a> {
    /// Each operator by its id.
    visits: Vec<Option<Visit<'a>>>,
    /// The ids, children first.
    order: Vec<usize>,
    tables: Vec<Table>,
    /// The ids of the operators being visited, outermost first.
    stack: Vec<usize>,
    /// (alias, its columns' expressions) of each derived table the
    /// binder merges into the query that reads it.
    merged: Vec<(String, Vec<(String, Expr)>)>,
    /// The binder's refusal of the FROM and JOIN shape, if any.
    refusal: Option<CompileError>,
    /// Whether the plan has an operator Quail's operators do not read.
    unplaced: bool,
    /// Whether that operator is one no operator of Quail's is, such as a
    /// LIMIT below a filter.
    unknown: bool,
}

impl<'a> Walk<'a> {
    fn visit(&mut self, plan: &'a LogicalPlan) -> usize {
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

    fn get(&self, id: usize) -> &Visit<'a> {
        self.visits[id].as_ref().expect("a visited operator")
    }

    /// The ids of the operators, children first.
    fn ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.order.iter().copied()
    }

    /// The tables an expression reads, through the derived tables merged
    /// into the query.
    fn call_tables(&self, expr: &Expr) -> BTreeSet<String> {
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
    fn written(&self, expr: &Expr, answers: &[Answer]) -> Expr {
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

    /// The terms of the plan's filters, as the query wrote them, then its
    /// tests of answer columns, in written order.
    fn where_terms(&self, answers: &[Answer]) -> Vec<Expr> {
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
            let found: Vec<Expr> = match &self.get(id).kind {
                Kind::Filter(predicate) => conjuncts(predicate),
                Kind::Ai(node) if !node.is_predicate() && node.semantics == Semantics::Rows => {
                    node.filter.iter().flat_map(conjuncts).collect()
                }
                _ => continue,
            };
            // each test of answers is in written order below
            terms.extend(
                found
                    .iter()
                    .filter(|term| !tests.iter().any(|test| test.test == **term))
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

    /// The query's `UNNEST`s, in written order.
    fn unnests(&self, answers: &[Answer]) -> Vec<Unnested> {
        self.where_terms(answers)
            .iter()
            .filter_map(unnest_term)
            .collect()
    }

    /// The answer columns the plan's AI columns add, with their calls.
    fn answers(&self) -> Vec<Answer> {
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
    fn selected(&self, expr: &Expr, answers: &[Answer]) -> Expr {
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
struct Answer {
    name: String,
    call: Expr,
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

// ---------------------------------------------------------------- binding

/// A bound ON or WHERE term, in written order.
enum Bound {
    Gate {
        call: ModelCall,
        inner: String,
        anti: bool,
        selectivity: Option<f64>,
        anchor: Option<String>,
    },
    Condition(Condition, Option<f64>),
    Asked(Asked, Option<f64>, Option<String>),
}

struct Binding {
    binder: Binder,
    bound: Vec<Bound>,
    /// Whether a term the lowering left in place binds, which the
    /// operators cannot read.
    unplaced: bool,
}

/// Whether a term is `a.x = b.y`, an equality of two tables' columns.
fn is_column_equality(term: &Expr) -> bool {
    match term {
        Expr::BinaryExpr(binary) if binary.op == datafusion::logical_expr::Operator::Eq => {
            match (binary.left.as_ref(), binary.right.as_ref()) {
                (Expr::Column(left), Expr::Column(right)) => left.relation != right.relation,
                _ => false,
            }
        }
        _ => false,
    }
}

impl Binding {
    /// Bind the relation's tables and terms in the order the query wrote
    /// them: the tables, each JOIN's ON, the ON AI predicates, then WHERE.
    fn bind_relation(
        &mut self,
        walk: &Walk<'_>,
        answers: &[Answer],
        unnests: &[Unnested],
    ) -> Result<(), CompileError> {
        for table in walk.tables.iter().filter(|table| !table.gate) {
            self.binder
                .add_table(&table.alias, &table.provider, table.schema.clone())?;
        }
        for id in walk.ids() {
            let Kind::Join(join) = &walk.get(id).kind else {
                continue;
            };
            let right = walk.get(walk.get(id).inputs[1]).tables.clone();
            if right.iter().any(|alias| {
                walk.tables
                    .iter()
                    .any(|table| table.alias == *alias && table.gate)
            }) {
                continue;
            }
            let joined = right.last().cloned().unwrap_or_default();
            let mut terms: Vec<Expr> = join
                .on
                .iter()
                .map(|(left, right)| left.clone().eq(right.clone()))
                .collect();
            if let Some(filter) = &join.filter {
                terms.extend(conjuncts(filter));
            }
            let mut equalities = Vec::new();
            for term in terms {
                let term = walk.written(&term, answers);
                // DataFusion pushes a WHERE test of the two sides into the
                // join; an equality of their columns is the join's key
                if !is_column_equality(&term) {
                    self.bind_where_term(&term)?;
                    continue;
                }
                match self.binder.bind_plain_on_term(&term, &joined)? {
                    OnTerm::Condition(condition) => {
                        self.bound.push(Bound::Condition(condition, None));
                    }
                    OnTerm::Equality(equality) => equalities.push(*equality),
                }
            }
            self.binder.add_equalities(&joined, equalities);
        }
        let mut asked: Vec<(usize, &AiNode)> = walk
            .ids()
            .filter_map(|id| match &walk.get(id).kind {
                Kind::Ai(node) if node.is_predicate() || node.semantics != Semantics::Rows => {
                    Some((id, *node))
                }
                _ => None,
            })
            .collect();
        // the ON predicates by their JOINs, then WHERE's by their terms
        asked.sort_by_key(|(_, node)| {
            node.written
                .as_ref()
                .map_or((true, usize::MAX, usize::MAX), |written| match written.on {
                    Some(_) => (false, written.index, written.index),
                    None => (true, written.position, written.index),
                })
        });
        let (on, rest): (Vec<_>, Vec<_>) = asked.into_iter().partition(|(_, node)| node.in_on());
        // one question per JOIN: its pairs are asked once
        for (_, node) in &on {
            let Some(written) = &node.written else {
                continue;
            };
            let num_asked = on
                .iter()
                .filter(|(_, other)| {
                    other
                        .written
                        .as_ref()
                        .is_some_and(|found| found.clause == written.clause)
                })
                .count();
            if num_asked > 1 {
                return compile_error(format!(
                    "JOIN {} has {num_asked} AI predicates in ON; ask one question per JOIN",
                    written.on.as_deref().unwrap_or_default()
                ));
            }
        }
        for (id, node) in on {
            self.bind_asked(walk, id, node)?;
        }
        for unnest in unnests {
            for (name, _) in &unnest.lists {
                self.binder.add_table(name, "", feature_schema())?;
            }
        }
        for (id, node) in rest {
            self.bind_asked(walk, id, node)?;
        }
        for term in walk.where_terms(answers) {
            if unnest_term(&term).is_none() {
                self.bind_where_term(&unnested_fields(&term))?;
            }
        }
        Ok(())
    }

    /// Bind an AI filter, AI join, or gate.
    fn bind_asked(
        &mut self,
        walk: &Walk<'_>,
        id: usize,
        node: &AiNode,
    ) -> Result<(), CompileError> {
        let join = node.in_on().then_some(true);
        let call = walk.written(&node.call, &[]);
        if node.semantics != Semantics::Rows {
            let Kind::Join(_) = &walk.get(walk.get(id).inputs[0]).kind else {
                self.unplaced = true;
                return Ok(());
            };
            let inner_id = walk.get(walk.get(id).inputs[0]).inputs[1];
            let inner = walk
                .get(inner_id)
                .tables
                .last()
                .cloned()
                .unwrap_or_default();
            let table = walk
                .tables
                .iter()
                .find(|table| table.alias == inner)
                .expect("a gated table");
            self.binder
                .add_table(&table.alias, &table.provider, table.schema.clone())?;
            let (call, options) =
                self.binder
                    .bind_ai_filter_call(&call, &["selectivity", "anchor"], Some(true))?;
            self.bound.push(Bound::Gate {
                call,
                inner,
                anti: node.semantics == Semantics::Anti,
                selectivity: options.selectivity,
                anchor: options.anchor,
            });
            return Ok(());
        }
        let (predicate, options) = if node.function() == "ai_score" {
            let answer = Answer {
                name: node.answer.clone(),
                call,
            };
            let term = walk.written(node.filter.as_ref().expect("a test"), &[answer]);
            let (test, options) = self
                .binder
                .bind_ai_score_test(&term, &["selectivity"], join)?;
            (Asked::Compare(test), options)
        } else {
            let (call, options) =
                self.binder
                    .bind_ai_filter_call(&call, &["selectivity", "anchor"], join)?;
            (Asked::Call(call), options)
        };
        self.bound
            .push(Bound::Asked(predicate, options.selectivity, options.anchor));
        Ok(())
    }

    /// Bind a WHERE term that is no AI filter: a plain condition, a test
    /// of a label, or a test of another column operator's answer.
    fn bind_where_term(&mut self, term: &Expr) -> Result<(), CompileError> {
        if let Some((condition, hint)) = self.binder.bind_plain_where_term(term)? {
            self.bound.push(Bound::Condition(condition, hint));
            return Ok(());
        }
        let (unhinted, hint) = self.binder.unwrap_hint(term)?;
        if let Some((classify, items)) = crate::binding::label_test(&unhinted) {
            let (test, selectivity) = self.binder.bind_label_test(classify, &items, hint)?;
            self.bound
                .push(Bound::Asked(Asked::Compare(test), selectivity, None));
            return Ok(());
        }
        // the binder's refusal of the term, or a term it reads that the
        // lowering left in place
        self.binder.refuse_where_term(term)?;
        self.unplaced = true;
        Ok(())
    }

    /// Each table's operators, named and placed.
    fn content(
        &mut self,
        walk: &Walk<'_>,
        columns: &[ValueExpression],
        allow_relational: bool,
    ) -> Result<RelationContent, CompileError> {
        let scans: Vec<Scan> = walk
            .tables
            .iter()
            .filter(|table| !table.gate)
            .map(|table| self.binder.scan(&table.alias))
            .collect();
        let mut terms = RelationTerms::new(scans, self.binder.equalities());
        for bound in std::mem::take(&mut self.bound) {
            match bound {
                Bound::Gate {
                    call,
                    inner,
                    anti,
                    selectivity,
                    anchor,
                } => {
                    let scan = self.binder.scan(&inner);
                    terms.gate(&call, scan, anti, anchor.as_ref(), selectivity)?;
                }
                Bound::Condition(condition, selectivity) => terms.condition(condition, selectivity),
                Bound::Asked(predicate, selectivity, anchor) => {
                    terms.asked(predicate, selectivity, anchor)?;
                }
            }
        }
        terms.content(
            columns,
            &self.binder.tested(),
            &self.binder.provider_schemas(),
            &self.binder.unnested(),
            allow_relational,
        )
    }
}

// ---------------------------------------------------------------- placement

/// One operator on a table's stack, lowest first.
#[derive(Debug, Clone)]
enum StackItem {
    /// The table's AI filters.
    Filters,
    /// A column operator: its kind and position in its list.
    Column(&'static str, usize),
    /// A test of answers: its position in the table's tests.
    Test(usize),
}

/// The operators of a region, placed as the plan holds them.
struct Placement<'w, 'a> {
    walk: &'w Walk<'a>,
    content: &'w RelationContent,
    found: Operators,
    /// (alias, name) of each column operator placed.
    placed: Vec<(String, String)>,
    joins: Vec<bool>,
    /// Each table's stack, by alias, in scan order.
    stacks: Vec<(String, Vec<StackItem>)>,
    /// The join spec placed at each AI join's id.
    join_at: BTreeMap<usize, usize>,
    /// The ON equalities placed on each join's id.
    join_on: BTreeMap<usize, Vec<quail_physical::Equality>>,
    /// The column operators placed at each AI column's id: (kind, position).
    columns_at: BTreeMap<usize, Vec<(&'static str, usize)>>,
}

impl<'w, 'a> Placement<'w, 'a> {
    fn new(walk: &'w Walk<'a>, content: &'w RelationContent) -> Self {
        Self {
            walk,
            content,
            found: Operators::default(),
            placed: Vec::new(),
            joins: vec![false; content.joins.len()],
            stacks: Vec::new(),
            join_at: BTreeMap::new(),
            join_on: BTreeMap::new(),
            columns_at: BTreeMap::new(),
        }
    }

    fn table(&self, alias: &str) -> Option<&'w TableContent> {
        self.content
            .tables
            .iter()
            .find(|table| table.alias == alias)
    }

    /// Whether `id` is an AI filter of one table.
    fn is_table_filter(&self, id: usize) -> bool {
        match &self.walk.get(id).kind {
            Kind::Ai(node) => {
                node.is_predicate()
                    && node.semantics == Semantics::Rows
                    && self.walk.call_tables(&node.call).len() < 2
            }
            _ => false,
        }
    }

    fn call_tables(&self, id: usize) -> BTreeSet<String> {
        match &self.walk.get(id).kind {
            Kind::Ai(node) => self.walk.call_tables(&node.call),
            _ => BTreeSet::new(),
        }
    }

    /// The input of `id`, passing through projections and plain filters.
    fn relation_input(&self, id: usize) -> Option<usize> {
        let mut below = *self.walk.get(id).inputs.first()?;
        loop {
            match &self.walk.get(below).kind {
                Kind::PassThrough | Kind::Filter(_) if !self.walk.get(below).inputs.is_empty() => {
                    below = self.walk.get(below).inputs[0];
                }
                _ => return Some(below),
            }
        }
    }

    /// The operators of the region; None when the plan holds an operator
    /// Quail's operators do not read here.
    fn operators(mut self) -> Option<Placed> {
        if self.walk.unplaced {
            return None;
        }
        let ids: Vec<usize> = self.walk.ids().collect();
        let mut noted: Vec<String> = Vec::new();
        for id in ids {
            let visit = self.walk.get(id);
            match &visit.kind {
                Kind::Scan(table) => {
                    let alias = self.walk.tables[*table].alias.clone();
                    let table = self.table(&alias)?;
                    self.found.scans.push(Scan {
                        provider: table.provider.clone(),
                        alias: table.alias.clone(),
                        column: table.column.clone(),
                        columns: Vec::new(),
                        predicates: table.operators.regular_predicates.clone(),
                        schema: table.schema.clone(),
                    });
                    self.stacks.push((alias, Vec::new()));
                }
                Kind::Filter(predicate) if contains_ai_call(predicate) => return None,
                Kind::Ai(node) if node.is_predicate() || node.semantics != Semantics::Rows => {
                    if self.is_table_filter(id) {
                        if self.continues_run(id) {
                            continue;
                        }
                        self.place_filters(id, &mut noted)?;
                    } else {
                        self.place_join(id)?;
                    }
                }
                Kind::Ai(node) => {
                    if self.continues_column_run(id) {
                        continue;
                    }
                    self.place_column(id, node)?;
                }
                _ => {}
            }
        }
        if self.joins.iter().any(|placed| !placed) {
            return None;
        }
        self.place_tests(&mut noted)?;
        Some(Placed {
            operators: self.found,
            stacks: self.stacks,
            join_at: self.join_at,
            join_on: self.join_on,
            columns_at: self.columns_at,
        })
    }

    /// Whether an AI filter's parent is another AI filter of the same
    /// table, which places the two as one run.
    fn continues_run(&self, id: usize) -> bool {
        self.walk.get(id).parent.is_some_and(|parent| {
            self.is_table_filter(parent) && self.call_tables(parent) == self.call_tables(id)
        })
    }

    fn place_filters(&mut self, top: usize, noted: &mut Vec<String>) -> Option<()> {
        let read = self.call_tables(top);
        let mut run = vec![top];
        let mut below = self.walk.get(top).inputs[0];
        while self.is_table_filter(below) && self.call_tables(below) == read {
            run.push(below);
            below = self.walk.get(below).inputs[0];
        }
        let [alias] = <[String; 1]>::try_from(read.into_iter().collect::<Vec<_>>()).ok()?;
        let predicates = self.table(&alias)?.operators.predicates.clone();
        if predicates.len() != run.len() {
            return None;
        }
        let before_joins = !self.walk.get(below).ai_join;
        let stop_key = run.iter().find_map(|id| match &self.walk.get(*id).kind {
            Kind::Ai(node) => match &node.fetch {
                Some(Fetch::PerKey { key, n: 1 }) => Some(key.clone()),
                _ => None,
            },
            _ => None,
        });
        for predicate in predicates {
            push(&mut self.found.filters, alias.clone(), predicate);
            push(
                &mut self.found.filters_before_joins,
                alias.clone(),
                before_joins,
            );
        }
        if let Some(key) = stop_key {
            self.found.stop_keys.insert(alias.clone(), key);
        }
        note(noted, &alias);
        self.stack_push(&alias, StackItem::Filters);
        Some(())
    }

    fn stack_push(&mut self, alias: &str, item: StackItem) {
        if let Some((_, stack)) = self
            .stacks
            .iter_mut()
            .rev()
            .find(|(found, _)| found == alias)
        {
            stack.push(item);
        }
    }

    fn place_join(&mut self, id: usize) -> Option<()> {
        let read = self.call_tables(id);
        let index = self
            .content
            .joins
            .iter()
            .enumerate()
            .position(|(index, spec)| {
                !self.joins[index]
                    && call_aliases(asked_call(&spec.predicate))
                        .into_iter()
                        .collect::<BTreeSet<String>>()
                        == read
            })?;
        self.joins[index] = true;
        self.join_at.insert(id, index);
        let spec = &self.content.joins[index];
        let input = self.relation_input(id)?;
        let on = match &self.walk.get(input).kind {
            Kind::Join(_) => spec.on.clone(),
            _ => Vec::new(),
        };
        self.join_on.insert(input, on.clone());
        let mut outer = input;
        while let Kind::Join(_) = &self.walk.get(outer).kind {
            outer = self.relation_input_at(outer, 0)?;
        }
        let mut outer_aliases: Vec<String> = Vec::new();
        for alias in &self.walk.get(outer).tables {
            if !outer_aliases.contains(alias) {
                outer_aliases.push(alias.clone());
            }
        }
        self.found.joins.push(SemanticJoin {
            predicate: spec.predicate.clone(),
            semantics: spec.semantics.clone(),
            selectivity: spec.selectivity,
            anchor: spec.anchor.clone(),
            on,
            outer_aliases,
        });
        Some(())
    }

    /// The `index`th input of `id`, passing through projections.
    fn relation_input_at(&self, id: usize, index: usize) -> Option<usize> {
        let mut below = *self.walk.get(id).inputs.get(index)?;
        while let Kind::PassThrough = &self.walk.get(below).kind
            && !self.walk.get(below).inputs.is_empty()
        {
            below = self.walk.get(below).inputs[0];
        }
        Some(below)
    }

    /// Whether an AI column of joined rows reads the same tables as the
    /// AI column it is the input of, which places the two as one run.
    fn continues_column_run(&self, id: usize) -> bool {
        if self.call_tables(id).len() != 2 {
            return false;
        }
        self.walk.get(id).parent.is_some_and(|parent| {
            matches!(&self.walk.get(parent).kind,
                Kind::Ai(node) if !node.is_predicate() && node.semantics == Semantics::Rows)
                && self.call_tables(parent) == self.call_tables(id)
        })
    }

    fn place_column(&mut self, id: usize, node: &AiNode) -> Option<()> {
        let kind = column_call_kind(&node.call)?;
        let read: Vec<String> = self.call_tables(id).into_iter().collect();
        let above_joins = self.walk.get(self.walk.get(id).inputs[0]).ai_join;
        if read.len() == 2 {
            // a run of classifications of the same joined rows, the first
            // written placed lowest
            let mut run = vec![(id, node)];
            let mut below = self.walk.get(id).inputs[0];
            while let Kind::Ai(next) = &self.walk.get(below).kind
                && !next.is_predicate()
                && next.semantics == Semantics::Rows
                && self.call_tables(below) == self.call_tables(id)
            {
                run.push((below, *next));
                below = self.walk.get(below).inputs[0];
            }
            let above_joins = self.walk.get(below).ai_join;
            for (at, found) in run {
                let (call, name) = self
                    .content
                    .joined_classifications
                    .iter()
                    .find(|(_, name)| *name == found.answer)
                    .or_else(|| {
                        self.content
                            .joined_classifications
                            .iter()
                            .find(|(_, name)| !self.placed.iter().any(|(_, placed)| placed == name))
                    })?
                    .clone();
                self.placed.push((read.join(","), name.clone()));
                self.found.classifies.push(SemanticColumn {
                    call,
                    name,
                    above_joins,
                });
                self.columns_at
                    .entry(at)
                    .or_default()
                    .push(("label", self.found.classifies.len() - 1));
            }
            return Some(());
        }
        let [alias] = <[String; 1]>::try_from(read).ok()?;
        let operators = &self.table(&alias)?.operators;
        let entries = match kind {
            "label" => &operators.labels,
            "extract" => &operators.extracts,
            "map" => &operators.maps,
            _ => &operators.sae_features,
        };
        let placed = |name: &String| {
            self.placed
                .iter()
                .any(|(owner, found)| *owner == alias && found == name)
        };
        // an extraction of several questions is one column per field
        let field_prefix = format!("{}.", node.answer);
        let mut chosen: Vec<(ModelCall, String)> = entries
            .iter()
            .filter(|(_, name)| *name == node.answer || name.starts_with(&field_prefix))
            .cloned()
            .collect();
        if chosen.is_empty() {
            // a column only a test reads: the next one not yet placed
            chosen.push(entries.iter().find(|(_, name)| !placed(name))?.clone());
        }
        for (call, name) in chosen {
            self.placed.push((alias.clone(), name.clone()));
            let column = SemanticColumn {
                call,
                name,
                above_joins,
            };
            let list = match kind {
                "label" => &mut self.found.classifies,
                "extract" => &mut self.found.extracts,
                "map" => &mut self.found.maps,
                _ => &mut self.found.sae_features,
            };
            list.push(column);
            let position = list.len() - 1;
            self.columns_at
                .entry(id)
                .or_default()
                .push((kind, position));
            self.stack_push(&alias, StackItem::Column(kind, position));
        }
        Some(())
    }

    fn column(&self, kind: &str, position: usize) -> &SemanticColumn {
        match kind {
            "label" => &self.found.classifies[position],
            "extract" => &self.found.extracts[position],
            "map" => &self.found.maps[position],
            _ => &self.found.sae_features[position],
        }
    }

    /// Each table's tests of answers, on its stack, and the operators'
    /// column filters.
    fn place_tests(&mut self, noted: &mut Vec<String>) -> Option<()> {
        let mut stacks = std::mem::take(&mut self.stacks);
        for (alias, stack) in &mut stacks {
            let Some(table) = self.table(alias) else {
                continue;
            };
            for (index, (condition, _)) in table.operators.computed_filters.iter().enumerate() {
                let position = self.test_position(stack, condition);
                stack.insert(position, StackItem::Test(index));
                note(noted, alias);
            }
        }
        for alias in noted.iter() {
            let Some((_, stack)) = stacks.iter().find(|(found, _)| found == alias) else {
                continue;
            };
            let table = self.table(alias)?;
            let mut tests: Vec<(usize, &SemanticColumn)> = Vec::new();
            for (at, item) in stack.iter().enumerate() {
                let StackItem::Test(index) = item else {
                    continue;
                };
                let read = condition_columns(&table.operators.computed_filters[*index].0);
                let producer = stack[..at].iter().rev().find_map(|below| match below {
                    StackItem::Column(kind, position) => {
                        let column = self.column(kind, *position);
                        column
                            .added_columns()
                            .iter()
                            .any(|name| read.contains(name))
                            .then_some(column)
                    }
                    _ => None,
                })?;
                tests.push((*index, producer));
            }
            if tests.is_empty() {
                continue;
            }
            tests.sort_by_key(|(_, producer)| producer.call.kind != "label");
            let asks = self
                .found
                .filters
                .iter()
                .find(|(found, _)| found == alias)
                .map(|(_, predicates)| predicates.len())
                .unwrap_or(0);
            let column_tests = tests
                .iter()
                .enumerate()
                .map(|(order, (index, producer))| {
                    let (condition, selectivity) = &table.operators.computed_filters[*index];
                    ColumnTest {
                        name: producer.name.clone(),
                        call: producer.call.clone(),
                        condition: condition.clone(),
                        selectivity: *selectivity,
                        position: asks + order,
                        before_joins: true,
                    }
                })
                .collect();
            self.found
                .column_filters
                .push((alias.clone(), column_tests));
        }
        self.stacks = stacks;
        Some(())
    }

    /// Where on a stack a test of answers goes: right above the lowest
    /// operator after which every answer column it reads is added, and
    /// above the tests placed before it; a test that reads no answer
    /// column goes right above the first extraction, SAE feature, or map.
    fn test_position(&self, stack: &[StackItem], condition: &Condition) -> usize {
        let answers: BTreeSet<String> = self.stack_columns(stack);
        let reads: BTreeSet<String> = condition_columns(condition)
            .into_iter()
            .filter(|column| answers.contains(column))
            .collect();
        let mut top = stack.len();
        while top > 0 {
            let at = top - 1;
            let descends = match &stack[at] {
                StackItem::Column(..) => {
                    let below = &stack[..at];
                    if reads.is_empty() {
                        self.has_answer_column(below)
                    } else {
                        reads.is_subset(&self.stack_columns(below))
                    }
                }
                _ => false,
            };
            if !descends {
                break;
            }
            top = at;
        }
        top
    }

    /// The answer columns the column operators of a stack add, down to its
    /// AI filters.
    fn stack_columns(&self, stack: &[StackItem]) -> BTreeSet<String> {
        let mut found = BTreeSet::new();
        for item in stack.iter().rev() {
            match item {
                StackItem::Column(kind, position) => {
                    found.extend(self.column(kind, *position).added_columns());
                }
                StackItem::Test(_) => {}
                StackItem::Filters => break,
            }
        }
        found
    }

    /// Whether the top of a stack, or an operator below it, is an
    /// extraction, SAE feature, or map column.
    fn has_answer_column(&self, stack: &[StackItem]) -> bool {
        for item in stack.iter().rev() {
            match item {
                StackItem::Column(kind, _) if *kind != "label" => return true,
                StackItem::Column(..) | StackItem::Test(_) => {}
                StackItem::Filters => return false,
            }
        }
        false
    }
}

fn note(aliases: &mut Vec<String>, alias: &str) {
    if !aliases.iter().any(|found| found == alias) {
        aliases.push(alias.to_string());
    }
}

// ---------------------------------------------------------------- result rows

/// What runs on result rows above the SELECT list.
struct ResultRows {
    /// The list names of each `UNNEST`, in written order.
    unnests: Vec<Vec<String>>,
    /// Whether each `UNNEST` keeps a row with no elements.
    outer: Vec<bool>,
    /// The SELECT list's output column names.
    output: Vec<String>,
    /// The tests of more than one table's columns.
    tests: Vec<Condition>,
}

impl ResultRows {
    /// The SELECT list the scans' rows are read into, and the operators on
    /// result rows above it, lowest first: each test where the columns it
    /// reads exist, each `UNNEST`, and the outer SELECT list.
    ///
    /// # Errors
    ///
    /// The SELECT list compares feature lists, an `UNNEST` does not read
    /// a list column, or a test reads a column no result row holds.
    fn operators(
        &self,
        columns: &[ValueExpression],
        structs: &[quail_physical::ExtractStruct],
        order: &[quail_physical::SortKey],
        distinct: bool,
    ) -> Result<(Projection, Vec<RowOperator>), CompileError> {
        let unnested: BTreeSet<String> = self.unnests.iter().flatten().cloned().collect();
        check_feature_lists(columns, order, distinct, &unnested)?;
        let reads_unnested = |expression: &ValueExpression| {
            column_references(expression)
                .iter()
                .any(|found| unnested.contains(&found.alias))
        };
        let mut expanded: Vec<(String, ValueExpression)> = Vec::new();
        for column in columns.iter().filter(|column| reads_unnested(column)) {
            let name = column_name(column, true);
            match expanded.iter_mut().find(|(found, _)| *found == name) {
                Some(entry) => entry.1 = column.clone(),
                None => expanded.push((name, column.clone())),
            }
        }
        let mut source: Vec<ValueExpression> = columns
            .iter()
            .filter(|column| {
                let name = column_name(column, true);
                !expanded.iter().any(|(found, _)| *found == name)
            })
            .cloned()
            .collect();
        let packing = packing(structs);
        let returned = packed_columns(
            source.iter().map(|column| column_name(column, true)),
            &packing,
        );
        let tests: Vec<Condition> = self
            .tests
            .iter()
            .map(|test| result_row_condition(test, &packing))
            .collect();
        let has_rows = !self.unnests.is_empty() || !tests.is_empty();
        if has_rows {
            for key in order {
                let SortExpression::Value(expression) = &key.expression;
                if !column_references(expression).is_empty()
                    && !reads_unnested(expression)
                    && !source.contains(expression)
                {
                    source.push(expression.clone());
                }
            }
        }
        source.extend(row_test_columns(&tests, &unnested, &source, &returned));
        let projection = Projection {
            columns: source.iter().cloned().map(Projected::Expression).collect(),
            structs: structs.to_vec(),
        };
        let mut names = packed_columns(
            source.iter().map(|column| column_name(column, true)),
            &packing,
        );
        let mut lists: BTreeMap<String, Vec<String>> = source
            .iter()
            .filter(|column| aliased_call(column, &["features"]).is_some())
            .map(|column| {
                (
                    column_name(column, true),
                    FEATURE_FIELDS.iter().map(ToString::to_string).collect(),
                )
            })
            .collect();
        let readable = |names: &[String]| -> BTreeSet<String> {
            let mut found: BTreeSet<String> = names.iter().cloned().collect();
            for (name, fields) in &packing {
                if found.contains(name) {
                    for (field, _) in fields {
                        found.insert(format!("{name}.{field}"));
                    }
                }
            }
            found
        };
        let mut chain: Vec<RowOperator> = Vec::new();
        let mut remaining = tests;
        let place =
            |chain: &mut Vec<RowOperator>, names: &[String], remaining: &mut Vec<Condition>| {
                let available = readable(names);
                let mut index = 0;
                while index < remaining.len() {
                    if result_columns_read(&remaining[index])
                        .iter()
                        .all(|name| available.contains(name))
                    {
                        chain.push(RowOperator::Filter {
                            condition: remaining.remove(index),
                            columns: names.to_vec(),
                        });
                    } else {
                        index += 1;
                    }
                }
            };
        place(&mut chain, &names, &mut remaining);
        let outer = if self.outer.is_empty() {
            vec![false; self.unnests.len()]
        } else {
            self.outer.clone()
        };
        for (unnest, keep_empty) in self.unnests.iter().zip(&outer) {
            names = unnest_columns(unnest, &names, &lists)?;
            lists.retain(|name, _| !unnest.contains(name));
            chain.push(RowOperator::Unnest {
                lists: unnest.clone(),
                outer: *keep_empty,
                columns: names.clone(),
            });
            place(&mut chain, &names, &mut remaining);
        }
        if let Some(test) = remaining.first() {
            let available = readable(&names);
            let mut missing: Vec<String> = result_columns_read(test)
                .into_iter()
                .filter(|name| !available.contains(name))
                .collect();
            missing.sort();
            missing.dedup();
            return compile_error(format!(
                "{} reads {}, which no result row holds",
                condition_text(test),
                repr_list(&missing)
            ));
        }
        if has_rows {
            let selected = if self.unnests.is_empty() {
                returned
            } else {
                self.output.clone()
            };
            chain.push(RowOperator::Projection {
                expressions: selected
                    .iter()
                    .map(
                        |name| match expanded.iter().find(|(found, _)| found == name) {
                            Some((_, column)) => Projected::Expression(column.clone()),
                            None => Projected::Name(name.clone()),
                        },
                    )
                    .collect(),
                columns: selected,
            });
        }
        Ok((projection, chain))
    }
}

/// The result columns after an `UNNEST` of `unnested`: each list column
/// as its field columns.
///
/// # Errors
///
/// The `UNNEST` names no list, a list twice, a column that is not a list
/// the rows below return, or makes a column name repeat.
fn unnest_columns(
    unnested: &[String],
    names: &[String],
    lists: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<String>, CompileError> {
    if unnested.is_empty() {
        return compile_error("UNNEST needs at least one list column");
    }
    if unnested.iter().collect::<BTreeSet<_>>().len() != unnested.len() {
        return compile_error(format!(
            "UNNEST names a list twice, got {}",
            repr_tuple(unnested)
        ));
    }
    for column in unnested {
        if !lists.contains_key(column) {
            let mut known: Vec<String> = lists.keys().cloned().collect();
            known.sort();
            return compile_error(format!(
                "UNNEST reads a list column the rows below return, got {}; the list columns \
                 are {}",
                repr_str(column),
                repr_list(&known)
            ));
        }
    }
    let expanded: Vec<String> = names
        .iter()
        .flat_map(
            |name| match lists.get(name).filter(|_| unnested.contains(name)) {
                Some(fields) => fields
                    .iter()
                    .map(|field| format!("{name}.{field}"))
                    .collect(),
                None => vec![name.clone()],
            },
        )
        .collect();
    if expanded.iter().collect::<BTreeSet<_>>().len() != expanded.len() {
        let listed: Vec<String> = expanded.iter().map(|name| repr_str(name)).collect();
        return compile_error(format!(
            "UNNEST field columns must not repeat a column name, got ({})",
            listed.join(", ")
        ));
    }
    Ok(expanded)
}

// ---------------------------------------------------------------- scan columns

/// What [`Placement`] placed where.
struct Placed {
    operators: Operators,
    /// Each table's stack, by alias, in scan order.
    stacks: Vec<(String, Vec<StackItem>)>,
    /// The join spec placed at each AI join's id.
    join_at: BTreeMap<usize, usize>,
    /// The ON equalities placed on each join's id.
    join_on: BTreeMap<usize, Vec<quail_physical::Equality>>,
    /// The column operators placed at each AI column's id.
    columns_at: BTreeMap<usize, Vec<(&'static str, usize)>>,
}

impl Placed {
    fn column(&self, found: &Operators, kind: &str, position: usize) -> SemanticColumn {
        match kind {
            "label" => found.classifies[position].clone(),
            "extract" => found.extracts[position].clone(),
            "map" => found.maps[position].clone(),
            _ => found.sae_features[position].clone(),
        }
    }
}

/// The columns each table's operators read, by alias, in first use
/// order, as Quail's scans load them.
#[derive(Default)]
struct Needed {
    by_alias: BTreeMap<String, Vec<String>>,
}

impl Needed {
    fn add(&mut self, references: Vec<ColumnRef>) {
        for reference in references {
            let columns = self.by_alias.entry(reference.alias).or_default();
            if !columns.contains(&reference.column) {
                columns.push(reference.column);
            }
        }
    }

    fn remove(&mut self, alias: &str, added: &[String]) {
        if let Some(columns) = self.by_alias.get_mut(alias) {
            columns.retain(|name| !added.contains(name));
        }
    }
}

/// Each scan's columns: the column values the operators above it read,
/// nearest the result first.
fn push_down_columns(
    found: &mut Operators,
    walk: &Walk<'_>,
    placed: &Placed,
    content: &RelationContent,
) {
    let mut valued: BTreeSet<(String, String)> = BTreeSet::new();
    let mut top = Needed::default();
    for key in &found.result.order {
        top.add(sort_refs(key));
    }
    for operator in found.after_projection.iter().rev() {
        match operator {
            RowOperator::Projection { expressions, .. } => {
                for column in expressions {
                    if let Projected::Expression(expression) = column {
                        top.add(value_refs(expression));
                    }
                }
            }
            RowOperator::Filter { condition, .. } => top.add(condition_refs(condition)),
            RowOperator::Unnest { .. } => {}
        }
    }
    for column in &found.projection.columns {
        if let Projected::Expression(expression) = column {
            top.add(value_refs(expression));
        }
    }
    let mut value = |references: Vec<ColumnRef>| {
        for reference in references {
            valued.insert((reference.alias, reference.column));
        }
    };
    // the columns the outermost SELECT list returns or computes
    let outermost = match found.after_projection.last() {
        Some(RowOperator::Projection { expressions, .. }) => expressions,
        _ => &found.projection.columns,
    };
    for column in outermost {
        match column {
            Projected::Expression(expression) => value(value_refs(expression)),
            Projected::Name(name) => {
                for inner in &found.projection.columns {
                    if let Projected::Expression(ValueExpression::ColumnRef(reference)) = inner
                        && reference_name(reference, true) == *name
                    {
                        value(vec![reference.clone()]);
                    }
                }
            }
        }
    }
    for key in &found.result.order {
        value(sort_refs(key));
    }
    for on in placed.join_on.values() {
        for equality in on {
            value(equality_refs(equality));
        }
    }
    for scan in &found.scans {
        for predicate in &scan.predicates {
            value(condition_refs(predicate));
        }
    }
    for column in found.extracts.iter().chain(&found.maps) {
        value(call_refs(&column.call));
    }
    let scan_ids: Vec<usize> = walk
        .ids()
        .filter(|id| matches!(walk.get(*id).kind, Kind::Scan(_)))
        .collect();
    for (index, scan_id) in scan_ids.iter().enumerate() {
        let alias = found.scans[index].alias.clone();
        let stack = &placed.stacks[index].1;
        let mut needed = Needed {
            by_alias: top.by_alias.clone(),
        };
        let ancestors = &walk.get(*scan_id).ancestors;
        let mut joined_run: Vec<SemanticColumn> = Vec::new();
        let flush = |needed: &mut Needed, run: &mut Vec<SemanticColumn>| {
            for column in run.drain(..).rev() {
                needed.add(call_refs(&column.call));
                let owner = call_aliases(&column.call)
                    .into_iter()
                    .next()
                    .unwrap_or_default();
                needed.remove(&owner, &column.added_columns());
            }
        };
        for id in ancestors {
            if *id == *scan_id {
                continue;
            }
            let visit = walk.get(*id);
            let on_stack = match &visit.kind {
                Kind::Ai(node) => {
                    let tables = walk.call_tables(&node.call);
                    tables.len() < 2 && tables.contains(&alias) && node.semantics == Semantics::Rows
                }
                _ => false,
            };
            if on_stack {
                continue;
            }
            if let Some(columns) = placed.columns_at.get(id)
                && walk
                    .call_tables(match &visit.kind {
                        Kind::Ai(node) => &node.call,
                        _ => unreachable!("a column at an AI node"),
                    })
                    .len()
                    == 2
            {
                for (kind, position) in columns {
                    joined_run.push(placed.column(found, kind, *position));
                }
                continue;
            }
            flush(&mut needed, &mut joined_run);
            if let Some(index) = placed.join_at.get(id) {
                needed.add(asked_refs(&content.joins[*index].predicate));
            }
            if let Some(on) = placed.join_on.get(id) {
                for equality in on {
                    needed.add(equality_refs(equality));
                }
            }
            if let Some(columns) = placed.columns_at.get(id) {
                for (kind, position) in columns.iter().rev() {
                    let column = placed.column(found, kind, *position);
                    needed.add(call_refs(&column.call));
                    let owner = call_aliases(&column.call)
                        .into_iter()
                        .next()
                        .unwrap_or_default();
                    needed.remove(&owner, &column.added_columns());
                }
            }
        }
        flush(&mut needed, &mut joined_run);
        let table = content.tables.iter().find(|table| table.alias == alias);
        for item in stack.iter().rev() {
            match item {
                StackItem::Filters => {
                    if let Some((_, predicates)) =
                        found.filters.iter().find(|(found, _)| *found == alias)
                    {
                        for predicate in predicates {
                            needed.add(asked_refs(&predicate.expression));
                        }
                    }
                }
                StackItem::Column(kind, position) => {
                    let column = placed.column(found, kind, *position);
                    needed.add(call_refs(&column.call));
                    needed.remove(&alias, &column.added_columns());
                }
                StackItem::Test(index) => {
                    if let Some(table) = table {
                        needed.add(condition_refs(&table.operators.computed_filters[*index].0));
                    }
                }
            }
        }
        let scan = &found.scans[index];
        for predicate in &scan.predicates {
            needed.add(condition_refs(predicate));
        }
        let sources = source_columns(&scan.column);
        let columns: Vec<String> = needed
            .by_alias
            .get(&alias)
            .map(|columns| {
                columns
                    .iter()
                    .filter(|column| {
                        !sources.contains(column)
                            || valued.contains(&(alias.clone(), (*column).clone()))
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        found.scans[index].columns = columns;
    }
}
