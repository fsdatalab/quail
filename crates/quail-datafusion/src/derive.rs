//! A model region's operators, read from its optimized DataFusion plan.
//!
//! [`derive_region`] walks the region's plan once, children first and
//! the left input before the right, which is the order Quail numbers
//! operators in: each table's scan, the `AiNode`s on it, then the joins
//! and the AI joins above them. It then
//!
//! 1. binds each table, ON and WHERE term, AI call, and SELECT list item
//!    with the [`Binder`], in the order the query wrote them, reading
//!    each `AiNode`'s call and written place and the tests it keeps;
//! 2. names each column operator, by its SELECT alias or a name of its
//!    own, and places each condition: at a table's scan, after the
//!    operator that computes the columns it reads, or on the result rows;
//! 3. assembles [`Operators`] in the plan's order, each `AiNode` giving
//!    one AI filter, AI join, gate, or column operator.
//!
//! The engine plans each region from these operators, and a region whose
//! shape they cannot hold is refused with [`UNSUPPORTED`] in the message.
//!
//! For example, `SELECT r.id, AI.CLASSIFY(...) AS sentiment FROM reviews
//! r WHERE AI.IF(...)` plans as an `AiNode` adding `sentiment` over an
//! `AiNode` filtering `r`, over the scan of `r`. The walk meets the scan,
//! the filter, then the classification, and the operators hold one
//! scan, one AI filter of `r`, and one classification named `sentiment`.

use std::collections::{BTreeMap, BTreeSet};

use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, ScalarValue};
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::logical_expr::{Distinct, Expr, Join, JoinType, LogicalPlan, SortExpr};
use quail_physical::{
    Asked, ColumnRef, Equality, Expression, FilterPredicate, GetField, ModelCall, Projected,
    SortExpression, SortKey, ValueExpression,
};
use quail_planner::expressions::{
    CompileError, FEATURE_FIELDS, UNSUPPORTED, aliased_call, asked_call, asked_refs, call_aliases,
    call_refs, column_alias_of, column_name, column_references, compile_error, condition_columns,
    condition_text, equality_refs, equality_text, packed_columns, packing, projected_call,
    reference_name, repr_list, repr_str, sort_refs, value_refs,
};
use quail_planner::operators::{
    ColumnTest, Operators, Projection, ResultModifiers, RowOperator, Scan, SemanticColumn,
    SemanticJoin, source_columns,
};
use quail_planner::prompts::Turn;

use crate::SPAN_SUFFIX;
use crate::ai::{AiNode, Fetch, Semantics, WrittenTest};
use crate::binding::{
    Binder, Selected, check_labels, label_test, output_name, string_literal, unsupported,
};
use crate::extension::as_node;
use crate::functions::column_call_kind;
use crate::lower::quail_function;
use crate::region::ModelRegion;
use serde_json::Value;

/// A model region's operators and the plan after DataFusion's optimizer
/// they are read from.
///
/// `turn` and `layout` are the session model's chat-turn text and prompt
/// layout.
///
/// # Errors
///
/// The region uses SQL outside Quail's language; a message that
/// contains [`UNSUPPORTED`] names a shape DataFusion can run instead.
pub fn derive_region(
    region: &ModelRegion,
    turn: &Turn,
    layout: &str,
) -> Result<(Operators, LogicalPlan), CompileError> {
    let plan =
        as_bound(region.plan.clone()).map_err(|error| CompileError::new(error.to_string()))?;
    Ok((derive_operators(&plan, turn, layout)?, plan))
}

/// `plan` with each struct or list literal in an `AiNode`'s expressions
/// written as the call it was folded from.
///
/// DataFusion's `SimplifyExpressions` runs over a region's operators and
/// folds a `named_struct`, `struct`, or `make_array` of literals into one
/// literal. The binder reads an AI call's options, categories, and
/// fields from the calls the query wrote.
fn as_bound(plan: LogicalPlan) -> datafusion::common::Result<LogicalPlan> {
    let read = |expr: Expr| {
        expr.transform_up(|inner| Ok(unfold(inner)))
            .map(|done| done.data)
    };
    plan.transform_up(|node| {
        let Some(ai) = crate::extension::as_node::<AiNode>(&node) else {
            return Ok(Transformed::no(node));
        };
        let mut ai = ai.clone();
        ai.call = read(ai.call)?;
        ai.guard = ai.guard.map(read).transpose()?;
        ai.filter = ai.filter.map(read).transpose()?;
        for test in &mut ai.tests {
            test.test = read(test.test.clone())?;
            test.written = read(test.written.clone())?;
        }
        Ok(Transformed::yes(crate::extension::extension_node(
            ai.checked()?,
        )))
    })
    .map(|done| done.data)
}

/// `expr` as the `named_struct`, `struct`, or `make_array` call of
/// literals it was folded from, when it is a struct or list literal.
fn unfold(expr: Expr) -> Transformed<Expr> {
    use datafusion::arrow::array::Array;
    use datafusion::logical_expr::expr::ScalarFunction as Call;
    let Expr::Literal(value, _) = &expr else {
        return Transformed::no(expr);
    };
    let element = |array: &dyn Array, index: usize| {
        ScalarValue::try_from_array(array, index)
            .map(|value| unfold(Expr::Literal(value, None)).data)
    };
    match value {
        ScalarValue::Struct(array) if array.len() == 1 => {
            let fields = array.fields();
            let positional = fields
                .iter()
                .enumerate()
                .all(|(index, field)| *field.name() == format!("c{index}"));
            let mut args = Vec::new();
            for (index, field) in fields.iter().enumerate() {
                let Ok(value) = element(array.column(index).as_ref(), 0) else {
                    return Transformed::no(expr);
                };
                if !positional {
                    args.push(Expr::Literal(
                        ScalarValue::Utf8(Some(field.name().clone())),
                        None,
                    ));
                }
                args.push(value);
            }
            let function = if positional {
                datafusion::functions::core::r#struct()
            } else {
                datafusion::functions::core::named_struct()
            };
            Transformed::yes(Expr::ScalarFunction(Call::new_udf(function, args)))
        }
        ScalarValue::List(array) if array.len() == 1 => {
            let values = array.value(0);
            let mut args = Vec::new();
            for index in 0..values.len() {
                let Ok(value) = element(values.as_ref(), index) else {
                    return Transformed::no(expr);
                };
                args.push(value);
            }
            Transformed::yes(Expr::ScalarFunction(Call::new_udf(
                datafusion::functions_nested::make_array::make_array_udf(),
                args,
            )))
        }
        _ => Transformed::no(expr),
    }
}

/// A model region's operators, read from its plan after DataFusion's
/// rules as the module explains.
///
/// # Errors
///
/// As [`derive_region`].
fn derive_operators(
    plan: &LogicalPlan,
    turn: &Turn,
    layout: &str,
) -> Result<Operators, CompileError> {
    let listed = with_select_list(plan);
    let top = read_top(&listed)?;
    refuse_grouped_input(top.relation)?;
    let fetch = read_fetch(&top).ok_or_else(|| unplaced(plan))?;
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
    let mut binder = Binder::new(turn, layout, &select);
    let relation = Relation::bind(&walk, &mut binder, &answers, &unnests)?;
    let Selected {
        mut columns,
        structs,
        output,
    } = binder.bind_select(&select)?;
    for (name, list) in unnests.iter().flat_map(|unnest| &unnest.lists) {
        let call = binder.bind_ai_sae_features(list)?;
        columns.push(ValueExpression::Alias(quail_physical::Alias {
            expression: Box::new(ValueExpression::ModelCall(call)),
            name: name.clone(),
        }));
    }
    let order = binder.bind_order(&order, &columns, &structs)?;
    binder.validate_query(&columns, &structs)?;
    let unplaced_term = relation.unplaced;
    let mut assembly = Assembly::new(&walk, &mut binder, relation, &columns, allow_relational)?;
    if assembly.place().is_none() {
        return Err(unplaced(plan));
    }
    let mut found = assembly.found.clone();
    if unplaced_term {
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
        tests: assembly.conditions.rows.clone(),
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
    found.fetch = fetch;
    if found
        .result
        .limit
        .or(found.fetch)
        .is_some_and(|limit| limit <= 0)
    {
        return compile_error("LIMIT must be a positive integer");
    }
    if found.result.offset < 0 {
        return compile_error("OFFSET must be a nonnegative integer");
    }
    if allow_relational && quail_planner::plan_statistics::plan_prompts(&found).is_empty() {
        return compile_error("the query has no AI predicate");
    }
    push_down_columns(&mut found, &walk, &assembly);
    Ok(found)
}

/// The refusal of a plan whose operators Quail's operators do not read.
fn unplaced(plan: &LogicalPlan) -> CompileError {
    CompileError::new(format!(
        "the placement of these operators {UNSUPPORTED}:\n{}",
        plan.display_indent()
    ))
}

/// Whether an operator is one that [`read_top`] reads above a SELECT
/// list: DataFusion's `push_down_limit` moves a LIMIT below the outer
/// SELECT list, and an ORDER BY of a column the SELECT list does not
/// return has one above it.
fn above_select_list(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Limit(_) | LogicalPlan::Sort(_) | LogicalPlan::Distinct(_)
    )
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
        LogicalPlan::Projection(projection) if above_select_list(&projection.input) => {
            let input = with_select_list(&projection.input);
            plan.with_new_exprs(plan.expressions(), vec![input])
                .unwrap_or_else(|_| plan.clone())
        }
        other => listed(other),
    }
}

// ---------------------------------------------------------------- the SELECT

/// The LIMIT, OFFSET, ORDER BY, DISTINCT, and SELECT list of one SELECT,
/// and the relation below them.
struct Top<'a> {
    limit: Option<i64>,
    offset: i64,
    distinct: bool,
    order: Vec<SortExpr>,
    /// The SELECT list, with an ORDER BY column it does not return read
    /// through the outer SELECT list.
    projection: Vec<Expr>,
    /// The input of the SELECT list.
    relation: &'a LogicalPlan,
}

/// Read the operators of a SELECT above its relation.
///
/// # Errors
///
/// The LIMIT or OFFSET is not a literal, the SELECT has a DISTINCT ON,
/// or it has no SELECT list.
fn read_top(plan: &LogicalPlan) -> Result<Top<'_>, CompileError> {
    let mut node = plan;
    let (mut limit, mut offset, mut distinct, mut order) = (None, 0, false, Vec::new());
    let mut outer_projection: Option<&Vec<Expr>> = None;
    loop {
        match node {
            LogicalPlan::Limit(found) => {
                use datafusion::logical_expr::{FetchType, SkipType};
                match found.get_skip_type() {
                    Ok(SkipType::Literal(skip)) => offset = skip as i64,
                    _ => return compile_error("OFFSET must be a nonnegative integer"),
                }
                match found.get_fetch_type() {
                    Ok(FetchType::Literal(fetch)) => limit = fetch.map(|fetch| fetch as i64),
                    _ => return compile_error("LIMIT must be a positive integer"),
                }
                node = &found.input;
            }
            LogicalPlan::Sort(found) => {
                // DataFusion moves a LIMIT over an ORDER BY into the sort
                if let Some(fetch) = found.fetch {
                    let fetch = fetch as i64;
                    limit = Some(limit.map_or(fetch, |found: i64| found.min(fetch)));
                }
                order = found.expr.clone();
                node = &found.input;
            }
            LogicalPlan::Distinct(Distinct::All(input)) => {
                distinct = true;
                node = input;
            }
            LogicalPlan::Distinct(Distinct::On(_)) => {
                return compile_error("DISTINCT ON is not supported; use DISTINCT");
            }
            LogicalPlan::Projection(found)
                if outer_projection.is_none() && above_select_list(&found.input) =>
            {
                outer_projection = Some(&found.expr);
                node = &found.input;
            }
            _ => break,
        }
    }
    let LogicalPlan::Projection(projection) = node else {
        return unsupported("a region without a SELECT list");
    };
    let mut expressions = projection.expr.clone();
    if let Some(outer) = outer_projection {
        // ORDER BY a column SELECT does not return: the inner
        // projection adds it, the outer one drops it; or a LIMIT that
        // DataFusion moved below the SELECT list
        expressions = outer
            .iter()
            .map(|expr| match expr {
                Expr::Column(column) => projection
                    .expr
                    .iter()
                    .find(|inner| output_name(inner) == column.name)
                    .cloned()
                    .unwrap_or_else(|| expr.clone()),
                other => other.clone(),
            })
            .collect();
    }
    Ok(Top {
        limit,
        offset,
        distinct,
        order,
        projection: expressions,
        relation: projection.input.as_ref(),
    })
}

/// The global fetch of the region's last `AiNode`, which a LIMIT became
/// (`PushDownLimitIntoAiNodes`); None, in the outer Option, when another
/// node has one, or an ORDER BY or DISTINCT reads the node's rows, as the
/// region's operators cannot place a LIMIT there.
fn read_fetch(top: &Top<'_>) -> Option<Option<i64>> {
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
    let reordered = fetch.is_some() && (!top.order.is_empty() || top.distinct);
    (!elsewhere && !reordered).then_some(fetch)
}

/// Refuse a SELECT whose input is a GROUP BY or a window.
///
/// DataFusion runs a GROUP BY or window above the region, or as an input
/// it computes for the region.
///
/// # Errors
///
/// The input groups or windows rows.
fn refuse_grouped_input(relation: &LogicalPlan) -> Result<(), CompileError> {
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

// ---------------------------------------------------------------- plan helpers

/// Whether an expression calls an AI function anywhere.
fn contains_ai_call(expr: &Expr) -> bool {
    expr.exists(|node| Ok(crate::is_ai_call(node)))
        .unwrap_or(false)
}

/// One `UNNEST` of `FROM`: each list's name and call, and whether a row
/// with no elements stays as one row of nulls.
struct Unnested {
    lists: Vec<(String, Expr)>,
    outer: bool,
}

/// The `UNNEST` a WHERE term of [`crate::take_unnests`] stands for, if it
/// is one.
fn unnest_term(term: &Expr) -> Option<Unnested> {
    let Expr::ScalarFunction(call) = term else {
        return None;
    };
    if call.name() != crate::UNNEST_TEST {
        return None;
    }
    let outer = matches!(
        call.args.first(),
        Some(Expr::Literal(ScalarValue::Boolean(Some(true)), _))
    );
    let lists = call.args[1..]
        .chunks(2)
        .filter_map(|pair| string_literal(&pair[0]).map(|name| (name, pair[1].clone())))
        .collect();
    Some(Unnested { lists, outer })
}

/// `expr` reading each unnested field as a column of its list's
/// relation: `quail_unnest_field(list, 'f', 'feature')` as `f.feature`.
fn unnested_fields(expr: &Expr) -> Expr {
    expr.clone()
        .transform_up(|node| {
            if let Expr::ScalarFunction(call) = &node
                && call.name() == crate::UNNEST_FIELD
                && let (Some(list), Some(field)) =
                    (string_literal(&call.args[1]), string_literal(&call.args[2]))
            {
                return Ok(datafusion::common::tree_node::Transformed::yes(
                    Expr::Column(Column::new(Some(list), field)),
                ));
            }
            Ok(datafusion::common::tree_node::Transformed::no(node))
        })
        .map(|found| found.data)
        .unwrap_or_else(|_| expr.clone())
}

/// The schema of an `AI.SAE_FEATURES` element, as Quail's `FEATURE_LIST`.
fn feature_schema() -> SchemaRef {
    std::sync::Arc::new(datafusion::arrow::datatypes::Schema::new(vec![
        datafusion::arrow::datatypes::Field::new("feature", DataType::Int32, true),
        datafusion::arrow::datatypes::Field::new("strength", DataType::Float32, true),
    ]))
}

/// Whether a plan reads a CTE or a FROM subquery.
fn has_subqueries(plan: &LogicalPlan) -> bool {
    plan.exists(|node| {
        Ok(matches!(node, LogicalPlan::SubqueryAlias(alias)
            if !matches!(alias.input.as_ref(), LogicalPlan::TableScan(_))))
    })
    .unwrap_or(false)
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

    /// Whether an EXISTS gate reads the table `alias`.
    fn is_gate(&self, alias: &str) -> bool {
        self.tables
            .iter()
            .any(|table| table.alias == alias && table.gate)
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

/// The relation's terms, bound in the order the query wrote them, each
/// where it runs.
#[derive(Default)]
struct Relation {
    /// alias -> its AI filters.
    filters: ByAlias<FilterPredicate>,
    /// Each AI join and gate, by its node.
    joins: BTreeMap<usize, JoinSpec>,
    /// alias -> its tests of a label: the classification, the labels it
    /// accepts, and the test's selectivity.
    label_tests: ByAlias<(ModelCall, Value, Option<f64>)>,
    /// The conditions that ask no model, and their HINT's selectivity.
    conditions: Vec<(ValueExpression, Option<f64>)>,
    /// alias -> the ON equalities of the JOIN of that table that no AI
    /// join has claimed, last JOIN first.
    equalities: Vec<(String, Vec<Equality>)>,
    /// The JOINed tables whose ON equalities an AI join has claimed.
    claimed: Vec<String>,
    /// The FROM table, then the JOINed tables.
    tables: Vec<String>,
    /// Whether a term the lowering left in place binds, which the
    /// operators cannot read.
    unplaced: bool,
}

/// Whether a term is `a.x = b.y`, an equality of two tables' columns.
fn is_column_equality(term: &Expr) -> bool {
    matches!(term, Expr::BinaryExpr(binary)
        if binary.op == datafusion::logical_expr::Operator::Eq
            && matches!((binary.left.as_ref(), binary.right.as_ref()),
                (Expr::Column(left), Expr::Column(right)) if left.relation != right.relation))
}

impl Relation {
    /// Bind the relation's tables and terms in the order the query wrote
    /// them: the tables, each JOIN's ON, the ON AI predicates, then WHERE.
    fn bind(
        walk: &Walk<'_>,
        binder: &mut Binder,
        answers: &[Answer],
        unnests: &[Unnested],
    ) -> Result<Self, CompileError> {
        let mut relation = Self::default();
        for table in walk.tables.iter().filter(|table| !table.gate) {
            binder.add_table(&table.alias, &table.provider, table.schema.clone())?;
            relation.tables.push(table.alias.clone());
        }
        for id in walk.ids() {
            let Kind::Join(join) = &walk.get(id).kind else {
                continue;
            };
            let right = &walk.get(walk.get(id).inputs[1]).tables;
            if right.iter().any(|alias| walk.is_gate(alias)) {
                continue;
            }
            let joined = right.last().cloned().unwrap_or_default();
            let mut terms: Vec<Expr> = join
                .on
                .iter()
                .map(|(left, right)| left.clone().eq(right.clone()))
                .collect();
            terms.extend(join.filter.iter().flat_map(split_conjunction).cloned());
            let mut equalities = Vec::new();
            for term in terms {
                let term = walk.written(&term, answers);
                // DataFusion pushes a WHERE test of the two sides into the
                // join; an equality of their columns is the join's key
                if is_column_equality(&term) {
                    equalities.push(binder.bind_equality(&term, &joined)?);
                } else {
                    relation.bind_where_term(binder, &term)?;
                }
            }
            if !equalities.is_empty() {
                relation.equalities.insert(0, (joined, equalities));
            }
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
            let written = node.written.as_ref().expect("an ON term's place");
            let num_asked = on
                .iter()
                .filter(|(_, other)| {
                    other.written.as_ref().map(|found| found.clause) == Some(written.clause)
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
            relation.bind_asked(walk, binder, id, node)?;
        }
        for (name, _) in unnests.iter().flat_map(|unnest| &unnest.lists) {
            binder.add_table(name, "", feature_schema())?;
        }
        for (id, node) in rest {
            relation.bind_asked(walk, binder, id, node)?;
        }
        for term in walk.where_terms(answers) {
            if unnest_term(&term).is_none() {
                relation.bind_where_term(binder, &unnested_fields(&term))?;
            }
        }
        Ok(relation)
    }

    /// Bind an AI filter, AI join, or gate.
    fn bind_asked(
        &mut self,
        walk: &Walk<'_>,
        binder: &mut Binder,
        id: usize,
        node: &AiNode,
    ) -> Result<(), CompileError> {
        let join = node.in_on().then_some(true);
        let call = walk.written(&node.call, &[]);
        if node.semantics != Semantics::Rows {
            let input = walk.get(walk.get(id).inputs[0]);
            let Kind::Join(_) = &input.kind else {
                self.unplaced = true;
                return Ok(());
            };
            let inner = walk.get(input.inputs[1]).tables.last().cloned();
            let table = walk
                .tables
                .iter()
                .find(|table| Some(&table.alias) == inner.as_ref())
                .expect("a gated table");
            binder.add_table(&table.alias, &table.provider, table.schema.clone())?;
            let (call, options) =
                binder.bind_ai_filter_call(&call, &["selectivity", "anchor"], Some(true))?;
            let anti = node.semantics == Semantics::Anti;
            let spec = gate(
                &call,
                &table.alias,
                anti,
                options.anchor,
                options.selectivity,
            )?;
            self.joins.insert(id, spec);
            return Ok(());
        }
        let (predicate, options) = if node.function() == "ai_score" {
            let answer = Answer {
                name: node.answer.clone(),
                call,
            };
            let term = walk.written(node.filter.as_ref().expect("a test"), &[answer]);
            let (test, options) = binder.bind_ai_score_test(&term, &["selectivity"], join)?;
            (Asked::Compare(test), options)
        } else {
            let (call, options) =
                binder.bind_ai_filter_call(&call, &["selectivity", "anchor"], join)?;
            (Asked::Call(call), options)
        };
        let read = call_aliases(asked_call(&predicate));
        if read.len() == 1 {
            if options.anchor.is_some() {
                return compile_error(
                    "anchor is a join option; a one-provider AI_FILTER takes only selectivity",
                );
            }
            let filter = FilterPredicate {
                expression: predicate,
                selectivity: options.selectivity,
            };
            push(&mut self.filters, &read[0], filter);
            return Ok(());
        }
        let spec = JoinSpec {
            predicate,
            semantics: "full",
            selectivity: options.selectivity,
            anchor: options.anchor,
            on: Vec::new(),
        };
        let (claimed, equalities) = (&mut self.claimed, &mut self.equalities);
        let spec = join_spec(spec, &read, &self.tables, claimed, equalities)?;
        self.joins.insert(id, spec);
        Ok(())
    }

    /// Bind a WHERE term that is no AI filter: a plain condition, a test
    /// of a label, or a test of another column operator's answer.
    fn bind_where_term(&mut self, binder: &mut Binder, term: &Expr) -> Result<(), CompileError> {
        if let Some(bound) = binder.bind_plain_where_term(term)? {
            self.conditions.extend(bound);
            return Ok(());
        }
        let (unhinted, hint) = binder.unwrap_hint(term)?;
        if let Some((classify, items)) = label_test(&unhinted) {
            let (call, labels, selectivity) = binder.bind_label_test(classify, &items, hint)?;
            let read = call_aliases(&call);
            if read.len() != 1 {
                return compile_error(format!(
                    "a filter on a label tests a one-document classification; this one \
                     classifies pairs of {}",
                    repr_tuple(&read)
                ));
            }
            push(&mut self.label_tests, &read[0], (call, labels, selectivity));
            return Ok(());
        }
        // the binder's refusal of the term, or a term it reads that the
        // lowering left in place
        binder.refuse_where_term(term)?;
        self.unplaced = true;
        Ok(())
    }
}

// ---------------------------------------------------------------- conditions

/// A condition that asks no model, with its HINT's selectivity and, for
/// a test of a label, the labels it accepts.
#[derive(Clone)]
struct Term {
    condition: ValueExpression,
    selectivity: Option<f64>,
    labels: Option<Value>,
}

/// The conditions of a relation, by where they run.
#[derive(Default)]
struct PlacedConditions {
    scans: ByAlias<ValueExpression>,
    computed: ByAlias<Term>,
    rows: Vec<ValueExpression>,
}

/// Place each condition where its columns exist: at a table's scan,
/// after the operator of one table that computes the columns it reads,
/// or on the result rows.
///
/// `computed` maps (alias, column) to the kind of the operator that
/// computes it; `unnested` holds the aliases of unnested lists.
fn place_conditions(
    conditions: Vec<Term>,
    computed: &BTreeMap<(String, String), &str>,
    unnested: &[String],
) -> Result<PlacedConditions, CompileError> {
    let mut placed = PlacedConditions::default();
    for term in conditions {
        let part = &term.condition;
        let references = column_references(part);
        let aliases: BTreeSet<&String> = references.iter().map(|found| &found.alias).collect();
        let owners: BTreeSet<Option<(&str, &String)>> = references
            .iter()
            .map(|found| {
                computed
                    .get(&(found.alias.clone(), found.column.clone()))
                    .map(|kind| (*kind, &found.column))
            })
            .collect();
        let kinds: BTreeSet<&str> = owners.iter().flatten().map(|(kind, _)| *kind).collect();
        let one_table = aliases.len() == 1 && !aliases.iter().any(|alias| unnested.contains(alias));
        if one_table && owners.len() == 1 && owners.contains(&None) {
            if term.selectivity.is_some() {
                return compile_error(format!(
                    "HINT estimates a test of a label or an answer; {} tests a source column, \
                     applied at the scan",
                    condition_text(part)
                ));
            }
            push(
                &mut placed.scans,
                aliases.first().expect("one alias"),
                part.clone(),
            );
        } else if one_table && kinds.len() == 1 {
            let alias = (*aliases.first().expect("one alias")).clone();
            push(&mut placed.computed, &alias, term);
        } else {
            if kinds.contains("label") && owners.iter().flatten().count() > 1 {
                return compile_error(format!(
                    "{} tests a label and another operator's column; test them in separate \
                     conditions",
                    condition_text(part)
                ));
            }
            if term.selectivity.is_some() {
                return compile_error(format!(
                    "HINT estimates a test that runs before a model call; {} runs on the result \
                     rows, after every model call",
                    condition_text(part)
                ));
            }
            placed.rows.push(term.condition);
        }
    }
    Ok(placed)
}

/// A list of names as Python writes a tuple of them.
fn repr_tuple(items: &[String]) -> String {
    let parts: Vec<String> = items.iter().map(|item| repr_str(item)).collect();
    match parts.as_slice() {
        [one] => format!("({one},)"),
        _ => format!("({})", parts.join(", ")),
    }
}

/// An EXISTS or NOT EXISTS gate over the table `inner`.
///
/// # Errors
///
/// The call does not read the inner table and one outer table, or the
/// anchor is not the outer table.
fn gate(
    call: &ModelCall,
    inner: &str,
    anti: bool,
    anchor: Option<String>,
    selectivity: Option<f64>,
) -> Result<JoinSpec, CompileError> {
    let aliases = call_aliases(call);
    let outer = match aliases.as_slice() {
        [first, second] if first == inner => second.clone(),
        [first, second] if second == inner => first.clone(),
        _ => {
            return compile_error(
                "the EXISTS predicate must reference the inner provider and exactly one outer \
                 provider",
            );
        }
    };
    if let Some(anchor) = anchor
        && anchor != outer
    {
        return compile_error(format!(
            "exists/anti always anchor on the outer table {} - the gate applies to its \
             documents - got anchor {}",
            repr_str(&outer),
            repr_str(&anchor)
        ));
    }
    Ok(JoinSpec {
        predicate: Asked::Call(call.clone()),
        semantics: if anti { "anti" } else { "exists" },
        selectivity,
        anchor: Some(outer),
        on: Vec::new(),
    })
}

/// An AI join over `aliases`, with the ON equalities of each JOINed
/// table it is the first to read.
///
/// # Errors
///
/// The predicate reads a table outside FROM and JOIN, its anchor is not
/// one of its tables, or an ON equality reads a table it does not.
fn join_spec(
    mut spec: JoinSpec,
    aliases: &[String],
    tables: &[String],
    claimed: &mut Vec<String>,
    equalities: &mut Vec<(String, Vec<Equality>)>,
) -> Result<JoinSpec, CompileError> {
    let outside: Vec<String> = aliases
        .iter()
        .filter(|alias| !tables.contains(alias))
        .cloned()
        .collect();
    if !outside.is_empty() {
        let mut sorted: Vec<String> = tables.to_vec();
        sorted.sort();
        sorted.dedup();
        return compile_error(format!(
            "the join prompt references {}, which are not the FROM table or JOINed tables of \
             this query ({})",
            repr_list(&outside),
            repr_list(&sorted)
        ));
    }
    if let Some(anchor) = &spec.anchor
        && !aliases.contains(anchor)
    {
        return compile_error(format!(
            "anchor {} is not a table of this join ({})",
            repr_str(anchor),
            repr_tuple(aliases)
        ));
    }
    for alias in &tables[1..] {
        if aliases.contains(alias) && !claimed.contains(alias) {
            claimed.push(alias.clone());
            if let Some(index) = equalities.iter().position(|(found, _)| found == alias) {
                spec.on.extend(equalities.remove(index).1);
            }
        }
    }
    if !spec.on.is_empty() && aliases.len() != 2 {
        return compile_error(format!(
            "join conditions are supported on two-table AI predicates; this one names {}",
            repr_tuple(aliases)
        ));
    }
    for condition in &spec.on {
        let sides = [&condition.left, &condition.right].map(column_alias_of);
        if !sides.iter().all(|side| aliases.contains(side)) {
            return compile_error(format!(
                "join condition {} names a table the AI predicate over {} does not",
                equality_text(condition),
                repr_tuple(aliases)
            ));
        }
    }
    Ok(spec)
}

/// Refuse JOINs that no AI predicate covers, or that do not connect to
/// the FROM table.
fn check_joins(
    equalities: &[(String, Vec<Equality>)],
    tables: &[String],
    joins: &[&JoinSpec],
    projected: &[(&ModelCall, String)],
) -> Result<(), CompileError> {
    let scores: Vec<&ModelCall> = projected
        .iter()
        .filter(|(call, _)| call.kind == "score")
        .map(|(call, _)| *call)
        .collect();
    if let Some((alias, conditions)) = equalities.first() {
        let first = equality_text(&conditions[0]);
        if scores
            .iter()
            .any(|score| call_aliases(score).contains(alias))
        {
            return compile_error(format!(
                "JOIN {alias} ON {first} projects a pair AI.SCORE; a projected pair score runs \
                 over CROSS JOIN, and an ON equality is supported when the score is compared \
                 in ON"
            ));
        }
        return unsupported(&format!(
            "JOIN {alias} ON {first} without an AI predicate over its pairs"
        ));
    }
    let mut predicates: Vec<BTreeSet<String>> = joins
        .iter()
        .filter(|join| join.semantics == "full")
        .map(|join| {
            call_aliases(asked_call(&join.predicate))
                .into_iter()
                .collect()
        })
        .collect();
    predicates.extend(
        scores
            .iter()
            .map(|score| call_aliases(score))
            .filter(|aliases| aliases.len() > 1)
            .map(|aliases| aliases.into_iter().collect()),
    );
    let (root, joined) = (&tables[0], &tables[1..]);
    let uncovered: Vec<String> = joined
        .iter()
        .filter(|alias| !predicates.iter().any(|found| found.contains(*alias)))
        .cloned()
        .collect();
    if !uncovered.is_empty() {
        return unsupported(&format!(
            "JOINed tables {} with no AI predicate over their pairs",
            repr_list(&uncovered)
        ));
    }
    let mut reached: BTreeSet<String> = BTreeSet::from([root.clone()]);
    let mut grew = true;
    while grew {
        grew = false;
        for found in &predicates {
            if !found.is_disjoint(&reached) && !found.is_subset(&reached) {
                reached.extend(found.iter().cloned());
                grew = true;
            }
        }
    }
    let disconnected: BTreeSet<&String> = joined
        .iter()
        .filter(|alias| !reached.contains(*alias))
        .collect();
    if !disconnected.is_empty() {
        let disconnected: Vec<String> = disconnected.into_iter().cloned().collect();
        return compile_error(format!(
            "join predicates do not connect {} to the FROM table {}: the predicates' tables \
             must form one connected graph with it",
            repr_list(&disconnected),
            repr_str(root)
        ));
    }
    Ok(())
}

/// Refuse AI.SCORE beside generative predicates, and a query that asks
/// no model.
fn check_calls(
    filters: &ByAlias<FilterPredicate>,
    joins: &[&JoinSpec],
    label_tests: &ByAlias<(ModelCall, Value, Option<f64>)>,
    projected: &[(&ModelCall, String)],
    tested: &[(ModelCall, String)],
    allow_relational: bool,
) -> Result<(), CompileError> {
    let scores: Vec<bool> = filters
        .iter()
        .flat_map(|(_, predicates)| predicates.iter().map(|found| &found.expression))
        .chain(joins.iter().map(|join| &join.predicate))
        .map(|asked| asked_call(asked).kind == "score")
        .chain(
            projected
                .iter()
                .filter(|(call, _)| call.kind == "score")
                .map(|_| true),
        )
        .collect();
    if scores.contains(&true) && scores.contains(&false) {
        return compile_error(
            "AI.SCORE cannot be mixed with generative AI predicates in one query",
        );
    }
    let asks = !(joins.is_empty()
        && filters.is_empty()
        && label_tests.is_empty()
        && projected.is_empty()
        && tested.is_empty());
    if !(allow_relational || asks) {
        return compile_error(
            "the query has no AI predicate; a plain scan belongs in the database the ids came \
             from",
        );
    }
    Ok(())
}

// ---------------------------------------------------------------- assembly

/// An AI join or gate, before it is placed.
#[derive(Debug, Clone)]
struct JoinSpec {
    predicate: Asked,
    semantics: &'static str,
    selectivity: Option<f64>,
    anchor: Option<String>,
    on: Vec<Equality>,
}

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

/// The region's operators: the bound terms, named, and placed where the
/// plan holds them.
struct Assembly<'w, 'a> {
    walk: &'w Walk<'a>,
    binder: &'w mut Binder,
    /// alias -> its AI filters, in written order.
    filters: ByAlias<FilterPredicate>,
    /// Each AI join and gate, by its node.
    joins: BTreeMap<usize, JoinSpec>,
    /// The call and name of each column operator that SELECT returns or a
    /// test reads, the first name of a call holding.
    names: Vec<(ModelCall, String)>,
    conditions: PlacedConditions,
    found: Operators,
    /// Each table's stack, by alias, in scan order.
    stacks: ByAlias<StackItem>,
    /// The aliases whose AI filters or tests are placed, in order.
    noted: Vec<String>,
    /// The ON equalities placed on each join's id.
    join_on: BTreeMap<usize, Vec<Equality>>,
    /// The column operators placed at each AI column's id: (kind, position).
    columns_at: BTreeMap<usize, Vec<(&'static str, usize)>>,
}

impl<'w, 'a> Assembly<'w, 'a> {
    /// Name each column operator and place the relation's conditions.
    ///
    /// `columns` is the SELECT list; `allow_relational` says whether the
    /// region reads a CTE or FROM subquery.
    ///
    /// # Errors
    ///
    /// A term does not fit the query's tables, or the query has no AI
    /// predicate where it needs one.
    fn new(
        walk: &'w Walk<'a>,
        binder: &'w mut Binder,
        relation: Relation,
        columns: &[ValueExpression],
        allow_relational: bool,
    ) -> Result<Self, CompileError> {
        let Relation {
            filters,
            joins,
            label_tests,
            mut conditions,
            equalities,
            tables,
            ..
        } = relation;
        let projected: Vec<(&ModelCall, String)> = columns
            .iter()
            .filter_map(|column| Some((aliased_call(column, &[])?, column_name(column, true))))
            .collect();
        let specs: Vec<&JoinSpec> = joins.values().collect();
        check_joins(&equalities, &tables, &specs, &projected)?;
        let tested = &binder.tested;
        check_calls(
            &filters,
            &specs,
            &label_tests,
            &projected,
            tested,
            allow_relational,
        )?;
        // a tested label takes SELECT's name, a test's, or one of its own
        let mut names: Vec<(ModelCall, String)> = Vec::new();
        for (alias, found) in &label_tests {
            let asks = lookup(&filters, alias).len();
            for (index, (call, _, _)) in found.iter().enumerate() {
                let name = projected
                    .iter()
                    .find(|(owner, _)| *owner == call)
                    .map(|(_, name)| name)
                    .or_else(|| {
                        tested
                            .iter()
                            .rev()
                            .find(|(owner, _)| owner == call)
                            .map(|(_, name)| name)
                    })
                    .cloned()
                    .unwrap_or_else(|| format!("__label_{alias}_{}", asks + index));
                names.push((call.clone(), name));
            }
        }
        names.extend(
            projected
                .iter()
                .map(|(call, name)| ((*call).clone(), name.clone())),
        );
        names.extend(tested.iter().cloned());
        let named = |call: &ModelCall| -> &String {
            let (_, name) = names
                .iter()
                .find(|(found, _)| found == call)
                .expect("a named call");
            name
        };
        // the columns each table's column operators add
        let mut computed: BTreeMap<(String, String), &str> = BTreeMap::new();
        for (call, _) in &names {
            let read = call_aliases(call);
            if read.len() == 1
                && ["label", "extract", "features", "map"].contains(&call.kind.as_str())
            {
                let name = named(call);
                computed.insert((read[0].clone(), name.clone()), &call.kind);
                if call.kind == "extract" {
                    computed.insert((read[0].clone(), format!("{name}{SPAN_SUFFIX}")), "extract");
                }
            }
        }
        let mut terms: Vec<Term> = conditions
            .drain(..)
            .map(|(condition, selectivity)| Term {
                condition,
                selectivity,
                labels: None,
            })
            .collect();
        for (alias, found) in &label_tests {
            for (call, labels, selectivity) in found {
                check_labels(named(call), labels, call)?;
                terms.push(Term {
                    condition: binder.label_condition(alias, named(call), labels)?,
                    selectivity: *selectivity,
                    labels: Some(labels.clone()),
                });
            }
        }
        let conditions = place_conditions(terms, &computed, &binder.unnested())?;
        Ok(Self {
            walk,
            binder,
            filters,
            joins,
            names,
            conditions,
            found: Operators::default(),
            stacks: Vec::new(),
            noted: Vec::new(),
            join_on: BTreeMap::new(),
            columns_at: BTreeMap::new(),
        })
    }

    /// Whether `id` is an AI filter of one table.
    fn is_table_filter(&self, id: usize) -> bool {
        matches!(&self.walk.get(id).kind, Kind::Ai(node)
            if node.is_predicate()
                && node.semantics == Semantics::Rows
                && self.walk.call_tables(&node.call).len() < 2)
    }

    /// Whether the executor's stop key of `alias`'s AI.IF filters keeps
    /// a per-key fetch's rows: the executor stops a key once a document
    /// passes the table's filters, so the region asks only those filters
    /// and joins nothing.
    fn stops_per_key(&self, alias: &str) -> bool {
        self.walk.ids().all(|id| match &self.walk.get(id).kind {
            Kind::Ai(node) => {
                node.function() == "ai_filter"
                    && self.is_table_filter(id)
                    && self.call_tables(id).iter().eq([alias])
            }
            Kind::Join(_) => false,
            _ => true,
        })
    }

    fn call_tables(&self, id: usize) -> BTreeSet<String> {
        match &self.walk.get(id).kind {
            Kind::Ai(node) => self.walk.call_tables(&node.call),
            _ => BTreeSet::new(),
        }
    }

    /// The `index`th input of `id`, passing through projections, and
    /// through plain filters when `filters` says so.
    fn input(&self, id: usize, index: usize, filters: bool) -> Option<usize> {
        let mut below = *self.walk.get(id).inputs.get(index)?;
        loop {
            let visit = self.walk.get(below);
            match &visit.kind {
                Kind::PassThrough if !visit.inputs.is_empty() => below = visit.inputs[0],
                Kind::Filter(_) if filters && !visit.inputs.is_empty() => below = visit.inputs[0],
                _ => return Some(below),
            }
        }
    }

    /// Place the region's operators in the plan's order; None when the
    /// plan holds an operator Quail's operators do not read here.
    fn place(&mut self) -> Option<()> {
        if self.walk.unplaced {
            return None;
        }
        for id in self.walk.ids() {
            match &self.walk.get(id).kind {
                Kind::Scan(table) => {
                    let alias = &self.walk.tables[*table].alias;
                    self.found.scans.push(Scan {
                        predicates: lookup(&self.conditions.scans, alias).to_vec(),
                        ..self.binder.scan(alias)
                    });
                    self.stacks.push((alias.clone(), Vec::new()));
                }
                Kind::Filter(predicate) if contains_ai_call(predicate) => return None,
                Kind::Ai(node) if node.is_predicate() || node.semantics != Semantics::Rows => {
                    if !self.is_table_filter(id) {
                        self.place_join(id)?;
                    } else if !self.walk.get(id).parent.is_some_and(|parent| {
                        self.is_table_filter(parent)
                            && self.call_tables(parent) == self.call_tables(id)
                    }) {
                        self.place_filters(id)?;
                    }
                }
                Kind::Ai(node) => self.place_column(id, node)?,
                _ => {}
            }
        }
        if self.found.joins.len() != self.joins.len() {
            return None;
        }
        self.place_tests()
    }

    /// Place a run of AI filters of one table, `top` the last.
    fn place_filters(&mut self, top: usize) -> Option<()> {
        let read = self.call_tables(top);
        let mut run = vec![top];
        let mut below = self.walk.get(top).inputs[0];
        while self.is_table_filter(below) && self.call_tables(below) == read {
            run.push(below);
            below = self.walk.get(below).inputs[0];
        }
        let [alias] = <[String; 1]>::try_from(read.into_iter().collect::<Vec<_>>()).ok()?;
        let predicates = lookup(&self.filters, &alias).to_vec();
        if predicates.len() != run.len() {
            return None;
        }
        let before_joins = !self.walk.get(below).ai_join;
        let stop_key = run.iter().find_map(|id| match &self.walk.get(*id).kind {
            Kind::Ai(AiNode {
                fetch: Some(Fetch::PerKey { key, n: 1 }),
                ..
            }) if self.stops_per_key(&alias) => Some(key.clone()),
            _ => None,
        });
        for predicate in predicates {
            push(&mut self.found.filters, &alias, predicate);
            push(&mut self.found.filters_before_joins, &alias, before_joins);
        }
        if let Some(key) = stop_key {
            self.found.stop_keys.insert(alias.clone(), key);
        }
        note(&mut self.noted, &alias);
        push(&mut self.stacks, &alias, StackItem::Filters);
        Some(())
    }

    fn place_join(&mut self, id: usize) -> Option<()> {
        let spec = self.joins.get(&id)?;
        let read: BTreeSet<String> = call_aliases(asked_call(&spec.predicate))
            .into_iter()
            .collect();
        if read != self.call_tables(id) {
            return None;
        }
        let input = self.input(id, 0, true)?;
        let on = match &self.walk.get(input).kind {
            Kind::Join(_) => spec.on.clone(),
            _ => Vec::new(),
        };
        self.join_on.insert(input, on.clone());
        let mut outer = input;
        while let Kind::Join(_) = &self.walk.get(outer).kind {
            outer = self.input(outer, 0, false)?;
        }
        let mut outer_aliases: Vec<String> = Vec::new();
        for alias in &self.walk.get(outer).tables {
            note(&mut outer_aliases, alias);
        }
        self.found.joins.push(SemanticJoin {
            predicate: spec.predicate.clone(),
            semantics: spec.semantics.to_string(),
            selectivity: spec.selectivity,
            anchor: spec.anchor.clone(),
            on,
            outer_aliases,
        });
        Some(())
    }

    /// The column operators an AI column node runs: its call, or an
    /// extraction's call per field, each with its name.
    fn bound_columns(&mut self, node: &AiNode) -> Option<(&'static str, Vec<(ModelCall, String)>)> {
        let kind = column_call_kind(&node.call)?;
        let call = self.walk.written(&node.call, &[]);
        let calls = match kind {
            "extract" => {
                let (_, members) = self.binder.bind_ai_extract(&call, &node.answer).ok()?;
                members
                    .into_iter()
                    .map(|member| match member {
                        ValueExpression::Alias(alias) => match *alias.expression {
                            ValueExpression::ModelCall(call) => Some(call),
                            _ => None,
                        },
                        _ => None,
                    })
                    .collect::<Option<Vec<ModelCall>>>()?
            }
            _ => vec![self.binder.bind_column_call(&call, kind).ok()?],
        };
        let named = calls
            .into_iter()
            .map(|call| {
                let (_, name) = self.names.iter().find(|(found, _)| *found == call)?;
                Some((call, name.clone()))
            })
            .collect::<Option<Vec<_>>>()?;
        Some((kind, named))
    }

    fn place_column(&mut self, id: usize, node: &AiNode) -> Option<()> {
        let read: Vec<String> = self.call_tables(id).into_iter().collect();
        let is_column = |kind: &Kind| matches!(kind, Kind::Ai(found) if !found.is_predicate() && found.semantics == Semantics::Rows);
        if read.len() == 2 {
            // a run of classifications of the same joined rows is placed
            // from its top, once
            let continues = |id: usize| {
                self.walk.get(id).parent.is_some_and(|parent| {
                    is_column(&self.walk.get(parent).kind)
                        && self.call_tables(parent) == self.call_tables(id)
                })
            };
            if continues(id) {
                return Some(());
            }
            let mut run = vec![(id, node)];
            let mut below = self.walk.get(id).inputs[0];
            while let Kind::Ai(next) = &self.walk.get(below).kind
                && is_column(&self.walk.get(below).kind)
                && self.call_tables(below) == self.call_tables(id)
            {
                run.push((below, *next));
                below = self.walk.get(below).inputs[0];
            }
            let above_joins = self.walk.get(below).ai_join;
            for (at, found) in run {
                let (_, columns) = self.bound_columns(found)?;
                for (call, name) in columns {
                    self.found.classifies.push(SemanticColumn {
                        call,
                        name,
                        above_joins,
                    });
                    let position = self.found.classifies.len() - 1;
                    self.columns_at
                        .entry(at)
                        .or_default()
                        .push(("label", position));
                }
            }
            return Some(());
        }
        let [alias] = <[String; 1]>::try_from(read).ok()?;
        let above_joins = self.walk.get(self.walk.get(id).inputs[0]).ai_join;
        let (kind, columns) = self.bound_columns(node)?;
        for (call, name) in columns {
            let list = self.list(kind);
            list.push(SemanticColumn {
                call,
                name,
                above_joins,
            });
            let position = list.len() - 1;
            self.columns_at
                .entry(id)
                .or_default()
                .push((kind, position));
            push(&mut self.stacks, &alias, StackItem::Column(kind, position));
        }
        Some(())
    }

    fn list(&mut self, kind: &str) -> &mut Vec<SemanticColumn> {
        match kind {
            "label" => &mut self.found.classifies,
            "extract" => &mut self.found.extracts,
            "map" => &mut self.found.maps,
            _ => &mut self.found.sae_features,
        }
    }

    fn column(&self, kind: &str, position: usize) -> &SemanticColumn {
        let list = match kind {
            "label" => &self.found.classifies,
            "extract" => &self.found.extracts,
            "map" => &self.found.maps,
            _ => &self.found.sae_features,
        };
        &list[position]
    }

    /// Each table's tests of answers, on its stack, and the operators'
    /// column filters.
    fn place_tests(&mut self) -> Option<()> {
        let mut stacks = std::mem::take(&mut self.stacks);
        for (alias, stack) in &mut stacks {
            for (index, term) in lookup(&self.conditions.computed, alias).iter().enumerate() {
                let position = self.test_position(stack, &term.condition);
                stack.insert(position, StackItem::Test(index));
                note(&mut self.noted, alias);
            }
        }
        for alias in &self.noted {
            let stack = lookup(&stacks, alias);
            let computed = lookup(&self.conditions.computed, alias);
            let mut tests: Vec<(usize, &SemanticColumn)> = Vec::new();
            for (at, item) in stack.iter().enumerate() {
                let StackItem::Test(index) = item else {
                    continue;
                };
                let read = condition_columns(&computed[*index].condition);
                let producer = stack[..at].iter().rev().find_map(|below| match below {
                    StackItem::Column(kind, position) => {
                        let column = self.column(kind, *position);
                        let adds = column.added_columns();
                        adds.iter()
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
            let asks = lookup(&self.found.filters, alias).len();
            let column_tests = tests
                .iter()
                .enumerate()
                .map(|(order, (index, producer))| ColumnTest {
                    name: producer.name.clone(),
                    call: producer.call.clone(),
                    condition: computed[*index].condition.clone(),
                    labels: computed[*index].labels.clone(),
                    selectivity: computed[*index].selectivity,
                    position: asks + order,
                    before_joins: true,
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
    fn test_position(&self, stack: &[StackItem], condition: &ValueExpression) -> usize {
        let reads: BTreeSet<String> = condition_columns(condition)
            .into_iter()
            .filter(|column| self.stack_columns(stack).contains(column))
            .collect();
        let mut top = stack.len();
        while let Some(StackItem::Column(..)) = top.checked_sub(1).map(|at| &stack[at]) {
            let below = &stack[..top - 1];
            let descends = if reads.is_empty() {
                self.has_answer_column(below)
            } else {
                reads.is_subset(&self.stack_columns(below))
            };
            if !descends {
                break;
            }
            top -= 1;
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
                StackItem::Filters => return false,
                _ => {}
            }
        }
        false
    }
}

/// Values kept per table alias, in the order the aliases first appear.
type ByAlias<T> = Vec<(String, Vec<T>)>;

/// Add `value` to `key`'s values, keeping keys in first use order.
fn push<V>(map: &mut ByAlias<V>, key: &str, value: V) {
    match map.iter_mut().find(|(found, _)| found == key) {
        Some((_, values)) => values.push(value),
        None => map.push((key.to_string(), vec![value])),
    }
}

fn lookup<'a, V>(map: &'a ByAlias<V>, key: &str) -> &'a [V] {
    map.iter()
        .find(|(found, _)| found == key)
        .map_or(&[], |(_, values)| values.as_slice())
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
    tests: Vec<ValueExpression>,
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
        let tests: Vec<ValueExpression> = self
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
        let place = |chain: &mut Vec<RowOperator>,
                     names: &[String],
                     remaining: &mut Vec<ValueExpression>| {
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

/// A column of the rows below as result rows hold it: a member of a
/// struct column as the struct's field.
fn result_reference(
    reference: &ColumnRef,
    members: &BTreeMap<String, (String, String)>,
) -> ColumnRef {
    match members.get(&reference.column) {
        Some((owner, name)) => ColumnRef {
            alias: owner.clone(),
            provider: String::new(),
            column: name.clone(),
        },
        None => reference.clone(),
    }
}

fn rewrite_reference(
    expression: &ValueExpression,
    members: &BTreeMap<String, (String, String)>,
) -> ValueExpression {
    match expression {
        ValueExpression::ColumnRef(found) => {
            ValueExpression::ColumnRef(result_reference(found, members))
        }
        ValueExpression::GetField(field) => ValueExpression::GetField(GetField {
            input: Box::new(rewrite_reference(&field.input, members)),
            field: field.field.clone(),
        }),
        ValueExpression::Value(value) => ValueExpression::Value(Expression {
            columns: value
                .columns
                .iter()
                .map(|found| result_reference(found, members))
                .collect(),
            ..value.clone()
        }),
        ValueExpression::Alias(alias) => ValueExpression::Alias(quail_physical::Alias {
            expression: Box::new(rewrite_reference(&alias.expression, members)),
            name: alias.name.clone(),
        }),
        other => other.clone(),
    }
}

/// Rewrite a condition to read result rows, where struct columns hold answers.
pub(crate) fn result_row_condition(
    condition: &ValueExpression,
    packing: &[(String, Vec<(String, String)>)],
) -> ValueExpression {
    let members: BTreeMap<String, (String, String)> = packing
        .iter()
        .flat_map(|(owner, names)| {
            names
                .iter()
                .map(move |(name, member)| (member.clone(), (owner.clone(), name.clone())))
        })
        .collect();
    rewrite_reference(condition, &members)
}

/// The result row columns a rewritten condition reads, by name.
pub(crate) fn result_columns_read(condition: &ValueExpression) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for reference in column_references(condition) {
        let name = reference_name(&reference, true);
        if !found.contains(&name) {
            found.push(name);
        }
    }
    found
}

/// The source columns the row tests read that the SELECT list does not
/// return, which it adds for them.
pub(crate) fn row_test_columns(
    row_tests: &[ValueExpression],
    unnested: &BTreeSet<String>,
    source: &[ValueExpression],
    returned: &[String],
) -> Vec<ValueExpression> {
    let mut present: BTreeSet<String> = returned.iter().cloned().collect();
    present.extend(source.iter().map(|column| column_name(column, true)));
    let mut extra = Vec::new();
    for test in row_tests {
        for reference in column_references(test) {
            let name = reference_name(&reference, true);
            if !reference.provider.is_empty()
                && !unnested.contains(&reference.alias)
                && !present.contains(&name)
            {
                present.insert(name);
                extra.push(ValueExpression::ColumnRef(reference));
            }
        }
    }
    extra
}

/// Refuse a DISTINCT or ORDER BY that compares AI.SAE_FEATURES lists.
///
/// # Errors
///
/// The query compares a feature list that is not unnested.
pub(crate) fn check_feature_lists(
    columns: &[ValueExpression],
    order: &[SortKey],
    distinct: bool,
    unnested: &BTreeSet<String>,
) -> Result<(), CompileError> {
    let names: BTreeSet<String> = columns
        .iter()
        .filter(|column| aliased_call(column, &["features"]).is_some())
        .map(|column| column_name(column, true))
        .filter(|name| !unnested.contains(name))
        .collect();
    if names.is_empty() {
        return Ok(());
    }
    if distinct {
        return compile_error("DISTINCT does not compare AI.SAE_FEATURES list columns");
    }
    for key in order {
        let refused = match &key.expression {
            SortExpression::Value(expression) => {
                let named = matches!(expression, ValueExpression::Alias(alias)
                    if names.contains(&alias.name));
                named
                    || aliased_call(expression, &["features"]).is_some()
                    || matches!(expression, ValueExpression::ModelCall(call)
                        if call.kind == "features")
            }
        };
        if refused {
            return compile_error("ORDER BY does not compare AI.SAE_FEATURES list columns");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- scan columns

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
fn push_down_columns(found: &mut Operators, walk: &Walk<'_>, assembly: &Assembly<'_, '_>) {
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
            RowOperator::Filter { condition, .. } => top.add(column_references(condition)),
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
    for on in assembly.join_on.values() {
        for equality in on {
            value(equality_refs(equality));
        }
    }
    for scan in &found.scans {
        for predicate in &scan.predicates {
            value(column_references(predicate));
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
        let stack = &assembly.stacks[index].1;
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
            if let Some(columns) = assembly.columns_at.get(id)
                && walk
                    .call_tables(match &visit.kind {
                        Kind::Ai(node) => &node.call,
                        _ => unreachable!("a column at an AI node"),
                    })
                    .len()
                    == 2
            {
                for (kind, position) in columns {
                    joined_run.push(assembly.column(kind, *position).clone());
                }
                continue;
            }
            flush(&mut needed, &mut joined_run);
            if let Some(spec) = assembly.joins.get(id) {
                needed.add(asked_refs(&spec.predicate));
            }
            if let Some(on) = assembly.join_on.get(id) {
                for equality in on {
                    needed.add(equality_refs(equality));
                }
            }
            if let Some(columns) = assembly.columns_at.get(id) {
                for (kind, position) in columns.iter().rev() {
                    let column = assembly.column(kind, *position).clone();
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
        let computed = lookup(&assembly.conditions.computed, &alias);
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
                    let column = assembly.column(kind, *position).clone();
                    needed.add(call_refs(&column.call));
                    needed.remove(&alias, &column.added_columns());
                }
                StackItem::Test(index) => {
                    needed.add(column_references(&computed[*index].condition));
                }
            }
        }
        let scan = &found.scans[index];
        for predicate in &scan.predicates {
            needed.add(column_references(predicate));
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
