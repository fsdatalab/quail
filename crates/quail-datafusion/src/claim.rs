//! The `ClaimModelRegions` analyzer rule.

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::datasource::{MemTable, provider_as_source};
use datafusion::logical_expr::{
    Aggregate, Expr, Extension, LogicalPlan, LogicalPlanBuilder, Projection, Window,
};
use datafusion::optimizer::AnalyzerRule;

use crate::functions::is_ai_call;
use crate::region::ModelRegion;

/// Replace each subtree of AI work that Quail can plan with one
/// `ModelRegion`, and leave the relational operators above it to
/// DataFusion.
///
/// The rule walks the plan from the root. DataFusion keeps an
/// `Aggregate`, a `Window`, and a `Sort` with no `LIMIT` above it, each
/// AI call of theirs moved into a `Projection` below them under a name
/// they then read; and a `Projection`, `Filter`, `Sort`, `Limit`,
/// `Distinct`, or `SubqueryAlias` that calls no AI function above one of
/// those. The first other operator whose whole subtree Quail can plan,
/// and which calls an AI function somewhere in that subtree, becomes a
/// model region; the rule does not look below it. A `Limit`, its
/// `Sort`, and a `Distinct` directly over the AI work stay in the
/// region, where Quail can stop the model early. A region's root is a
/// SELECT list: a region with none gets a `Projection` of the columns
/// the operators above it read.
#[derive(Debug, Default)]
pub struct ClaimModelRegions;

impl ClaimModelRegions {
    /// Create the rule.
    pub fn new() -> Self {
        Self
    }
}

impl AnalyzerRule for ClaimModelRegions {
    fn name(&self) -> &str {
        "claim_model_regions"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        let mut extracted = Extracted::default();
        // a moved call keeps the name the SELECT list gives it
        plan.apply(|node| {
            if let LogicalPlan::Projection(projection) = node {
                for expr in &projection.expr {
                    // the call itself, or the column a GROUP BY makes of it
                    let Expr::Alias(alias) = expr else { continue };
                    let named = match alias.expr.as_ref() {
                        Expr::Column(column) => Some(column.name.clone()),
                        inner if is_ai_call(inner) => Some(inner.schema_name().to_string()),
                        _ => None,
                    };
                    if let Some(named) = named {
                        extracted.aliases.push((named, alias.name.clone()));
                    }
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        extracted.claim(plan, None, false)
    }
}

/// The AI calls moved out of DataFusion's operators so far.
#[derive(Debug, Default)]
struct Extracted {
    num_calls: usize,
    /// The SELECT list's names, by the name of the call or column each
    /// one renames.
    aliases: Vec<(String, String)>,
    /// The names given to moved calls.
    used: Vec<String>,
}

impl Extracted {
    /// Move each AI call of an `Aggregate`, `Window`, or `Sort` into a
    /// `Projection` below it.
    fn extract(&mut self, node: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
        if !matches!(
            node,
            LogicalPlan::Aggregate(_) | LogicalPlan::Window(_) | LogicalPlan::Sort(_)
        ) {
            return Ok(Transformed::no(node));
        }
        let mut calls: Vec<Expr> = Vec::new();
        for expr in node.expressions() {
            for call in ai_calls(&expr)? {
                if !calls.contains(&call) {
                    calls.push(call);
                }
            }
        }
        if calls.is_empty() {
            return Ok(Transformed::no(node));
        }
        let named = self.name(calls);
        let read = |expr: &Expr| read_calls(expr, &named);
        let exprs: Vec<Expr> = node.expressions().iter().map(read).collect::<Result<_>>()?;
        // the projection passes on the input columns an aggregate reads,
        // or every input column a window or sort passes on
        let input = node.inputs()[0].clone();
        let mut read_columns: Vec<Expr> = Vec::new();
        if !matches!(node, LogicalPlan::Aggregate(_)) {
            read_columns.extend(input.schema().columns().into_iter().map(Expr::Column));
        }
        for expr in &exprs {
            for column in expr.column_refs() {
                if input.schema().has_column(column) {
                    let column = Expr::Column(column.clone());
                    if !read_columns.contains(&column) {
                        read_columns.push(column);
                    }
                }
            }
        }
        read_columns.extend(named.iter().map(|(call, name)| call.clone().alias(name)));
        let projection =
            LogicalPlan::Projection(Projection::try_new(read_columns, Arc::new(input))?);
        let rewritten = match &node {
            LogicalPlan::Aggregate(aggregate) => {
                // the operators above read each output by its original name
                let keep = |exprs: &[Expr]| -> Result<Vec<Expr>> {
                    exprs
                        .iter()
                        .map(|expr| read(expr)?.alias_if_changed(expr.schema_name().to_string()))
                        .collect()
                };
                LogicalPlan::Aggregate(Aggregate::try_new(
                    Arc::new(projection),
                    keep(&aggregate.group_expr)?,
                    keep(&aggregate.aggr_expr)?,
                )?)
            }
            LogicalPlan::Window(window) => {
                let keep = window
                    .window_expr
                    .iter()
                    .map(|expr| read(expr)?.alias_if_changed(expr.schema_name().to_string()))
                    .collect::<Result<Vec<_>>>()?;
                LogicalPlan::Window(Window::try_new(keep, Arc::new(projection))?)
            }
            _ => node.with_new_exprs(exprs, vec![projection])?,
        };
        Ok(Transformed::yes(rewritten))
    }

    /// Move the AI calls of a `Projection` over a `Window`, and the
    /// window's own, into a `Projection` below the window, when each
    /// reads only the window's input columns; the window passes their
    /// columns on. A projection that reads a window value in an AI call
    /// is unchanged.
    fn push_below_window(&mut self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let LogicalPlan::Projection(projection) = &plan else {
            return Ok(plan);
        };
        let LogicalPlan::Window(window) = projection.input.as_ref() else {
            return Ok(plan);
        };
        let below = window.input.schema();
        let mut calls: Vec<Expr> = Vec::new();
        for expr in projection.expr.iter().chain(&window.window_expr) {
            for call in ai_calls(expr)? {
                if !call
                    .column_refs()
                    .iter()
                    .all(|column| below.has_column(column))
                {
                    return Ok(plan);
                }
                if !calls.contains(&call) {
                    calls.push(call);
                }
            }
        }
        if calls.is_empty() {
            return Ok(plan);
        }
        let named = self.name(calls);
        let mut exprs: Vec<Expr> = below.columns().into_iter().map(Expr::Column).collect();
        exprs.extend(named.iter().map(|(call, name)| call.clone().alias(name)));
        let input = LogicalPlan::Projection(Projection::try_new(exprs, Arc::clone(&window.input))?);
        let keep = |expr: &Expr| -> Result<Expr> {
            read_calls(expr, &named)?.alias_if_changed(expr.schema_name().to_string())
        };
        let window = Window::try_new(
            window.window_expr.iter().map(keep).collect::<Result<_>>()?,
            Arc::new(input),
        )?;
        let exprs = projection.expr.iter().map(keep).collect::<Result<_>>()?;
        Ok(LogicalPlan::Projection(Projection::try_new(
            exprs,
            Arc::new(LogicalPlan::Window(window)),
        )?))
    }

    /// A column name for each call: the SELECT list's name for it, or a
    /// new hidden one.
    fn name(&mut self, calls: Vec<Expr>) -> Vec<(Expr, String)> {
        calls
            .into_iter()
            .map(|call| {
                let key = call.schema_name().to_string();
                let named = self
                    .aliases
                    .iter()
                    .find(|(found, name)| *found == key && !self.used.contains(name))
                    .map(|(_, name)| name.clone());
                let name = named.unwrap_or_else(|| {
                    self.num_calls += 1;
                    format!("__quail_call_{}", self.num_calls)
                });
                self.used.push(name.clone());
                (call, name)
            })
            .collect()
    }

    /// Claim the model regions of `plan`, as [`ClaimModelRegions`] explains.
    ///
    /// `needed` holds the columns the operator above reads from `plan`, or
    /// None when it reads every one; `limited` says a `Limit` is directly
    /// above.
    fn claim(
        &mut self,
        plan: LogicalPlan,
        needed: Option<Vec<Column>>,
        limited: bool,
    ) -> Result<LogicalPlan> {
        if !calls_ai(&plan)? {
            return Ok(plan);
        }
        let plan = self.push_below_window(plan)?;
        if kept_by_datafusion(&plan, limited)? {
            let plan = match plan {
                LogicalPlan::Aggregate(_) | LogicalPlan::Window(_) | LogicalPlan::Sort(_) => {
                    self.extract(plan)?.data
                }
                plan => plan,
            };
            let inputs: Vec<LogicalPlan> = plan.inputs().into_iter().cloned().collect();
            let limit = limits(&plan, limited);
            let claimed = inputs
                .into_iter()
                .map(|input| {
                    let read = read_from(&plan, &input, needed.as_deref());
                    self.claim(input, read, limit)
                })
                .collect::<Result<Vec<_>>>()?;
            return plan.with_new_exprs(plan.expressions(), claimed);
        }
        if quail_plans(&plan)? {
            let (bound, inputs) = split_inputs(select_list(plan, needed)?)?;
            let mut region = ModelRegion::new(bound);
            for (name, input) in inputs {
                region.inputs.push(self.claim(input, None, false)?);
                region.input_names.push(name);
            }
            return Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(region),
            }));
        }
        plan.map_children(|child| {
            let read = child.schema().columns();
            self.claim(child, Some(read), false).map(Transformed::yes)
        })
        .map(|claimed| claimed.data)
    }
}

/// Whether DataFusion runs this operator, as [`ClaimModelRegions`] explains.
fn kept_by_datafusion(plan: &LogicalPlan, limited: bool) -> Result<bool> {
    // an aggregate, window, or unlimited sort runs in DataFusion with its
    // AI calls moved below it
    match plan {
        LogicalPlan::Aggregate(_) | LogicalPlan::Window(_) => return Ok(true),
        // a sort with a fetch is a LIMIT, which DataFusion's planner folds in
        LogicalPlan::Sort(sort) if !limited && sort.fetch.is_none() => return Ok(true),
        _ => {}
    }
    for expr in plan.expressions() {
        if expr.exists(|inner| Ok(is_ai_call(inner)))? {
            return Ok(false);
        }
    }
    Ok(match plan {
        LogicalPlan::Projection(_)
        | LogicalPlan::Filter(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Distinct(_)
        | LogicalPlan::SubqueryAlias(_) => {
            let input = plan.inputs()[0];
            calls_ai(input)? && kept_by_datafusion(input, limits(plan, limited))?
        }
        _ => false,
    })
}

/// The outermost AI calls in `expr`, each once; an unnested field counts
/// whole, with the AI call it reads.
fn ai_calls(expr: &Expr) -> Result<Vec<Expr>> {
    let mut calls: Vec<Expr> = Vec::new();
    expr.apply(|inner| {
        let unnested = matches!(inner, Expr::ScalarFunction(call)
            if call.name() == crate::UNNEST_FIELD && inner.exists(|e| Ok(is_ai_call(e)))?);
        if unnested || is_ai_call(inner) {
            if !calls.contains(inner) {
                calls.push(inner.clone());
            }
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(calls)
}

/// `expr` with each named call replaced by its column.
fn read_calls(expr: &Expr, named: &[(Expr, String)]) -> Result<Expr> {
    Ok(expr
        .clone()
        .transform_down(
            |inner| match named.iter().find(|(call, _)| *call == inner) {
                Some((_, name)) => Ok(Transformed::new(
                    Expr::Column(Column::from_name(name)),
                    true,
                    TreeNodeRecursion::Jump,
                )),
                None => Ok(Transformed::no(inner)),
            },
        )?
        .data)
}

/// Whether a `Limit` is above `plan`'s input: `plan` is one, or passes
/// one on from above without changing the rows.
fn limits(plan: &LogicalPlan, limited: bool) -> bool {
    match plan {
        LogicalPlan::Limit(_) => true,
        LogicalPlan::Projection(_) | LogicalPlan::SubqueryAlias(_) => limited,
        _ => false,
    }
}

/// The columns of `input` that `plan` reads, when the operator above
/// `plan` reads `needed` of its columns.
fn read_from(
    plan: &LogicalPlan,
    input: &LogicalPlan,
    needed: Option<&[Column]>,
) -> Option<Vec<Column>> {
    let mut read: Vec<Column> = Vec::new();
    let mut add = |column: Column| {
        if !read.contains(&column) {
            read.push(column);
        }
    };
    for expr in plan.expressions() {
        for column in expr.column_refs() {
            add(column.clone());
        }
    }
    match plan {
        // these pass their input's columns on
        LogicalPlan::Filter(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Distinct(_)
        | LogicalPlan::Window(_) => needed?.iter().cloned().for_each(&mut add),
        LogicalPlan::SubqueryAlias(_) => {
            // the alias renames the qualifier; the input's columns keep their names
            let names: Vec<String> = needed?.iter().map(|column| column.name.clone()).collect();
            for (qualifier, field) in input.schema().iter() {
                if names.contains(field.name()) {
                    add(Column::new(qualifier.cloned(), field.name()));
                }
            }
        }
        _ => {}
    }
    Some(
        read.into_iter()
            .filter(|column| input.schema().has_column(column))
            .collect(),
    )
}

/// Take each CTE or FROM subquery that groups or windows rows out of a
/// region's plan: the plan reads it as a table of the same name, and
/// DataFusion computes it as the region's input.
///
/// Returns the plan and each input's table name and plan.
fn split_inputs(plan: LogicalPlan) -> Result<(LogicalPlan, Vec<(String, LogicalPlan)>)> {
    let mut inputs: Vec<(String, LogicalPlan)> = Vec::new();
    let plan = plan
        .transform_down(|node| {
            let LogicalPlan::SubqueryAlias(alias) = &node else {
                return Ok(Transformed::no(node));
            };
            let groups = alias.input.exists(|inner| {
                Ok(matches!(
                    inner,
                    LogicalPlan::Aggregate(_) | LogicalPlan::Window(_)
                ))
            })?;
            if !groups {
                return Ok(Transformed::no(node));
            }
            let body = alias.input.as_ref().clone();
            let mut name = alias.alias.table().to_string();
            if inputs.iter().any(|(found, _)| *found == name) {
                name = format!("{name}_{}", inputs.len());
            }
            let schema = Arc::new(body.schema().as_arrow().clone());
            let Ok(table) = MemTable::try_new(schema, vec![vec![]]) else {
                return Ok(Transformed::no(node));
            };
            let read =
                LogicalPlanBuilder::scan(name.as_str(), provider_as_source(Arc::new(table)), None)?
                    .alias(alias.alias.clone())?
                    .build()?;
            inputs.push((name, body));
            Ok(Transformed::new(read, true, TreeNodeRecursion::Jump))
        })?
        .data;
    Ok((plan, inputs))
}

/// A region's plan with a SELECT list at its root: the plan itself when
/// it has one, or a `Projection` of the `needed` columns over it.
fn select_list(plan: LogicalPlan, needed: Option<Vec<Column>>) -> Result<LogicalPlan> {
    let mut root = &plan;
    loop {
        match root {
            LogicalPlan::Projection(_) => return Ok(plan),
            LogicalPlan::Limit(found) => root = &found.input,
            LogicalPlan::Sort(found) => root = &found.input,
            LogicalPlan::Distinct(found) => root = found.input(),
            _ => break,
        }
    }
    let mut columns = needed.unwrap_or_else(|| plan.schema().columns());
    if columns.is_empty() {
        // Quail returns at least one column; DataFusion ignores it
        columns = plan.schema().columns().into_iter().take(1).collect();
    }
    let exprs = columns.into_iter().map(Expr::Column).collect();
    Ok(LogicalPlan::Projection(Projection::try_new(
        exprs,
        Arc::new(plan),
    )?))
}

/// Whether any operator in the subtree, or an EXISTS subquery in it,
/// calls an AI function.
fn calls_ai(plan: &LogicalPlan) -> Result<bool> {
    plan.exists(|node| {
        for expr in node.expressions() {
            let found = expr.exists(|inner| match inner {
                Expr::Exists(exists) => calls_ai(&exists.subquery.subquery),
                other => Ok(is_ai_call(other)),
            })?;
            if found {
                return Ok(true);
            }
        }
        Ok(false)
    })
}

/// Whether Quail can plan every operator in the subtree.
fn quail_plans(plan: &LogicalPlan) -> Result<bool> {
    let unsupported = plan.exists(|node| {
        if !is_quail_operator(node) {
            return Ok(true);
        }
        for expr in node.expressions() {
            if expr.exists(|inner| Ok(is_subquery(inner)))? {
                return Ok(true);
            }
        }
        Ok(false)
    })?;
    Ok(!unsupported)
}

/// Whether Quail's planner has an operator for this node.
fn is_quail_operator(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Projection(_)
            | LogicalPlan::Filter(_)
            | LogicalPlan::Window(_)
            | LogicalPlan::Aggregate(_)
            | LogicalPlan::Sort(_)
            | LogicalPlan::Join(_)
            | LogicalPlan::Limit(_)
            | LogicalPlan::Distinct(_)
            | LogicalPlan::SubqueryAlias(_)
            | LogicalPlan::TableScan(_)
            | LogicalPlan::Unnest(_)
    )
}

/// Whether an expression is a subquery Quail does not plan: anything
/// but an EXISTS gate, whose subquery Quail's binder checks.
fn is_subquery(expr: &Expr) -> bool {
    match expr {
        Expr::Exists(exists) => !quail_plans(&exists.subquery.subquery).unwrap_or(false),
        Expr::InSubquery(_) | Expr::ScalarSubquery(_) => true,
        _ => false,
    }
}
