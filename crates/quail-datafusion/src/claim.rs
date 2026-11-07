//! The `ClaimModelRegions` analyzer rule.

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, DataFusionError, Result, TableReference, plan_err};
use datafusion::datasource::{MemTable, provider_as_source, source_as_provider};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Aggregate, Expr, Extension, Filter, LogicalPlan, LogicalPlanBuilder, Projection, TableScan,
    Window,
};
use datafusion::optimizer::AnalyzerRule;

use crate::functions::{is_ai_call, is_quail_call};
use crate::identified::IdentifiedTable;
use crate::region::ModelRegion;
use quail_planner::expressions::UNSUPPORTED;

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

/// Remove the name DataFusion gives an AI call written by a dotted
/// name, such as `AI.IF(...)`, where a SELECT alias replaces it or
/// where it names no column: inside a `WHERE` or `ON` test, or
/// under the SELECT alias.
fn drop_written_names(plan: LogicalPlan) -> Result<LogicalPlan> {
    fn written(expr: &Expr) -> Option<&Expr> {
        match expr {
            Expr::Alias(alias) if is_ai_call(&alias.expr) => Some(alias.expr.as_ref()),
            _ => None,
        }
    }
    fn unnamed(expr: Expr) -> Result<Transformed<Expr>> {
        expr.transform_up(|inner| match written(&inner) {
            Some(call) => Ok(Transformed::yes(call.clone())),
            None => Ok(Transformed::no(inner)),
        })
    }
    plan.transform_up(|node| {
        let tests = matches!(node, LogicalPlan::Filter(_) | LogicalPlan::Join(_));
        node.map_expressions(|expr| {
            if tests {
                return unnamed(expr);
            }
            match expr {
                Expr::Alias(mut alias) => match written(&alias.expr) {
                    Some(call) => {
                        alias.expr = Box::new(call.clone());
                        Ok(Transformed::yes(Expr::Alias(alias)))
                    }
                    None => Ok(Transformed::no(Expr::Alias(alias))),
                },
                other => Ok(Transformed::no(other)),
            }
        })
    })
    .map(|done| done.data)
}

impl AnalyzerRule for ClaimModelRegions {
    fn name(&self) -> &str {
        "claim_model_regions"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        let plan = drop_written_names(plan)?;
        let mut extracted = Extracted::default();
        // the call each written name, such as `AI.CLASSIFY(...)`, names
        let mut written: Vec<(String, String)> = Vec::new();
        plan.apply(|node| {
            for expr in node.expressions() {
                if let Expr::Alias(alias) = &expr
                    && is_ai_call(&alias.expr)
                {
                    written.push((alias.name.clone(), alias.expr.schema_name().to_string()));
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        let call_named = |name: &str| {
            written
                .iter()
                .find(|(found, _)| found == name)
                .map_or_else(|| name.to_string(), |(_, call)| call.clone())
        };
        // a moved call keeps the name the SELECT list gives it
        plan.apply(|node| {
            if let LogicalPlan::Projection(projection) = node {
                for expr in &projection.expr {
                    // the call itself, or the column a GROUP BY makes of it
                    let Expr::Alias(alias) = expr else { continue };
                    let named = match alias.expr.as_ref() {
                        Expr::Column(column) => Some(call_named(&column.name)),
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
    /// Whether the plan being claimed has had its filters pushed down.
    pushed: bool,
}

impl Extracted {
    /// Move each AI call of an `Aggregate`, `Window`, or `Sort` into a
    /// `Projection` below it.
    fn extract(&mut self, node: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
        if !matches!(
            node,
            LogicalPlan::Aggregate(_)
                | LogicalPlan::Window(_)
                | LogicalPlan::Sort(_)
                | LogicalPlan::Filter(_)
                | LogicalPlan::Projection(_)
        ) {
            return Ok(Transformed::no(node));
        }
        let mut calls: Vec<Expr> = Vec::new();
        for expr in node.expressions() {
            // a SELECT list item that is a model call already sits where
            // a region reads it
            if matches!(node, LogicalPlan::Projection(_))
                && ai_calls(&expr)?.first() == Some(expr.clone().unalias_nested().data).as_ref()
            {
                continue;
            }
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
            LogicalPlan::Projection(found) => {
                // each output keeps its name
                let keep = found
                    .expr
                    .iter()
                    .map(|expr| read(expr)?.alias_if_changed(expr.schema_name().to_string()))
                    .collect::<Result<Vec<_>>>()?;
                LogicalPlan::Projection(Projection::try_new(keep, Arc::new(projection))?)
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
            let selected = select_list(plan.clone(), needed.clone())?;
            let pushed = precompute_plain_expressions(push_plain_terms(selected)?)?;
            // a FROM or JOIN subquery the region does not read through runs
            // before the region, as its own input
            let (bound, inputs) = split_inputs(pushed)?;
            if !calls_ai(&bound)? {
                // every model call is in a subquery that runs first: DataFusion
                // runs this part, and the subqueries are claimed below it
                return self.hand_back(plan, needed, limited, None);
            }
            match refusal(&bound) {
                None => {
                    let mut region = ModelRegion::new(bound);
                    for (name, input) in inputs {
                        region.inputs.push(self.claim(input, None, false)?);
                        region.input_names.push(name);
                    }
                    return Ok(LogicalPlan::Extension(Extension {
                        node: Arc::new(region),
                    }));
                }
                Some(refusal) => return self.hand_back(plan, needed, limited, Some(refusal)),
            }
        }
        self.hand_back(plan, needed, limited, None)
    }

    /// Let DataFusion run the part of `plan` that Quail does not bind.
    ///
    /// First DataFusion's `PushDownFilter` moves each filter as far down
    /// as SQL allows, such as a test of one table below a join, and the
    /// result is claimed again. Then DataFusion keeps `plan`'s operator,
    /// its AI calls moved into a projection below it, and its inputs are
    /// claimed. `refusal` is Quail's reason it cannot run `plan`
    /// whole; it is the error when a model call is still outside every
    /// region.
    fn hand_back(
        &mut self,
        plan: LogicalPlan,
        needed: Option<Vec<Column>>,
        limited: bool,
        refusal: Option<String>,
    ) -> Result<LogicalPlan> {
        // the reason Quail cannot run this operator explains a failure
        // better than one found below it
        self.hand_back_parts(plan, needed, limited, refusal.clone())
            .map_err(|error| match refusal {
                Some(refusal) => DataFusionError::Plan(refusal),
                None => error,
            })
    }

    fn hand_back_parts(
        &mut self,
        plan: LogicalPlan,
        needed: Option<Vec<Column>>,
        limited: bool,
        refusal: Option<String>,
    ) -> Result<LogicalPlan> {
        if !self.pushed {
            let pushed = push_down_filters(&plan);
            if pushed != plan {
                self.pushed = true;
                let claimed = self.claim(pushed, needed, limited);
                self.pushed = false;
                return claimed;
            }
        }
        // a filter's terms with a subquery stay in DataFusion; the rest
        // run below it, where a region can take them
        if let LogicalPlan::Filter(filter) = &plan {
            let (above, below): (Vec<Expr>, Vec<Expr>) = split_conjunction(&filter.predicate)
                .into_iter()
                .cloned()
                .partition(|term| term.exists(|inner| Ok(is_subquery(inner))).unwrap_or(true));
            if !above.is_empty() && !below.is_empty() {
                let lower = LogicalPlan::Filter(Filter::try_new(
                    conjunction(below).expect("a term"),
                    Arc::clone(&filter.input),
                )?);
                let split = LogicalPlan::Filter(Filter::try_new(
                    conjunction(above).expect("a term"),
                    Arc::new(lower),
                )?);
                return self.claim(split, needed, limited);
            }
        }
        let plan = self.extract(plan)?.data;
        let inputs: Vec<LogicalPlan> = plan.inputs().into_iter().cloned().collect();
        let limit = limits(&plan, limited);
        let claimed = inputs
            .into_iter()
            .map(|input| {
                let read = read_from(&plan, &input, needed.as_deref());
                self.claim(input, read, limit)
            })
            .collect::<Result<Vec<_>>>()?;
        let kept = plan.with_new_exprs(plan.expressions(), claimed)?;
        if calls_ai(&kept)? {
            return plan_err!(
                "{}",
                refusal.unwrap_or_else(|| format!(
                    "a model call in this operator {UNSUPPORTED}:\n{}",
                    plan.display()
                ))
            );
        }
        Ok(kept)
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
        | LogicalPlan::Window(_)
        | LogicalPlan::Join(_) => needed?.iter().cloned().for_each(&mut add),
        // a union's inputs line up by position, not by name
        LogicalPlan::Union(_) => return None,
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

/// Quail's refusal of `plan` as a region, when the refusal is of a shape
/// DataFusion can run instead; any other refusal is the query's error,
/// reported when the region is planned.
fn refusal(plan: &LogicalPlan) -> Option<String> {
    // the chat turn and layout change the prompts, not what Quail reads
    let turn: quail_planner::prompts::Turn = (String::new(), String::new());
    let region = ModelRegion::new(plan.clone());
    match crate::derive::derive_region(&region, &turn, "ai-if") {
        Err(error) if error.0.contains(UNSUPPORTED) => Some(error.0),
        _ => None,
    }
}

/// The CTE name and body of a FROM subquery: `x` over the CTE `clauses`,
/// or the subquery itself.
pub(crate) fn subquery_body(
    alias: &datafusion::logical_expr::SubqueryAlias,
) -> (String, &LogicalPlan) {
    match alias.input.as_ref() {
        LogicalPlan::SubqueryAlias(cte) => (cte.alias.table().to_string(), cte.input.as_ref()),
        body => (alias.alias.table().to_string(), body),
    }
}

/// The aliases of the JOIN inputs a region reads through, and the CTE
/// each reads: each subquery that selects and filters one table without
/// a model call, and is read once. Any other FROM or JOIN subquery runs
/// before the region, as its own input.
///
/// For example, `FROM contracts a JOIN kept b ON AI.IF(...b.text...)`,
/// with `kept` as `SELECT c.id, c.text FROM contracts c WHERE AI.IF(...)`,
/// reads `kept` through: the region scans `contracts` as `c` under `b`.
pub(crate) fn merged_aliases(plan: &LogicalPlan) -> Vec<String> {
    let mut readers: std::collections::BTreeMap<String, usize> = Default::default();
    let _ = plan.apply(|node| {
        // each read of a CTE is one copy of its body under the CTE's name
        if let LogicalPlan::SubqueryAlias(alias) = node
            && !matches!(
                alias.input.as_ref(),
                LogicalPlan::TableScan(_) | LogicalPlan::SubqueryAlias(_)
            )
        {
            *readers.entry(alias.alias.table().to_string()).or_default() += 1;
        }
        Ok(TreeNodeRecursion::Continue)
    });
    let mut merged = Vec::new();
    let _ = plan.apply(|node| {
        let LogicalPlan::Join(join) = node else {
            return Ok(TreeNodeRecursion::Continue);
        };
        for side in [join.left.as_ref(), join.right.as_ref()] {
            let LogicalPlan::SubqueryAlias(alias) = side else {
                continue;
            };
            let (name, body) = subquery_body(alias);
            let LogicalPlan::Projection(projection) = body else {
                continue;
            };
            let base = match projection.input.as_ref() {
                LogicalPlan::Filter(filter) => filter.input.as_ref(),
                other => other,
            };
            let is_table = matches!(base, LogicalPlan::SubqueryAlias(found)
                if matches!(found.input.as_ref(), LogicalPlan::TableScan(_)));
            let calls_model = projection
                .expr
                .iter()
                .any(|expr| expr.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(false));
            if is_table && readers.get(&name).copied().unwrap_or(0) <= 1 && !calls_model {
                merged.push(alias.alias.table().to_string());
                // `kept b` reads the CTE `kept` through `b`
                if name != alias.alias.table() {
                    merged.push(name);
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    merged
}

/// `plan` with DataFusion's `PushDownFilter` applied, or `plan` when the
/// rule fails.
///
/// A filter the rule moves below a table's alias returns above it, its
/// columns named by the alias: Quail reads a table as an alias over its
/// scan.
fn push_down_filters(plan: &LogicalPlan) -> LogicalPlan {
    let rules: Vec<Arc<dyn datafusion::optimizer::OptimizerRule + Send + Sync>> = vec![Arc::new(
        datafusion::optimizer::push_down_filter::PushDownFilter::new(),
    )];
    let Ok(pushed) = datafusion::optimizer::Optimizer::with_rules(rules).optimize(
        plan.clone(),
        &datafusion::optimizer::OptimizerContext::new(),
        |_, _| {},
    ) else {
        return plan.clone();
    };
    lift_filters_above_aliases(&pushed).unwrap_or_else(|_| plan.clone())
}

/// `plan` with each filter that DataFusion moved below a table's alias
/// back above it, its columns named by the alias: Quail reads a table as
/// an alias over its scan, and SQL names its columns by the alias.
///
/// # Errors
///
/// A lifted filter does not plan.
pub(crate) fn lift_filters_above_aliases(plan: &LogicalPlan) -> Result<LogicalPlan> {
    plan.clone()
        .transform_up(|node| {
            let LogicalPlan::SubqueryAlias(alias) = &node else {
                return Ok(Transformed::no(node));
            };
            let LogicalPlan::Filter(filter) = alias.input.as_ref() else {
                return Ok(Transformed::no(node));
            };
            if !matches!(filter.input.as_ref(), LogicalPlan::TableScan(_)) {
                return Ok(Transformed::no(node));
            }
            let table = alias.alias.clone();
            let predicate = filter
                .predicate
                .clone()
                .transform_up(|expr| match expr {
                    Expr::Column(column) => Ok(Transformed::yes(Expr::Column(Column::new(
                        Some(table.clone()),
                        column.name,
                    )))),
                    other => Ok(Transformed::no(other)),
                })?
                .data;
            let read = LogicalPlanBuilder::from(filter.input.as_ref().clone())
                .alias(table)?
                .filter(predicate)?
                .build()?;
            Ok(Transformed::yes(read))
        })
        .map(|lifted| lifted.data)
}

/// Move each plain filter term about one table down to that table's
/// alias, where it becomes part of the table's input.
///
/// A plain term calls no model and has no subquery. It moves down only
/// through filters and inner joins, which keep a row whatever happens
/// to the other tables' rows; a term about two tables stays.
fn push_plain_terms(plan: LogicalPlan) -> Result<LogicalPlan> {
    plan.transform_down(|node| {
        let LogicalPlan::Filter(filter) = &node else {
            return Ok(Transformed::no(node));
        };
        let mut kept = Vec::new();
        let mut moved: Vec<(TableReference, Expr)> = Vec::new();
        for term in split_conjunction(&filter.predicate) {
            match plain_table(term) {
                Some(table) if reaches_alias(&filter.input, &table) => {
                    moved.push((table, term.clone()))
                }
                _ => kept.push(term.clone()),
            }
        }
        if moved.is_empty() {
            return Ok(Transformed::no(node));
        }
        let mut input = filter.input.as_ref().clone();
        for (table, term) in moved {
            input = place_term(input, &table, term)?;
        }
        let placed = match conjunction(kept) {
            Some(predicate) => LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input))?),
            None => input,
        };
        Ok(Transformed::yes(placed))
    })
    .map(|pushed| pushed.data)
}

/// Compute each plain expression of a region that reads one table as a
/// column of that table's input, so DataFusion evaluates it.
///
/// A plain expression is not a column, calls no model, aggregate,
/// window, or subquery, and reads columns of one table alias whose part
/// of the plan calls no model. The alias's part gains a projection with
/// the expression as `__quail_expr_<n>`, and the region reads that
/// column; a SELECT item keeps its name.
fn precompute_plain_expressions(plan: LogicalPlan) -> Result<LogicalPlan> {
    let mut tables: Vec<TableReference> = Vec::new();
    plan.apply(|node| {
        if let LogicalPlan::SubqueryAlias(alias) = node
            && !calls_ai(node)?
            && single_scan(node).is_some()
        {
            tables.push(alias.alias.clone());
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if tables.is_empty() {
        return Ok(plan);
    }
    // (table, expression, column name), in the order first seen
    let mut computed: Vec<(TableReference, Expr, String)> = Vec::new();
    let mut find = |expr: &Expr| {
        let _ = expr.apply(|inner| {
            let Some(table) = plain_table(inner) else {
                return Ok(TreeNodeRecursion::Continue);
            };
            // a struct field is a column path the region reads as it is
            let field = matches!(inner, Expr::ScalarFunction(call) if call.name() == "get_field");
            if matches!(inner, Expr::Column(_) | Expr::Literal(..) | Expr::Alias(_))
                || field
                || !tables.contains(&table)
                || inner.exists(|found| {
                    Ok(matches!(
                        found,
                        Expr::AggregateFunction(_) | Expr::WindowFunction(_)
                    ))
                })?
            {
                return Ok(TreeNodeRecursion::Continue);
            }
            if !computed.iter().any(|(_, found, _)| found == inner) {
                let name = format!("__quail_expr_{}", computed.len());
                computed.push((table, inner.clone(), name));
            }
            Ok(TreeNodeRecursion::Jump)
        });
    };
    plan.apply(|node| {
        if let LogicalPlan::SubqueryAlias(alias) = node
            && tables.contains(&alias.alias)
        {
            return Ok(TreeNodeRecursion::Jump);
        }
        for expr in node.expressions() {
            if let LogicalPlan::Filter(_) = node {
                // a whole plain term is the input's filter; only a model
                // call's arguments are computed
                for term in split_conjunction(&expr) {
                    if plain_table(term).is_none() {
                        find(term);
                    }
                }
            } else {
                find(&expr);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if computed.is_empty() {
        return Ok(plan);
    }
    let replace_inner = |expr: Expr| -> Result<Transformed<Expr>> {
        expr.transform_down(
            |inner| match computed.iter().find(|(_, found, _)| *found == inner) {
                Some((table, _, name)) => Ok(Transformed::new(
                    Expr::Column(Column::new(Some(table.clone()), name)),
                    true,
                    TreeNodeRecursion::Jump,
                )),
                None => Ok(Transformed::no(inner)),
            },
        )
    };
    // a filter's whole plain terms stay as written; they are the input's
    let replace = |expr: Expr, filter: bool| -> Result<Transformed<Expr>> {
        if !filter {
            return replace_inner(expr);
        }
        let mut changed = false;
        let terms = split_conjunction(&expr)
            .into_iter()
            .cloned()
            .map(|term| {
                if plain_table(&term).is_some() {
                    return Ok(term);
                }
                let found = replace_inner(term)?;
                changed |= found.transformed;
                Ok(found.data)
            })
            .collect::<Result<Vec<_>>>()?;
        let joined = conjunction(terms).expect("a term");
        Ok(if changed {
            Transformed::yes(joined)
        } else {
            Transformed::no(expr)
        })
    };
    plan.transform_up(|node| {
        if let LogicalPlan::SubqueryAlias(alias) = &node
            && tables.contains(&alias.alias)
        {
            let added: Vec<Expr> = computed
                .iter()
                .filter(|(table, _, _)| *table == alias.alias)
                .map(|(_, expr, name)| expr.clone().alias(name))
                .collect();
            if added.is_empty() {
                return Ok(Transformed::no(node));
            }
            let table = alias.alias.clone();
            let mut exprs: Vec<Expr> = node
                .schema()
                .columns()
                .into_iter()
                .map(Expr::Column)
                .collect();
            exprs.extend(added);
            let projected = LogicalPlanBuilder::from(node)
                .project(exprs)?
                .alias(table)?
                .build()?;
            return Ok(Transformed::yes(projected));
        }
        let is_projection = matches!(node, LogicalPlan::Projection(_));
        let is_filter = matches!(node, LogicalPlan::Filter(_));
        let mut changed = false;
        let exprs = node
            .expressions()
            .into_iter()
            .map(|expr| {
                let name = expr.schema_name().to_string();
                let found = replace(expr, is_filter)?;
                changed |= found.transformed;
                // a SELECT item keeps the name it had
                Ok(
                    if is_projection && found.transformed && !matches!(found.data, Expr::Alias(_)) {
                        found.data.alias(name)
                    } else {
                        found.data
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let inputs: Vec<LogicalPlan> = node.inputs().into_iter().cloned().collect();
        let rebuilt = node.with_new_exprs(exprs, inputs)?;
        Ok(Transformed::new(
            rebuilt,
            changed,
            TreeNodeRecursion::Continue,
        ))
    })
    .map(|done| done.data)
}

/// The one table a plain term reads, or None for a term that calls a
/// model, has a subquery, or reads several tables or none.
fn plain_table(term: &Expr) -> Option<TableReference> {
    let special = term
        .exists(|inner| Ok(is_quail_call(inner) || is_subquery_expr(inner)))
        .unwrap_or(true);
    if special {
        return None;
    }
    let mut tables = term
        .column_refs()
        .into_iter()
        .map(|column| column.relation.clone());
    let first = tables.next()??;
    tables
        .all(|table| table.as_ref() == Some(&first))
        .then_some(first)
}

/// Whether `plan` reaches the alias `table` through filters and inner
/// joins only, and the alias's subtree calls no model.
fn reaches_alias(plan: &LogicalPlan, table: &TableReference) -> bool {
    match plan {
        // a term moves only to a table DataFusion can compute as an input
        LogicalPlan::SubqueryAlias(alias) => {
            alias.alias == *table && !calls_ai(plan).unwrap_or(true)
        }
        LogicalPlan::Filter(filter) => reaches_alias(&filter.input, table),
        LogicalPlan::Join(join) if join.join_type == datafusion::logical_expr::JoinType::Inner => {
            reaches_alias(&join.left, table) || reaches_alias(&join.right, table)
        }
        _ => false,
    }
}

/// `plan` with `term` as a filter directly above the alias `table`.
fn place_term(plan: LogicalPlan, table: &TableReference, term: Expr) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::SubqueryAlias(ref alias) if alias.alias == *table => {
            Ok(LogicalPlan::Filter(Filter::try_new(term, Arc::new(plan))?))
        }
        LogicalPlan::Filter(filter) => {
            let input = place_term(filter.input.as_ref().clone(), table, term)?;
            Ok(LogicalPlan::Filter(Filter::try_new(
                filter.predicate,
                Arc::new(input),
            )?))
        }
        LogicalPlan::Join(join) => {
            let (left, right) = if reaches_alias(&join.left, table) {
                (
                    place_term(join.left.as_ref().clone(), table, term)?,
                    join.right.as_ref().clone(),
                )
            } else {
                (
                    join.left.as_ref().clone(),
                    place_term(join.right.as_ref().clone(), table, term)?,
                )
            };
            LogicalPlan::Join(join.clone())
                .with_new_exprs(LogicalPlan::Join(join).expressions(), vec![left, right])
        }
        other => Ok(other),
    }
}

fn is_subquery_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Exists(_) | Expr::InSubquery(_) | Expr::ScalarSubquery(_)
    )
}

/// The one table a subtree reads, or None when it reads several or none.
fn single_scan(plan: &LogicalPlan) -> Option<TableScan> {
    let mut scans = Vec::new();
    plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            scans.push(scan.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .ok()?;
    match <[TableScan; 1]>::try_from(scans) {
        Ok([scan]) => Some(scan),
        Err(_) => None,
    }
}

/// Whether a subtree is a table the runner reads by name: an identified
/// table's scan, alone or under its alias.
fn read_by_name(plan: &LogicalPlan) -> bool {
    let scan = match plan {
        LogicalPlan::TableScan(scan) => scan,
        LogicalPlan::SubqueryAlias(alias) => match alias.input.as_ref() {
            LogicalPlan::TableScan(scan) => scan,
            _ => return false,
        },
        _ => return false,
    };
    source_as_provider(&scan.source)
        .is_ok_and(|provider| provider.downcast_ref::<IdentifiedTable>().is_some())
}

/// Whether DataFusion computes this part of a region as an input: it
/// calls no model, reads one table, and is more than an identified table
/// the runner reads by name.
fn plain_input(plan: &LogicalPlan) -> Result<Option<TableScan>> {
    if read_by_name(plan) || calls_ai(plan)? {
        return Ok(None);
    }
    let subquery = plan.exists(|node| {
        Ok(node.expressions().iter().any(|expr| {
            expr.exists(|inner| Ok(is_subquery_expr(inner)))
                .unwrap_or(true)
        }))
    })?;
    if subquery {
        return Ok(None);
    }
    Ok(single_scan(plan))
}

/// Take out of a region's plan each part DataFusion computes as an
/// input: a part that calls no model and reads one table, such as a
/// table's plain filters, each CTE or FROM subquery that groups or
/// windows rows, and each one the region does not read through
/// ([`merged_aliases`]). The plan reads each input as a table of its name.
///
/// Returns the plan and each input's table name and plan.
fn split_inputs(plan: LogicalPlan) -> Result<(LogicalPlan, Vec<(String, LogicalPlan)>)> {
    let mut inputs: Vec<(String, LogicalPlan)> = Vec::new();
    let merged = merged_aliases(&plan);
    let mut root = true;
    let plan = plan
        .transform_down(|node| {
            // a subquery at the root is the region itself
            let at_root = std::mem::take(&mut root);
            if let Some(scan) = plain_input(&node)? {
                return cut_input(node, &scan, &mut inputs);
            }
            if let LogicalPlan::TableScan(scan) = &node {
                return table_input(scan, &mut inputs);
            }
            let LogicalPlan::SubqueryAlias(alias) = &node else {
                return Ok(Transformed::no(node));
            };
            // a subquery the region does not read through runs before the
            // query that reads it, as an input: its own model region when it
            // calls a model, or DataFusion's when it groups or windows
            let groups = alias.input.exists(|inner| {
                Ok(matches!(
                    inner,
                    LogicalPlan::Aggregate(_) | LogicalPlan::Window(_)
                ))
            })?;
            let table = matches!(alias.input.as_ref(), LogicalPlan::TableScan(_));
            let staged = !at_root && !table && !merged.contains(&alias.alias.table().to_string());
            if !groups && !staged {
                return Ok(Transformed::no(node));
            }
            // `FROM clauses a` reads the CTE `clauses` under the alias `a`;
            // every read of one CTE is one input, which runs once
            let (mut name, body) = match alias.input.as_ref() {
                LogicalPlan::SubqueryAlias(cte) => {
                    (cte.alias.table().to_string(), cte.input.as_ref().clone())
                }
                body => (alias.alias.table().to_string(), body.clone()),
            };
            let read_before = inputs
                .iter()
                .any(|(found, plan)| *found == name && *plan == body);
            if !read_before && inputs.iter().any(|(found, _)| *found == name) {
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
            if !read_before {
                inputs.push((name, body));
            }
            Ok(Transformed::new(read, true, TreeNodeRecursion::Jump))
        })?
        .data;
    Ok((plan, inputs))
}

/// Replace `node`, a plain part of a region that reads `scan`'s table,
/// with a scan of an input DataFusion computes from it.
///
/// The input is named `<table>__<alias>`; its id column stays a primary
/// key when the part returns it.
fn cut_input(
    node: LogicalPlan,
    scan: &TableScan,
    inputs: &mut Vec<(String, LogicalPlan)>,
) -> Result<Transformed<LogicalPlan>> {
    let schema = node.schema();
    let qualifiers: Vec<_> = schema
        .iter()
        .map(|(qualifier, _)| qualifier.cloned())
        .collect();
    let Some(Some(qualifier)) = qualifiers.first().cloned() else {
        return Ok(Transformed::no(node));
    };
    if qualifiers
        .iter()
        .any(|found| found.as_ref() != Some(&qualifier))
    {
        return Ok(Transformed::no(node));
    }
    // not the table's own name: two aliases of one table read different rows
    let name = format!(
        "{}{}{}",
        scan.table_name.table(),
        quail_planner::operators::INPUT_SEPARATOR,
        qualifier.table()
    );
    let fields = Arc::new(schema.as_arrow().clone());
    let mut placeholder = MemTable::try_new(Arc::clone(&fields), vec![vec![]])?;
    let provider = source_as_provider(&scan.source)?;
    let key = provider.constraints().and_then(|constraints| {
        constraints.iter().find_map(|constraint| match constraint {
            datafusion::common::Constraint::PrimaryKey(indices) if indices.len() == 1 => {
                Some(provider.schema().field(indices[0]).name().clone())
            }
            _ => None,
        })
    });
    if let Some(index) = key.and_then(|column| fields.index_of(&column).ok()) {
        placeholder =
            placeholder.with_constraints(datafusion::common::Constraints::new_unverified(vec![
                datafusion::common::Constraint::PrimaryKey(vec![index]),
            ]));
    }
    let read = LogicalPlanBuilder::scan(
        name.as_str(),
        provider_as_source(Arc::new(placeholder)),
        None,
    )?
    .alias(qualifier)?
    .build()?;
    inputs.push((name, node));
    Ok(Transformed::new(read, true, TreeNodeRecursion::Jump))
}

/// A scan of a table without an identity, as a scan of a region input of
/// the table's name: DataFusion scans the table and passes its rows. A
/// table read twice, as in a self join, is one input.
fn table_input(
    scan: &TableScan,
    inputs: &mut Vec<(String, LogicalPlan)>,
) -> Result<Transformed<LogicalPlan>> {
    let provider = source_as_provider(&scan.source)?;
    if provider.downcast_ref::<IdentifiedTable>().is_some() {
        return Ok(Transformed::no(LogicalPlan::TableScan(scan.clone())));
    }
    let name = scan.table_name.table().to_string();
    let schema = provider.schema();
    let mut placeholder = MemTable::try_new(Arc::clone(&schema), vec![vec![]])?;
    if let Some(constraints) = provider.constraints() {
        placeholder = placeholder.with_constraints(constraints.clone());
    }
    let read = LogicalPlanBuilder::scan_with_filters(
        scan.table_name.clone(),
        provider_as_source(Arc::new(placeholder)),
        scan.projection.clone(),
        scan.filters.clone(),
    )?
    .build()?;
    if !inputs.iter().any(|(found, _)| *found == name) {
        let body =
            LogicalPlanBuilder::scan(scan.table_name.clone(), Arc::clone(&scan.source), None)?
                .build()?;
        inputs.push((name, body));
    }
    Ok(Transformed::new(read, true, TreeNodeRecursion::Jump))
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
/// but an EXISTS gate, whose subquery the region's lowering reads.
fn is_subquery(expr: &Expr) -> bool {
    match expr {
        Expr::Exists(exists) => !quail_plans(&exists.subquery.subquery).unwrap_or(false),
        Expr::InSubquery(_) | Expr::ScalarSubquery(_) => true,
        _ => false,
    }
}
