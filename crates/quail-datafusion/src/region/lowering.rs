//! Lower AI calls into plan nodes before optimization.
//! Move calls below operators that DataFusion runs.

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, DataFusionError, Result, plan_err};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Aggregate, Expr, Filter, LogicalPlan, LogicalPlanBuilder, Projection, Window,
};
use quail_planner::error::UNSUPPORTED;

use super::claim::{
    calls_ai, calls_ai_function, is_subquery, kept_by_datafusion, quail_plans, refusal, select_list,
};
use super::inputs::{precompute_plain_expressions, push_plain_terms, read_inputs, split_inputs};
use crate::functions::is_ai_call;

/// Lower the AI calls of each part of `plan` that Quail can plan into
/// `AiNode`s, in place, as the module explains.
///
/// # Errors
///
/// A model call is in an operator Quail cannot plan.
pub(crate) fn lower_regions(plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
    let plan = drop_written_names(plan)?;
    let plan = plan.transform_up(exists_terms_as_in)?.data;
    let mut extracted = Extracted {
        config: config.clone(),
        ..Extracted::default()
    };
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
    extracted.claim(plan, None)
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
    /// The session's configuration, which a trial of a part runs in.
    config: ConfigOptions,
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

    /// Claim the model regions of `plan`, as [`super::claim::ClaimModelRegions`] explains.
    ///
    /// `needed` holds the columns the operator above reads from `plan`, or
    /// None when it reads every one.
    fn claim(&mut self, plan: LogicalPlan, needed: Option<Vec<Column>>) -> Result<LogicalPlan> {
        if !calls_ai(&plan)? {
            return Ok(plan);
        }
        let plan = self.push_below_window(plan)?;
        if kept_by_datafusion(&plan)? {
            let plan = match plan {
                LogicalPlan::Aggregate(_) | LogicalPlan::Window(_) | LogicalPlan::Sort(_) => {
                    self.extract(plan)?.data
                }
                plan => plan,
            };
            let inputs: Vec<LogicalPlan> = plan.inputs().into_iter().cloned().collect();
            let claimed = inputs
                .into_iter()
                .map(|input| {
                    let read = read_from(&plan, &input, needed.as_deref());
                    self.claim(input, read)
                })
                .collect::<Result<Vec<_>>>()?;
            return plan.with_new_exprs(plan.expressions(), claimed);
        }
        if quail_plans(&plan)? {
            let selected = select_list(plan.clone(), needed.clone())?;
            let pushed = precompute_plain_expressions(push_plain_terms(selected)?)?;
            // a FROM or JOIN subquery the region does not read through runs
            // before the region, as its own input
            let (bound, inputs, reads) = split_inputs(pushed)?;
            if !calls_ai(&bound)? {
                // every model call is in a subquery that runs first: DataFusion
                // runs this part, and the subqueries are claimed below it
                return self.hand_back(plan, needed, None);
            }
            match refusal(&bound, &self.config)? {
                None => {
                    let mut claimed = Vec::new();
                    for (name, input) in inputs {
                        claimed.push((name, self.claim(input, None)?));
                    }
                    let lowered = crate::lower_ai_calls(bound)?;
                    return read_inputs(lowered, &reads, &claimed);
                }
                Some(refusal) => return self.hand_back(plan, needed, Some(refusal)),
            }
        }
        self.hand_back(plan, needed, None)
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
        refusal: Option<String>,
    ) -> Result<LogicalPlan> {
        // the reason Quail cannot run this operator explains a failure
        // better than one found below it
        self.hand_back_parts(plan, needed, refusal.clone())
            .map_err(|error| match refusal {
                Some(refusal) => DataFusionError::Plan(refusal),
                None => error,
            })
    }

    /// `plan` with the model regions of each uncorrelated `IN` subquery
    /// claimed: DataFusion runs the `IN` as a semi-join, whose subquery
    /// side is then a part of its own. [`lower_regions`] reads a
    /// correlated `EXISTS` of AI calls on the subquery's table alone as
    /// such an `IN` ([`exists_terms_as_in`]).
    fn claim_in_subqueries(&mut self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let mut failed = None;
        let claimed = plan.map_expressions(|expr| {
            expr.transform_down(|inner| match inner {
                Expr::InSubquery(mut found)
                    if found.subquery.outer_ref_columns.is_empty()
                        && calls_ai_function(&found.subquery.subquery)? =>
                {
                    let body = found.subquery.subquery.as_ref().clone();
                    match self.claim(body, None) {
                        Ok(body) => {
                            found.subquery.subquery = Arc::new(body);
                            Ok(Transformed::yes(Expr::InSubquery(found)))
                        }
                        Err(error) => {
                            failed = Some(error);
                            Ok(Transformed::no(Expr::InSubquery(found)))
                        }
                    }
                }
                other => Ok(Transformed::no(other)),
            })
        })?;
        match failed {
            Some(error) => Err(error),
            None => Ok(claimed.data),
        }
    }

    fn hand_back_parts(
        &mut self,
        plan: LogicalPlan,
        needed: Option<Vec<Column>>,
        refusal: Option<String>,
    ) -> Result<LogicalPlan> {
        if !self.pushed {
            let pushed = push_down_filters(&plan);
            if pushed != plan {
                self.pushed = true;
                let claimed = self.claim(pushed, needed);
                self.pushed = false;
                return claimed;
            }
        }
        let plan = self.claim_in_subqueries(plan)?;
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
                return self.claim(split, needed);
            }
        }
        let plan = self.extract(plan)?.data;
        let inputs: Vec<LogicalPlan> = plan.inputs().into_iter().cloned().collect();
        let claimed = inputs
            .into_iter()
            .map(|input| {
                let read = read_from(&plan, &input, needed.as_deref());
                self.claim(input, read)
            })
            .collect::<Result<Vec<_>>>()?;
        let kept = plan.with_new_exprs(plan.expressions(), claimed)?;
        // a part below that Quail lowered holds its calls in `AiNode`s
        if calls_ai_function(&kept)? {
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

/// The outermost AI calls in `expr`, each once.
fn ai_calls(expr: &Expr) -> Result<Vec<Expr>> {
    let mut calls: Vec<Expr> = Vec::new();
    expr.apply(|inner| {
        if is_ai_call(inner) {
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

/// `plan` with each term of its filter that is a correlated `EXISTS` of
/// AI calls on the subquery's table alone read as an `IN` subquery
/// ([`exists_as_in`]). An EXISTS is TRUE or FALSE and an IN may be NULL,
/// so only a filter's term, which drops both, reads one as the other.
fn exists_terms_as_in(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Filter(filter) = &plan else {
        return Ok(Transformed::no(plan));
    };
    let terms: Vec<Expr> = split_conjunction(&filter.predicate)
        .into_iter()
        .cloned()
        .map(exists_as_in)
        .collect();
    if conjunction(terms.clone()).as_ref() == Some(&filter.predicate) {
        return Ok(Transformed::no(plan));
    }
    Ok(Transformed::yes(LogicalPlan::Filter(Filter::try_new(
        conjunction(terms).expect("a term"),
        Arc::clone(&filter.input),
    )?)))
}

/// `expr` as an uncorrelated `IN` subquery when it is an `EXISTS` whose
/// subquery is a SELECT list over a filter of one equality of an outer
/// column and an inner one, and terms that read no outer column, with an
/// AI call among them; else `expr` as it is.
///
/// For example, `EXISTS (SELECT 1 FROM reviews r WHERE r.product =
/// p.number AND AI.IF(...r.body...))` is `p.number IN (SELECT r.product
/// FROM reviews r WHERE AI.IF(...r.body...))`: both keep the outer rows
/// with an inner row that has their key and passes the rest.
fn exists_as_in(expr: Expr) -> Expr {
    let Expr::Exists(exists) = &expr else {
        return expr;
    };
    if exists.negated {
        return expr;
    }
    let LogicalPlan::Projection(projection) = exists.subquery.subquery.as_ref() else {
        return expr;
    };
    let LogicalPlan::Filter(filter) = projection.input.as_ref() else {
        return expr;
    };
    let outer = |found: &Expr| {
        found
            .exists(|inner| Ok(matches!(inner, Expr::OuterReferenceColumn(..))))
            .unwrap_or(true)
    };
    let mut keys = Vec::new();
    let mut rest = Vec::new();
    for term in split_conjunction(&filter.predicate) {
        let key = match term {
            Expr::BinaryExpr(binary) if binary.op == datafusion::logical_expr::Operator::Eq => {
                match (binary.left.as_ref(), binary.right.as_ref()) {
                    (Expr::OuterReferenceColumn(_, outer), inner @ Expr::Column(_))
                    | (inner @ Expr::Column(_), Expr::OuterReferenceColumn(_, outer)) => {
                        Some((Expr::Column(outer.clone()), inner.clone()))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        match key {
            Some(key) => keys.push(key),
            None if outer(term) => return expr,
            None => rest.push(term.clone()),
        }
    }
    let [(key, read)] = keys.as_slice() else {
        return expr;
    };
    let asks = rest
        .iter()
        .any(|term| term.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(false));
    if !asks {
        return expr;
    }
    let body = match conjunction(rest) {
        Some(predicate) => match Filter::try_new(predicate, Arc::clone(&filter.input)) {
            Ok(kept) => LogicalPlan::Filter(kept),
            Err(_) => return expr,
        },
        None => filter.input.as_ref().clone(),
    };
    let Ok(selected) = Projection::try_new(vec![read.clone()], Arc::new(body)) else {
        return expr;
    };
    Expr::InSubquery(datafusion::logical_expr::expr::InSubquery::new(
        Box::new(key.clone()),
        datafusion::logical_expr::Subquery {
            subquery: Arc::new(LogicalPlan::Projection(selected)),
            outer_ref_columns: Vec::new(),
            spans: exists.subquery.spans.clone(),
        },
        false,
    ))
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
fn lift_filters_above_aliases(plan: &LogicalPlan) -> Result<LogicalPlan> {
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
