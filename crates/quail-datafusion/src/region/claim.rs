//! Model regions: where Quail lowers AI calls, and the `ClaimModelRegions`
//! optimizer rule.
//!
//! Before DataFusion's optimizer, [`lower_regions`] finds each part of a
//! query that Quail can plan and lowers its AI calls into `AiNode`s
//! there ([`crate::LowerAiCalls`]). It walks the plan from the root.
//! DataFusion keeps each `Aggregate`, `Window`, `Sort`, `Limit`,
//! `Distinct`, and `Unnest`, each AI call of theirs moved into a `Projection` below
//! them under a name they then read; and a `Projection`, `Filter`, or
//! `SubqueryAlias` that calls no AI function above one of those. The
//! first other operator whose whole subtree Quail can plan, and which
//! calls an AI function somewhere in that subtree, is lowered; the walk
//! does not look below it. DataFusion's optimizer makes a `Limit` over
//! the AI work the fetch of the `AiNode` it reads
//! ([`crate::PushDownLimitIntoAiNodes`]), and a `Distinct` a per-key
//! fetch ([`crate::PushDownDistinct`]), where Quail can stop the model
//! early. The part's root is a SELECT list: a part with none gets a
//! `Projection` of the columns the operators above it read.
//!
//! DataFusion's optimizer then runs over the whole plan, `AiNode`s
//! included. Last, [`ClaimModelRegions`] replaces each such part with
//! one `ModelRegion`: the same walk over the optimized plan, where the
//! first operator DataFusion does not keep starts a region, which holds
//! the `AiNode`s and the operators that run with them on the GPU. The
//! rest of the part's SELECT list, each `UNNEST`, and each test of the
//! result rows run in DataFusion above the region ([`lift_relational`]).
//! The region returns the id column of each table it reads and its
//! answers; DataFusion joins the tables' other columns back on their
//! ids ([`join_back`]).
use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, DataFusionError, Result, ScalarValue, TableReference, plan_err};
use datafusion::datasource::{MemTable, provider_as_source, source_as_provider};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Aggregate, Expr, Filter, LogicalPlan, LogicalPlanBuilder, Projection, SubqueryAlias, TableScan,
    TableScanBuilder, Window,
};
use datafusion::optimizer::{
    Analyzer, Optimizer, OptimizerConfig, OptimizerContext, OptimizerRule,
};

use crate::ai::AiNode;
use crate::extension::{as_node, extension_node};
use crate::functions::{is_ai_call, is_quail_call};
use crate::region::ModelRegion;
use crate::region::identified::IdentifiedTable;
use crate::region::join_back::join_back;
use crate::region::lift::lift_relational;
use crate::region::shared::{ROW_COLUMN, SharedInput, row_field};
use quail_planner::expressions::{CompileError, UNSUPPORTED};

/// The name of [`ClaimModelRegions`], which runs after every other
/// optimizer rule.
pub const CLAIM_RULE: &str = "claim_model_regions";

/// Replace each part of the optimized plan that holds `AiNode`s with one
/// `ModelRegion`, and leave the relational operators above it to
/// DataFusion, as the module explains.
///
/// DataFusion's optimizer runs each rule once per pass, and a region's
/// operators are out of the later passes' reach, so the rule first runs
/// the rules listed before it until the plan stops changing.
///
/// A region's inputs are the parts DataFusion computes for it: each
/// table's plain filters, and each CTE or FROM subquery that groups or
/// windows rows or that the region does not read through.
#[derive(Debug)]
pub struct ClaimModelRegions {
    rules: Vec<Arc<dyn OptimizerRule + Send + Sync>>,
}

impl ClaimModelRegions {
    /// The claim after `rules`, the optimizer rules listed before it.
    pub fn new(rules: Vec<Arc<dyn OptimizerRule + Send + Sync>>) -> Self {
        Self { rules }
    }
}

impl OptimizerRule for ClaimModelRegions {
    fn name(&self) -> &str {
        CLAIM_RULE
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if !calls_ai(&plan)? {
            return Ok(Transformed::no(plan));
        }
        let optimized =
            Optimizer::with_rules(self.rules.clone()).optimize(plan, config, |_, _| {})?;
        let claimed = claim_regions(optimized, None)?;
        drop_unread_probabilities(claimed).map(Transformed::yes)
    }
}

/// `plan` with each classification that asks for its labels'
/// probabilities asking for none where nothing reads them.
///
/// A region's probabilities column is not a column of its DataFusion
/// schema: only a region that is the whole query returns it, as the
/// column after its label in the SELECT list. A classification only a
/// test reads, or one in a region below operators DataFusion runs,
/// scores its labels without the probabilities readout.
fn drop_unread_probabilities(plan: LogicalPlan) -> Result<LogicalPlan> {
    let whole = as_node::<ModelRegion>(&plan).is_some();
    plan.transform_up(|node| {
        let Some(region) = as_node::<ModelRegion>(&node) else {
            return Ok(Transformed::no(node));
        };
        let returned: Vec<String> = if whole {
            region
                .plan
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect()
        } else {
            Vec::new()
        };
        let mut unread = Vec::new();
        region.plan.apply(|found| {
            if let Some(ai) = as_node::<AiNode>(found)
                && ai.function() == "ai_classify"
                && !returned.contains(&ai.answer)
                && let Some(call) = without_probabilities(&ai.call)
                && !unread.iter().any(|(known, _)| *known == call.0)
            {
                unread.push(call);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        if unread.is_empty() {
            return Ok(Transformed::no(node));
        }
        let mut region = region.clone();
        for (old, new) in &unread {
            region.plan = crate::rules::projection::replace_call(region.plan, old, new)?;
        }
        Ok(Transformed::yes(extension_node(region)))
    })
    .map(|done| done.data)
}

/// A classification call unfolded, and the same call with its
/// `probabilities` option false, when the option is true.
fn without_probabilities(call: &Expr) -> Option<(Expr, Expr)> {
    use datafusion::logical_expr::expr::ScalarFunction;
    let Expr::ScalarFunction(unfolded) = crate::derive::unfolded(call.clone()) else {
        return None;
    };
    let Some(Expr::ScalarFunction(config)) = unfolded.args.get(2) else {
        return None;
    };
    if config.name() != "named_struct" {
        return None;
    }
    let mut options = config.args.clone();
    let mut asked = false;
    for pair in options.chunks_mut(2) {
        let [key, value] = pair else {
            return None;
        };
        let named = matches!(key, Expr::Literal(found, _)
            if found.try_as_str().flatten().is_some_and(|name| name.eq_ignore_ascii_case("probabilities")));
        if named && matches!(value, Expr::Literal(ScalarValue::Boolean(Some(true)), _)) {
            *value = Expr::Literal(ScalarValue::Boolean(Some(false)), None);
            asked = true;
        }
    }
    if !asked {
        return None;
    }
    let mut args = unfolded.args.clone();
    args[2] = Expr::ScalarFunction(ScalarFunction::new_udf(Arc::clone(&config.func), options));
    let pruned = Expr::ScalarFunction(ScalarFunction::new_udf(Arc::clone(&unfolded.func), args));
    Some((Expr::ScalarFunction(unfolded), pruned))
}

/// Claim the model regions of an optimized plan.
///
/// `fetch` is the most rows the operators above read from `plan`, when
/// a `Limit` above passes them on through projections only; a region
/// there may stop the model once it has that many.
fn claim_regions(plan: LogicalPlan, fetch: Option<usize>) -> Result<LogicalPlan> {
    if !calls_ai(&plan)? {
        return Ok(plan);
    }
    let mut refused = None;
    if !kept_by_datafusion(&plan)? {
        let selected = select_list(plan.clone(), None)?;
        let (bound, inputs, _) = split_inputs(selected)?;
        let numbered_inputs = numbered(&bound);
        let (bound, above) = lift_relational(bound)?;
        let names: Vec<String> = inputs.iter().map(|(name, _)| name.clone()).collect();
        let (bound, back) = join_back(bound, &names)?;
        if calls_ai(&bound)? && quail_plans(&bound)? {
            refused = refusal_of(&bound);
            if refused.is_none() {
                let mut region = ModelRegion::new(bound);
                // a fetch counts the rows the region returns, not the
                // rows a lifted test or UNNEST keeps of them
                region.fetch = fetch.filter(|_| above.keeps_rows());
                let shared = back.inputs();
                let mut read = Vec::new();
                for (name, input) in inputs {
                    let mut input = claim_regions(input, None)?;
                    let numbered = numbered_inputs.contains(&name);
                    if numbered || shared.contains(&name) {
                        input = extension_node(SharedInput::try_new(input, numbered)?);
                    }
                    read.push((name.clone(), input.clone()));
                    region.inputs.push(input);
                    region.input_names.push(name);
                }
                return above.build(back.build(extension_node(region), &read)?);
            }
        }
    }
    // DataFusion runs this operator; its inputs are claimed below it
    if as_node::<AiNode>(&plan).is_some()
        || plan
            .expressions()
            .iter()
            .any(|expr| expr.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(false))
    {
        return plan_err!(
            "{}",
            refused.unwrap_or_else(|| format!(
                "a model call in this operator {UNSUPPORTED}:\n{}",
                plan.display()
            ))
        );
    }
    let passed = passed_fetch(&plan, fetch);
    let inputs = plan
        .inputs()
        .into_iter()
        .cloned()
        .map(|input| claim_regions(input, passed))
        .collect::<Result<Vec<_>>>()?;
    plan.with_new_exprs(plan.expressions(), inputs)
}

/// The most rows `plan`'s operator reads from its input, when it is a
/// `Limit`, or passes `fetch` on from above without changing the rows.
fn passed_fetch(plan: &LogicalPlan, fetch: Option<usize>) -> Option<usize> {
    use datafusion::logical_expr::{FetchType, SkipType};
    match plan {
        LogicalPlan::Limit(limit) => match (limit.get_skip_type(), limit.get_fetch_type()) {
            (Ok(SkipType::Literal(skip)), Ok(FetchType::Literal(Some(rows)))) => {
                Some(skip + fetch.map_or(rows, |above| above.min(rows)))
            }
            _ => None,
        },
        LogicalPlan::Projection(_) | LogicalPlan::SubqueryAlias(_) => fetch,
        _ => None,
    }
}

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

    /// Claim the model regions of `plan`, as [`ClaimModelRegions`] explains.
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

/// Whether DataFusion runs this operator, as the module explains: a
/// `DISTINCT` past DataFusion's optimizer is an `Aggregate`.
fn kept_by_datafusion(plan: &LogicalPlan) -> Result<bool> {
    if as_node::<AiNode>(plan).is_some() {
        return Ok(false);
    }
    // these run in DataFusion with their AI calls moved below them
    if matches!(
        plan,
        LogicalPlan::Aggregate(_)
            | LogicalPlan::Window(_)
            | LogicalPlan::Sort(_)
            | LogicalPlan::Limit(_)
            | LogicalPlan::Distinct(_)
            | LogicalPlan::Unnest(_)
    ) {
        return Ok(true);
    }
    for expr in plan.expressions() {
        if expr.exists(|inner| Ok(is_ai_call(inner)))? {
            return Ok(false);
        }
    }
    Ok(match plan {
        LogicalPlan::Projection(_) | LogicalPlan::Filter(_) | LogicalPlan::SubqueryAlias(_) => {
            let input = plan.inputs()[0];
            calls_ai(input)? && kept_by_datafusion(input)?
        }
        _ => false,
    })
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

/// Quail's refusal of `plan` as a region, when the refusal is of a shape
/// DataFusion can run instead; any other refusal is the query's error,
/// reported when the region is planned.
///
/// The trial lowers `plan`, coerces its types, and runs the default
/// optimizer rules over it ([`crate::rules::LOGICAL_RULES`]), as the
/// session would.
///
/// # Errors
///
/// Type coercion fails and Quail's binder names the cause, such as a
/// test of a feature list that is not unnested; DataFusion's own error
/// would follow from the same query.
fn refusal(plan: &LogicalPlan, config: &ConfigOptions) -> Result<Option<String>> {
    // a part that does not lower or optimize fails when its region plans
    let Ok(lowered) = crate::lower_ai_calls(plan.clone()) else {
        return Ok(None);
    };
    let coerced = match Analyzer::new().execute_and_check(lowered.clone(), config, |_, _| {}) {
        Ok(coerced) => coerced,
        Err(_) => {
            let rules = crate::rules::QUAIL_RULES
                .iter()
                .map(|name| crate::rules::logical_rule(name))
                .collect::<Result<Vec<_>>>()?;
            let placed = Optimizer::with_rules(rules)
                .optimize(lowered, &OptimizerContext::new(), |_, _| {})
                .and_then(|placed| select_list(placed, None))
                .and_then(|selected| lift_relational(selected).map(|(region, _)| region));
            return match placed.map(|placed| derive_error(&placed)) {
                Ok(Some(error)) => plan_err!("{}", error.0),
                _ => Ok(None),
            };
        }
    };
    let context = OptimizerContext::new_with_config_options(Arc::new(config.clone()));
    let Ok(optimized) =
        Optimizer::with_rules(crate::rules::default_rules()).optimize(coerced, &context, |_, _| {})
    else {
        return Ok(None);
    };
    let selected = select_list(optimized, None)?;
    let (region, _) = lift_relational(selected)?;
    Ok(refusal_of(&region))
}

/// Quail's binder's error for `plan` as a region, if it has one.
fn derive_error(plan: &LogicalPlan) -> Option<CompileError> {
    // the chat turn and layout change the prompts, not what Quail reads
    let turn: quail_planner::prompts::Turn = (String::new(), String::new());
    let region = ModelRegion::new(plan.clone());
    crate::derive::derive_region(&region, &turn, "ai-if").err()
}

/// Quail's refusal of `plan`, a region's operators after DataFusion's
/// optimizer, when the refusal is of a shape DataFusion can run instead.
fn refusal_of(plan: &LogicalPlan) -> Option<String> {
    derive_error(plan)
        .filter(CompileError::is_unsupported)
        .map(|error| error.0)
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
/// each reads: each subquery that selects and filters one table, with
/// AI.IF filters but no other model call, and is read once. Any other
/// FROM or JOIN subquery runs before the region, as its own input.
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
            // DataFusion's optimizer removes a SELECT list that passes its
            // input's columns on, and lowers an AI.IF filter to an `AiNode`
            let (mut base, mut calls_model) = match body {
                LogicalPlan::Projection(projection) => (
                    projection.input.as_ref(),
                    projection
                        .expr
                        .iter()
                        .any(|expr| expr.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(false)),
                ),
                other => (other, false),
            };
            loop {
                base = match base {
                    LogicalPlan::Filter(filter) => filter.input.as_ref(),
                    LogicalPlan::Extension(_) => match as_node::<AiNode>(base) {
                        Some(node) => {
                            calls_model |= !node.is_predicate()
                                || node.semantics != crate::ai::Semantics::Rows;
                            node.input.as_ref()
                        }
                        None => break,
                    },
                    _ => break,
                };
            }
            let is_table = match base {
                LogicalPlan::SubqueryAlias(found) => {
                    let mut scanned = found.input.as_ref();
                    while let LogicalPlan::Filter(filter) = scanned {
                        scanned = filter.input.as_ref();
                    }
                    matches!(scanned, LogicalPlan::TableScan(_))
                }
                _ => false,
            };
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
    // a filter or limit DataFusion moved into the scan is the input's
    scan.filters.is_empty()
        && scan.fetch.is_none()
        && source_as_provider(&scan.source)
            .is_ok_and(|provider| provider.downcast_ref::<IdentifiedTable>().is_some())
}

/// The inputs a region's plan reads with [`ROW_COLUMN`], by name: those
/// without an id column of their own.
fn numbered(plan: &LogicalPlan) -> Vec<String> {
    let mut found = Vec::new();
    let _ = plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && scan.source.schema().index_of(ROW_COLUMN).is_ok()
        {
            found.push(scan.table_name.table().to_string());
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

/// The schema of a placeholder for a region input: the input's columns,
/// then [`ROW_COLUMN`] when `numbered`, as [`SharedInput`] returns them.
fn placeholder_schema(
    schema: &datafusion::arrow::datatypes::Schema,
    numbered: bool,
) -> Arc<datafusion::arrow::datatypes::Schema> {
    let mut fields: Vec<_> = schema.fields().iter().cloned().collect();
    if numbered {
        fields.push(Arc::new(row_field()));
    }
    Arc::new(datafusion::arrow::datatypes::Schema::new(fields))
}

/// Whether DataFusion computes this part of a region as an input: it
/// calls no model, reads one table, and is more than an identified table
/// the runner reads by name.
fn plain_input(plan: &LogicalPlan) -> Result<Option<TableScan>> {
    // DataFusion computes a part's subqueries too, such as a filter
    // `stars > (SELECT AVG(stars) ...)` its optimizer moved to the table
    if read_by_name(plan) || calls_ai(plan)? {
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
/// Returns the plan, each input's table name and plan, and the parts the
/// inputs replaced.
fn split_inputs(plan: LogicalPlan) -> Result<Split> {
    let mut inputs: Vec<(String, LogicalPlan)> = Vec::new();
    let mut reads: Vec<Read> = Vec::new();
    let merged = merged_aliases(&plan);
    let mut root = true;
    let plan = plan
        .transform_down(|node| {
            // a subquery at the root is the region itself
            let at_root = std::mem::take(&mut root);
            let cut = if let Some(scan) = plain_input(&node)? {
                Some(cut_input(node.clone(), &scan, &mut inputs)?)
            } else if let LogicalPlan::TableScan(scan) = &node {
                Some(table_input(scan, &mut inputs)?)
            } else {
                None
            };
            if let Some(cut) = cut {
                if cut.transformed {
                    reads.push(Read {
                        read: cut.data.clone(),
                        original: node,
                        body: None,
                    });
                }
                return Ok(cut);
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
            // a CTE's rows have no id column; the region numbers them
            let schema = placeholder_schema(body.schema().as_arrow(), true);
            let Ok(table) = MemTable::try_new(schema, vec![vec![]]) else {
                return Ok(Transformed::no(node));
            };
            let read =
                LogicalPlanBuilder::scan(name.as_str(), provider_as_source(Arc::new(table)), None)?
                    .alias(alias.alias.clone())?
                    .build()?;
            reads.push(Read {
                read: read.clone(),
                original: node.clone(),
                body: Some(name.clone()),
            });
            if !read_before {
                inputs.push((name, body));
            }
            Ok(Transformed::new(read, true, TreeNodeRecursion::Jump))
        })?
        .data;
    Ok((plan, inputs, reads))
}

/// A region's plan, each input's table name and plan, and the parts of
/// the region the inputs replaced, as [`split_inputs`] returns them.
type Split = (LogicalPlan, Vec<(String, LogicalPlan)>, Vec<Read>);

/// A part of a region that [`split_inputs`] replaced with a scan of an
/// input.
#[derive(Debug)]
struct Read {
    /// The scan that replaced it.
    read: LogicalPlan,
    /// The part it replaced.
    original: LogicalPlan,
    /// The input that is the body of a subquery it replaced, if it is one.
    body: Option<String>,
}

/// `plan` with each scan of an input that [`split_inputs`] made replaced
/// by the part it replaced again, a subquery over its claimed body.
fn read_inputs(
    plan: LogicalPlan,
    reads: &[Read],
    claimed: &[(String, LogicalPlan)],
) -> Result<LogicalPlan> {
    plan.transform_down(|node| {
        let Some(found) = reads.iter().find(|read| read.read == node) else {
            return Ok(Transformed::no(node));
        };
        let body = found
            .body
            .as_ref()
            .and_then(|name| claimed.iter().find(|(input, _)| input == name));
        let original = match (body, &found.original) {
            (Some((_, body)), LogicalPlan::SubqueryAlias(alias)) => {
                let inner = match alias.input.as_ref() {
                    LogicalPlan::SubqueryAlias(cte) => LogicalPlan::SubqueryAlias(
                        SubqueryAlias::try_new(Arc::new(body.clone()), cte.alias.clone())?,
                    ),
                    _ => body.clone(),
                };
                LogicalPlan::SubqueryAlias(SubqueryAlias::try_new(
                    Arc::new(inner),
                    alias.alias.clone(),
                )?)
            }
            _ => found.original.clone(),
        };
        Ok(Transformed::new(original, true, TreeNodeRecursion::Jump))
    })
    .map(|done| done.data)
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
    // the input's name ends with the alias after the last separator, which
    // an alias with the separator in it would make ambiguous; the region
    // then reads the table itself
    if qualifier
        .table()
        .contains(quail_planner::operators::INPUT_SEPARATOR)
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
    let provider = source_as_provider(&scan.source)?;
    let key = provider.constraints().and_then(|constraints| {
        constraints.iter().find_map(|constraint| match constraint {
            datafusion::common::Constraint::PrimaryKey(indices) if indices.len() == 1 => {
                Some(indices[0])
            }
            _ => None,
        })
    });
    // the region reads each document by its id, which DataFusion's
    // projection pruning drops when no operator reads it
    let key = key.map(|index| (index, provider.schema().field(index).name().clone()));
    let node = match &key {
        Some((index, name)) => with_key(node, *index, name)?,
        None => node,
    };
    let key = key.map(|(_, name)| name);
    let fields = placeholder_schema(node.schema().as_arrow(), key.is_none());
    let mut placeholder = MemTable::try_new(Arc::clone(&fields), vec![vec![]])?;
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

/// `plan`, a part that reads one table, returning the table's column
/// `name` at `index` too: each scan reads it, and each `Projection`
/// passes it on.
fn with_key(plan: LogicalPlan, index: usize, name: &str) -> Result<LogicalPlan> {
    plan.transform_up(|node| match node {
        LogicalPlan::TableScan(scan) => match scan.projection.clone() {
            Some(mut projection) if !projection.contains(&index) => {
                projection.push(index);
                let read = TableScanBuilder::new(scan.table_name, scan.source)
                    .with_projection(Some(projection))
                    .with_filters(scan.filters)
                    .with_fetch(scan.fetch)
                    .build()?;
                Ok(Transformed::yes(LogicalPlan::TableScan(read)))
            }
            _ => Ok(Transformed::no(LogicalPlan::TableScan(scan))),
        },
        LogicalPlan::Projection(projection) => {
            let found: Vec<Column> = projection
                .input
                .schema()
                .iter()
                .filter(|(_, field)| field.name() == name)
                .map(|(qualifier, field)| Column::new(qualifier.cloned(), field.name()))
                .collect();
            match <[Column; 1]>::try_from(found) {
                Ok([column]) if !projection.schema.has_column(&column) => {
                    let mut exprs = projection.expr.clone();
                    exprs.push(Expr::Column(column));
                    Ok(Transformed::yes(LogicalPlan::Projection(
                        Projection::try_new(exprs, Arc::clone(&projection.input))?,
                    )))
                }
                _ => Ok(Transformed::no(LogicalPlan::Projection(projection))),
            }
        }
        other => other.recompute_schema().map(Transformed::yes),
    })
    .map(|done| done.data)
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
    let keyed = provider.constraints().is_some_and(|constraints| {
        constraints.iter().any(|constraint| {
            matches!(constraint, datafusion::common::Constraint::PrimaryKey(indices) if indices.len() == 1)
        })
    });
    // a table without an id column has its rows numbered
    let mut placeholder = MemTable::try_new(placeholder_schema(&schema, !keyed), vec![vec![]])?;
    let mut projection = scan.projection.clone();
    if keyed && let Some(constraints) = provider.constraints() {
        placeholder = placeholder.with_constraints(constraints.clone());
    } else if let Some(read) = &mut projection {
        read.push(schema.fields().len());
    }
    let read = LogicalPlanBuilder::scan_with_filters(
        scan.table_name.clone(),
        provider_as_source(Arc::new(placeholder)),
        projection,
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
    if matches!(plan, LogicalPlan::Projection(_)) {
        return Ok(plan);
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

/// Whether any operator in the subtree is an `AiNode`, or calls an AI
/// function in its expressions or in an EXISTS or IN subquery.
fn calls_ai(plan: &LogicalPlan) -> Result<bool> {
    Ok(plan.exists(|node| Ok(as_node::<AiNode>(node).is_some()))? || calls_ai_function(plan)?)
}

/// Whether any operator in the subtree, or an EXISTS or IN subquery in
/// it, calls an AI function that is not yet lowered into an `AiNode`.
fn calls_ai_function(plan: &LogicalPlan) -> Result<bool> {
    plan.exists(|node| {
        // an `AiNode`'s expressions hold its lowered call
        if as_node::<AiNode>(node).is_some() {
            return Ok(false);
        }
        for expr in node.expressions() {
            let found = expr.exists(|inner| match inner {
                Expr::Exists(exists) => calls_ai_function(&exists.subquery.subquery),
                Expr::InSubquery(found) => calls_ai_function(&found.subquery.subquery),
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
    ) || as_node::<AiNode>(plan).is_some()
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
