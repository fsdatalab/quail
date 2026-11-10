//! Claim optimized model regions and choose their boundaries.
//! Leave unsupported operators and relational result work in DataFusion.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, ScalarValue, plan_err};
use datafusion::execution::SessionState;
use datafusion::logical_expr::{Expr, LogicalPlan, Projection};
use datafusion::optimizer::{Analyzer, Optimizer, OptimizerContext};
use quail_planner::error::CompileError;

use super::ModelRegion;
use super::inputs::{numbered, split_inputs};
use super::join_back::join_back;
use super::lift::lift_relational;
use super::shared::SharedInput;
use crate::ai::AiNode;
use crate::extension::{as_node, extension_node};
use crate::functions::is_ai_call;

pub(crate) use super::lowering::lower_regions;

/// Replace each part of an optimized plan that holds `AiNode`s with one
/// `ModelRegion`, leave the relational operators above it to DataFusion,
/// and optimize what the claim built around the regions.
///
/// `plan` is the plan as the session's optimizer left it: its rules have
/// moved every filter, limit, and guard they can into the `AiNode`s. A
/// region hides its operators from DataFusion, so the claim comes after
/// those rules, not among them: [`crate::QuailQueryPlanner`] calls it
/// before physical planning, and a host with its own query planner does
/// the same.
///
/// A region's inputs are the parts DataFusion computes for it: each
/// table's plain filters, and each CTE or FROM subquery that groups or
/// windows rows or that the region does not read through. The claim
/// joins a region's answers back to the columns its tables hold, and
/// the session's optimizer then runs once more, over the regions as
/// single operators, to prune and place that work.
///
/// # Errors
///
/// A model call is where no region can run it, or `session` is not
/// DataFusion's `SessionState`, whose optimizer the claim runs.
pub fn claim_model_regions(plan: LogicalPlan, session: &dyn Session) -> Result<LogicalPlan> {
    let Some(state) = session.as_any().downcast_ref::<SessionState>() else {
        return plan_err!("claiming model regions needs DataFusion's SessionState");
    };
    // EXPLAIN holds the plan it explains, which the optimizer left optimized
    match plan {
        LogicalPlan::Explain(mut explain) => {
            explain.plan = Arc::new(claim_model_regions(
                Arc::unwrap_or_clone(explain.plan),
                session,
            )?);
            return Ok(LogicalPlan::Explain(explain));
        }
        LogicalPlan::Analyze(mut analyze) => {
            analyze.input = Arc::new(claim_model_regions(
                Arc::unwrap_or_clone(analyze.input),
                session,
            )?);
            return Ok(LogicalPlan::Analyze(analyze));
        }
        _ => {}
    }
    if !calls_ai(&plan)? {
        // the analyzer has already made a fallback column's region
        return number_regions(drop_unread_probabilities(plan)?);
    }
    let claimed = number_regions(drop_unread_probabilities(claim_regions(plan, None)?)?)?;
    state.optimize(&claimed)
}

/// Number occurrences, not plan contents: identical sibling regions still run
/// separately. Repeated readers of a SharedInput keep the same numbered subtree.
/// Numbering is query-local, so inspection and execution agree even though
/// SharedInput's process-global keys differ between compilations.
fn number_regions(plan: LogicalPlan) -> Result<LogicalPlan> {
    fn visit(
        plan: LogicalPlan,
        next: &mut usize,
        shared: &mut HashMap<u64, LogicalPlan>,
    ) -> Result<LogicalPlan> {
        plan.transform_down_with_subqueries(|node| {
            if let Some(input) = as_node::<SharedInput>(&node) {
                let numbered = if let Some(known) = shared.get(&input.key) {
                    known.clone()
                } else {
                    let mut input = input.clone();
                    input.input = visit(input.input, next, shared)?;
                    let key = input.key;
                    let numbered = extension_node(input);
                    shared.insert(key, numbered.clone());
                    numbered
                };
                return Ok(Transformed::new(numbered, true, TreeNodeRecursion::Jump));
            }
            if let Some(region) = as_node::<ModelRegion>(&node) {
                let mut region = region.clone();
                region.id = Some(*next);
                *next += 1;
                return Ok(Transformed::yes(extension_node(region)));
            }
            Ok(Transformed::no(node))
        })
        .map(|done| done.data)
    }
    visit(plan, &mut 0, &mut HashMap::new())
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
    // Expression subqueries are not plan inputs. Claim their model work
    // explicitly, including scalar subqueries that DataFusion keeps intact.
    let plan = plan
        .map_expressions(|expr| {
            expr.transform_up(|expr| match expr {
                Expr::ScalarSubquery(mut found) => {
                    found.subquery =
                        Arc::new(claim_regions(found.subquery.as_ref().clone(), None)?);
                    Ok(Transformed::yes(Expr::ScalarSubquery(found)))
                }
                Expr::InSubquery(mut found) => {
                    found.subquery.subquery = Arc::new(claim_regions(
                        found.subquery.subquery.as_ref().clone(),
                        None,
                    )?);
                    Ok(Transformed::yes(Expr::InSubquery(found)))
                }
                Expr::Exists(mut found) => {
                    found.subquery.subquery = Arc::new(claim_regions(
                        found.subquery.subquery.as_ref().clone(),
                        None,
                    )?);
                    Ok(Transformed::yes(Expr::Exists(found)))
                }
                other => Ok(Transformed::no(other)),
            })
        })?
        .data;
    if !calls_ai(&plan)? {
        return Ok(plan);
    }
    if ai_only()
        && let Some((chain, restore)) = isolate_chain(&plan)?
    {
        let claimed = claim_regions(chain, fetch)?;
        return Ok(LogicalPlan::Projection(Projection::try_new(
            restore,
            Arc::new(claimed),
        )?));
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
                region.fetch = fetch
                    .filter(|_| above.keeps_rows() && !crate::ai::any_filters_on_or(&region.plan));
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
                "This operator cannot contain a model call:\n{}",
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

/// Check a refused placeholder trial against the real lowered inputs. Optimizer
/// rewrites can move computed columns into those inputs, making the native plan
/// bind even when the placeholder-only trial could not see them.
pub(super) fn native_plan_fits(plan: &LogicalPlan, config: &ConfigOptions) -> bool {
    let context = OptimizerContext::new_with_config_options(Arc::new(config.clone()));
    let claimed = Analyzer::new()
        .execute_and_check(plan.clone(), config, |_, _| {})
        .and_then(|plan| {
            Optimizer::with_rules(crate::rules::default_rules()).optimize(plan, &context, |_, _| {})
        })
        .and_then(|plan| claim_regions(plan, None));
    let Ok(claimed) = claimed else { return false };
    if calls_ai(&claimed).unwrap_or(true) {
        return false;
    }
    !claimed
        .exists(|node| {
            Ok(as_node::<ModelRegion>(node)
                .is_some_and(|region| derive_error(&region.plan).is_some()))
        })
        .unwrap_or(true)
}

/// Seal a fallback column before optimization can absorb the surrounding SQL
/// expression into it again. Its rows are normal staged DataFusion inputs.
pub(super) fn claim_isolated(
    bound: LogicalPlan,
    inputs: Vec<(String, LogicalPlan)>,
    config: &ConfigOptions,
) -> Result<LogicalPlan> {
    let lowered = crate::lower_ai_calls(bound)?;
    let coerced = Analyzer::new().execute_and_check(lowered, config, |_, _| {})?;
    let context = OptimizerContext::new_with_config_options(Arc::new(config.clone()));
    let optimized = Optimizer::with_rules(crate::rules::default_rules()).optimize(
        coerced,
        &context,
        |_, _| {},
    )?;
    let bound = select_list(optimized, None)?;
    let numbered_inputs = numbered(&bound);
    let (bound, above) = lift_relational(bound)?;
    let names = inputs
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let (bound, back) = join_back(bound, &names)?;
    if let Some(error) = derive_error(&bound) {
        return plan_err!("{}", error.message());
    }
    let shared = back.inputs();
    let mut region = ModelRegion::new(bound);
    let mut read = Vec::new();
    for (name, input) in inputs {
        // Lowering's native trial optimizes a copy; these staged inputs still
        // need those same rules before they can be claimed early.
        let input = Analyzer::new().execute_and_check(input, config, |_, _| {})?;
        let input = Optimizer::with_rules(crate::rules::default_rules()).optimize(
            input,
            &context,
            |_, _| {},
        )?;
        let mut input = claim_regions(input, None)?;
        let numbered = numbered_inputs.contains(&name);
        if numbered || shared.contains(&name) {
            input = extension_node(SharedInput::try_new(input, numbered)?);
        }
        read.push((name.clone(), input.clone()));
        region.inputs.push(input);
        region.input_names.push(name);
    }
    above.build(back.build(extension_node(region), &read)?)
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

/// Whether DataFusion runs this operator, as the module explains: a
/// `DISTINCT` past DataFusion's optimizer is an `Aggregate`.
pub(super) fn kept_by_datafusion(plan: &LogicalPlan) -> Result<bool> {
    if as_node::<AiNode>(plan).is_some() {
        return Ok(false);
    }
    if ai_only() {
        return Ok(true);
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

/// Quail's refusal of `plan` as a region, when the refusal is of a shape
/// DataFusion can run instead; any other refusal is the query's error,
/// reported when the region is planned.
///
/// The trial lowers `plan`, coerces its types, and runs the default
/// optimizer rules over it ([`crate::rules::default_rule_names`]), as the
/// session would.
///
/// # Errors
///
/// Type coercion fails and Quail's binder names the cause, such as a
/// test of a feature list that is not unnested; DataFusion's own error
/// would follow from the same query.
pub(super) fn refusal(plan: &LogicalPlan, config: &ConfigOptions) -> Result<Option<String>> {
    // The original SQL has already bound. Failure to construct a model region
    // is a reason to isolate its calls, not permission to claim a broken plan.
    let lowered = match crate::lower_ai_calls(plan.clone()) {
        Ok(lowered) => lowered,
        Err(error) => return Ok(Some(error.to_string())),
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
                Ok(Some(error)) if error.is_unsupported() => Ok(Some(error.message().to_string())),
                Ok(Some(error)) => plan_err!("{}", error.message()),
                _ => Ok(None),
            };
        }
    };
    let context = OptimizerContext::new_with_config_options(Arc::new(config.clone()));
    let optimized = match Optimizer::with_rules(crate::rules::default_rules()).optimize(
        coerced,
        &context,
        |_, _| {},
    ) {
        Ok(optimized) => optimized,
        Err(error) => return Ok(Some(error.to_string())),
    };
    match select_list(optimized, None).and_then(lift_relational) {
        Ok((region, _)) => Ok(refusal_of(&region)),
        // SQL was already bound by DataFusion. A schema failure while moving
        // its expressions is a planner limitation, not a user argument error.
        Err(error) => Ok(Some(error.to_string())),
    }
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
        .map(|error| error.to_string())
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

/// A region's plan with a SELECT list at its root: the plan itself when
/// it has one, or a `Projection` of the `needed` columns over it.
pub(super) fn select_list(plan: LogicalPlan, needed: Option<Vec<Column>>) -> Result<LogicalPlan> {
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
pub(super) fn calls_ai(plan: &LogicalPlan) -> Result<bool> {
    plan.exists(|node| {
        if as_node::<AiNode>(node).is_some() {
            return Ok(true);
        }
        for expr in node.expressions() {
            if expr.exists(|inner| match inner {
                Expr::Exists(found) => calls_ai(&found.subquery.subquery),
                Expr::InSubquery(found) => calls_ai(&found.subquery.subquery),
                Expr::ScalarSubquery(found) => calls_ai(&found.subquery),
                other => Ok(is_ai_call(other)),
            })? {
                return Ok(true);
            }
        }
        Ok(false)
    })
}

/// Whether any operator in the subtree, or an EXISTS or IN subquery in
/// it, calls an AI function that is not yet lowered into an `AiNode`.
pub(super) fn calls_ai_function(plan: &LogicalPlan) -> Result<bool> {
    plan.exists(|node| {
        // an `AiNode`'s expressions hold its lowered call
        if as_node::<AiNode>(node).is_some() {
            return Ok(false);
        }
        for expr in node.expressions() {
            let found = expr.exists(|inner| match inner {
                Expr::Exists(exists) => calls_ai_function(&exists.subquery.subquery),
                Expr::InSubquery(found) => calls_ai_function(&found.subquery.subquery),
                Expr::ScalarSubquery(found) => calls_ai_function(&found.subquery),
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
pub(super) fn quail_plans(plan: &LogicalPlan) -> Result<bool> {
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
pub(super) fn is_subquery(expr: &Expr) -> bool {
    match expr {
        Expr::Exists(exists) => !quail_plans(&exists.subquery.subquery).unwrap_or(false),
        Expr::InSubquery(_) | Expr::ScalarSubquery(_) => true,
        _ => false,
    }
}

/// SPIKE: whether regions hold AI operators only.
fn ai_only() -> bool {
    std::env::var("QUAIL_AI_ONLY").is_ok()
}

static NEXT_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// SPIKE: the run of `AiNode`s at `plan`'s root over its input as an
/// opaque relation DataFusion computes, and the columns that restore the
/// run's own names. None for an AI join or a run already over such a
/// relation.
fn isolate_chain(plan: &LogicalPlan) -> Result<Option<(LogicalPlan, Vec<Expr>)>> {
    let mut nodes = Vec::new();
    let mut bottom = plan;
    while let Some(node) = as_node::<AiNode>(bottom) {
        nodes.push(node);
        bottom = node.input.as_ref();
    }
    if nodes.is_empty() || matches!(bottom, LogicalPlan::Join(_)) {
        return Ok(None);
    }
    if let LogicalPlan::SubqueryAlias(alias) = bottom
        && as_node::<SharedInput>(alias.input.as_ref()).is_some()
    {
        return Ok(None);
    }
    let columns = bottom.schema().columns();
    let renamed: Vec<Expr> = columns
        .iter()
        .enumerate()
        .map(|(index, column)| Expr::Column(column.clone()).alias(format!("column_{index}")))
        .collect();
    let number = NEXT_ROWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let input = datafusion::logical_expr::LogicalPlanBuilder::from(bottom.clone())
        .project(renamed)?
        .build()?;
    let input = extension_node(SharedInput::try_new(input, false)?);
    let mut input = datafusion::logical_expr::LogicalPlanBuilder::from(input)
        .alias(format!("__quail_rows_{number}"))?
        .build()?;
    let replacements = input.schema().columns();
    let remap = |expr: Expr| -> Result<Expr> {
        Ok(expr
            .transform_up(|expr| {
                if let Expr::Column(column) = &expr
                    && let Some(index) = columns.iter().position(|found| found == column)
                {
                    return Ok(Transformed::yes(Expr::Column(replacements[index].clone())));
                }
                Ok(Transformed::no(expr))
            })?
            .data)
    };
    for node in nodes.iter().rev() {
        input = extension_node(node.map_expressions(&remap)?.with_input(input)?);
    }
    let restore = plan
        .schema()
        .iter()
        .zip(input.schema().columns())
        .map(|((qualifier, field), column)| {
            Expr::Column(column).alias_qualified(qualifier.cloned(), field.name())
        })
        .collect();
    Ok(Some((input, restore)))
}
