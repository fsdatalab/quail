//! Claim optimized model regions and choose their boundaries.
//! Leave unsupported operators and relational result work in DataFusion.

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, ScalarValue, plan_err};
use datafusion::logical_expr::{Expr, LogicalPlan, Projection};
use datafusion::optimizer::{
    Analyzer, Optimizer, OptimizerConfig, OptimizerContext, OptimizerRule,
};
use quail_planner::error::{CompileError, UNSUPPORTED};

use super::ModelRegion;
use super::inputs::{numbered, split_inputs};
use super::join_back::join_back;
use super::lift::lift_relational;
use super::shared::SharedInput;
use crate::ai::AiNode;
use crate::extension::{as_node, extension_node};
use crate::functions::is_ai_call;

pub(crate) use super::lowering::lower_regions;

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

/// Whether DataFusion runs this operator, as the module explains: a
/// `DISTINCT` past DataFusion's optimizer is an `Aggregate`.
pub(super) fn kept_by_datafusion(plan: &LogicalPlan) -> Result<bool> {
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
pub(super) fn refusal(plan: &LogicalPlan, config: &ConfigOptions) -> Result<Option<String>> {
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
                Ok(Some(error)) => plan_err!("{}", error.message()),
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
    Ok(plan.exists(|node| Ok(as_node::<AiNode>(node).is_some()))? || calls_ai_function(plan)?)
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
