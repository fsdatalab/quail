//! Claim optimized model regions and choose their boundaries.
//! Leave unsupported operators and relational result work in DataFusion.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, ScalarValue, plan_err};
use datafusion::execution::SessionState;
use datafusion::logical_expr::{Expr, LogicalPlan, Projection};
use datafusion::optimizer::Optimizer;
use quail_planner::error::CompileError;

use super::ModelRegion;
use super::inputs::{numbered, split_inputs};
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
/// windows rows or that the region does not read through. A region
/// returns the columns of its tables that the query above reads, beside
/// its answers. The session's optimizer then runs once more, over the
/// regions as single operators, to prune and place the work above them.
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
    NEXT_ROWS.with(|next| next.set(0));
    let plan = super::unread::drop_unread_nodes(plan)?;
    let claimed = number_regions(drop_unread_probabilities(claim_regions(plan, None)?)?)?;
    // DataFusion 55.2's two leaf-expression rules fail on the join the
    // claim builds above a region: they move a struct field read below the
    // join and then name the region's `r.id` beside the table's. The
    // session's optimizer has already run them over the whole plan.
    let rules = state
        .optimizers()
        .iter()
        // Quail's rules act on `AiNode`s, which the claim has replaced
        .filter(|rule| {
            !["extract_leaf_expressions", "push_down_leaf_projections"].contains(&rule.name())
                && !crate::rules::QUAIL_RULES.contains(&rule.name())
                && !rule.name().starts_with("derive")
        })
        .cloned()
        .collect();
    state
        .analyzer()
        .execute_and_check(claimed, state.config_options(), |_, _| {})
        .and_then(|plan| Optimizer::with_rules(rules).optimize(plan, state, |_, _| {}))
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
    // a projection that only passes columns on is removed by the
    // optimizer run that follows the claim
    let mut root = &plan;
    while let LogicalPlan::Projection(projection) = root {
        if !projection
            .expr
            .iter()
            .all(|expr| matches!(expr.clone().unalias_nested().data, Expr::Column(_)))
        {
            break;
        }
        root = projection.input.as_ref();
    }
    let whole = as_node::<ModelRegion>(root).is_some();
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
    claim_needed(plan, fetch, None)
}

/// [`claim_regions`], where the operator above reads only the `needed`
/// columns of `plan`, or every one when None.
fn claim_needed(
    plan: LogicalPlan,
    fetch: Option<usize>,
    needed: Option<Vec<Column>>,
) -> Result<LogicalPlan> {
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
    if let Some((chain, restore)) = isolate_chain(&plan)? {
        // the run's own names for the needed columns, by position
        let renamed = needed.as_ref().map(|needed| {
            plan.schema()
                .columns()
                .iter()
                .zip(chain.schema().columns())
                .filter(|(original, _)| needed.contains(original))
                .map(|(_, column)| column)
                .collect::<Vec<_>>()
        });
        let restore = match &needed {
            Some(needed) => plan
                .schema()
                .columns()
                .into_iter()
                .zip(restore)
                .filter(|(original, _)| needed.contains(original))
                .map(|(_, expr)| expr)
                .collect(),
            None => restore,
        };
        let mut claimed = claim_needed(chain, fetch, renamed)?;
        let names: Vec<String> = restore
            .iter()
            .map(|expr| match expr {
                Expr::Alias(alias) => alias.name.clone(),
                other => other.schema_name().to_string(),
            })
            .collect();
        let shared = names
            .iter()
            .any(|name| names.iter().filter(|other| *other == name).count() > 1);
        if shared {
            claimed = without_filters(claimed)?;
        }
        return Ok(LogicalPlan::Projection(Projection::try_new(
            restore,
            Arc::new(claimed),
        )?));
    }
    if as_node::<AiNode>(&plan).is_some() {
        // a region returns an extraction's spans beside its answer
        let needed = needed.map(|mut needed| {
            let answers: Vec<Column> = needed
                .iter()
                .filter_map(|column| column.name.strip_suffix(crate::SPAN_SUFFIX))
                .map(Column::from_name)
                .filter(|answer| plan.schema().has_column(answer))
                .collect();
            for answer in answers {
                if !needed.contains(&answer) {
                    needed.push(answer);
                }
            }
            needed
        });
        let refused = match claim_run(&plan, fetch, needed.clone(), false)? {
            Ok(claimed) => return Ok(claimed),
            Err(refused) => refused,
        };
        // DataFusion can compute an expression the region does not read
        if let Ok(claimed) = claim_run(&plan, fetch, needed.clone(), true)? {
            return Ok(claimed);
        }
        // a run no region takes is tried in smaller parts
        return match simpler_run(&plan)? {
            Some(simpler) => {
                let fetch = fetch.filter(|_| as_node::<AiNode>(&simpler).is_some());
                claim_needed(simpler, fetch, needed)
            }
            None => plan_err!("{refused}"),
        };
    }
    // DataFusion runs this operator; its inputs are claimed below it
    if plan
        .expressions()
        .iter()
        .any(|expr| expr.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(false))
    {
        return plan_err!(
            "This operator cannot contain a model call:\n{}",
            plan.display()
        );
    }
    let passed = passed_fetch(&plan, fetch);
    let inputs = plan
        .inputs()
        .into_iter()
        .cloned()
        .map(|input| {
            let read = super::lowering::read_from(&plan, &input, needed.as_deref());
            claim_needed(input, passed, read)
        })
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

/// Quail's binder's error for `plan` as a region, if it has one.
fn derive_error(plan: &LogicalPlan) -> Option<CompileError> {
    // the chat turn and layout change the prompts, not what Quail reads
    let turn: quail_planner::prompts::Turn = (String::new(), String::new());
    let region = ModelRegion::new(plan.clone());
    crate::derive::derive_region(&region, &turn, "ai-if").err()
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
pub(crate) fn calls_ai_function(plan: &LogicalPlan) -> Result<bool> {
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
            | LogicalPlan::Join(_)
            | LogicalPlan::SubqueryAlias(_)
            | LogicalPlan::TableScan(_)
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

thread_local! {
    /// How many inputs the claim on this thread has named: a query's
    /// inputs are numbered from zero each time it is claimed.
    static NEXT_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// `plan` with each of its regions taking no dynamic filter
/// ([`ModelRegion::takes_filters`]).
fn without_filters(plan: LogicalPlan) -> Result<LogicalPlan> {
    plan.transform_down(|node| match as_node::<ModelRegion>(&node) {
        Some(region) if region.takes_filters => {
            let mut region = region.clone();
            region.takes_filters = false;
            Ok(Transformed::yes(extension_node(region)))
        }
        _ => Ok(Transformed::no(node)),
    })
    .map(|done| done.data)
}

/// The run of `AiNode`s at `plan`'s root over its input as rows
/// DataFusion computes, and the columns that restore the run's own
/// names above it.
///
/// None when the run reads its input as it stands: an AI join's two
/// sides, one table ([`is_table`]), or rows already cut off.
fn isolate_chain(plan: &LogicalPlan) -> Result<Option<(LogicalPlan, Vec<Expr>)>> {
    let mut nodes = Vec::new();
    let mut bottom = plan;
    while let Some(node) = as_node::<AiNode>(bottom) {
        nodes.push(node);
        bottom = node.input.as_ref();
    }
    let Some(lowest) = nodes.last() else {
        return Ok(None);
    };
    if is_ai_join(lowest, bottom) || is_table(bottom) {
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
    let number = NEXT_ROWS.with(|next| next.replace(next.get() + 1));
    // rows of one relation keep its name
    let mut qualifiers: Vec<String> = columns
        .iter()
        .map(|column| {
            column
                .relation
                .as_ref()
                .map_or_else(String::new, |relation| relation.table().to_string())
        })
        .collect();
    // a computed column has no relation of its own
    qualifiers.retain(|qualifier| !qualifier.is_empty());
    qualifiers.dedup();
    let name = match qualifiers.as_slice() {
        [one] => one.clone(),
        _ => format!("__quail_rows_{number}"),
    };
    let input = datafusion::logical_expr::LogicalPlanBuilder::from(bottom.clone())
        .project(renamed)?
        .build()?;
    let input = extension_node(SharedInput::try_new(input, false)?);
    let mut input = datafusion::logical_expr::LogicalPlanBuilder::from(input)
        .alias(name)?
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
        input = extension_node(node.map_expressions(remap)?.with_input(input)?);
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

/// Whether `plan` reads one table and calls no model: a region reads it
/// as an input with the table's own id column.
fn is_table(plan: &LogicalPlan) -> bool {
    let mut num_scans = 0;
    let _ = plan.apply(|node| {
        if matches!(node, LogicalPlan::TableScan(_)) {
            num_scans += 1;
        }
        Ok(TreeNodeRecursion::Continue)
    });
    // the region reads a table by its alias, which names its columns
    let mut top = plan;
    while let LogicalPlan::Filter(filter) = top {
        top = filter.input.as_ref();
    }
    matches!(
        top,
        LogicalPlan::SubqueryAlias(_) | LogicalPlan::TableScan(_)
    ) && num_scans == 1
        && !calls_ai(plan).unwrap_or(true)
}

/// One model region for the run of `AiNode`s at `plan`'s root, or the
/// binder's refusal of the run. With `precompute`, DataFusion computes
/// the run's expressions of one table's columns before the region.
fn claim_run(
    plan: &LogicalPlan,
    fetch: Option<usize>,
    needed: Option<Vec<Column>>,
    precompute: bool,
) -> Result<std::result::Result<LogicalPlan, String>> {
    // a run the claim cannot shape into a region is refused like one the
    // binder cannot read
    let shaped = (|| {
        let selected = select_list(plan.clone(), needed)?;
        // each expression of one table's columns as a column of that
        // table's input
        let selected = if precompute {
            super::inputs::precompute_plain_expressions(selected)?
        } else {
            selected
        };
        let (bound, inputs) = split_inputs(selected)?;
        let numbered_inputs = numbered(&bound);
        let (bound, above) = lift_relational(bound)?;
        Ok::<_, datafusion::common::DataFusionError>((bound, inputs, numbered_inputs, above))
    })();
    let (bound, inputs, numbered_inputs, above) = match shaped {
        Ok(shaped) => shaped,
        Err(error) => return Ok(Err(error.strip_backtrace())),
    };
    if !quail_plans(&bound)? {
        return Ok(Err(format!(
            "This operator cannot contain a model call:\n{}",
            plan.display()
        )));
    }
    // a shape the binder does not read may run in smaller parts; any
    // other error is the query's own, reported when the region is planned
    if let Some(error) = derive_error(&bound).filter(CompileError::is_unsupported) {
        return Ok(Err(error.to_string()));
    }
    let mut region = ModelRegion::new(bound);
    // a fetch counts the rows the region returns, not the rows a lifted
    // test or UNNEST keeps of them
    region.fetch =
        fetch.filter(|_| above.keeps_rows() && !crate::ai::any_filters_on_or(&region.plan));
    for (name, input) in inputs {
        let mut input = claim_regions(input, None)?;
        if numbered_inputs.contains(&name) {
            input = extension_node(SharedInput::try_new(input, true)?);
        }
        region.inputs.push(input);
        region.input_names.push(name);
    }
    above.build(extension_node(region)).map(Ok)
}

/// A plan that returns the rows of the run of `AiNode`s at `plan`'s root
/// and asks less of one region, or None when there is nothing simpler.
///
/// A run of several nodes gives up its hint tests, and then its top
/// node, which becomes a run of its own over the others' rows. A single node gives up its options: its
/// filter and fetch become a `Filter` and `Limit` above it, and its
/// guard, hint, and hint test, which only save work, are dropped. A node
/// with no options has its arguments computed below it.
fn simpler_run(plan: &LogicalPlan) -> Result<Option<LogicalPlan>> {
    let Some(node) = as_node::<AiNode>(plan) else {
        return Ok(None);
    };
    let pass_on = |input: &LogicalPlan| -> Result<LogicalPlan> {
        let columns = input.schema().columns().into_iter().map(Expr::Column);
        Ok(LogicalPlan::Projection(Projection::try_new(
            columns.collect(),
            Arc::new(input.clone()),
        )?))
    };
    if as_node::<AiNode>(node.input.as_ref()).is_some() {
        // a hint test only saves work, and one that reads another call's
        // answer is the usual reason a run of several calls is refused
        let mut nodes = Vec::new();
        let mut bottom = plan;
        while let Some(found) = as_node::<AiNode>(bottom) {
            nodes.push(found);
            bottom = found.input.as_ref();
        }
        if nodes.iter().any(|found| !found.tests.is_empty()) {
            let mut run = bottom.clone();
            for found in nodes.iter().rev() {
                let mut found = (*found).clone();
                found.tests = Vec::new();
                run = extension_node(found.with_input(run)?);
            }
            return Ok(Some(run));
        }
        let below = pass_on(node.input.as_ref())?;
        return Ok(Some(extension_node(node.with_input(below)?)));
    }
    if node.semantics != crate::ai::Semantics::Rows {
        return Ok(None);
    }
    let plain = node.filter.is_none()
        && node.guard.is_none()
        && node.hint.is_none()
        && node.fetch.is_none()
        && node.projection.is_none()
        && node.tests.is_empty();
    if plain {
        return computed_arguments(node);
    }
    let mut stripped = node.clone();
    stripped.filter = None;
    stripped.guard = None;
    stripped.hint = None;
    stripped.fetch = None;
    stripped.projection = None;
    stripped.tests = Vec::new();
    let mut above = extension_node(stripped.checked()?);
    if let Some(filter) = &node.filter {
        above = LogicalPlan::Filter(datafusion::logical_expr::Filter::try_new(
            filter.clone(),
            Arc::new(above),
        )?);
    }
    if let Some(crate::ai::Fetch::Global(rows)) = &node.fetch {
        above = datafusion::logical_expr::LogicalPlanBuilder::from(above)
            .limit(0, Some(*rows))?
            .build()?;
    }
    let columns = plan.schema().columns().into_iter().map(Expr::Column);
    Ok(Some(LogicalPlan::Projection(Projection::try_new(
        columns.collect(),
        Arc::new(above),
    )?)))
}

/// `node`, which has no options, over a SELECT list that computes each
/// argument of its call that is not a column; None when every argument
/// is one already.
fn computed_arguments(node: &AiNode) -> Result<Option<LogicalPlan>> {
    let mut exprs: Vec<Expr> = node
        .input
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    exprs.push(node.call.clone().alias(&node.answer));
    let written = Projection::try_new(exprs, Arc::clone(&node.input))?;
    let computed = crate::rules::lower::precompute_prompt_arguments(written.clone())?;
    if computed == written {
        return Ok(None);
    }
    let lowered = crate::lower_ai_calls(LogicalPlan::Projection(computed))?;
    // the node's own columns, without the computed arguments
    let columns = datafusion::logical_expr::UserDefinedLogicalNodeCore::schema(node).columns();
    if !columns
        .iter()
        .all(|column| lowered.schema().has_column(column))
    {
        return Ok(None);
    }
    let columns: Vec<Expr> = columns.into_iter().map(Expr::Column).collect();
    Ok(Some(LogicalPlan::Projection(Projection::try_new(
        columns,
        Arc::new(lowered),
    )?)))
}

/// Whether `node` over `input` is an AI join or gate: its call reads both
/// sides of an inner join, which the region pairs itself.
fn is_ai_join(node: &AiNode, input: &LogicalPlan) -> bool {
    let LogicalPlan::Join(join) = input else {
        return false;
    };
    if node.semantics != crate::ai::Semantics::Rows {
        return true;
    }
    if join.join_type != datafusion::logical_expr::JoinType::Inner {
        return false;
    }
    let read = node.call.column_refs();
    let reads = |side: &LogicalPlan| read.iter().any(|column| side.schema().has_column(column));
    reads(&join.left) && reads(&join.right)
}
