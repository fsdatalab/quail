//! Claim optimized model regions and choose their boundaries.
//! Leave unsupported operators and relational result work in DataFusion.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, ScalarValue, plan_err};
use datafusion::execution::SessionState;
use datafusion::logical_expr::expr_rewriter::replace_col;
use datafusion::logical_expr::{Expr, LogicalPlan, Projection, UserDefinedLogicalNodeCore};
use datafusion::optimizer::Optimizer;
use quail_planner::error::CompileError;

use super::inputs::{numbered, split_inputs};
use super::lift::lift_relational;
use super::{ModelRegion, calls_ai};
use crate::ai::AiNode;
use crate::functions::is_ai_call;
use crate::plan::{as_node, extension_node};
use crate::shared::SharedInput;

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
    let claimed = number_regions(drop_unread_probabilities(claim_regions(plan)?)?)?;
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
                    let rows = visit(input.inputs()[0].clone(), next, shared)?;
                    let key = input.key;
                    let numbered = extension_node(input.with_exprs_and_inputs(vec![], vec![rows])?);
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
                && !returned.iter().any(|name| name == ai.answer())
                && let Some(call) = without_probabilities(&ai.call())
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
    let Expr::ScalarFunction(unfolded) = crate::functions::options::unfolded(call.clone()) else {
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
fn claim_regions(plan: LogicalPlan) -> Result<LogicalPlan> {
    claim_needed(plan, None)
}

/// [`claim_regions`], where the operator above reads only the `needed`
/// columns of `plan`, or every one when None.
fn claim_needed(plan: LogicalPlan, needed: Option<Vec<Column>>) -> Result<LogicalPlan> {
    if !calls_ai(&plan)? {
        return Ok(plan);
    }
    // Expression subqueries are not plan inputs. Claim their model work
    // explicitly, including scalar subqueries that DataFusion keeps intact.
    let plan = plan
        .map_subqueries(|subquery| {
            let LogicalPlan::Subquery(mut found) = subquery else {
                return Ok(Transformed::no(subquery));
            };
            found.subquery = Arc::new(claim_regions(Arc::unwrap_or_clone(found.subquery))?);
            Ok(Transformed::yes(LogicalPlan::Subquery(found)))
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
        let claimed = claim_needed(chain, renamed)?;
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
                .filter_map(|column| {
                    column
                        .name
                        .strip_suffix(quail_planner::constants::SPAN_SUFFIX)
                })
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
        let refused = match claim_run(&plan, needed.clone(), false)? {
            Ok(claimed) => return Ok(claimed),
            Err(refused) => refused,
        };
        // DataFusion can compute an expression the region does not read
        if let Ok(claimed) = claim_run(&plan, needed.clone(), true)? {
            return Ok(claimed);
        }
        // a run no region takes is tried in smaller parts
        return match simpler_run(&plan)? {
            Some(simpler) => claim_needed(simpler, needed),
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
    let inputs = plan
        .inputs()
        .into_iter()
        .cloned()
        .map(|input| {
            let read = read_from(&plan, &input, needed.as_deref());
            claim_needed(input, read)
        })
        .collect::<Result<Vec<_>>>()?;
    crate::plan::with_inputs(&plan, inputs)
}

/// Quail's binder's error for `plan` as a region, if it has one.
fn derive_error(plan: &LogicalPlan) -> Option<CompileError> {
    // the chat turn and layout change the prompts, not what Quail reads
    let turn: quail_planner::prompts::Turn = (String::new(), String::new());
    let region = ModelRegion::new(plan.clone());
    crate::region::derive::derive_region(&region, &turn, "ai-if").err()
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

/// The run of `AiNode`s at `plan`'s root over its input as rows
/// DataFusion computes, and the columns that restore the run's own
/// names above it.
///
/// None when the run reads its input as it stands: an AI join's two
/// sides, one table ([`is_table`]), or rows already cut off.
fn isolate_chain(plan: &LogicalPlan) -> Result<Option<(LogicalPlan, Vec<Expr>)>> {
    let (nodes, bottom) = AiNode::run(plan);
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
    let renamed: HashMap<&Column, &Column> = columns.iter().zip(&replacements).collect();
    let remap = |expr: Expr| replace_col(expr, &renamed);
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
        // an AI join keeps its join's filter when the binder can read it
        // there; DataFusion otherwise tests the joined rows above
        let kept = lift_relational(bound.clone(), true)?;
        let lifted = lift_relational(bound, false)?;
        let (bound, above) = if kept.0 != lifted.0
            && derive_error(&kept.0).is_none_or(|error| !error.is_unsupported())
        {
            kept
        } else {
            lifted
        };
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
    for (name, input) in inputs {
        let mut input = claim_regions(input)?;
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
    let (nodes, bottom) = AiNode::run(plan);
    if let [_, _, ..] = nodes.as_slice() {
        // a hint test only saves work, and one that reads another call's
        // answer is the usual reason a run of several calls is refused
        let untested: Vec<Option<AiNode>> =
            nodes.iter().map(|found| found.without_tests()).collect();
        if untested.iter().any(Option::is_some) {
            let mut run = bottom.clone();
            for (found, untested) in nodes.iter().zip(untested).rev() {
                let found = untested.unwrap_or_else(|| (*found).clone());
                run = extension_node(found.with_input(run)?);
            }
            return Ok(Some(run));
        }
        let below = pass_on(node.inputs()[0])?;
        return Ok(Some(extension_node(node.with_input(below)?)));
    }
    if node.is_join() {
        return Ok(None);
    }
    if node.is_bare() {
        return computed_arguments(node);
    }
    let above = node.options_above()?;
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
        .input_schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    let asked = node.call();
    exprs.push(asked.clone().alias(node.answer()));
    if let Expr::ScalarFunction(call) = &asked
        && call.name() == "ai_extract"
    {
        // an extraction's node also returns its spans
        let spans = Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
            crate::functions::quail_function("ai_extract_span"),
            call.args.clone(),
        ));
        exprs.push(spans.alias(format!(
            "{}{}",
            node.answer(),
            quail_planner::constants::SPAN_SUFFIX
        )));
    }
    let written = Projection::try_new(exprs, Arc::new(node.inputs()[0].clone()))?;
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
    if node.is_join() {
        return true;
    }
    if join.join_type != datafusion::logical_expr::JoinType::Inner {
        return false;
    }
    let call = node.call();
    let read = call.column_refs();
    let reads = |side: &LogicalPlan| read.iter().any(|column| side.schema().has_column(column));
    reads(&join.left) && reads(&join.right)
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
