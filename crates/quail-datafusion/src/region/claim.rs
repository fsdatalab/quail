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
use crate::ai::{AiNode, Semantics};
use crate::functions::is_ai_call;
use crate::plan::{as_node, extension_node};
use crate::shared::SharedInput;

pub fn claim_model_regions(plan: LogicalPlan, session: &dyn Session) -> Result<LogicalPlan> {
    let Some(state) = session.as_any().downcast_ref::<SessionState>() else {
        return plan_err!("claiming model regions needs DataFusion's SessionState");
    };
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
        return number_regions(drop_unread_probabilities(plan)?);
    }
    NEXT_ROWS.with(|next| next.set(0));
    let plan = super::unread::drop_unread_nodes(plan)?;
    let claimed = number_regions(drop_unread_probabilities(claim_regions(plan)?)?)?;
    let rules = state
        .optimizers()
        .iter()
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

fn drop_unread_probabilities(plan: LogicalPlan) -> Result<LogicalPlan> {
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

fn claim_regions(plan: LogicalPlan) -> Result<LogicalPlan> {
    claim_needed(plan, None)
}

fn claim_needed(plan: LogicalPlan, needed: Option<Vec<Column>>) -> Result<LogicalPlan> {
    if !calls_ai(&plan)? {
        return Ok(plan);
    }
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
        if let Ok(claimed) = claim_run(&plan, needed.clone(), true)? {
            return Ok(claimed);
        }
        return match simpler_run(&plan)? {
            Some(simpler) => claim_needed(simpler, needed),
            None => plan_err!("{refused}"),
        };
    }
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

fn derive_error(plan: &LogicalPlan) -> Option<CompileError> {
    let turn: quail_planner::prompts::Turn = (String::new(), String::new());
    let region = ModelRegion::new(plan.clone());
    crate::region::derive::derive_region(&region, &turn, "ai-if").err()
}

pub(super) fn select_list(plan: LogicalPlan, needed: Option<Vec<Column>>) -> Result<LogicalPlan> {
    if matches!(plan, LogicalPlan::Projection(_)) {
        return Ok(plan);
    }
    let mut columns = needed.unwrap_or_else(|| plan.schema().columns());
    if columns.is_empty() {
        columns = plan.schema().columns().into_iter().take(1).collect();
    }
    let exprs = columns.into_iter().map(Expr::Column).collect();
    Ok(LogicalPlan::Projection(Projection::try_new(
        exprs,
        Arc::new(plan),
    )?))
}

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

fn is_quail_operator(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Projection(_)
            | LogicalPlan::Join(_)
            | LogicalPlan::SubqueryAlias(_)
            | LogicalPlan::TableScan(_)
    ) || as_node::<AiNode>(plan).is_some()
}

pub(super) fn is_subquery(expr: &Expr) -> bool {
    match expr {
        Expr::Exists(exists) => !quail_plans(&exists.subquery.subquery).unwrap_or(false),
        Expr::InSubquery(_) | Expr::ScalarSubquery(_) => true,
        _ => false,
    }
}

thread_local! {
    static NEXT_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

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
    let mut qualifiers: Vec<String> = columns
        .iter()
        .map(|column| {
            column
                .relation
                .as_ref()
                .map_or_else(String::new, |relation| relation.table().to_string())
        })
        .collect();
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

fn is_table(plan: &LogicalPlan) -> bool {
    let mut num_scans = 0;
    let _ = plan.apply(|node| {
        if matches!(node, LogicalPlan::TableScan(_)) {
            num_scans += 1;
        }
        Ok(TreeNodeRecursion::Continue)
    });
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

fn claim_run(
    plan: &LogicalPlan,
    needed: Option<Vec<Column>>,
    precompute: bool,
) -> Result<std::result::Result<LogicalPlan, String>> {
    let shaped = (|| {
        let selected = select_list(plan.clone(), needed)?;
        let selected = if precompute {
            super::inputs::precompute_plain_expressions(selected)?
        } else {
            selected
        };
        let (bound, inputs) = split_inputs(selected)?;
        let numbered_inputs = numbered(&bound);
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
    if node.semantics() != Semantics::Rows {
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

fn is_ai_join(node: &AiNode, input: &LogicalPlan) -> bool {
    let LogicalPlan::Join(join) = input else {
        return false;
    };
    if node.semantics() != Semantics::Rows {
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
        LogicalPlan::Filter(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Distinct(_)
        | LogicalPlan::Window(_)
        | LogicalPlan::Join(_) => needed?.iter().cloned().for_each(&mut add),
        LogicalPlan::Union(_) => return None,
        LogicalPlan::SubqueryAlias(_) => {
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
