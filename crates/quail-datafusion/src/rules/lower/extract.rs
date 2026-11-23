use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Aggregate, Expr, JoinType, LogicalPlan, LogicalPlanBuilder, Projection, Window,
};

use super::{Columns, calls_ai_function, is_join_predicate, lower_node, names_input};
use crate::functions::is_ai_call;
use crate::functions::options::unwrap_hint;

pub(super) fn lower_plan(plan: LogicalPlan) -> Result<LogicalPlan> {
    let plan = drop_written_names(plan)?;
    let mut extracted = Extracted::of(&plan)?;
    let mut columns = Columns::of(&plan)?;
    extracted.lower(plan, &mut columns)
}

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

fn terms_above(plan: LogicalPlan) -> Result<LogicalPlan> {
    let LogicalPlan::Join(join) = &plan else {
        return Ok(plan);
    };
    if join.join_type != JoinType::Inner {
        return Ok(plan);
    }
    let mut kept = Vec::new();
    let mut above = Vec::new();
    for term in join.filter.iter().flat_map(split_conjunction) {
        let calls = !ai_calls(term)?.is_empty();
        if calls && !is_join_predicate(term) {
            above.push(term.clone());
        } else {
            kept.push(term.clone());
        }
    }
    let Some(above) = conjunction(above) else {
        return Ok(plan);
    };
    let mut below = join.clone();
    below.filter = conjunction(kept);
    LogicalPlanBuilder::from(LogicalPlan::Join(below))
        .filter(above)?
        .build()
}

#[derive(Debug, Default)]
struct Extracted {
    num_calls: usize,
    num_inputs: usize,
    aliases: Vec<(String, String)>,
    used: Vec<String>,
    reserved: Vec<String>,
}

impl Extracted {
    fn of(plan: &LogicalPlan) -> Result<Self> {
        let mut extracted = Self::default();
        let mut written: Vec<(String, String)> = Vec::new();
        plan.apply(|node| {
            for (qualifier, field) in node.schema().iter() {
                extracted.reserved.push(field.name().clone());
                if let Some(qualifier) = qualifier {
                    extracted.reserved.push(qualifier.table().to_string());
                }
            }
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
        plan.apply(|node| {
            if let LogicalPlan::Projection(projection) = node {
                for expr in &projection.expr {
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
        Ok(extracted)
    }

    fn lower(&mut self, plan: LogicalPlan, columns: &mut Columns) -> Result<LogicalPlan> {
        if !calls_ai_function(&plan)? {
            return Ok(plan);
        }
        let plan = plan
            .map_uncorrelated_subqueries(|subquery| {
                let LogicalPlan::Subquery(mut found) = subquery else {
                    return Ok(Transformed::no(subquery));
                };
                found.subquery =
                    Arc::new(self.lower(Arc::unwrap_or_clone(found.subquery), columns)?);
                Ok(Transformed::yes(LogicalPlan::Subquery(found)))
            })?
            .data;
        if let LogicalPlan::Values(values) = &plan {
            let mut rows = Vec::new();
            for row in &values.values {
                let exprs: Vec<Expr> = row
                    .iter()
                    .zip(values.schema.iter())
                    .map(|(expr, (qualifier, field))| {
                        expr.clone()
                            .alias_qualified(qualifier.cloned(), field.name())
                    })
                    .collect();
                let one = LogicalPlanBuilder::empty(true).project(exprs)?.build()?;
                rows.push(self.lower(one, columns)?);
            }
            let mut rows = rows.into_iter();
            let Some(first) = rows.next() else {
                return Ok(plan);
            };
            let mut union = LogicalPlanBuilder::from(first);
            for row in rows {
                union = union.union(row)?;
            }
            return union.build();
        }
        self.num_inputs += 1;
        let plan = self.join_keys_below(plan)?;
        let plan = terms_above(plan)?;
        if let Some(join) = super::outer_join::lower_outer_join(&plan, self.num_inputs, |input| {
            self.lower(input, columns)
        })? {
            return Ok(join);
        }
        let inputs = plan
            .inputs()
            .into_iter()
            .cloned()
            .map(|input| self.lower(input, columns))
            .collect::<Result<Vec<_>>>()?;
        let plan = crate::plan::with_inputs(&plan, inputs)?;
        let plan = self.lower_arguments(plan, columns)?;
        let plan = match plan {
            LogicalPlan::Aggregate(_) | LogicalPlan::Window(_) | LogicalPlan::Sort(_) => {
                self.extract(plan)?
                    .data
                    .map_children(|child| lower_node(child, columns))?
                    .data
            }
            plan => lower_node(plan, columns)?.data,
        };
        let nested = !matches!(plan, LogicalPlan::Extension(_))
            && plan
                .expressions()
                .iter()
                .any(|expr| expr.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(false));
        if nested && matches!(plan, LogicalPlan::Filter(_) | LogicalPlan::Projection(_)) {
            return Ok(self
                .extract_calls(plan, true)?
                .data
                .map_children(|child| lower_node(child, columns))?
                .data);
        }
        Ok(plan)
    }

    fn lower_arguments(
        &mut self,
        mut plan: LogicalPlan,
        columns: &mut Columns,
    ) -> Result<LogicalPlan> {
        loop {
            if plan.inputs().len() != 1 {
                return Ok(plan);
            }
            let mut inner: Vec<Expr> = Vec::new();
            for expr in plan.expressions() {
                expr.apply(|outer| {
                    let Expr::ScalarFunction(call) = outer else {
                        return Ok(TreeNodeRecursion::Continue);
                    };
                    if !is_ai_call(outer) {
                        return Ok(TreeNodeRecursion::Continue);
                    }
                    for argument in &call.args {
                        argument.apply(|found| {
                            let innermost = is_ai_call(found)
                                && matches!(found, Expr::ScalarFunction(call)
                                if !call.args.iter().any(|argument| {
                                    argument.exists(|deeper| Ok(is_ai_call(deeper))).unwrap_or(true)
                                }));
                            if innermost && !inner.contains(found) {
                                inner.push(found.clone());
                            }
                            Ok(TreeNodeRecursion::Continue)
                        })?;
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
            }
            if inner.is_empty() {
                return Ok(plan);
            }
            let input = plan.inputs()[0].clone();
            let named = self.name(inner, input.schema());
            let mut exprs: Vec<Expr> = input
                .schema()
                .columns()
                .into_iter()
                .map(Expr::Column)
                .collect();
            exprs.extend(named.iter().map(|(call, name)| call.clone().alias(name)));
            let below = LogicalPlan::Projection(Projection::try_new(exprs, Arc::new(input))?);
            let below = lower_node(below, columns)?.data;
            let exprs = plan
                .expressions()
                .iter()
                .map(|expr| read_calls(expr, &named))
                .collect::<Result<Vec<_>>>()?;
            plan = plan.with_new_exprs(exprs, vec![below])?;
        }
    }

    fn join_keys_below(&mut self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let LogicalPlan::Join(join) = &plan else {
            return Ok(plan);
        };
        let side = |expr: &Expr| {
            let columns = expr.column_refs();
            let within = |input: &LogicalPlan| {
                !columns.is_empty()
                    && columns
                        .iter()
                        .all(|column| input.schema().has_column(column))
            };
            match (within(&join.left), within(&join.right)) {
                (true, false) => Some(0),
                (false, true) => Some(1),
                _ => None,
            }
        };
        let mut sides = [(*join.left).clone(), (*join.right).clone()];
        let mut moved = false;
        let mut below = |this: &mut Self, expr: &Expr| -> Result<Expr> {
            let calls = ai_calls(expr)?;
            let read: Vec<Option<usize>> = calls.iter().map(&side).collect();
            if calls.is_empty() || read.contains(&None) {
                return Ok(expr.clone());
            }
            let mut named = Vec::new();
            for index in [0usize, 1] {
                let mine: Vec<Expr> = calls
                    .iter()
                    .zip(&read)
                    .filter(|(_, found)| **found == Some(index))
                    .map(|(call, _)| call.clone())
                    .collect();
                if mine.is_empty() {
                    continue;
                }
                let added = this.name(mine, sides[index].schema());
                let mut selected: Vec<Expr> = sides[index]
                    .schema()
                    .iter()
                    .map(|(qualifier, field)| {
                        Expr::Column(Column::new(qualifier.cloned(), field.name()))
                    })
                    .collect();
                selected.extend(added.iter().map(|(call, name)| call.clone().alias(name)));
                sides[index] = LogicalPlanBuilder::from(sides[index].clone())
                    .project(selected)?
                    .build()?;
                named.extend(added);
            }
            moved = true;
            read_calls(expr, &named)
        };
        let mut keys: (Vec<Expr>, Vec<Expr>) = (Vec::new(), Vec::new());
        for (left, right) in &join.on {
            keys.0.push(below(self, left)?);
            keys.1.push(below(self, right)?);
        }
        let mut rest: Vec<Expr> = Vec::new();
        for term in join.filter.iter().flat_map(split_conjunction) {
            rest.push(below(self, term)?);
        }
        if !moved {
            return Ok(plan);
        }
        let [left, right] = sides;
        LogicalPlanBuilder::from(left)
            .join_with_expr_keys(right, join.join_type, keys, conjunction(rest))?
            .build()
    }

    fn fresh_call(&mut self) -> String {
        loop {
            self.num_calls += 1;
            let name = format!("__quail_call_{}", self.num_calls);
            if !self.reserved.contains(&name)
                && !self
                    .reserved
                    .contains(&format!("{name}{}", quail_planner::constants::SPAN_SUFFIX))
                && !self.used.contains(&name)
            {
                self.reserved.push(name.clone());
                return name;
            }
        }
    }

    fn extract(&mut self, node: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
        self.extract_calls(node, false)
    }

    fn extract_calls(
        &mut self,
        node: LogicalPlan,
        whole: bool,
    ) -> Result<Transformed<LogicalPlan>> {
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
            if !whole
                && matches!(node, LogicalPlan::Projection(_))
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
        let named = self.name(calls, node.inputs()[0].schema());
        let read = |expr: &Expr| read_calls(expr, &named);
        let exprs: Vec<Expr> = node.expressions().iter().map(read).collect::<Result<_>>()?;
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
        let projection = Projection::try_new(read_columns, Arc::new(input))?;
        let projection = LogicalPlan::Projection(projection);
        let rewritten = match &node {
            LogicalPlan::Aggregate(aggregate) => {
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

    fn name(
        &mut self,
        calls: Vec<Expr>,
        input: &datafusion::common::DFSchema,
    ) -> Vec<(Expr, String)> {
        calls
            .into_iter()
            .map(|call| {
                let key = call.schema_name().to_string();
                let named = self
                    .aliases
                    .iter()
                    .find(|(found, name)| {
                        *found == key && !self.used.contains(name) && !names_input(input, name)
                    })
                    .map(|(_, name)| name.clone());
                let name = named.unwrap_or_else(|| {
                    let mut name = self.fresh_call();
                    if crate::functions::function_name(&call) == Some("ai_extract_span") {
                        name.push_str(quail_planner::constants::SPAN_SUFFIX);
                    }
                    name
                });
                self.used.push(name.clone());
                (call, name)
            })
            .collect()
    }
}

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
        .data
        .transform_up(|expr| {
            if crate::functions::function_name(&expr) != Some("hint") {
                return Ok(Transformed::no(expr));
            }
            let (test, _) = unwrap_hint(&expr)
                .map_err(|error| datafusion::common::DataFusionError::Plan(error.to_string()))?;
            Ok(Transformed::yes(test))
        })?
        .data)
}
