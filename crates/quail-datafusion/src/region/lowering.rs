//! Lower AI calls into plan nodes before optimization.
//! Move calls below operators that DataFusion runs.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{
    Aggregate, Expr, Filter, JoinType, LogicalPlan, LogicalPlanBuilder, Projection, Window,
};

use super::claim::calls_ai_function;
use crate::functions::is_ai_call;

/// `plan` with every AI call as an `AiNode` below the operator that held
/// it.
///
/// # Errors
///
/// A rewritten operator's schema does not fit DataFusion's rules.
pub(crate) fn lower_regions(plan: LogicalPlan) -> Result<LogicalPlan> {
    let plan = drop_written_names(plan)?;
    let plan = plan.transform_up(exists_terms_as_in)?.data;
    let mut extracted = Extracted::default();
    // the call each written name, such as `AI.CLASSIFY(...)`, names
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
    let mut columns = crate::rules::lower::Columns::of(&plan)?;
    extracted.lower(plan, &mut columns)
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

/// An inner join with each ON term that calls the model, and is not the
/// join's own AI predicate, as a WHERE term above it: an inner join
/// keeps a pair exactly when every ON term is TRUE.
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
        if calls && !crate::rules::lower::is_join_predicate(term) {
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

/// The AI calls moved out of DataFusion's operators so far.
#[derive(Debug, Default)]
struct Extracted {
    num_calls: usize,
    num_inputs: usize,
    /// The SELECT list's names, by the name of the call or column each
    /// one renames.
    aliases: Vec<(String, String)>,
    /// The names given to moved calls.
    used: Vec<String>,
    /// User column and relation names that generated names must not shadow.
    reserved: Vec<String>,
}

impl Extracted {
    /// `plan` with every AI call as an `AiNode` below the operator that
    /// held it.
    fn lower(
        &mut self,
        plan: LogicalPlan,
        columns: &mut crate::rules::lower::Columns,
    ) -> Result<LogicalPlan> {
        if !calls_ai_function(&plan)? {
            return Ok(plan);
        }
        // an expression's subquery is a plan of its own
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
        // VALUES has no input: each row is a SELECT list over one empty row
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
        if let Some(join) = super::join::lower_outer_join(&plan, self.num_inputs, |input| {
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
        let plan = plan.with_new_exprs(plan.expressions(), inputs)?;
        let plan = self.lower_arguments(plan, columns)?;
        // an operator that is no filter, inner join, or SELECT list reads
        // its calls from a SELECT list below it
        let plan = match plan {
            LogicalPlan::Aggregate(_) | LogicalPlan::Window(_) | LogicalPlan::Sort(_) => {
                self.extract(plan)?
                    .data
                    .map_children(|child| crate::rules::lower::lower_node(child, columns))?
                    .data
            }
            plan => crate::rules::lower::lower_node(plan, columns)?.data,
        };
        // a call inside a larger expression is read from a SELECT list below
        let nested = !matches!(plan, LogicalPlan::Extension(_))
            && plan
                .expressions()
                .iter()
                .any(|expr| expr.exists(|inner| Ok(is_ai_call(inner))).unwrap_or(false));
        if nested && matches!(plan, LogicalPlan::Filter(_) | LogicalPlan::Projection(_)) {
            return Ok(self
                .extract_calls(plan, true)?
                .data
                .map_children(|child| crate::rules::lower::lower_node(child, columns))?
                .data);
        }
        Ok(plan)
    }

    /// `plan` with each AI call that is an argument of another read as
    /// a column of a SELECT list below it, innermost first.
    fn lower_arguments(
        &mut self,
        mut plan: LogicalPlan,
        columns: &mut crate::rules::lower::Columns,
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
            let below = crate::rules::lower::lower_node(below, columns)?.data;
            let exprs = plan
                .expressions()
                .iter()
                .map(|expr| read_calls(expr, &named))
                .collect::<Result<Vec<_>>>()?;
            plan = plan.with_new_exprs(exprs, vec![below])?;
        }
    }

    /// A join with each AI call of one input read from a SELECT list
    /// below that input, so the join reads columns: for example `JOIN v
    /// ON AI.MAP(i.text, ...) = v.name` reads the map's answer from a
    /// SELECT list over `i`, and so does a test of one input in ON, such
    /// as `ON r.id = v.rid AND AI.IF(PROMPT('{0}', r.body))`.
    ///
    /// A call that reads both inputs is the join's own AI predicate and
    /// stays. The join keeps the added columns in its output, as the
    /// operators above may read the same answers.
    fn join_keys_below(&mut self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let LogicalPlan::Join(join) = &plan else {
            return Ok(plan);
        };
        // which input an expression reads: 0, 1, or neither alone
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
        // `expr` with each of its AI calls read as a column of the SELECT
        // list below the input the call reads, named as the lowering
        // names a call anywhere else; unchanged when a call reads both
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
                    .contains(&format!("{name}{}", crate::SPAN_SUFFIX))
                && !self.used.contains(&name)
            {
                self.reserved.push(name.clone());
                return name;
            }
        }
    }

    /// Move each AI call of an `Aggregate`, `Window`, or `Sort` into a
    /// `Projection` below it.
    fn extract(&mut self, node: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
        self.extract_calls(node, false)
    }

    /// [`Self::extract`]; with `whole`, also a SELECT list's items that
    /// are a call and nothing else.
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
            // a SELECT list item that is a model call already sits where
            // a region reads it
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
        let projection = Projection::try_new(read_columns, Arc::new(input))?;
        let projection = LogicalPlan::Projection(projection);
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

    /// A column name for each call: the SELECT list's name for it, or a
    /// new hidden one.
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
                        *found == key
                            && !self.used.contains(name)
                            && !crate::rules::lower::names_input(input, name)
                    })
                    .map(|(_, name)| name.clone());
                let name = named.unwrap_or_else(|| {
                    let mut name = self.fresh_call();
                    // a span column is its extraction's answer name with the suffix
                    if crate::functions::function_name(&call) == Some("ai_extract_span") {
                        name.push_str(crate::SPAN_SUFFIX);
                    }
                    name
                });
                self.used.push(name.clone());
                (call, name)
            })
            .collect()
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
        .data
        .transform_up(|expr| {
            if crate::functions::function_name(&expr) != Some("hint") {
                return Ok(Transformed::no(expr));
            }
            let mut binder =
                crate::derive::binding::Binder::new(&(String::new(), String::new()), "ai-if");
            let (test, _) = binder
                .unwrap_hint(&expr)
                .map_err(|error| datafusion::common::DataFusionError::Plan(error.to_string()))?;
            Ok(Transformed::yes(test))
        })?
        .data)
}

/// The columns of `input` that `plan` reads, when the operator above
/// `plan` reads `needed` of its columns.
pub(super) fn read_from(
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
