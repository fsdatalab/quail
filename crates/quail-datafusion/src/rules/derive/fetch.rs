//! The fetch steps of [`super::Derive`]: how many rows of an `AiNode`
//! are enough for the operator above it.
//!
//! An `AiNode`'s fetch keeps the first `n` rows that pass its filter, in
//! all or for each value of some key columns. Four operators need only
//! so many rows of their input:
//!
//! ```text
//! Limit_k(AI(R; n))                 ⇒ AI(R; n = min(k, n))
//! Limit_{k,m}(AI(R; n))             ⇒ Limit_{k,m}(AI(R; n = min(k + m, n)))
//! δ(π_K(AI(R)))                     ⇒ δ(π_K(AI(R; n = 1 per K)))
//! S ⋉_{J = K} AI(R)                 ⇒ S ⋉_{J = K} AI(R; n = 1 per K)
//! π_K(σ_{c ≥ k}(γ_{K; count(*) → c}(AI(R))))
//!                                   ⇒ … AI(R; n = k per K) …
//! ```
//!
//! A `LIMIT` with no offset leaves the plan; every other operator stays,
//! as the executor may return more rows of a key than it needs before
//! it stops the key.
//!
//! The need reaches its node through the SELECT lists and derived-table
//! aliases between them, which return one row for each row, with the key
//! columns followed by name. A per-key need also passes a `Filter` that
//! reads only columns of the node's input: the filter moves below the node, as a
//! fetch counts the rows left after it. Any other operator can change
//! how many rows there are, and stops the need.
//!
//! The node takes the fetch when it accepts one ([`AiNode::accepts_fetch`])
//! and has none of the other kind. A `DISTINCT` gives it to any node of
//! rows, keyed by columns of the node's input or by the node's answer
//! alone, such as each label of a classification. A semi-join and a
//! count threshold give it only where the executor stops a key: the
//! node, and every `AiNode` below it, is an `AI.IF` over the same one
//! table, with no join, fetch, guard, or `HINT`.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::logical_expr::expr::BinaryExpr;
use datafusion::logical_expr::{
    Distinct, Expr, FetchType, Filter, JoinType, Limit, LogicalPlan, Operator, SkipType,
};

use crate::ai::{AiNode, Fetch, Semantics};
use crate::extension::{as_node, extension_node};
use crate::rules::distinct::is_distinct;

/// How many rows of a plan are enough for the operator above it.
enum Enough {
    /// The first `n` rows.
    Rows(usize),
    /// One row of each value of `keys`, for a `DISTINCT`.
    Distinct(Vec<Column>),
    /// `n` rows of each value of `keys`, where the executor stops a key.
    PerKey(Vec<Column>, usize),
}

/// Give the `AiNode` under each `DISTINCT` one row per value of the
/// `DISTINCT` columns.
pub(super) fn distinct_fetches(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    plan.transform_up(|node| {
        let input = match &node {
            LogicalPlan::Distinct(Distinct::All(input)) => input,
            LogicalPlan::Aggregate(aggregate) if is_distinct(aggregate) => &aggregate.input,
            _ => return Ok(Transformed::no(node)),
        };
        let keys = input.schema().columns();
        match enough(input, Enough::Distinct(keys), Vec::new())? {
            Some((input, _)) => Ok(Transformed::yes(
                node.with_new_exprs(node.expressions(), vec![input])?,
            )),
            None => Ok(Transformed::no(node)),
        }
    })
}

/// Give a `LIMIT` to the `AiNode` whose rows it reads, as its fetch.
pub(super) fn limit_fetches(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    plan.transform_down(|plan| {
        let LogicalPlan::Limit(limit) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let (Ok(SkipType::Literal(skip)), Ok(FetchType::Literal(Some(fetch)))) =
            (limit.get_skip_type(), limit.get_fetch_type())
        else {
            return Ok(Transformed::no(plan));
        };
        // DataFusion's `eliminate_limit` replaces a LIMIT 0 with no rows
        if fetch == 0 {
            return Ok(Transformed::no(plan));
        }
        let Some((input, changed)) = enough(&limit.input, Enough::Rows(skip + fetch), Vec::new())?
        else {
            return Ok(Transformed::no(plan));
        };
        if skip == 0 {
            return Ok(Transformed::yes(input));
        }
        if !changed {
            return Ok(Transformed::no(plan));
        }
        Ok(Transformed::yes(LogicalPlan::Limit(Limit {
            skip: limit.skip.clone(),
            fetch: limit.fetch.clone(),
            input: Arc::new(input),
        })))
    })
}

/// Give the AI filter on a semi-join's right side one row per join key,
/// and the AI filter under a count threshold `k` rows per group key.
pub(super) fn key_fetches(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    plan.transform_up(|plan| {
        let found = match semi_join_keys(&plan) {
            Some(found) => Some(found),
            None => count_threshold_keys(&plan),
        };
        let Some((path, keys, n)) = found else {
            return Ok(Transformed::no(plan));
        };
        // the plan's operators from the one that needs the rows down to
        // the input it needs them of
        let mut below = &plan;
        let mut above = Vec::new();
        for index in &path {
            above.push((below, *index));
            below = below.inputs()[*index];
        }
        let Some((mut input, _)) = enough(below, Enough::PerKey(keys, n), Vec::new())? else {
            return Ok(Transformed::no(plan));
        };
        for (operator, index) in above.into_iter().rev() {
            let mut inputs: Vec<LogicalPlan> = operator.inputs().into_iter().cloned().collect();
            inputs[index] = input;
            input = operator.with_new_exprs(operator.expressions(), inputs)?;
        }
        Ok(Transformed::yes(input))
    })
}

/// The path to a semi-join's filtered side, the join keys there, and
/// the one row of each the join needs.
fn semi_join_keys(plan: &LogicalPlan) -> Option<(Vec<usize>, Vec<Column>, usize)> {
    let LogicalPlan::Join(join) = plan else {
        return None;
    };
    let side = match join.join_type {
        JoinType::LeftSemi => 1,
        JoinType::RightSemi => 0,
        _ => return None,
    };
    if join.filter.is_some() || join.on.is_empty() {
        return None;
    }
    let keys: Vec<Column> = join
        .on
        .iter()
        .map(|(left, right)| match if side == 1 { right } else { left } {
            Expr::Column(column) => Some(column.clone()),
            _ => None,
        })
        .collect::<Option<_>>()?;
    Some((vec![side], keys, 1))
}

/// The path from the SELECT list over a `HAVING COUNT(*) >= k` to the
/// grouped rows, the group keys, and `k`, when nothing else reads the
/// count.
fn count_threshold_keys(plan: &LogicalPlan) -> Option<(Vec<usize>, Vec<Column>, usize)> {
    let LogicalPlan::Projection(projection) = plan else {
        return None;
    };
    let LogicalPlan::Filter(filter) = projection.input.as_ref() else {
        return None;
    };
    let LogicalPlan::Aggregate(aggregate) = filter.input.as_ref() else {
        return None;
    };
    let [count] = aggregate.aggr_expr.as_slice() else {
        return None;
    };
    if !is_count_star(count) {
        return None;
    }
    let schema = aggregate.schema.as_ref();
    let count_column = Column::from(schema.qualified_field(aggregate.group_expr.len()));
    if projection
        .expr
        .iter()
        .any(|expr| expr.column_refs().contains(&count_column))
    {
        return None;
    }
    let threshold = at_least(&filter.predicate, &count_column)?;
    let keys: Vec<Column> = aggregate
        .group_expr
        .iter()
        .map(|expr| match expr {
            Expr::Column(column) => Some(column.clone()),
            _ => None,
        })
        .collect::<Option<_>>()?;
    (!keys.is_empty()).then_some((vec![0, 0, 0], keys, threshold))
}

/// `plan` with `need` as the fetch of the `AiNode` whose rows it
/// returns, and whether that changed the node; None when the need does
/// not reach a node or the node does not take it, as the module
/// explains. `filters` are the plain filters the need passed, outermost
/// first, which move below the node.
fn enough(
    plan: &LogicalPlan,
    need: Enough,
    mut filters: Vec<Expr>,
) -> Result<Option<(LogicalPlan, bool)>> {
    // the key columns as the operator's input names them
    let keys_below = |name: &dyn Fn(&Column) -> Option<Column>, need: Enough| -> Option<Enough> {
        let below = |keys: Vec<Column>| keys.iter().map(name).collect::<Option<Vec<_>>>();
        Some(match need {
            Enough::Rows(rows) => Enough::Rows(rows),
            Enough::Distinct(keys) => Enough::Distinct(below(keys)?),
            Enough::PerKey(keys, rows) => Enough::PerKey(below(keys)?, rows),
        })
    };
    match plan {
        LogicalPlan::Projection(projection) => {
            let name = |key: &Column| -> Option<Column> {
                let index = projection.schema.index_of_column(key).ok()?;
                match projection.expr[index].clone().unalias() {
                    Expr::Column(column) => Some(column),
                    _ => None,
                }
            };
            let Some(need) = keys_below(&name, need) else {
                return Ok(None);
            };
            let Some((input, changed)) = enough(&projection.input, need, filters)? else {
                return Ok(None);
            };
            let rebuilt = plan.with_new_exprs(plan.expressions(), vec![input])?;
            Ok(Some((rebuilt, changed)))
        }
        LogicalPlan::SubqueryAlias(alias) => {
            let name = |key: &Column| -> Option<Column> {
                let index = alias.schema.index_of_column(key).ok()?;
                Some(Column::from(alias.input.schema().qualified_field(index)))
            };
            let Some(need) = keys_below(&name, need) else {
                return Ok(None);
            };
            let Some((input, changed)) = enough(&alias.input, need, filters)? else {
                return Ok(None);
            };
            let rebuilt = plan.with_new_exprs(plan.expressions(), vec![input])?;
            Ok(Some((rebuilt, changed)))
        }
        LogicalPlan::Filter(filter) if !matches!(need, Enough::Rows(_)) => {
            filters.push(filter.predicate.clone());
            enough(&filter.input, need, filters)
        }
        LogicalPlan::Extension(_) => {
            let Some(node) = as_node::<AiNode>(plan) else {
                return Ok(None);
            };
            if !node.accepts_fetch() {
                return Ok(None);
            }
            // a filter moves below the node only when the node's input has
            // every column it reads: not the node's answer, and not a
            // column a SELECT list between them computes
            let input = node.input.schema();
            let reads_other = |expr: &Expr| {
                expr.column_refs()
                    .iter()
                    .any(|column| !input.has_column(column))
            };
            if filters.iter().any(reads_other) {
                return Ok(None);
            }
            let fetch = match need {
                Enough::Rows(rows) => match &node.fetch {
                    Some(Fetch::PerKey { .. }) => return Ok(None),
                    Some(Fetch::Global(found)) if *found <= rows => {
                        return Ok(Some((plan.clone(), false)));
                    }
                    Some(Fetch::Global(_)) | None => Fetch::Global(rows),
                },
                Enough::Distinct(keys) => {
                    if node.semantics != Semantics::Rows || node.fetch.is_some() {
                        return Ok(None);
                    }
                    let Some(key) = distinct_key(node, &keys) else {
                        return Ok(None);
                    };
                    Fetch::PerKey { key, n: 1 }
                }
                Enough::PerKey(keys, n) => {
                    if !stops_per_key(node) {
                        return Ok(None);
                    }
                    let Some(key) = input_key(node, &keys) else {
                        return Ok(None);
                    };
                    Fetch::PerKey { key, n }
                }
            };
            let mut below = node.input.as_ref().clone();
            for predicate in filters.into_iter().rev() {
                below = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(below))?);
            }
            let mut fetched = node.with_input(below)?;
            fetched.fetch = Some(fetch);
            Ok(Some((extension_node(fetched.checked()?), true)))
        }
        _ => Ok(None),
    }
}

/// The names of `keys` when each is a column of the node's input whose
/// name no other column there has: the executor groups documents by the
/// key's column names.
fn input_key(node: &AiNode, keys: &[Column]) -> Option<Vec<String>> {
    let input = node.input.schema();
    let mut names: Vec<String> = Vec::new();
    for key in keys {
        let named = input
            .fields()
            .iter()
            .filter(|field| field.name() == &key.name)
            .count();
        if !input.has_column(key) || named != 1 {
            return None;
        }
        if !names.contains(&key.name) {
            names.push(key.name.clone());
        }
    }
    (!names.is_empty()).then_some(names)
}

/// The key of a `DISTINCT` over `keys`: the node's returned answer
/// alone, or columns of its input ([`input_key`]).
fn distinct_key(node: &AiNode, keys: &[Column]) -> Option<Vec<String>> {
    let returns_answer = node
        .returned_fields()
        .iter()
        .any(|field| field.name() == &node.answer);
    if let [column] = keys
        && column.relation.is_none()
        && column.name == node.answer
        && returns_answer
    {
        return Some(vec![node.answer.clone()]);
    }
    input_key(node, keys)
}

/// Whether `expr` counts every row: `count(*)`, which DataFusion writes as
/// `count(Int64(1))`, with no filter, DISTINCT, or ORDER BY.
fn is_count_star(expr: &Expr) -> bool {
    let expr = match expr {
        Expr::Alias(alias) => alias.expr.as_ref(),
        other => other,
    };
    let Expr::AggregateFunction(function) = expr else {
        return false;
    };
    let params = &function.params;
    function.func.name() == "count"
        && !params.distinct
        && params.filter.is_none()
        && params.order_by.is_empty()
        && matches!(params.args.as_slice(), [Expr::Literal(value, _)] if !value.is_null())
}

/// The positive integer `k` when `predicate` is `count >= k`, `count > k
/// - 1`, or the same written the other way round.
fn at_least(predicate: &Expr, count: &Column) -> Option<usize> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = predicate else {
        return None;
    };
    let is_count = |expr: &Expr| matches!(expr, Expr::Column(column) if column == count);
    let integer = |expr: &Expr| match expr {
        Expr::Literal(value, _) => value.to_string().parse::<i64>().ok(),
        _ => None,
    };
    let bound = if is_count(left) {
        match op {
            Operator::GtEq => integer(right)?,
            Operator::Gt => integer(right)? + 1,
            _ => return None,
        }
    } else if is_count(right) {
        match op {
            Operator::LtEq => integer(left)?,
            Operator::Lt => integer(left)? + 1,
            _ => return None,
        }
    } else {
        return None;
    };
    usize::try_from(bound).ok().filter(|bound| *bound > 0)
}

/// Whether the executor stops `node` per key: it and every `AiNode`
/// below it are AI.IF filters of rows over the same one table, with no
/// fetch or `HINT`, and nothing below joins.
fn stops_per_key(node: &AiNode) -> bool {
    let tables = |node: &AiNode| -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        for column in node.call.column_refs() {
            if let Some(table) = &column.relation
                && !found.contains(&table.table().to_string())
            {
                found.push(table.table().to_string());
            }
        }
        found
    };
    let table = tables(node);
    let filters = |found: &AiNode| {
        found.function() == "ai_filter"
            && found.semantics == Semantics::Rows
            && found.fetch.is_none()
            && found.hint.is_none()
            && found.guard.is_none()
            && tables(found) == table
    };
    if table.len() != 1 || !filters(node) {
        return false;
    }
    let mut plain = true;
    let _ = node.input.apply(|below| {
        let joins = matches!(below, LogicalPlan::Join(_));
        let other_call = as_node::<AiNode>(below).is_some_and(|found| !filters(found));
        let unknown =
            matches!(below, LogicalPlan::Extension(_)) && as_node::<AiNode>(below).is_none();
        if joins || other_call || unknown {
            plain = false;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    plain
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::datasource::provider_as_source;
    use datafusion::logical_expr::expr::ScalarFunction;
    use datafusion::logical_expr::{Expr, LogicalPlanBuilder, col, lit};
    use datafusion::optimizer::{Optimizer, OptimizerContext};

    use super::*;
    use crate::rules::derive::Derive;

    fn kind_reviews(hint: Option<Expr>) -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("body", DataType::Utf8, true),
        ]));
        let reviews = LogicalPlanBuilder::scan(
            "reviews",
            provider_as_source(Arc::new(EmptyTable::new(schema))),
            None,
        )
        .unwrap()
        .build()
        .unwrap();
        let call = Expr::ScalarFunction(ScalarFunction::new_udf(
            crate::functions::quail_functions()
                .into_iter()
                .find(|found| found.name() == "ai_filter")
                .unwrap(),
            vec![lit("Is {0} kind?"), col("reviews.body")],
        ));
        let mut node = AiNode::try_new(
            Arc::new(reviews),
            call,
            "kind".into(),
            vec![Field::new("kind", DataType::Boolean, true)],
        )
        .unwrap();
        node.filter = Some(col("kind"));
        node.projection = Some(Vec::new());
        node.hint = hint;
        extension_node(node.checked().unwrap())
    }

    fn pushed(plan: LogicalPlan) -> String {
        Optimizer::with_rules(vec![Arc::new(Derive::default())])
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
            .display_indent()
            .to_string()
    }

    #[test]
    fn a_limit_over_an_ai_filter_becomes_its_fetch() {
        let plan = LogicalPlanBuilder::from(kind_reviews(None))
            .project(vec![col("reviews.id")])
            .unwrap()
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(!text.contains("Limit"), "{text}");
        assert!(text.contains("fetch: 3"), "{text}");
    }

    #[test]
    fn a_limit_with_an_offset_stays_and_fetches_its_offset_rows_too() {
        let plan = LogicalPlanBuilder::from(kind_reviews(None))
            .limit(2, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(text.contains("Limit: skip=2, fetch=3"), "{text}");
        assert!(text.contains("fetch: 5"), "{text}");
    }

    #[test]
    fn a_limit_stays_above_a_filter_or_a_hint() {
        let filtered = LogicalPlanBuilder::from(kind_reviews(None))
            .filter(col("reviews.id").gt(lit(3)))
            .unwrap()
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let hinted = LogicalPlanBuilder::from(kind_reviews(Some(lit(0.5))))
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        for plan in [filtered, hinted] {
            let text = pushed(plan);
            assert!(text.contains("Limit: skip=0, fetch=3"), "{text}");
            assert!(!text.contains("fetch: 3"), "{text}");
        }
    }
}
