//! Which fetch an `AiNode` can take, for [`super::demand`].
//!
//! An `AiNode`'s fetch keeps the first `n` rows that pass its filter, in
//! all or for each value of some key columns:
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
//! A `DISTINCT` gives a per-key fetch to any node of rows, keyed by
//! columns of the node's input or by the node's answer alone, such as
//! each label of a classification. A semi-join and a count threshold
//! give it only where the executor stops a key ([`stops_per_key`]).

use datafusion::common::Column;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::expr::BinaryExpr;
use datafusion::logical_expr::{Expr, LogicalPlan, Operator};

use crate::ai::{AiNode, Semantics};
use crate::extension::as_node;

/// The names of `keys` when each is a column of the node's input whose
/// name no other column there has: the executor groups documents by the
/// key's column names.
pub(super) fn input_key(node: &AiNode, keys: &[Column]) -> Option<Vec<String>> {
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
/// alone, or columns of its input ([`input_key`]). An `AI.IF` is keyed
/// by its input only: the executor stops a filter per group of input
/// rows, not per answer.
pub(super) fn distinct_key(node: &AiNode, keys: &[Column]) -> Option<Vec<String>> {
    let returns_answer = node.function() != "ai_filter"
        && node
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
pub(super) fn is_count_star(expr: &Expr) -> bool {
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
pub(super) fn at_least(predicate: &Expr, count: &Column) -> Option<usize> {
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
pub(super) fn stops_per_key(node: &AiNode) -> bool {
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

    use std::sync::Arc;

    use super::*;
    use crate::extension::extension_node;
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
    fn a_limit_over_a_filter_of_the_input_fetches_after_the_filter_moves_below() {
        let plan = LogicalPlanBuilder::from(kind_reviews(None))
            .filter(col("reviews.id").gt(lit(3)))
            .unwrap()
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(!text.contains("Limit"), "{text}");
        let fetch = text.find("fetch: 3").expect("a fetch");
        assert!(
            fetch < text.find("Filter: reviews.id > Int32(3)").unwrap(),
            "{text}"
        );
    }

    #[test]
    fn a_limit_stays_above_a_hint() {
        let plan = LogicalPlanBuilder::from(kind_reviews(Some(lit(0.5))))
            .limit(0, Some(3))
            .unwrap()
            .build()
            .unwrap();
        let text = pushed(plan);
        assert!(text.contains("Limit: skip=0, fetch=3"), "{text}");
        assert!(!text.contains("fetch: 3"), "{text}");
    }
}
