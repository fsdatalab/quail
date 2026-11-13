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
