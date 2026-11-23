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
//! give it only where the executor stops a key. The node says which
//! fetch it takes ([`crate::ai::AiNode::per_key_fetch`]); this module
//! reads the counts the operators above it ask for.

use datafusion::common::Column;
use datafusion::logical_expr::expr::BinaryExpr;
use datafusion::logical_expr::{Expr, Operator};

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
