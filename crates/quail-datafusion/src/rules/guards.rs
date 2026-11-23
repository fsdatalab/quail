//! What a guard may read, and how many rows it passes.
//!
//! An `AiNode`'s guard `g` says which rows the model is asked about; the
//! answer of every other row is NULL, `f_g = CASE WHEN g THEN f END`.
//! [`crate::Derive`]'s demand pass guards a column node with where its
//! answer is read: a SELECT list that reads it only as `CASE WHEN c THEN
//! a END` needs the model only where `c` is TRUE,
//!
//! ```text
//! π_{B, (CASE WHEN c THEN a END) → z}(AI_{f → a}(R))
//!     ⇒ π_{B, (CASE WHEN c THEN a END) → z}(AI_{f → a}(R; g = g ∧ c))
//! ```
//!
//! For example, `SELECT r.id, CASE WHEN r.long THEN AI.MAP(...) END AS
//! summary` writes summaries of the long reviews only. The executor
//! evaluates a guard on the rows the node reads, so a guard keeps only
//! the terms [`crate::ai::AiNode::reads_call_table`] accepts.
//!
//! An AI test in an OR is guarded too: it is asked only where the rest
//! of the OR is not TRUE.
//!
//! A filter `a OR b` keeps every row where `a` is TRUE, whatever `b` is.
//! When `b` is an `AI.IF` node's answer and `a` tests the input alone, the
//! model need not answer for those rows: with the guard `a IS NOT TRUE`,
//! their answer is NULL, and `a OR NULL` is still TRUE.
//!
//! ```text
//! AI_{f → b}(R; p = a ∨ b) ⇒ AI_{f → b}(R; g = (a IS NOT TRUE), p = a ∨ b)
//! ```
//!
//! The lowering writes `WHERE r.flagged OR AI.IF(...)` as such a node:
//! its filter is the OR, with the answer in the call's place, and it
//! returns none of the columns it adds. The rewrite is the logical form
//! of `π_Q(σ_{a ∨ b}(AI_{f → b}(R))) ⇒ π_Q(σ_{a ∨ b}(AI_{f → b}(R; g =
//! g ∧ (a IS NOT TRUE))))`, with `g` TRUE, once `derive` has made the OR the
//! node's filter. The rule guards an `AiNode` when:
//!
//! - the node is an `AI.IF` of rows with no guard, `HINT`, or fetch,
//!   and it returns no column it adds, so nothing above it reads `b`;
//! - a term of its filter is an OR, one of whose terms is `b`, and no
//!   other term of the filter reads `b`;
//! - the other terms of the OR, `a`, read only columns of the one table
//!   the call reads, or are answers of the earlier `AI.IF`s of the OR.
//!
//! An OR of several `AI.IF`s, `a ∨ f_1 ∨ … ∨ f_k`, lowers to one node per
//! call: each `f_j` but the last returns its answer `b_j` and drops no
//! row, in written order, below the last, whose filter is `a ∨ b_1 ∨ … ∨
//! b_k`. The rule guards each earlier node with the terms before it,
//!
//! ```text
//! AI_{f_j → b_j}(R) ⇒ AI_{f_j → b_j}(R; g = a IS NOT TRUE ∧ b_1 IS NOT TRUE ∧ … ∧ b_{j-1} IS NOT TRUE)
//! ```
//!
//! leaving the first unguarded when there is no `a`, and the last with
//! every other term. A guard writes `t_1 OR t_2 IS NOT TRUE` as `t_1 IS
//! NOT TRUE AND t_2 IS NOT TRUE`, which is the same test.
//!
//! A term of the OR may read an answer as `NOT b`; the guards are the
//! same, with `NOT b` in the place of `b`.
//!
//! The executor's AI filter asks the model about the rows where the
//! guard is TRUE and passes each row where `a` is TRUE without asking.
//!
//! The guards are written in the order the query wrote the calls, but
//! the calls depend on no order: the planner orders the calls of one OR
//! by cost, as it orders ANDed filters, and the executor asks each call
//! only about the rows no call of the OR before it passed. The binder

use datafusion::logical_expr::Expr;

/// The estimated fraction of rows where `guard` is TRUE, as the planner
/// estimates a filter's: a `HINT`'s selectivity where the guard gives
/// one, the selectivity of an earlier `AI.IF` whose answer it reads, by
/// the answer's name in `answers`, and the default selectivity for any
/// other test. `x IS NOT TRUE` and `NOT x` pass the rest of `x`'s rows,
/// an AND the product of its terms', and an OR the rows any term passes,
/// each term taken as independent.
pub(crate) fn guard_selectivity(guard: &Expr, answers: &[(String, Option<f64>)]) -> f64 {
    use datafusion::logical_expr::{BinaryExpr, Operator};
    use quail_planner::statistics::effective_selectivity;
    match guard {
        Expr::IsNotTrue(inner) | Expr::Not(inner) => 1.0 - guard_selectivity(inner, answers),
        Expr::BinaryExpr(BinaryExpr { left, op, right }) if *op == Operator::And => {
            guard_selectivity(left, answers) * guard_selectivity(right, answers)
        }
        Expr::BinaryExpr(BinaryExpr { left, op, right }) if *op == Operator::Or => {
            1.0 - (1.0 - guard_selectivity(left, answers))
                * (1.0 - guard_selectivity(right, answers))
        }
        Expr::Column(column) if column.relation.is_none() => effective_selectivity(
            answers
                .iter()
                .find(|(name, _)| *name == column.name)
                .and_then(|(_, selectivity)| *selectivity),
        ),
        Expr::ScalarFunction(call) if call.name() == "hint" => {
            effective_selectivity(call.args.get(1).and_then(hinted_selectivity))
        }
        _ => effective_selectivity(None),
    }
}

/// The selectivity a `HINT`'s options give, `{'selectivity': s}`.
fn hinted_selectivity(options: &Expr) -> Option<f64> {
    let Expr::ScalarFunction(call) = options else {
        return None;
    };
    if call.name() != "named_struct" {
        return None;
    }
    call.args.chunks(2).find_map(|pair| match pair {
        [Expr::Literal(key, _), Expr::Literal(value, _)]
            if key.try_as_str().flatten() == Some("selectivity") =>
        {
            value.to_string().parse().ok()
        }
        _ => None,
    })
}
