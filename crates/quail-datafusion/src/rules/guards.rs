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
//! the terms [`reads_call_table`] accepts.

use datafusion::common::tree_node::TreeNode;
use datafusion::logical_expr::Expr;

use crate::ai::AiNode;

/// Whether `test` reads columns of the node's input, of the one table its
/// call reads, and calls no AI function.
pub(crate) fn reads_call_table(test: &Expr, node: &AiNode) -> bool {
    let tables = |expr: &Expr| -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        for column in expr.column_refs() {
            let table = column
                .relation
                .as_ref()
                .map(|table| table.table().to_string());
            if let Some(table) = table
                && !found.contains(&table)
            {
                found.push(table);
            }
        }
        found
    };
    let call = tables(&node.call);
    let read = tables(test);
    let input = node.input.schema();
    call.len() == 1
        && read == call
        // an unqualified column is another node's answer, not the table's
        && test
            .column_refs()
            .iter()
            .all(|column| column.relation.is_some() && input.has_column(column))
        && !test
            .exists(|inner| Ok(crate::is_ai_call(inner)))
            .unwrap_or(true)
}

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
