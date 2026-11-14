//! The guard of an AI test in an OR, which [`crate::Derive`]'s demand
//! pass sets: the test is asked
//! only where the rest of the OR is not TRUE.
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
//! reads a guard on another call's answer as that instruction
//! ([`OrParts`]).

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::utils::{disjunction, split_binary, split_conjunction};
use datafusion::logical_expr::{Expr, LogicalPlan, Operator};

use crate::ai::{AiNode, Semantics};
use crate::extension::as_node;

/// The terms of an `AI.IF` node's filter OR other than its answer `b`,
/// when the filter is `a ∨ b` as the module describes, and whether the
/// OR reads the answer as `NOT b`.
pub(crate) fn or_terms(node: &AiNode) -> Option<(Expr, bool)> {
    let filter = node.filter.as_ref()?;
    let reads_answer = |term: &Expr| {
        term.column_refs()
            .iter()
            .any(|column| column.relation.is_none() && node.is_field(&column.name))
    };
    let conjuncts = split_conjunction(filter);
    let mut found = None;
    for conjunct in &conjuncts {
        let terms = split_binary(conjunct, Operator::Or);
        let (answers, others): (Vec<&Expr>, Vec<&Expr>) = terms
            .into_iter()
            .partition(|term| node.answer_term(term).is_some());
        if answers.is_empty() || others.is_empty() {
            if reads_answer(conjunct) {
                return None;
            }
            continue;
        }
        if found.is_some() || others.iter().any(|term| reads_answer(term)) {
            return None;
        }
        // the OR reads the answer one way: `b OR NOT b` is no test of it
        let [answer] = answers.as_slice() else {
            return None;
        };
        let negated = node.answer_term(answer) == Some(true);
        found = disjunction(others.into_iter().cloned()).map(|rest| (rest, negated));
    }
    found
}

/// The parts of the OR an `AI.IF` node filters on.
pub(crate) struct OrParts {
    /// The terms that ask no model, as one OR; None when it has none.
    pub(crate) plain: Option<Expr>,
    /// The answer of each other `AI.IF` of the OR, and whether the OR
    /// reads it as `NOT`.
    pub(crate) others: Vec<(String, bool)>,
    /// Whether the OR reads the node's own answer as `NOT`.
    pub(crate) negated: bool,
}

/// The parts of the OR `node` filters on, when its filter is one, as the
/// module describes.
pub(crate) fn or_parts(node: &AiNode) -> Option<OrParts> {
    let (rest, negated) = or_terms(node)?;
    let answers = answers_below(&node.input);
    let mut plain = Vec::new();
    let mut others = Vec::new();
    for term in split_binary(&rest, Operator::Or) {
        let (read, reversed) = match term {
            Expr::Not(inner) => (inner.as_ref(), true),
            other => (other, false),
        };
        match read.try_as_col() {
            Some(column) if column.relation.is_none() && answers.contains(&column.name) => {
                if others.iter().any(|(name, _)| *name == column.name) {
                    return None;
                }
                others.push((column.name.clone(), reversed));
            }
            _ => plain.push(term.clone()),
        }
    }
    Some(OrParts {
        plain: disjunction(plain),
        others,
        negated,
    })
}

/// The answers of the `AI.IF` nodes in `plan` that return their answer
/// and drop no row: the other `AI.IF`s of an OR.
fn answers_below(plan: &LogicalPlan) -> Vec<String> {
    let mut found = Vec::new();
    let _ = plan.apply(|below| {
        if let Some(asked) = as_node::<AiNode>(below)
            && asked.semantics == Semantics::Rows
            && asked.function() == "ai_filter"
            && asked.filter.is_none()
        {
            found.push(asked.answer.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}
