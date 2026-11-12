//! A filter step of [`crate::Derive`]: an extraction whose
//! answer must contain a literal reads only the documents that contain
//! it.
//!
//! `AI.EXTRACT` copies each answer from its document: an answer is NULL
//! or a substring of the document `d`. A `LIKE` pattern `x` of an answer
//! names literal text `s` that every string it matches contains, so a
//! document without `s` cannot give a passing answer:
//!
//! ```text
//! σ_{a LIKE x}(AI_{extract(d) → a}(R))
//!     ⇒ σ_{a LIKE x}(AI_{extract(d) → a}(σ_{d ILIKE '%s%'}(R)))
//! ```
//!
//! For example, `WHERE lease.end_date LIKE '%2027%'` extracts end dates
//! only from the contracts that contain `2027`. The test of the answer
//! stays: containing `s` does not make the answer match `x`.
//!
//! `s` is the longest run of the pattern between its wildcards, then the
//! longest word of that run: whitespace and case are where a decoded
//! answer may differ from its document, so the precheck ignores case
//! (`ILIKE`) and reads no whitespace. The rule rewrites an extraction
//! node of rows with no fetch whose filter has a term `get_field(a, f)
//! LIKE x` or `ILIKE x`, not negated, with a literal pattern.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::expr::Like;
use datafusion::logical_expr::expr_rewriter::unnormalize_col;
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::logical_expr::{Expr, Filter, LogicalPlan};

use crate::ai::{AiNode, Semantics};
use crate::extension::{as_node, extension_node};

/// Test each extraction's document for the literal its answer's `LIKE`
/// needs, as the module explains.
pub(super) fn substring_prechecks(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    plan.transform_up(precheck)
}

fn precheck(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let Some(node) = as_node::<AiNode>(&plan) else {
        return Ok(Transformed::no(plan));
    };
    if node.function() != "ai_extract" || node.semantics != Semantics::Rows || node.fetch.is_some()
    {
        return Ok(Transformed::no(plan));
    }
    let Expr::ScalarFunction(call) = &node.call else {
        return Ok(Transformed::no(plan));
    };
    let Some(document) = call.args.first() else {
        return Ok(Transformed::no(plan));
    };
    let Some(filter) = &node.filter else {
        return Ok(Transformed::no(plan));
    };
    let Some(literal) = split_conjunction(filter)
        .into_iter()
        .filter_map(|term| required_literal(term, node))
        .max_by_key(|literal| literal.chars().count())
    else {
        return Ok(Transformed::no(plan));
    };
    let test = Expr::Like(Like::new(
        false,
        Box::new(document.clone()),
        Box::new(Expr::Literal(
            ScalarValue::Utf8(Some(format!("%{}%", escaped(&literal)))),
            None,
        )),
        Some('\\'),
        true,
    ));
    if tests_below(&node.input, &test) {
        return Ok(Transformed::no(plan));
    }
    let input = LogicalPlan::Filter(Filter::try_new(test, Arc::clone(&node.input))?);
    Ok(Transformed::yes(extension_node(node.with_input(input)?)))
}

/// The literal text every answer `term` accepts contains, when `term` is
/// a `LIKE` of a field of the node's answer.
fn required_literal(term: &Expr, node: &AiNode) -> Option<String> {
    let Expr::Like(Like {
        negated: false,
        expr,
        pattern,
        escape_char,
        ..
    }) = term
    else {
        return None;
    };
    let Expr::ScalarFunction(field) = expr.as_ref() else {
        return None;
    };
    let reads_answer = field.name() == "get_field"
        && matches!(field.args.first(), Some(Expr::Column(column))
            if column.relation.is_none() && column.name == node.answer);
    if !reads_answer {
        return None;
    }
    let Expr::Literal(ScalarValue::Utf8(Some(pattern)), _) = pattern.as_ref() else {
        return None;
    };
    let run = literal_runs(pattern, escape_char.unwrap_or('\\'))
        .into_iter()
        .max_by_key(|run| run.chars().count())?;
    run.split_whitespace()
        .max_by_key(|word| word.chars().count())
        .map(ToString::to_string)
}

/// The runs of literal text between a `LIKE` pattern's wildcards.
fn literal_runs(pattern: &str, escape: char) -> Vec<String> {
    let mut runs = vec![String::new()];
    let mut chars = pattern.chars();
    while let Some(found) = chars.next() {
        match found {
            '%' | '_' => runs.push(String::new()),
            found if found == escape => {
                if let Some(next) = chars.next() {
                    runs.last_mut().expect("a run").push(next);
                }
            }
            found => runs.last_mut().expect("a run").push(found),
        }
    }
    runs.retain(|run| !run.is_empty());
    runs
}

/// `text` as literal text in a `LIKE` pattern escaped with `\`.
fn escaped(text: &str) -> String {
    let mut found = String::new();
    for character in text.chars() {
        if matches!(character, '%' | '_' | '\\') {
            found.push('\\');
        }
        found.push(character);
    }
    found
}

/// Whether a filter at or below `plan` already makes `test`, read by
/// column names: DataFusion's `push_down_filter` renames a test's table
/// as it moves it below a subquery alias.
fn tests_below(plan: &LogicalPlan, test: &Expr) -> bool {
    let named = |expr: &Expr| unnormalize_col(expr.clone());
    let wanted = named(test);
    let mut found = false;
    let _ = plan.apply(|node| {
        let predicates: Vec<&Expr> = match node {
            LogicalPlan::Filter(filter) => split_conjunction(&filter.predicate),
            LogicalPlan::TableScan(scan) => scan.filters.iter().collect(),
            LogicalPlan::Projection(_) | LogicalPlan::SubqueryAlias(_) => Vec::new(),
            _ => return Ok(TreeNodeRecursion::Stop),
        };
        if predicates
            .into_iter()
            .any(|predicate| named(predicate) == wanted)
        {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_patterns_literal_runs_skip_its_wildcards_and_escapes() {
        assert_eq!(literal_runs("%20_7%", '\\'), ["20", "7"]);
        assert_eq!(literal_runs("a\\%b%c", '\\'), ["a%b", "c"]);
        assert_eq!(literal_runs("%%", '\\'), Vec::<String>::new());
        assert_eq!(escaped("50%_\\"), "50\\%\\_\\\\");
    }
}
