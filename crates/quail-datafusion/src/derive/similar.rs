//! AI calls in one region that are written almost the same.
//!
//! The planner asks the model once for calls written the same way
//! ([`crate::MergeRepeatedAiCalls`]). Two calls that differ only in a
//! few characters of a prompt are most likely one call written twice by
//! mistake: the model then answers both, and each answer may differ.
//! [`similar_calls`] finds such pairs: the same function over the same
//! columns and options, whose text arguments differ only in
//!
//! - case, whitespace, or punctuation ([`Likeness::Spelling`]), which
//!   [`crate::derive::derive_region`] refuses; or
//! - at most two letter edits of a text at least 12 characters long,
//!   with the same digits ([`Likeness::Typo`]), which Quail warns about.
//!
//! For example, `AI.IF(PROMPT('Is {0} rude?', r.body))` and `AI.IF(
//! PROMPT('is {0} rude', r.body))` differ in spelling, and `'Is the
//! review about the ending? {0}'` and `'Is the reveiw about the ending?
//! {0}'` by a typo. `'Is it from 2027? {0}'` and `'Is it from 2028?
//! {0}'` differ in a digit and are two calls.

use datafusion::common::ScalarValue;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::{Expr, LogicalPlan, lit};

use crate::ai::AiNode;
use crate::extension::as_node;

/// How alike two calls' texts are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Likeness {
    /// A few letters apart.
    Typo,
    /// The same but for case, whitespace, and punctuation.
    Spelling,
}

/// Two calls of one region written almost the same: how alike they are,
/// and the first text in which they differ, as each call writes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Similar {
    pub likeness: Likeness,
    pub first: String,
    pub second: String,
}

impl Similar {
    /// The message that names both calls and what to do.
    pub fn message(&self) -> String {
        let reason = match self.likeness {
            Likeness::Spelling => "differ only in case, whitespace, or punctuation",
            Likeness::Typo => "differ in a few letters",
        };
        format!(
            "two AI calls over the same columns {reason}, so the model answers \
             each: {:?} and {:?}. Write the call the same way each time to ask \
             the model once",
            self.first, self.second
        )
    }
}

/// The pairs of AI calls in `plan` written almost the same, in plan
/// order.
pub fn similar_calls(plan: &LogicalPlan) -> Vec<Similar> {
    let mut calls: Vec<Expr> = Vec::new();
    let _ = plan.apply(|node| {
        if let Some(ai) = as_node::<AiNode>(node)
            && !calls.contains(&ai.call)
        {
            calls.push(ai.call.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    });
    let mut found = Vec::new();
    for (index, first) in calls.iter().enumerate() {
        for second in &calls[index + 1..] {
            if let Some((likeness, (first, second))) = likeness(first, second) {
                found.push(Similar {
                    likeness,
                    first,
                    second,
                });
            }
        }
    }
    found
}

/// How alike two calls are, and the first texts in which they differ;
/// None when they differ in more than their texts' spelling or a typo.
fn likeness(first: &Expr, second: &Expr) -> Option<(Likeness, (String, String))> {
    let (first_shape, first_texts) = shape(first);
    let (second_shape, second_texts) = shape(second);
    if first_shape != second_shape || first_texts.len() != second_texts.len() {
        return None;
    }
    let mut least: Option<Likeness> = None;
    let mut differing: Option<(String, String)> = None;
    for (left, right) in first_texts.iter().zip(&second_texts) {
        if left == right {
            continue;
        }
        let pair = if normalized(left) == normalized(right) {
            Likeness::Spelling
        } else if is_typo(left, right) {
            Likeness::Typo
        } else {
            return None;
        };
        least = Some(least.map_or(pair, |found| found.min(pair)));
        differing.get_or_insert_with(|| (left.clone(), right.clone()));
    }
    least.zip(differing)
}

/// `expr` with each text literal blanked, and the texts in order.
fn shape(expr: &Expr) -> (Expr, Vec<String>) {
    let mut texts = Vec::new();
    let blank = expr
        .clone()
        .transform_down(|inner| match &inner {
            Expr::Literal(ScalarValue::Utf8(Some(text)), _) => {
                texts.push(text.clone());
                Ok(Transformed::yes(lit("")))
            }
            _ => Ok(Transformed::no(inner)),
        })
        .map(|done| done.data)
        .unwrap_or_else(|_| expr.clone());
    (blank, texts)
}

/// `text` in lower case, without whitespace or punctuation.
fn normalized(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || *c == '{' || *c == '}')
        .flat_map(char::to_lowercase)
        .collect()
}

/// Whether two texts at least 12 characters long are at most two letter
/// edits apart, with the same digits and placeholders.
fn is_typo(left: &str, right: &str) -> bool {
    let (left, right) = (normalized(left), normalized(right));
    let not_letters =
        |text: &str| -> String { text.chars().filter(|c| !c.is_alphabetic()).collect() };
    if left.chars().count().min(right.chars().count()) < 12
        || not_letters(&left) != not_letters(&right)
    {
        return false;
    }
    (1..=2).contains(&datafusion::common::utils::datafusion_strsim::levenshtein(
        &left, &right,
    ))
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::col;

    use super::*;
    use crate::functions::quail_functions;

    fn call(template: &str) -> Expr {
        let functions = quail_functions();
        let find = |name: &str| {
            functions
                .iter()
                .find(|function| function.name() == name)
                .unwrap()
                .clone()
        };
        let prompt = find("prompt").call(vec![lit(template), col("r.body")]);
        find("ai_filter").call(vec![prompt])
    }

    #[test]
    fn calls_that_differ_in_spelling_are_alike() {
        assert_eq!(
            likeness(&call("Is {0} rude?"), &call("is {0}  rude")).map(|found| found.0),
            Some(Likeness::Spelling)
        );
    }

    #[test]
    fn calls_a_typo_apart_are_alike() {
        assert_eq!(
            likeness(
                &call("Is the review about the ending? {0}"),
                &call("Is the reveiw about the ending? {0}")
            )
            .map(|found| found.0),
            Some(Likeness::Typo)
        );
    }

    #[test]
    fn calls_that_differ_in_a_digit_or_a_word_are_not() {
        assert_eq!(
            likeness(&call("Is it from 2027? {0}"), &call("Is it from 2028? {0}")),
            None
        );
        assert_eq!(
            likeness(
                &call("Is the review positive? {0}"),
                &call("Is the review negative? {0}")
            ),
            None
        );
        assert_eq!(
            likeness(&call("Is {0} rude?"), &call("Is {0} crude?")),
            None
        );
    }
}
