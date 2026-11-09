//! Conditions decided on partial answers.
//!
//! While the model writes an answer, the tokens decoded so far, `w`,
//! limit the complete answers still possible, `C(w)`. A condition `q` of
//! the answer is decided TRUE at `w` when `q(z)` is TRUE for every `z` in
//! `C(w)`, and decided not TRUE when `q(z)` is FALSE or NULL for every
//! `z` in `C(w)`; otherwise it is undecided. Undecided is a state of the
//! executor, not a NULL.
//!
//! [`Completions`] describes `C(w)`: the text so far, whether the answer
//! may still be NULL, and how it may go on. An `AI.MAP` answer goes on
//! with any text within its answer limit. An `AI.EXTRACT` answer copies
//! a span of its document, so it goes on with the text after one of the
//! places where the document holds the text so far.
//!
//! [`decide`] reads a condition as a DataFusion expression and finds,
//! for each part, which of TRUE, FALSE, and NULL some answer in `C(w)`
//! gives. It reads these shapes of the answer `a`, with literals:
//!
//! - `a LIKE 'x%'` and `a LIKE 'x'`, `STARTS_WITH(a, 'x')`, `LEFT(a, k)
//!   = 'x'`, and `SUBSTR(a, 1, k) = 'x'`;
//! - `a = 'x'`, `a <> 'x'`, and `a IN ('x', ...)`;
//! - `a IS NULL` and `a IS NOT NULL`;
//! - `LENGTH(a)`, `CHARACTER_LENGTH(a)`, or `CHAR_LENGTH(a)` compared
//!   with an integer;
//! - AND, OR, NOT, and `IS [NOT] TRUE` or `FALSE` of these.
//!
//! Any other part may give any value, so a condition made of it stays
//! undecided until the answer is complete. Parts are combined one at a
//! time, ignoring that two parts read the same answer, so a decision is
//! sound but may come later than it could.

use datafusion::common::ScalarValue;
use datafusion::logical_expr::expr::{InList, Like, ScalarFunction};
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use quail_planner::error::CompileError;
use quail_planner::physical::Expression;

/// What a condition is at a partial answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// TRUE for every complete answer still possible.
    True,
    /// FALSE or NULL for every complete answer still possible.
    NotTrue,
    /// TRUE for some complete answer and not for another.
    Undecided,
}

/// The complete answers still possible after a partial answer, `C(w)`.
#[derive(Debug, Clone)]
pub struct Completions {
    /// The answer's text so far.
    pub prefix: String,
    /// Whether the answer is complete: then `C(w)` is the text alone.
    pub complete: bool,
    /// Whether the answer may still be NULL, as an extraction that may
    /// still answer that it found nothing.
    pub may_be_null: bool,
    /// The texts the answer may go on with, each cut anywhere; None for
    /// any text.
    pub tails: Option<Vec<String>>,
    /// The most characters the answer may add; None for no limit.
    pub max_chars: Option<usize>,
}

impl Completions {
    /// The answers of a model that may write any text after `prefix`.
    pub fn text(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            complete: false,
            may_be_null: false,
            tails: None,
            max_chars: None,
        }
    }

    fn chars(text: &str) -> usize {
        text.chars().count()
    }

    /// Whether some answer adds exactly `rest` and possibly more.
    fn can_add(&self, rest: &str) -> bool {
        if rest.is_empty() {
            return true;
        }
        if self.complete || self.max_chars.is_some_and(|max| Self::chars(rest) > max) {
            return false;
        }
        match &self.tails {
            None => true,
            Some(tails) => tails.iter().any(|tail| tail.starts_with(rest)),
        }
    }

    /// Whether some answer adds at least one character.
    fn can_grow(&self) -> bool {
        if self.complete || self.max_chars == Some(0) {
            return false;
        }
        match &self.tails {
            None => true,
            Some(tails) => tails.iter().any(|tail| !tail.is_empty()),
        }
    }

    /// The fewest and the most characters an answer has; None for no
    /// most.
    fn lengths(&self) -> (usize, Option<usize>) {
        let low = Self::chars(&self.prefix);
        if self.complete {
            return (low, Some(low));
        }
        let longest = self.tails.as_ref().map(|tails| {
            tails
                .iter()
                .map(|tail| Self::chars(tail))
                .max()
                .unwrap_or(0)
        });
        let added = match (longest, self.max_chars) {
            (Some(tail), Some(max)) => Some(tail.min(max)),
            (found, None) | (None, found) => found,
        };
        (low, added.map(|added| low + added))
    }

    /// Which values some answer gives a test of its text: TRUE and FALSE
    /// from the text, and NULL when the answer may be NULL.
    fn text_test(&self, can_true: bool, can_false: bool) -> Possible {
        Possible {
            t: can_true,
            f: can_false,
            n: self.may_be_null,
        }
    }

    fn starts_with(&self, literal: &str) -> Possible {
        let prefix = &self.prefix;
        if prefix.starts_with(literal) {
            return self.text_test(true, false);
        }
        let can_true = literal
            .strip_prefix(prefix.as_str())
            .is_some_and(|rest| self.can_add(rest));
        // the text so far is an answer that does not start with it
        self.text_test(can_true, true)
    }

    fn equals(&self, literal: &str) -> Possible {
        let can_true = literal
            .strip_prefix(self.prefix.as_str())
            .is_some_and(|rest| self.can_add(rest));
        let can_false = self.prefix != literal || self.can_grow();
        self.text_test(can_true, can_false)
    }

    fn length(&self, op: Operator, bound: i64) -> Possible {
        let (low, high) = self.lengths();
        let holds = |length: i64| match op {
            Operator::Eq => length == bound,
            Operator::NotEq => length != bound,
            Operator::Lt => length < bound,
            Operator::LtEq => length <= bound,
            Operator::Gt => length > bound,
            _ => length >= bound,
        };
        let low = low as i64;
        // the lengths are every integer from low to high
        let (can_true, can_false) = match high {
            Some(high) => {
                let high = high as i64;
                let any = |want: bool| (low..=high).any(|length| holds(length) == want);
                if high - low <= 4096 {
                    (any(true), any(false))
                } else {
                    (true, true)
                }
            }
            None => match op {
                Operator::Eq => (bound >= low, true),
                Operator::NotEq => (true, bound >= low),
                Operator::Lt | Operator::LtEq => (holds(low), true),
                _ => (true, !holds(low)),
            },
        };
        self.text_test(can_true, can_false)
    }
}

/// Which of TRUE, FALSE, and NULL some complete answer gives a condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Possible {
    t: bool,
    f: bool,
    n: bool,
}

const ANY: Possible = Possible {
    t: true,
    f: true,
    n: true,
};

impl Possible {
    fn not(self) -> Self {
        Self {
            t: self.f,
            f: self.t,
            n: self.n,
        }
    }

    fn and(self, other: Self) -> Self {
        Self {
            t: self.t && other.t,
            f: self.f || other.f,
            n: (self.n && (other.t || other.n)) || (other.n && (self.t || self.n)),
        }
    }

    fn or(self, other: Self) -> Self {
        self.not().and(other.not()).not()
    }

    fn constant(value: Option<bool>) -> Self {
        Self {
            t: value == Some(true),
            f: value == Some(false),
            n: value.is_none(),
        }
    }
}

/// `expr` without casts.
fn uncast(expr: &Expr) -> &Expr {
    match expr {
        Expr::Cast(cast) => uncast(&cast.expr),
        Expr::TryCast(cast) => uncast(&cast.expr),
        Expr::Alias(alias) => uncast(&alias.expr),
        other => other,
    }
}

fn text_literal(expr: &Expr) -> Option<String> {
    match uncast(expr) {
        Expr::Literal(value, _) => value.try_as_str().flatten().map(str::to_string),
        _ => None,
    }
}

fn integer_literal(expr: &Expr) -> Option<i64> {
    match uncast(expr) {
        Expr::Literal(value, _) => {
            match value.cast_to(&datafusion::arrow::datatypes::DataType::Int64) {
                Ok(ScalarValue::Int64(Some(found))) => Some(found),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The arguments of a call of one of `names`.
fn call_args<'e>(expr: &'e Expr, names: &[&str]) -> Option<&'e [Expr]> {
    match uncast(expr) {
        Expr::ScalarFunction(ScalarFunction { func, args }) if names.contains(&func.name()) => {
            Some(args)
        }
        _ => None,
    }
}

/// The operator with its sides swapped: `1 < a` is `a > 1`.
fn swapped(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => other,
    }
}

/// Reads a condition over the answers in `C(w)`, as the module explains.
struct Reader<'a, F: Fn(&Expr) -> bool> {
    is_answer: F,
    answers: &'a Completions,
}

impl<F: Fn(&Expr) -> bool> Reader<'_, F> {
    fn answer(&self, expr: &Expr) -> bool {
        (self.is_answer)(uncast(expr))
    }

    fn possible(&self, expr: &Expr) -> Possible {
        let answers = self.answers;
        match expr {
            Expr::Alias(alias) => self.possible(&alias.expr),
            Expr::Literal(ScalarValue::Boolean(value), _) => Possible::constant(*value),
            Expr::Not(inner) => self.possible(inner).not(),
            Expr::IsTrue(inner)
            | Expr::IsNotFalse(inner)
            | Expr::IsFalse(inner)
            | Expr::IsNotTrue(inner) => {
                let found = self.possible(inner);
                let (yes, no) = match expr {
                    Expr::IsTrue(_) => (found.t, found.f || found.n),
                    Expr::IsNotTrue(_) => (found.f || found.n, found.t),
                    Expr::IsFalse(_) => (found.f, found.t || found.n),
                    _ => (found.t || found.n, found.f),
                };
                Possible {
                    t: yes,
                    f: no,
                    n: false,
                }
            }
            Expr::IsNull(inner) if self.answer(inner) => Possible {
                t: answers.may_be_null,
                f: true,
                n: false,
            },
            Expr::IsNotNull(inner) if self.answer(inner) => Possible {
                t: true,
                f: answers.may_be_null,
                n: false,
            },
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
                Operator::And => self.possible(left).and(self.possible(right)),
                Operator::Or => self.possible(left).or(self.possible(right)),
                _ => self
                    .compared(left, *op, right)
                    .or_else(|| self.compared(right, swapped(*op), left))
                    .unwrap_or(ANY),
            },
            Expr::Like(Like {
                negated,
                expr: subject,
                pattern,
                escape_char: None,
                case_insensitive: false,
            }) if self.answer(subject) => {
                let Some(pattern) = text_literal(pattern) else {
                    return ANY;
                };
                let found = match pattern.find(['%', '_']) {
                    None => answers.equals(&pattern),
                    Some(at) if at == pattern.len() - 1 && pattern.ends_with('%') => {
                        answers.starts_with(&pattern[..at])
                    }
                    Some(_) => return ANY,
                };
                if *negated { found.not() } else { found }
            }
            Expr::InList(InList {
                expr: subject,
                list,
                negated,
            }) if self.answer(subject) => {
                let mut found = Possible::constant(Some(false));
                for item in list {
                    let Some(literal) = text_literal(item) else {
                        return ANY;
                    };
                    found = found.or(answers.equals(&literal));
                }
                if *negated { found.not() } else { found }
            }
            _ => match call_args(expr, &["starts_with"]) {
                Some([subject, literal]) if self.answer(subject) => match text_literal(literal) {
                    Some(literal) => answers.starts_with(&literal),
                    None => ANY,
                },
                _ => ANY,
            },
        }
    }

    /// `left op right` with the answer, or a function of it, on the left
    /// and a literal on the right; None for another shape.
    fn compared(&self, left: &Expr, op: Operator, right: &Expr) -> Option<Possible> {
        let answers = self.answers;
        let equality = matches!(op, Operator::Eq | Operator::NotEq);
        let negate = |found: Possible| {
            if op == Operator::NotEq {
                found.not()
            } else {
                found
            }
        };
        if equality && self.answer(left) {
            return Some(negate(answers.equals(&text_literal(right)?)));
        }
        if let Some([subject]) = call_args(left, &["length", "character_length", "char_length"])
            && self.answer(subject)
        {
            let bound = integer_literal(right)?;
            return matches!(
                op,
                Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq
            )
            .then(|| answers.length(op, bound));
        }
        let cut = match (
            call_args(left, &["left"]),
            call_args(left, &["substr", "substring"]),
        ) {
            (Some([subject, count]), _) => Some((subject, count)),
            (_, Some([subject, start, count])) if integer_literal(start) == Some(1) => {
                Some((subject, count))
            }
            _ => None,
        };
        if equality
            && let Some((subject, count)) = cut
            && self.answer(subject)
        {
            let count = integer_literal(count)?;
            let literal = text_literal(right)?;
            let length = literal.chars().count() as i64;
            let found = if count <= 0 {
                answers
                    .equals("")
                    .and(Possible::constant(Some(literal.is_empty())))
            } else if length == count {
                answers.starts_with(&literal)
            } else if length < count {
                answers.equals(&literal)
            } else {
                answers.text_test(false, true)
            };
            return Some(negate(found));
        }
        None
    }
}

/// `expr` at the answers in `answers`, with `is_answer` telling which
/// expression reads the answer.
pub fn decide_expr(
    expr: &Expr,
    is_answer: impl Fn(&Expr) -> bool,
    answers: &Completions,
) -> Decision {
    let found = Reader { is_answer, answers }.possible(expr);
    if !found.t {
        Decision::NotTrue
    } else if !found.f && !found.n {
        Decision::True
    } else {
        Decision::Undecided
    }
}

/// `condition`, an [`Expression`], at the answers in `answers`, where
/// the column named `answer` holds the answer.
///
/// # Errors
///
/// The expression does not decode.
pub fn decide(
    condition: &Expression,
    answer: &str,
    answers: &Completions,
) -> Result<Decision, CompileError> {
    let expr = crate::derive::expressions::decode(condition)?;
    let positions: Vec<String> = condition
        .columns
        .iter()
        .enumerate()
        .filter(|(_, column)| {
            column.column == answer || column.name(false) == answer || column.name(true) == answer
        })
        .map(|(index, _)| format!("#{index}"))
        .collect();
    let is_answer = |expr: &Expr| {
        matches!(expr, Expr::Column(column)
            if column.relation.is_none() && positions.contains(&column.name))
    };
    Ok(decide_expr(&expr, is_answer, answers))
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::{col, lit};
    use datafusion::prelude::{character_length, left, starts_with};

    use super::*;

    fn at(expr: &Expr, answers: &Completions) -> Decision {
        decide_expr(expr, |found| *found == col("a"), answers)
    }

    fn text(prefix: &str) -> Completions {
        Completions::text(prefix)
    }

    #[test]
    fn a_prefix_test_is_decided_once_the_prefix_settles_it() {
        let test = col("a").like(lit("Mar%"));
        assert_eq!(at(&test, &text("Jan")), Decision::NotTrue);
        assert_eq!(at(&test, &text("Mar")), Decision::True);
        assert_eq!(at(&test, &text("March 3")), Decision::True);
        assert_eq!(at(&test, &text("Ma")), Decision::Undecided);
        assert_eq!(at(&test, &text("")), Decision::Undecided);
        assert_eq!(at(&!test.clone(), &text("Jan")), Decision::True);
        let same = starts_with(col("a"), lit("Mar"));
        assert_eq!(at(&same, &text("Jan")), Decision::NotTrue);
    }

    #[test]
    fn a_left_cut_equal_to_a_literal_is_a_prefix_test() {
        let test = left(col("a"), lit(3_i64)).eq(lit("yes"));
        assert_eq!(at(&test, &text("no,")), Decision::NotTrue);
        assert_eq!(at(&test, &text("yes, because")), Decision::True);
        assert_eq!(at(&test, &text("y")), Decision::Undecided);
        let shorter = left(col("a"), lit(5_i64)).eq(lit("yes"));
        assert_eq!(at(&shorter, &text("yes")), Decision::Undecided);
        assert_eq!(at(&shorter, &text("yes!")), Decision::NotTrue);
    }

    #[test]
    fn an_equality_is_decided_not_true_once_the_prefix_leaves_the_literal() {
        let test = col("a").eq(lit("yes"));
        assert_eq!(at(&test, &text("no")), Decision::NotTrue);
        assert_eq!(at(&test, &text("ye")), Decision::Undecided);
        // the answer may still go on past the literal
        assert_eq!(at(&test, &text("yes")), Decision::Undecided);
        let complete = Completions {
            complete: true,
            ..text("yes")
        };
        assert_eq!(at(&test, &complete), Decision::True);
        let listed = col("a").in_list(vec![lit("yes"), lit("no")], false);
        assert_eq!(at(&listed, &text("maybe")), Decision::NotTrue);
        assert_eq!(at(&listed, &text("n")), Decision::Undecided);
    }

    #[test]
    fn a_null_test_is_decided_by_whether_the_answer_may_be_null() {
        let found = Completions {
            may_be_null: true,
            ..text("")
        };
        assert_eq!(at(&col("a").is_not_null(), &found), Decision::Undecided);
        assert_eq!(at(&col("a").is_not_null(), &text("x")), Decision::True);
        assert_eq!(at(&col("a").is_null(), &text("x")), Decision::NotTrue);
        // a prefix test of a NULL answer is NULL, so not TRUE
        let test = col("a").like(lit("Mar%"));
        let null_or_mar = Completions {
            may_be_null: true,
            ..text("Mar")
        };
        assert_eq!(at(&test, &null_or_mar), Decision::Undecided);
    }

    #[test]
    fn a_length_bound_is_decided_with_the_answer_limit() {
        let short = character_length(col("a")).lt(lit(5_i64));
        assert_eq!(at(&short, &text("abcde")), Decision::NotTrue);
        assert_eq!(at(&short, &text("ab")), Decision::Undecided);
        let limited = Completions {
            max_chars: Some(2),
            ..text("ab")
        };
        assert_eq!(at(&short, &limited), Decision::True);
        let long = character_length(col("a")).gt_eq(lit(10_i64));
        assert_eq!(at(&long, &limited), Decision::NotTrue);
        assert_eq!(at(&long, &text("ab")), Decision::Undecided);
    }

    #[test]
    fn a_copied_span_goes_on_only_with_the_document_text() {
        let span = Completions {
            tails: Some(vec!["ch 3, 2020".into(), "y 2021".into()]),
            ..text("Mar")
        };
        let test = col("a").like(lit("March%"));
        assert_eq!(at(&test, &span), Decision::Undecided);
        let other = col("a").like(lit("Marx%"));
        assert_eq!(at(&other, &span), Decision::NotTrue);
        let date = col("a").eq(lit("May 2021"));
        assert_eq!(at(&date, &span), Decision::NotTrue);
        let year = col("a").eq(lit("Mary 2021"));
        assert_eq!(at(&year, &span), Decision::Undecided);
    }

    #[test]
    fn a_combination_is_decided_from_its_parts() {
        let test = col("a")
            .like(lit("Mar%"))
            .or(col("a").like(lit("Apr%")))
            .and(col("a").is_not_null());
        assert_eq!(at(&test, &text("Jan")), Decision::NotTrue);
        assert_eq!(at(&test, &text("Apr 1")), Decision::True);
        // a part that reads another column may be anything
        let mixed = col("a").like(lit("Mar%")).and(col("b"));
        assert_eq!(at(&mixed, &text("Mar")), Decision::Undecided);
        assert_eq!(at(&mixed, &text("Jan")), Decision::NotTrue);
        let other = col("a").like(lit("M_r%"));
        assert_eq!(at(&other, &text("Jan")), Decision::Undecided);
    }
}
