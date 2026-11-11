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
//! Any other part may give any value, and parts are combined one at a
//! time, ignoring that two parts read the same answer. So a condition
//! these shapes leave undecided is then evaluated by DataFusion on each
//! complete answer, when `C(w)` is a short list: a finished answer, or
//! an extraction with few places left to copy from. That decides any
//! expression of the answer alone, such as `a LIKE '%x%'` or
//! `UPPER(a) = 'YES'`.

use std::sync::{Arc, OnceLock};

use datafusion::arrow::array::{ArrayRef, AsArray, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::pruning::PruningStatistics;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, DFSchema, ScalarValue};
use datafusion::logical_expr::expr::{InList, Like, ScalarFunction};
use datafusion::logical_expr::simplify::SimplifyContext;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::optimizer::simplify_expressions::ExprSimplifier;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::pruning::PruningPredicateBuilder;
use datafusion::prelude::SessionContext;
use quail_planner::error::CompileError;
use quail_planner::physical::Expression;

/// The most complete answers DataFusion evaluates a condition on.
const MAX_LISTED: usize = 1024;

/// The column that holds each listed answer.
pub(crate) const LISTED: &str = "answer";

/// The schema of a batch of answers: the one column [`LISTED`].
fn answer_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(LISTED, DataType::Utf8, true)]))
}

/// The session whose functions and type rules plan a test of an answer.
fn context() -> &'static SessionContext {
    static CONTEXT: OnceLock<SessionContext> = OnceLock::new();
    CONTEXT.get_or_init(SessionContext::new)
}

/// The least string greater than every string that starts with `text`:
/// `text` with its last character that has a successor raised by one,
/// and nothing after it. None when no character has one.
fn after(text: &str) -> Option<String> {
    let mut characters: Vec<char> = text.chars().collect();
    while let Some(last) = characters.pop() {
        let next = (u32::from(last) + 1..=u32::from(char::MAX)).find_map(char::from_u32);
        if let Some(next) = next {
            characters.push(next);
            return Some(characters.into_iter().collect());
        }
    }
    None
}

/// The answers that start with a text, as the range of strings
/// `PruningPredicate` reads: one container of rows.
struct Range {
    least: String,
    /// None when no string bounds the answers from above.
    greatest: Option<String>,
}

impl PruningStatistics for Range {
    fn min_values(&self, _column: &Column) -> Option<ArrayRef> {
        Some(Arc::new(StringArray::from(vec![Some(self.least.clone())])))
    }

    fn max_values(&self, _column: &Column) -> Option<ArrayRef> {
        Some(Arc::new(StringArray::from(vec![self.greatest.clone()])))
    }

    fn num_containers(&self) -> usize {
        1
    }

    fn null_counts(&self, _column: &Column) -> Option<ArrayRef> {
        None
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        None
    }

    fn contained(
        &self,
        _column: &Column,
        _values: &std::collections::HashSet<ScalarValue>,
    ) -> Option<datafusion::arrow::array::BooleanArray> {
        None
    }
}

/// Whether `test`, which reads the answer as its one column, column 0
/// named [`LISTED`] of type `data_type`, can be TRUE for a complete
/// answer still possible; None when DataFusion does not decide it.
///
/// A short list of answers is evaluated one by one. An answer that may
/// go on with any text is the range of strings that start with the text
/// so far, which `PruningPredicate` reads.
pub fn can_be_true(
    test: &Arc<dyn PhysicalExpr>,
    data_type: &DataType,
    answers: &Completions,
) -> Option<bool> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        LISTED,
        data_type.clone(),
        true,
    )]));
    if let Some(listed) = answers.listed() {
        let mut found: Vec<Option<String>> = listed.into_iter().map(Some).collect();
        if answers.may_be_null {
            found.push(None);
        }
        let column: ArrayRef = Arc::new(StringArray::from(found));
        let column = datafusion::arrow::compute::cast(&column, data_type).ok()?;
        let batch = RecordBatch::try_new(schema, vec![column]).ok()?;
        let values = test
            .evaluate(&batch)
            .ok()?
            .into_array(batch.num_rows())
            .ok()?;
        return Some(
            values
                .as_boolean_opt()?
                .iter()
                .any(|value| value == Some(true)),
        );
    }
    if !matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    ) {
        return None;
    }
    let range = Range {
        least: answers.prefix.clone(),
        greatest: after(&answers.prefix),
    };
    let predicate = PruningPredicateBuilder::new()
        .with_file_schema(schema)
        .try_build(Arc::clone(test))
        .ok()?;
    predicate.prune(&range).ok()?.first().copied()
}

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

    /// Every complete answer still possible, when they are few: the text
    /// so far, and the text with each cut of each tail. None when the
    /// answer may go on with any text, or there are more than
    /// [`MAX_LISTED`]. An extraction strips the whitespace around its
    /// span, so the list can hold answers the model cannot give, which
    /// only delays a decision.
    fn listed(&self) -> Option<Vec<String>> {
        if self.complete {
            return Some(vec![self.prefix.clone()]);
        }
        let mut found = std::collections::BTreeSet::from([self.prefix.clone()]);
        for tail in self.tails.as_ref()? {
            for (added, (index, character)) in tail.char_indices().enumerate() {
                if self.max_chars.is_some_and(|max| added >= max) {
                    break;
                }
                let cut = &tail[..index + character.len_utf8()];
                found.insert(format!("{}{cut}", self.prefix));
                if found.len() > MAX_LISTED {
                    return None;
                }
            }
        }
        Some(found.into_iter().collect())
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

    /// Whether an answer holds `literal` somewhere: decided TRUE once the
    /// text so far holds it, whatever follows.
    fn contains(&self, literal: &str) -> Possible {
        if self.prefix.contains(literal) {
            return self.text_test(true, false);
        }
        // the text so far is an answer that does not hold it
        self.text_test(self.can_grow(), true)
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
                let inner = pattern
                    .strip_prefix('%')
                    .and_then(|rest| rest.strip_suffix('%'))
                    .filter(|inner| !inner.is_empty() && !inner.contains(['%', '_']));
                let found = match (inner, pattern.find(['%', '_'])) {
                    (Some(inner), _) => answers.contains(inner),
                    (None, None) => answers.equals(&pattern),
                    (None, Some(at)) if at == pattern.len() - 1 && pattern.ends_with('%') => {
                        answers.starts_with(&pattern[..at])
                    }
                    _ => return ANY,
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

    /// `expr` reading the answer as the column [`LISTED`]; None when it
    /// reads any other column.
    fn over_answer(&self, expr: &Expr) -> Option<Expr> {
        let mut other = false;
        let read = expr
            .clone()
            .transform_up(|found| match &found {
                Expr::Column(_) if self.answer(&found) => {
                    Ok(Transformed::yes(Expr::Column(Column::from_name(LISTED))))
                }
                Expr::Column(_) => {
                    other = true;
                    Ok(Transformed::no(found))
                }
                _ => Ok(Transformed::no(found)),
            })
            .ok()?
            .data;
        (!other).then_some(read)
    }

    /// Which values `expr` can give an answer that starts with the text
    /// so far, as DataFusion's `PruningPredicate` decides from the
    /// least and greatest such texts: whether `expr` can be TRUE, and
    /// whether its negation can. None when the answer is complete,
    /// `expr` reads a column that is not the answer, or DataFusion does
    /// not plan it.
    ///
    /// Every answer that continues the text sorts from the text itself
    /// up to the text with its last character raised by one, so a test a
    /// range of strings cannot pass, such as `a = 'x'` or `a < 'x'`, is
    /// decided for a map's open-ended answer. A shape the predicate does
    /// not read counts as possible.
    fn ranged(&self, expr: &Expr) -> Option<Possible> {
        if self.answers.complete {
            return None;
        }
        let read = self.over_answer(expr)?;
        let schema = answer_schema();
        let fields = DFSchema::try_from(Arc::clone(&schema)).ok()?;
        let range = Range {
            least: self.answers.prefix.clone(),
            greatest: after(&self.answers.prefix),
        };
        let possible = |expr: Expr| -> Option<bool> {
            let physical = context().create_physical_expr(expr, &fields).ok()?;
            let predicate = PruningPredicateBuilder::new()
                .with_file_schema(Arc::clone(&schema))
                .try_build(physical)
                .ok()?;
            predicate.prune(&range).ok()?.first().copied()
        };
        let simplify = SimplifyContext::builder()
            .with_schema(Arc::new(fields.clone()))
            .build();
        // the simplifier moves a NOT into the comparison it negates
        let negated = ExprSimplifier::new(simplify)
            .simplify(Expr::Not(Box::new(read.clone())))
            .ok()?;
        Some(Possible {
            t: possible(read)?,
            f: possible(negated)?,
            n: self.answers.may_be_null,
        })
    }

    /// Which values `expr` gives the complete answers still possible,
    /// each one evaluated by DataFusion; None when the answers are too
    /// many to list ([`Completions::listed`]), `expr` reads a column that
    /// is not the answer, or DataFusion does not evaluate it.
    fn evaluated(&self, expr: &Expr) -> Option<Possible> {
        let mut answers: Vec<Option<String>> =
            self.answers.listed()?.into_iter().map(Some).collect();
        if self.answers.may_be_null {
            answers.push(None);
        }
        let read = self.over_answer(expr)?;
        let schema = answer_schema();
        let column: ArrayRef = Arc::new(StringArray::from(answers));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).ok()?;
        let physical = context()
            .create_physical_expr(read, &DFSchema::try_from(schema).ok()?)
            .ok()?;
        let values = physical
            .evaluate(&batch)
            .ok()?
            .into_array(batch.num_rows())
            .ok()?;
        let values = values.as_boolean_opt()?;
        let mut found = Possible {
            t: false,
            f: false,
            n: false,
        };
        for value in values {
            match value {
                Some(true) => found.t = true,
                Some(false) => found.f = true,
                None => found.n = true,
            }
        }
        Some(found)
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

/// The texts of `answers` in one letter case, `upper` or lower; None when
/// a text is not ASCII, where a letter's case can depend on the letters
/// after it.
pub(crate) fn in_case(answers: &Completions, upper: bool) -> Option<Completions> {
    let ascii =
        answers.prefix.is_ascii() && answers.tails.iter().flatten().all(|tail| tail.is_ascii());
    let fold = |text: &String| {
        if upper {
            text.to_ascii_uppercase()
        } else {
            text.to_ascii_lowercase()
        }
    };
    ascii.then(|| Completions {
        prefix: fold(&answers.prefix),
        tails: answers
            .tails
            .as_ref()
            .map(|tails| tails.iter().map(fold).collect()),
        ..answers.clone()
    })
}

/// A test that reads the answer only in one letter case, as the same
/// test of the answer's texts in that case: `lower(a) LIKE 'neg%'` at
/// the answers that start with `Ne` is `a LIKE 'neg%'` at those that
/// start with `ne`. An ASCII `ILIKE` reads the answer in lower case.
/// None when the test reads the answer as written, or in both cases.
fn case_folded(
    expr: &Expr,
    is_answer: &impl Fn(&Expr) -> bool,
    answers: &Completions,
) -> Option<(Expr, Completions)> {
    use datafusion::functions::string::expr_fn::lower;
    let insensitive = expr
        .clone()
        .transform_up(|found| match found {
            Expr::Like(like) if like.case_insensitive => {
                let pattern = match like.pattern.as_ref() {
                    Expr::Literal(value, _) => value.try_as_str().flatten(),
                    _ => None,
                };
                match pattern.filter(|pattern| pattern.is_ascii()) {
                    Some(pattern) => Ok(Transformed::yes(Expr::Like(Like::new(
                        like.negated,
                        Box::new(lower(*like.expr)),
                        Box::new(datafusion::logical_expr::lit(pattern.to_ascii_lowercase())),
                        like.escape_char,
                        false,
                    )))),
                    None => Ok(Transformed::no(Expr::Like(like))),
                }
            }
            other => Ok(Transformed::no(other)),
        })
        .ok()?
        .data;
    let (mut lowered, mut raised, mut as_written) = (false, false, false);
    let read = insensitive
        .transform_down(|found| {
            if let Expr::ScalarFunction(call) = &found
                && let [argument] = call.args.as_slice()
                && is_answer(argument)
                && matches!(call.name(), "lower" | "upper")
            {
                if call.name() == "lower" {
                    lowered = true;
                } else {
                    raised = true;
                }
                return Ok(Transformed::new(
                    argument.clone(),
                    true,
                    datafusion::common::tree_node::TreeNodeRecursion::Jump,
                ));
            }
            as_written |= is_answer(&found);
            Ok(Transformed::no(found))
        })
        .ok()?
        .data;
    if as_written || lowered == raised {
        return None;
    }
    Some((read, in_case(answers, raised)?))
}

/// `expr` at the answers in `answers`, with `is_answer` telling which
/// expression reads the answer.
pub fn decide_expr(
    expr: &Expr,
    is_answer: impl Fn(&Expr) -> bool,
    answers: &Completions,
) -> Decision {
    let folded = case_folded(expr, &is_answer, answers);
    let (expr, answers) = match &folded {
        Some((expr, answers)) => (expr, answers),
        None => (expr, answers),
    };
    let reader = Reader { is_answer, answers };
    let mut found = reader.possible(expr);
    // the shapes above ignore that two parts read one answer, and read
    // any other shape as any value: a short list of answers settles both
    if found.t && (found.f || found.n) {
        if let Some(listed) = reader.evaluated(expr) {
            found = listed;
        } else if let Some(ranged) = reader.ranged(expr) {
            // both readings hold, so a value either rules out is out
            found = Possible {
                t: found.t && ranged.t,
                f: found.f && ranged.f,
                n: found.n && ranged.n,
            };
        }
    }
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
    fn a_test_of_an_answer_in_one_letter_case_is_decided_on_its_text_so_far() {
        use datafusion::functions::string::expr_fn::lower;
        let starts = col("a").ilike(lit("neg%"));
        assert_eq!(at(&starts, &text("Po")), Decision::NotTrue);
        assert_eq!(at(&starts, &text("NEG")), Decision::True);
        assert_eq!(at(&starts, &text("N")), Decision::Undecided);
        let holds = col("a").ilike(lit("%refund%"));
        assert_eq!(at(&holds, &text("A REFUND was")), Decision::True);
        assert_eq!(at(&holds, &text("The cust")), Decision::Undecided);
        let equals = lower(col("a")).eq(lit("acme"));
        assert_eq!(at(&equals, &text("B")), Decision::NotTrue);
        assert_eq!(at(&equals, &text("AC")), Decision::Undecided);
        // a letter outside ASCII may change case with what follows it
        assert_eq!(at(&starts, &text("Σ")), Decision::Undecided);
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
    fn datafusion_decides_any_test_of_a_short_list_of_answers() {
        use datafusion::prelude::upper;
        // a finished answer is one answer
        let done = Completions {
            complete: true,
            ..text("yes, kind")
        };
        let loud = upper(col("a")).eq(lit("YES, KIND"));
        assert_eq!(at(&loud, &done), Decision::True);
        assert_eq!(at(&loud, &text("yes")), Decision::Undecided);
        // a span that can only go on as "2027." never holds "2026"
        let copied = Completions {
            tails: Some(vec!["27.".into()]),
            ..text("20")
        };
        let holds = |year: &str| col("a").like(lit(format!("%{year}%")));
        assert_eq!(at(&holds("2026"), &copied), Decision::NotTrue);
        assert_eq!(at(&holds("2027"), &copied), Decision::Undecided);
        assert_eq!(at(&holds("20"), &copied), Decision::True);
        // an answer that may be NULL is not certainly TRUE
        let nullable = Completions {
            may_be_null: true,
            ..copied.clone()
        };
        assert_eq!(at(&holds("20"), &nullable), Decision::Undecided);
        // two parts that read the one answer are decided together
        let either = col("a").like(lit("%7%")).or(col("a").not_like(lit("%7%")));
        assert_eq!(at(&either, &copied), Decision::True);
    }

    #[test]
    fn a_range_of_strings_decides_a_comparison_of_open_ended_text() {
        // every answer that starts with "N" sorts after "Mz" and before "O"
        let before = col("a").lt(lit("Mz"));
        assert_eq!(at(&before, &text("N")), Decision::NotTrue);
        assert_eq!(at(&col("a").gt(lit("Mz")), &text("N")), Decision::True);
        assert_eq!(at(&before, &text("M")), Decision::Undecided);
        // a NOT is read as the comparison it negates
        assert_eq!(at(&!before.clone(), &text("N")), Decision::True);
        // an answer that may be NULL is not certainly TRUE
        let nullable = Completions {
            may_be_null: true,
            ..text("N")
        };
        assert_eq!(at(&col("a").gt(lit("Mz")), &nullable), Decision::Undecided);
        assert_eq!(after("ab"), Some("ac".into()));
        assert_eq!(after(""), None);
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
        // a pattern with a wildcard inside is read by its literal start
        let other = col("a").like(lit("M_r%"));
        assert_eq!(at(&other, &text("Jan")), Decision::NotTrue);
        assert_eq!(at(&other, &text("Mar")), Decision::Undecided);
    }
}
