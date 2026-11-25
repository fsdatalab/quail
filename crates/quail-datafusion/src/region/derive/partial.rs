use quail_planner::constants::MAX_LISTED;
use std::sync::{Arc, OnceLock};

use datafusion::arrow::array::{ArrayRef, AsArray, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, DFSchema, ScalarValue};
use datafusion::logical_expr::expr::{InList, Like, ScalarFunction};
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::prelude::SessionContext;
use quail_planner::error::CompileError;
use quail_planner::physical::Expression;

pub(crate) const LISTED: &str = "answer";

fn answer_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(LISTED, DataType::Utf8, true)]))
}

fn context() -> &'static SessionContext {
    static CONTEXT: OnceLock<SessionContext> = OnceLock::new();
    CONTEXT.get_or_init(SessionContext::new)
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    True,
    NotTrue,
    Undecided,
}

#[derive(Debug, Clone)]
pub struct Completions {
    pub prefix: String,
    pub complete: bool,
    pub may_be_null: bool,
    pub tails: Option<Vec<String>>,
    pub max_chars: Option<usize>,
}

impl Completions {
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

    fn can_grow(&self) -> bool {
        if self.complete || self.max_chars == Some(0) {
            return false;
        }
        match &self.tails {
            None => true,
            Some(tails) => tails.iter().any(|tail| !tail.is_empty()),
        }
    }

    fn listed(&self) -> Option<Vec<String>> {
        if !self.can_grow() {
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
        self.text_test(can_true, true)
    }

    fn contains(&self, literal: &str) -> Possible {
        if self.prefix.contains(literal) {
            return self.text_test(true, false);
        }
        self.text_test(self.can_grow(), true)
    }

    fn ordered(&self, op: Operator, literal: &str) -> Possible {
        let least = self.prefix.as_str();
        let reaches = |strictly: bool| {
            if !self.can_grow() {
                return if strictly {
                    least > literal
                } else {
                    least >= literal
                };
            }
            after(least).is_none_or(|greatest| greatest.as_str() > literal)
        };
        let (can_true, can_false) = match op {
            Operator::Lt => (least < literal, reaches(false)),
            Operator::LtEq => (least <= literal, reaches(true)),
            Operator::Gt => (reaches(true), least <= literal),
            _ => (reaches(false), least < literal),
        };
        self.text_test(can_true, can_false)
    }

    fn in_case(&self, upper: bool) -> Option<Self> {
        let ascii =
            self.prefix.is_ascii() && self.tails.iter().flatten().all(|tail| tail.is_ascii());
        let fold = |text: &String| {
            if upper {
                text.to_ascii_uppercase()
            } else {
                text.to_ascii_lowercase()
            }
        };
        ascii.then(|| Self {
            prefix: fold(&self.prefix),
            tails: self
                .tails
                .as_ref()
                .map(|tails| tails.iter().map(fold).collect()),
            max_chars: self
                .max_chars
                .filter(|_| self.tails.is_some() || !self.can_grow()),
            ..self.clone()
        })
    }

    fn cut(&self, count: i64) -> Self {
        let count = usize::try_from(count).unwrap_or(0);
        let held = Self::chars(&self.prefix);
        let first = |text: &str, count: usize| text.chars().take(count).collect::<String>();
        if count <= held {
            return Self {
                prefix: first(&self.prefix, count),
                complete: true,
                may_be_null: self.may_be_null,
                tails: None,
                max_chars: None,
            };
        }
        let room = count - held;
        Self {
            tails: self
                .tails
                .as_ref()
                .map(|tails| tails.iter().map(|tail| first(tail, room)).collect()),
            max_chars: Some(self.max_chars.map_or(room, |max| max.min(room))),
            ..self.clone()
        }
    }

    fn equals(&self, literal: &str) -> Possible {
        let can_true = literal
            .strip_prefix(self.prefix.as_str())
            .is_some_and(|rest| self.can_add(rest));
        let can_false = self.prefix != literal || self.can_grow();
        self.text_test(can_true, can_false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Possible {
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

#[derive(Debug, Clone)]
pub enum Partial {
    Known(ScalarValue),
    Text(Completions),
    Count {
        low: i64,
        high: Option<i64>,
        may_be_null: bool,
    },
    Truth(Possible),
    Any,
}

impl Partial {
    fn text(&self) -> Option<String> {
        match self {
            Partial::Known(value) => value.try_as_str().flatten().map(str::to_string),
            _ => None,
        }
    }

    fn integer(&self) -> Option<i64> {
        match self {
            Partial::Known(value) if value.data_type().is_integer() => {
                match value.cast_to(&DataType::Int64) {
                    Ok(ScalarValue::Int64(found)) => found,
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn cast(self, data_type: &DataType) -> Partial {
        let string = matches!(
            data_type,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        );
        match self {
            Partial::Known(value) => value
                .cast_to(data_type)
                .map_or(Partial::Any, Partial::Known),
            Partial::Text(_) if string => self,
            Partial::Count { .. } if data_type.is_integer() => self,
            _ => Partial::Any,
        }
    }

    fn truth(&self) -> Possible {
        match self {
            Partial::Truth(found) => *found,
            Partial::Known(ScalarValue::Boolean(value)) => Possible::constant(*value),
            _ => ANY,
        }
    }

    fn is_null(&self) -> Option<Possible> {
        let may_be_null = match self {
            Partial::Known(value) => return Some(Possible::constant(Some(value.is_null()))),
            Partial::Text(found) => found.may_be_null,
            Partial::Count { may_be_null, .. } => *may_be_null,
            Partial::Truth(found) => {
                return Some(Possible {
                    t: found.n,
                    f: found.t || found.f,
                    n: false,
                });
            }
            Partial::Any => return None,
        };
        Some(Possible {
            t: may_be_null,
            f: true,
            n: false,
        })
    }
}

pub trait PartialEval: Sync {
    fn partial_eval(&self, args: &[Partial]) -> Partial;
}

struct LetterCase {
    upper: bool,
}

impl PartialEval for LetterCase {
    fn partial_eval(&self, args: &[Partial]) -> Partial {
        match args {
            [Partial::Text(found)] => found
                .in_case(self.upper)
                .map_or(Partial::Any, Partial::Text),
            _ => Partial::Any,
        }
    }
}

fn first_characters(found: &Completions, count: &Partial) -> Partial {
    match count.integer() {
        Some(count) if count >= 0 => Partial::Text(found.cut(count)),
        _ => Partial::Any,
    }
}

struct Left;

impl PartialEval for Left {
    fn partial_eval(&self, args: &[Partial]) -> Partial {
        match args {
            [Partial::Text(found), count] => first_characters(found, count),
            _ => Partial::Any,
        }
    }
}

struct Substr;

impl PartialEval for Substr {
    fn partial_eval(&self, args: &[Partial]) -> Partial {
        match args {
            [Partial::Text(found), start, count] if start.integer() == Some(1) => {
                first_characters(found, count)
            }
            _ => Partial::Any,
        }
    }
}

struct Length;

impl PartialEval for Length {
    fn partial_eval(&self, args: &[Partial]) -> Partial {
        match args {
            [Partial::Text(found)] => {
                let (low, high) = found.lengths();
                Partial::Count {
                    low: low as i64,
                    high: high.map(|high| high as i64),
                    may_be_null: found.may_be_null,
                }
            }
            _ => Partial::Any,
        }
    }
}

struct StartsWith;

impl PartialEval for StartsWith {
    fn partial_eval(&self, args: &[Partial]) -> Partial {
        match args {
            [Partial::Text(found), literal] => match literal.text() {
                Some(literal) => Partial::Truth(found.starts_with(&literal)),
                None => Partial::Any,
            },
            _ => Partial::Any,
        }
    }
}

fn partial_function(name: &str) -> Option<&'static dyn PartialEval> {
    Some(match name {
        "lower" => &LetterCase { upper: false },
        "upper" => &LetterCase { upper: true },
        "left" => &Left,
        "substr" | "substring" => &Substr,
        "length" | "character_length" | "char_length" => &Length,
        "starts_with" => &StartsWith,
        _ => return None,
    })
}

fn count_test(low: i64, high: Option<i64>, op: Operator, bound: i64) -> Option<(bool, bool)> {
    let holds = |length: i64| match op {
        Operator::Eq => length == bound,
        Operator::NotEq => length != bound,
        Operator::Lt => length < bound,
        Operator::LtEq => length <= bound,
        Operator::Gt => length > bound,
        _ => length >= bound,
    };
    if !matches!(
        op,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
    ) {
        return None;
    }
    Some(match high {
        Some(high) if high - low <= 4096 => {
            let any = |want: bool| (low..=high).any(|length| holds(length) == want);
            (any(true), any(false))
        }
        Some(_) => (true, true),
        None => match op {
            Operator::Eq => (bound >= low, true),
            Operator::NotEq => (true, bound >= low),
            Operator::Lt | Operator::LtEq => (holds(low), true),
            _ => (true, !holds(low)),
        },
    })
}

fn compare(left: &Partial, op: Operator, right: &Partial) -> Possible {
    match (left, right) {
        (Partial::Text(found), literal) => match (literal.text(), op) {
            (Some(literal), Operator::Eq) => found.equals(&literal),
            (Some(literal), Operator::NotEq) => found.equals(&literal).not(),
            (Some(literal), Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq) => {
                found.ordered(op, &literal)
            }
            _ => ANY,
        },
        (
            Partial::Count {
                low,
                high,
                may_be_null,
            },
            bound,
        ) => match bound
            .integer()
            .and_then(|bound| count_test(*low, *high, op, bound))
        {
            Some((t, f)) => Possible {
                t,
                f,
                n: *may_be_null,
            },
            None => ANY,
        },
        (Partial::Known(_), Partial::Text(_) | Partial::Count { .. }) => {
            compare(right, op.swap().unwrap_or(op), left)
        }
        _ => ANY,
    }
}

fn like(subject: &Partial, pattern: &Partial, case_insensitive: bool) -> Possible {
    let (Partial::Text(found), Some(pattern)) = (subject, pattern.text()) else {
        return ANY;
    };
    if pattern.contains('\\') {
        return ANY;
    }
    let (found, pattern) = if case_insensitive {
        match (found.in_case(false), pattern.is_ascii()) {
            (Some(found), true) => (found, pattern.to_ascii_lowercase()),
            _ => return ANY,
        }
    } else {
        (found.clone(), pattern)
    };
    let inner = pattern
        .strip_prefix('%')
        .and_then(|rest| rest.strip_suffix('%'))
        .filter(|inner| !inner.is_empty() && !inner.contains(['%', '_']));
    match (inner, pattern.find(['%', '_'])) {
        (Some(inner), _) => found.contains(inner),
        (None, None) => found.equals(&pattern),
        (None, Some(at)) if at == pattern.len() - 1 && pattern.ends_with('%') => {
            found.starts_with(&pattern[..at])
        }
        (None, Some(at)) => Possible {
            f: true,
            ..found.starts_with(&pattern[..at])
        },
    }
}

struct Reader<'a, F: Fn(&Expr) -> bool> {
    is_answer: F,
    answers: &'a Completions,
}

impl<F: Fn(&Expr) -> bool> Reader<'_, F> {
    fn value(&self, expr: &Expr) -> Partial {
        let truth = |expr: &Expr| self.value(expr).truth();
        match expr {
            Expr::Alias(alias) => self.value(&alias.expr),
            Expr::Cast(cast) => self.value(&cast.expr).cast(cast.field.data_type()),
            Expr::TryCast(cast) => self.value(&cast.expr).cast(cast.field.data_type()),
            Expr::Literal(value, _) => Partial::Known(value.clone()),
            Expr::Column(_) if (self.is_answer)(expr) => Partial::Text(self.answers.clone()),
            Expr::ScalarFunction(ScalarFunction { func, args }) => {
                match partial_function(func.name()) {
                    Some(function) => {
                        let args: Vec<Partial> = args.iter().map(|arg| self.value(arg)).collect();
                        function.partial_eval(&args)
                    }
                    None => Partial::Any,
                }
            }
            Expr::Not(inner) => Partial::Truth(truth(inner).not()),
            Expr::IsTrue(inner)
            | Expr::IsNotFalse(inner)
            | Expr::IsFalse(inner)
            | Expr::IsNotTrue(inner) => {
                let found = truth(inner);
                let (yes, no) = match expr {
                    Expr::IsTrue(_) => (found.t, found.f || found.n),
                    Expr::IsNotTrue(_) => (found.f || found.n, found.t),
                    Expr::IsFalse(_) => (found.f, found.t || found.n),
                    _ => (found.t || found.n, found.f),
                };
                Partial::Truth(Possible {
                    t: yes,
                    f: no,
                    n: false,
                })
            }
            Expr::IsNull(inner) => Partial::Truth(self.value(inner).is_null().unwrap_or(ANY)),
            Expr::IsNotNull(inner) => {
                Partial::Truth(self.value(inner).is_null().map_or(ANY, Possible::not))
            }
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => Partial::Truth(match op {
                Operator::And => truth(left).and(truth(right)),
                Operator::Or => truth(left).or(truth(right)),
                _ => compare(&self.value(left), *op, &self.value(right)),
            }),
            Expr::Like(Like {
                negated,
                expr: subject,
                pattern,
                escape_char: None,
                case_insensitive,
            }) => {
                let found = like(
                    &self.value(subject),
                    &self.value(pattern),
                    *case_insensitive,
                );
                Partial::Truth(if *negated { found.not() } else { found })
            }
            Expr::InList(InList {
                expr: subject,
                list,
                negated,
            }) => {
                let subject = self.value(subject);
                let mut found = Possible::constant(Some(false));
                for item in list {
                    found = found.or(compare(&subject, Operator::Eq, &self.value(item)));
                }
                Partial::Truth(if *negated { found.not() } else { found })
            }
            _ => Partial::Any,
        }
    }

    fn over_answer(&self, expr: &Expr) -> Option<Expr> {
        let mut other = false;
        let read = expr
            .clone()
            .transform_up(|found| match &found {
                Expr::Column(_) if (self.is_answer)(&found) => {
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
}

pub fn decide_expr(
    expr: &Expr,
    is_answer: impl Fn(&Expr) -> bool,
    answers: &Completions,
) -> Decision {
    let reader = Reader { is_answer, answers };
    let found = reader
        .evaluated(expr)
        .unwrap_or_else(|| reader.value(expr).truth());
    if !found.t {
        Decision::NotTrue
    } else if !found.f && !found.n {
        Decision::True
    } else {
        Decision::Undecided
    }
}

pub fn decide(
    condition: &Expression,
    answer: &str,
    answers: &Completions,
) -> Result<Decision, CompileError> {
    let expr = crate::region::derive::expressions::decode(condition)?;
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
        let done = Completions {
            complete: true,
            ..text("yes, kind")
        };
        let loud = upper(col("a")).eq(lit("YES, KIND"));
        assert_eq!(at(&loud, &done), Decision::True);
        assert_eq!(at(&loud, &text("yes")), Decision::Undecided);
        let copied = Completions {
            tails: Some(vec!["27.".into()]),
            ..text("20")
        };
        let holds = |year: &str| col("a").like(lit(format!("%{year}%")));
        assert_eq!(at(&holds("2026"), &copied), Decision::NotTrue);
        assert_eq!(at(&holds("2027"), &copied), Decision::Undecided);
        assert_eq!(at(&holds("20"), &copied), Decision::True);
        let nullable = Completions {
            may_be_null: true,
            ..copied.clone()
        };
        assert_eq!(at(&holds("20"), &nullable), Decision::Undecided);
        let either = col("a").like(lit("%7%")).or(col("a").not_like(lit("%7%")));
        assert_eq!(at(&either, &copied), Decision::True);
    }

    #[test]
    fn a_range_of_strings_decides_a_comparison_of_open_ended_text() {
        let before = col("a").lt(lit("Mz"));
        assert_eq!(at(&before, &text("N")), Decision::NotTrue);
        assert_eq!(at(&col("a").gt(lit("Mz")), &text("N")), Decision::True);
        assert_eq!(at(&before, &text("M")), Decision::Undecided);
        assert_eq!(at(&!before.clone(), &text("N")), Decision::True);
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
        let mixed = col("a").like(lit("Mar%")).and(col("b"));
        assert_eq!(at(&mixed, &text("Mar")), Decision::Undecided);
        assert_eq!(at(&mixed, &text("Jan")), Decision::NotTrue);
        let other = col("a").like(lit("M_r%"));
        assert_eq!(at(&other, &text("Jan")), Decision::NotTrue);
        assert_eq!(at(&other, &text("Mar")), Decision::Undecided);
    }

    #[test]
    fn a_part_the_evaluator_cannot_read_exactly_is_left_undecided() {
        use datafusion::arrow::datatypes::DataType;
        use datafusion::functions::string::expr_fn::upper;
        use datafusion::functions::unicode::expr_fn::{left, substr};
        use datafusion::logical_expr::cast;
        let escaped = col("a").like(lit("a\\%"));
        assert_eq!(at(&escaped, &text("a%")), Decision::Undecided);
        let number = cast(col("a"), DataType::Int32).eq(lit(5));
        assert_eq!(at(&number, &text("0")), Decision::Undecided);
        let none = left(col("a"), lit(0)).eq(lit(""));
        assert_eq!(at(&none, &text("x")), Decision::True);
        let end = left(col("a"), lit(-1)).eq(lit("x"));
        assert_eq!(at(&end, &text("x")), Decision::Undecided);
        let rest = substr(col("a"), lit(2)).eq(lit(""));
        assert_eq!(at(&rest, &text("I")), Decision::Undecided);
        let half = character_length(col("a")).lt(lit(2.5));
        assert_eq!(at(&half, &text("ab")), Decision::Undecided);
        let sharp = upper(col("a")).eq(lit("SS"));
        let one = Completions {
            max_chars: Some(1),
            ..text("")
        };
        assert_eq!(at(&sharp, &one), Decision::Undecided);
    }
}
