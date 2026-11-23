//! Token, attention, and KV work, as `quail/common/work.py` counts it.
//!
//! Each count is a `Quantity`: one number, or one number per document.
//! `quail/common` passes floats and numpy arrays through the same
//! arithmetic. `Quantity` does the same with numpy's broadcasting, and
//! its sums round as numpy's and Python's do.

use crate::specs::ModelSpec;
use std::ops::{Add, Div, Mul, Sub};

/// Tokens, full and sliding attention pairs, and KV and state movement.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Work {
    pub tokens: Quantity,
    pub pairs: Quantity,
    pub kv_written: Quantity,
    pub kv_read: Quantity,
    pub sliding_pairs: Quantity,
    pub sliding_kv_read: Quantity,
    pub state_written: Quantity,
    pub state_read: Quantity,
    /// Tokens the model's SAE encodes, for `AI.SAE_FEATURES`.
    pub sae_encoded: Quantity,
    /// Answer rows the model's head reads out.
    pub readout_rows: Quantity,
    /// Head rows read, summed over the answer rows: an answer row that
    /// reads the whole vocabulary counts the vocabulary's size.
    pub readout_head_rows: Quantity,
}

impl Add for &Work {
    type Output = Work;
    fn add(self, other: &Work) -> Work {
        Work {
            tokens: &self.tokens + &other.tokens,
            pairs: &self.pairs + &other.pairs,
            kv_written: &self.kv_written + &other.kv_written,
            kv_read: &self.kv_read + &other.kv_read,
            sliding_pairs: &self.sliding_pairs + &other.sliding_pairs,
            sliding_kv_read: &self.sliding_kv_read + &other.sliding_kv_read,
            state_written: &self.state_written + &other.state_written,
            state_read: &self.state_read + &other.state_read,
            sae_encoded: &self.sae_encoded + &other.sae_encoded,
            readout_rows: &self.readout_rows + &other.readout_rows,
            readout_head_rows: &self.readout_head_rows + &other.readout_head_rows,
        }
    }
}

impl Mul<f64> for &Work {
    type Output = Work;
    fn mul(self, count: f64) -> Work {
        Work {
            tokens: &self.tokens * count,
            pairs: &self.pairs * count,
            kv_written: &self.kv_written * count,
            kv_read: &self.kv_read * count,
            sliding_pairs: &self.sliding_pairs * count,
            sliding_kv_read: &self.sliding_kv_read * count,
            state_written: &self.state_written * count,
            state_read: &self.state_read * count,
            sae_encoded: &self.sae_encoded * count,
            readout_rows: &self.readout_rows * count,
            readout_head_rows: &self.readout_head_rows * count,
        }
    }
}

impl Work {
    /// This work with `rows` answer rows read out, each reading
    /// `head_rows` rows of the model's head, or the whole vocabulary
    /// when `head_rows` is 0.
    pub fn with_readout(mut self, rows: f64, head_rows: i64, vocab: i64) -> Work {
        let head = if head_rows != 0 { head_rows } else { vocab };
        self.readout_rows = Quantity::Scalar(rows);
        self.readout_head_rows = Quantity::Scalar(rows * head as f64);
        self
    }

    /// The work's counts, each summed over documents, in field order.
    pub fn values(&self) -> Vec<f64> {
        [
            &self.tokens,
            &self.pairs,
            &self.kv_written,
            &self.kv_read,
            &self.sliding_pairs,
            &self.sliding_kv_read,
            &self.state_written,
            &self.state_read,
            &self.sae_encoded,
            &self.readout_rows,
            &self.readout_head_rows,
        ]
        .iter()
        .map(|value| value.total())
        .collect()
    }

    /// The work of counts in the order [`Work::values`] gives them; no
    /// work for an empty list.
    pub fn from_values(values: &[f64]) -> Work {
        let at = |index: usize| Quantity::Scalar(values.get(index).copied().unwrap_or(0.0));
        Work {
            tokens: at(0),
            pairs: at(1),
            kv_written: at(2),
            kv_read: at(3),
            sliding_pairs: at(4),
            sliding_kv_read: at(5),
            state_written: at(6),
            state_read: at(7),
            sae_encoded: at(8),
            readout_rows: at(9),
            readout_head_rows: at(10),
        }
    }

    /// Sum work over document arrays, leaving scalar counts unchanged.
    pub fn total(&self) -> Work {
        let sum = |value: &Quantity| Quantity::Scalar(value.total());
        Work {
            tokens: sum(&self.tokens),
            pairs: sum(&self.pairs),
            kv_written: sum(&self.kv_written),
            kv_read: sum(&self.kv_read),
            sliding_pairs: sum(&self.sliding_pairs),
            sliding_kv_read: sum(&self.sliding_kv_read),
            state_written: sum(&self.state_written),
            state_read: sum(&self.state_read),
            sae_encoded: sum(&self.sae_encoded),
            readout_rows: sum(&self.readout_rows),
            readout_head_rows: sum(&self.readout_head_rows),
        }
    }

    /// Whether every count is no larger than the other record's.
    ///
    /// # Panics
    ///
    /// A count is an array; Python refuses to compare those too.
    pub fn dominates(&self, other: &Work) -> bool {
        let fields = |work: &Work| {
            [
                work.tokens.clone(),
                work.pairs.clone(),
                work.kv_written.clone(),
                work.kv_read.clone(),
                work.sliding_pairs.clone(),
                work.sliding_kv_read.clone(),
                work.state_written.clone(),
                work.state_read.clone(),
                work.sae_encoded.clone(),
                work.readout_rows.clone(),
                work.readout_head_rows.clone(),
            ]
        };
        fields(self)
            .iter()
            .zip(fields(other).iter())
            .all(|(mine, theirs)| {
                mine.scalar().expect("scalar work") <= theirs.scalar().expect("scalar work")
            })
    }
}

/// One per document of `shape`: `1.0 + 0.0 * np.asarray(shape)`.
fn ones(value: f64, shape: &Quantity) -> Quantity {
    value + &(0.0 * shape)
}

/// Causal attention pairs, each token's own key included.
pub fn triangle(n: &Quantity, window: i64) -> Quantity {
    if window != 0 {
        let visible = n.minimum(&Quantity::Scalar(window as f64));
        &(&visible * n) - &(&(&visible * &(&visible - 1.0)) / 2.0)
    } else {
        &(n * &(n + 1.0)) / 2.0
    }
}

/// Compute one document and its first suffix from nothing.
pub fn scan(prefix: &Quantity, suffix: &Quantity, window: i64) -> Work {
    let n = prefix + suffix;
    Work {
        tokens: n.clone(),
        pairs: triangle(&n, 0),
        kv_written: n.clone(),
        sliding_pairs: if window != 0 {
            triangle(&n, window)
        } else {
            Quantity::Scalar(0.0)
        },
        state_written: ones(1.0, &n),
        ..Work::default()
    }
}

/// Attach one suffix to a prefix already in KV.
pub fn ask(prefix: &Quantity, suffix: &Quantity, window: i64) -> Work {
    Work {
        tokens: suffix.clone(),
        pairs: &(suffix * prefix) + &triangle(suffix, 0),
        kv_written: suffix.clone(),
        kv_read: prefix.clone(),
        sliding_pairs: if window != 0 {
            &triangle(&(prefix + suffix), window) - &triangle(prefix, window)
        } else {
            Quantity::Scalar(0.0)
        },
        sliding_kv_read: if window != 0 {
            prefix.minimum(&Quantity::Scalar((window - 1) as f64))
        } else {
            Quantity::Scalar(0.0)
        },
        state_read: ones(1.0, prefix),
        ..Work::default()
    }
}

/// Attach several independent suffixes to one prefix in KV.
pub fn stream(prefix: &Quantity, suffixes: &[Quantity], window: i64) -> Work {
    let mut tokens = Quantity::Scalar(0.0);
    let mut pairs = Quantity::Scalar(0.0);
    let mut sliding_pairs = Quantity::Scalar(0.0);
    let mut count = 0.0;
    for suffix in suffixes {
        count += 1.0;
        tokens = &tokens + suffix;
        pairs = &pairs + &(&(suffix * prefix) + &triangle(suffix, 0));
        if window != 0 {
            sliding_pairs = &sliding_pairs
                + &(&triangle(&(prefix + suffix), window) - &triangle(prefix, window));
        }
    }
    Work {
        kv_written: tokens.clone(),
        tokens,
        pairs,
        kv_read: prefix.clone(),
        sliding_pairs,
        sliding_kv_read: if window != 0 {
            prefix.minimum(&Quantity::Scalar((window - 1) as f64))
        } else {
            Quantity::Scalar(0.0)
        },
        state_read: ones(count, prefix),
        ..Work::default()
    }
}

/// One set-attention pass: a state, then its questions, nothing kept.
pub fn packed(state: &Quantity, questions: &[f64], window: i64) -> Work {
    let asked = python_sum(questions.iter().copied());
    let n = state + asked;
    let question_pairs = sum_quantities(
        questions
            .iter()
            .map(|question| *question * &(state + *question)),
    );
    let pairs = &(state * &n) + &question_pairs;
    let read = if questions.len() == 1 {
        n.clone()
    } else {
        &(&n + state) + asked
    };
    Work {
        tokens: n.clone(),
        pairs,
        kv_written: n.clone(),
        kv_read: read,
        sliding_pairs: if window != 0 {
            triangle(&n, window)
        } else {
            Quantity::Scalar(0.0)
        },
        ..Work::default()
    }
}

/// Append accepted tokens over sequential rounds, rereading KV each round.
pub fn decode(prefix: &Quantity, rounds: &[f64], window: i64) -> Work {
    let mut work = Work::default();
    let mut prefix = prefix.clone();
    for tokens in rounds {
        let mut step = ask(&prefix, &Quantity::Scalar(*tokens), window);
        step.state_written = step.state_read.clone();
        work = &work + &step;
        prefix = &prefix + *tokens;
    }
    if prefix.is_array() {
        let ones = prefix.map(|_| 1.0);
        work = Work {
            tokens: &work.tokens * &ones,
            pairs: work.pairs,
            kv_written: &work.kv_written * &ones,
            kv_read: work.kv_read,
            sliding_pairs: work.sliding_pairs,
            sliding_kv_read: work.sliding_kv_read,
            state_written: work.state_written,
            state_read: work.state_read,
            sae_encoded: Quantity::Scalar(0.0),
            readout_rows: work.readout_rows,
            readout_head_rows: work.readout_head_rows,
        };
    }
    work
}

/// One number, or one number per document.
#[derive(Debug, Clone, PartialEq)]
pub enum Quantity {
    Scalar(f64),
    Array(Vec<f64>),
}

impl Default for Quantity {
    fn default() -> Self {
        Quantity::Scalar(0.0)
    }
}

impl From<f64> for Quantity {
    fn from(value: f64) -> Self {
        Quantity::Scalar(value)
    }
}

impl From<Vec<f64>> for Quantity {
    fn from(values: Vec<f64>) -> Self {
        Quantity::Array(values)
    }
}

impl Quantity {
    /// Apply `f` element by element, broadcasting a scalar.
    ///
    /// # Panics
    ///
    /// Two arrays have different lengths.
    pub fn zip_with(&self, other: &Quantity, f: impl Fn(f64, f64) -> f64) -> Quantity {
        match (self, other) {
            (Quantity::Scalar(a), Quantity::Scalar(b)) => Quantity::Scalar(f(*a, *b)),
            (Quantity::Array(a), Quantity::Scalar(b)) => {
                Quantity::Array(a.iter().map(|x| f(*x, *b)).collect())
            }
            (Quantity::Scalar(a), Quantity::Array(b)) => {
                Quantity::Array(b.iter().map(|y| f(*a, *y)).collect())
            }
            (Quantity::Array(a), Quantity::Array(b)) => {
                assert_eq!(a.len(), b.len(), "arrays of different lengths");
                Quantity::Array(a.iter().zip(b).map(|(x, y)| f(*x, *y)).collect())
            }
        }
    }

    /// Apply `f` to each element.
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Quantity {
        match self {
            Quantity::Scalar(a) => Quantity::Scalar(f(*a)),
            Quantity::Array(a) => Quantity::Array(a.iter().map(|x| f(*x)).collect()),
        }
    }

    /// The element-wise smaller of two quantities, as `np.minimum`.
    pub fn minimum(&self, other: &Quantity) -> Quantity {
        self.zip_with(other, |a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        })
    }

    /// The scalar, or None for an array.
    pub fn scalar(&self) -> Option<f64> {
        match self {
            Quantity::Scalar(value) => Some(*value),
            Quantity::Array(_) => None,
        }
    }

    /// Whether this is an array.
    pub fn is_array(&self) -> bool {
        matches!(self, Quantity::Array(_))
    }

    /// The sum, as `float(np.sum(value))` rounds it.
    pub fn total(&self) -> f64 {
        match self {
            Quantity::Scalar(value) => *value,
            Quantity::Array(values) => numpy_sum(values),
        }
    }

    /// Whether the value is nonzero, as Python's truth test of a float.
    ///
    /// # Panics
    ///
    /// The value is an array, whose truth Python refuses to test.
    pub fn is_nonzero(&self) -> bool {
        self.scalar()
            .map(|value| value != 0.0)
            .expect("the truth of an array is ambiguous")
    }
}

macro_rules! arithmetic {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait<&Quantity> for &Quantity {
            type Output = Quantity;
            fn $method(self, other: &Quantity) -> Quantity {
                self.zip_with(other, |a, b| a $op b)
            }
        }
        impl $trait<Quantity> for Quantity {
            type Output = Quantity;
            fn $method(self, other: Quantity) -> Quantity {
                (&self).$method(&other)
            }
        }
        impl $trait<f64> for &Quantity {
            type Output = Quantity;
            fn $method(self, other: f64) -> Quantity {
                self.map(|a| a $op other)
            }
        }
        impl $trait<f64> for Quantity {
            type Output = Quantity;
            fn $method(self, other: f64) -> Quantity {
                self.map(|a| a $op other)
            }
        }
        impl $trait<&Quantity> for f64 {
            type Output = Quantity;
            fn $method(self, other: &Quantity) -> Quantity {
                other.map(|b| self $op b)
            }
        }
        impl $trait<Quantity> for f64 {
            type Output = Quantity;
            fn $method(self, other: Quantity) -> Quantity {
                other.map(|b| self $op b)
            }
        }
    };
}

arithmetic!(Add, add, +);
arithmetic!(Sub, sub, -);
arithmetic!(Mul, mul, *);
arithmetic!(Div, div, /);

/// numpy's pairwise summation of a contiguous float64 array, as
/// `np.sum` rounds it (checked against numpy 2.3).
pub fn numpy_sum(values: &[f64]) -> f64 {
    pairwise_sum(values)
}

/// numpy's `pairwise_sum` over one block of values.
fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        let mut total = 0.0;
        for value in values {
            total += value;
        }
        total
    } else if n <= BLOCK {
        let mut partial = [0.0; 8];
        partial.copy_from_slice(&values[..8]);
        let whole = n - n % 8;
        let mut index = 8;
        while index < whole {
            for lane in 0..8 {
                partial[lane] += values[index + lane];
            }
            index += 8;
        }
        let mut total = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
            + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        for value in &values[whole..] {
            total += value;
        }
        total
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
    }
}

/// Python 3.12's `sum()` of floats: Neumaier's compensated summation.
pub fn python_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let next = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// Python's `sum()` of quantities: compensated when every value is a
/// scalar, and element by element, in order, from 0 when any is an
/// array, as numpy adds arrays.
pub fn sum_quantities(values: impl IntoIterator<Item = Quantity>) -> Quantity {
    let values: Vec<Quantity> = values.into_iter().collect();
    if values.iter().all(|value| !value.is_array()) {
        return Quantity::Scalar(python_sum(values.iter().filter_map(Quantity::scalar)));
    }
    let mut total = Quantity::Scalar(0.0);
    for value in values {
        total = &total + &value;
    }
    total
}

/// The work of one filter evaluation over a document of the mean
/// length; the first filter computes the document, a later one reads
/// its KV. A set-attention model reads the document again every pass.
pub fn filter_work(question_tokens: i64, mean_tokens: f64, model: &ModelSpec, first: bool) -> Work {
    let mean = Quantity::Scalar(mean_tokens);
    let question = Quantity::Scalar(question_tokens as f64);
    if model.attention_mask == "set" {
        packed(&mean, &[question_tokens as f64], model.sliding_window)
    } else if first {
        scan(&mean, &question, model.sliding_window)
    } else {
        ask(&mean, &question, model.sliding_window)
    }
}

#[cfg(test)]
mod quantity_tests {
    use super::*;

    #[test]
    fn compensated_summation_keeps_a_small_term_beside_large_ones() {
        assert_eq!(python_sum([1e16, 1.0, -1e16]), 1.0);
    }

    #[test]
    fn a_scalar_broadcasts_over_an_array() {
        let documents = Quantity::Array(vec![1.0, 2.0]);
        assert_eq!(&documents * 2.0, Quantity::Array(vec![2.0, 4.0]));
        assert_eq!(
            documents.minimum(&Quantity::Scalar(1.5)),
            Quantity::Array(vec![1.0, 1.5])
        );
    }
}
