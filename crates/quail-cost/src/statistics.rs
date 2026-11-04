//! Per-table document-length summaries, as `AliasStats` in
//! `quail/planner/statistics/tables.py` holds them.

use std::collections::BTreeMap;

use crate::quantity::{Quantity, python_sum};
use crate::work::triangle;

/// Python's `int / int`: the quotient correctly rounded to the nearest
/// float, ties to even, however large the integers.
///
/// # Panics
///
/// `denominator` is zero.
pub fn int_true_divide(numerator: i128, denominator: i128) -> f64 {
    assert!(denominator != 0, "division by zero");
    const EXACT: i128 = 1 << 53;
    if numerator.abs() <= EXACT && denominator.abs() <= EXACT {
        return numerator as f64 / denominator as f64;
    }
    let negative = (numerator < 0) != (denominator < 0);
    let (n, d) = (numerator.unsigned_abs(), denominator.unsigned_abs());
    // Scale so the quotient has 55 significant bits: 53 kept, one to
    // round on, and one more so the remainder decides ties.
    let shift = 55 - (n.ilog2() as i32 - d.ilog2() as i32);
    let (scaled_n, scaled_d) = if shift >= 0 {
        (n << shift, d)
    } else {
        (n, d << -shift)
    };
    let mut quotient = scaled_n / scaled_d;
    let inexact = scaled_n % scaled_d != 0;
    let mut exponent = -shift;
    while quotient >= 1 << 55 {
        let low = quotient & 1;
        quotient >>= 1;
        exponent += 1;
        quotient |= low;
    }
    // quotient has 55 bits; keep 53 with round half to even
    let round_bits = quotient & 0b11;
    let mut mantissa = quotient >> 2;
    let half = round_bits & 0b10 != 0;
    let rest = round_bits & 0b01 != 0 || inexact;
    if half && (rest || mantissa & 1 == 1) {
        mantissa += 1;
    }
    let value = mantissa as f64 * 2f64.powi(exponent + 2);
    if negative { -value } else { value }
}

/// Python's `round(x)` of a float: the nearest integer, ties to even.
pub fn round_half_even(value: f64) -> i64 {
    value.round_ties_even() as i64
}

/// Length sums and counts below the attention window for one table.
#[derive(Debug, Clone, PartialEq)]
pub struct AliasStats {
    pub count: i64,
    pub total: i64,
    pub squared: i128,
    pub maximum: i64,
    pub window: i64,
    /// (length, documents) for each document length below the window.
    pub short_counts: Vec<(i64, i64)>,
}

impl AliasStats {
    /// Summarize a column of document token lengths, as
    /// `summarize_alias` does; `short_counts` is sorted by length.
    pub fn summarize(lengths: &[i64], window: i64) -> AliasStats {
        let mut short = BTreeMap::new();
        if window != 0 {
            for length in lengths.iter().filter(|length| **length < window) {
                *short.entry(*length).or_insert(0_i64) += 1;
            }
        }
        AliasStats {
            count: lengths.len() as i64,
            total: lengths.iter().sum(),
            squared: lengths
                .iter()
                .map(|length| *length as i128 * *length as i128)
                .sum(),
            maximum: lengths.iter().copied().max().unwrap_or(0),
            window,
            short_counts: short.into_iter().collect(),
        }
    }

    /// The mean document length.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            int_true_divide(self.total as i128, self.count as i128)
        }
    }

    /// The variance of the document lengths.
    pub fn variance(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let count = self.count as i128;
        let total = self.total as i128;
        int_true_divide(self.squared * count - total * total, count * count)
    }

    /// Sliding pairs summed over the document lengths plus an offset.
    pub fn window_pairs(&self, offset: f64) -> f64 {
        let window = self.window;
        let mut linear = window as f64 * (self.total as f64 + offset * self.count as f64);
        linear -= (self.count * window * (window - 1)) as f64 / 2.0;
        let half = (window * (window - 1)) as f64 / 2.0;
        linear
            + python_sum(
                self.short_counts
                    .iter()
                    .filter(|(length, _)| (*length as f64 + offset) < window as f64)
                    .map(|(length, count)| {
                        let shifted = *length as f64 + offset;
                        *count as f64
                            * (triangle(&Quantity::Scalar(shifted), 0).total()
                                - window as f64 * shifted
                                + half)
                    }),
            )
    }

    /// Retained prefix keys visible to the first suffix token, summed.
    pub fn window_reads(&self, offset: f64) -> f64 {
        let cap = self.window - 1;
        (self.count * cap) as f64
            - python_sum(
                self.short_counts
                    .iter()
                    .filter(|(length, _)| (*length as f64 + offset) < cap as f64)
                    .map(|(length, count)| *count as f64 * ((cap - length) as f64 - offset)),
            )
    }

    /// The summary of the fraction of the documents expected to remain.
    pub fn scale(&self, fraction: f64) -> AliasStats {
        AliasStats {
            count: round_half_even(self.count as f64 * fraction),
            total: round_half_even(self.total as f64 * fraction),
            squared: (self.squared as f64 * fraction).round_ties_even() as i128,
            maximum: self.maximum,
            window: self.window,
            short_counts: self
                .short_counts
                .iter()
                .map(|(length, count)| (*length, round_half_even(*count as f64 * fraction)))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_large_integer_quotient_rounds_once() {
        // the values Python's 10**22 / 3 and (2**60 + 3) / (2**55 + 1) give
        assert_eq!(int_true_divide(10_i128.pow(22), 3), 3.3333333333333335e21);
        assert_eq!(int_true_divide((1 << 60) + 3, (1 << 55) + 1), 32.0);
        assert_eq!(int_true_divide(-7, 2), -3.5);
    }
}
