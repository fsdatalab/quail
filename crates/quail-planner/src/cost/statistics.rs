use std::collections::BTreeMap;

use crate::cost::work::triangle;
use crate::cost::work::{Quantity, python_sum};

pub fn int_true_divide(numerator: i128, denominator: i128) -> f64 {
    assert!(denominator != 0, "division by zero");
    const EXACT: i128 = 1 << 53;
    if numerator.abs() <= EXACT && denominator.abs() <= EXACT {
        return numerator as f64 / denominator as f64;
    }
    let negative = (numerator < 0) != (denominator < 0);
    let (n, d) = (numerator.unsigned_abs(), denominator.unsigned_abs());
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

pub fn round_half_even(value: f64) -> i64 {
    value.round_ties_even() as i64
}

#[derive(Debug, Clone, PartialEq)]
pub struct AliasStats {
    pub count: i64,
    pub total: i64,
    pub squared: i128,
    pub maximum: i64,
    pub window: i64,
    pub short_counts: Vec<(i64, i64)>,
}

impl AliasStats {
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

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            int_true_divide(self.total as i128, self.count as i128)
        }
    }

    pub fn variance(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let count = self.count as i128;
        let total = self.total as i128;
        int_true_divide(self.squared * count - total * total, count * count)
    }

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
        assert_eq!(int_true_divide(10_i128.pow(22), 3), 3.3333333333333335e21);
        assert_eq!(int_true_divide((1 << 60) + 3, (1 << 55) + 1), 32.0);
        assert_eq!(int_true_divide(-7, 2), -3.5);
    }
}
