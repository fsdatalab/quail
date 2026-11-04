//! A count that is one number, or one number per document.
//!
//! The Python cost model passes floats and numpy arrays through the
//! same arithmetic. `Quantity` does the same with numpy's
//! broadcasting, and its sums round as numpy's and Python's do.

use std::ops::{Add, Div, Mul, Sub};

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

#[cfg(test)]
mod tests {
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
