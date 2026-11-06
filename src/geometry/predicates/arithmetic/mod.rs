//! The arithmetic that a predicate is written in once and run twice: first in floating point,
//! which says when it is certain of a sign, and then, when it is not, exactly.
//!
//! A formula that is generic over [`Arithmetic`] cannot differ between the two, so the filter can
//! never disagree with the exact path about what is being computed.

#[cfg(test)]
mod test;

use super::{HALF_EPSILON, expansion::Expansion};
use core::cmp::Ordering;

/// The operations that predicates are built from, with the sign of the result when it is certain.
pub(super) trait Arithmetic: Clone {
    /// A number that is known exactly.
    #[cfg(test)]
    fn exactly(value: f64) -> Self;
    /// The difference of two floats, exactly if the arithmetic is.
    fn difference(a: f64, b: f64) -> Self;
    fn sum(self, other: Self) -> Self;
    fn minus(self, other: Self) -> Self;
    fn times(&self, other: &Self) -> Self;
    fn negated(self) -> Self;
    /// The same number in the fewest components, which exact arithmetic gains by and an estimate
    /// has nothing to gain by.
    fn compact(self) -> Self;
    /// The sign of the number if rounding cannot have changed it, and `None` if it could have.
    ///
    /// Exact arithmetic is always certain, and a zero is only ever certain when it is exact.
    fn certain_sign(&self) -> Option<Ordering>;
}

impl Arithmetic for Expansion {
    #[cfg(test)]
    fn exactly(value: f64) -> Self {
        Self::new(value)
    }
    fn difference(a: f64, b: f64) -> Self {
        Self::difference(a, b)
    }
    fn sum(self, other: Self) -> Self {
        Self::sum(self, other)
    }
    fn minus(self, other: Self) -> Self {
        Self::minus(self, other)
    }
    fn times(&self, other: &Self) -> Self {
        Self::times(self, other)
    }
    fn negated(self) -> Self {
        Self::negated(self)
    }
    fn compact(self) -> Self {
        self.compressed()
    }
    fn certain_sign(&self) -> Option<Ordering> {
        Some(self.sign())
    }
}

/// A floating-point number with a bound on how far rounding can have moved it.
///
/// The bound is kept as a count of rounding errors, `units`, relative to a `permanent`: the sum of
/// the magnitudes of the terms that the number is built from, which is what its error scales with.
/// Take every floating-point operation to be correctly rounded with relative error at most `ε`,
/// with no overflow or underflow, and count to first order in `ε`. The error of the number is at
/// most `units · ε · permanent`.
/// - A float is exact: no error, and its magnitude is its permanent.
/// - A difference of two floats has one rounding, relative to its own magnitude: error 1.
/// - A sum has the larger of the two errors, each relative to its own permanent and so at most
///   that of the two permanents added, and its own rounding, relative to the magnitude of the sum,
///   which is at most the sum of the permanents: error `max + 1` relative to the permanents added.
/// - A product has the errors of its factors added, since the relative error of a product is the
///   sum of those of its factors, and its own rounding: error `a + b + 1` relative to the product
///   of the permanents.
/// - Negation changes nothing.
///
/// This is the count that the derived bound of `incircle` uses, applied to any formula. A sign is
/// certain when the number is farther from zero than `(units + 1) · ε · permanent`. The extra
/// unit is a margin that covers the terms of order `ε²` that the count leaves out, which are at
/// most about `units² · ε²` times the permanent, so they are smaller than the margin by a factor
/// of `1 / (units² · ε)`, which is about `10¹¹` for counts in the hundreds, as the largest that
/// arise are. It also covers the rounding of the bound itself.
///
/// Counting this way is cruder than Shewchuk's analysis of each predicate, so the bound is not
/// tight. That only sends more cases to the exact path, and a bound that is too small would be
/// wrong.
#[derive(Clone, Copy, Debug)]
pub(super) struct Estimate {
    value: f64,
    permanent: f64,
    units: f64,
}

impl Estimate {
    /// The number, which is only an approximation.
    #[cfg(test)]
    pub fn value(&self) -> f64 {
        self.value
    }
    /// The most that rounding can have moved the number, to first order.
    #[cfg(test)]
    pub fn error(&self) -> f64 {
        self.units * HALF_EPSILON * self.permanent
    }
}

impl Arithmetic for Estimate {
    #[cfg(test)]
    fn exactly(value: f64) -> Self {
        Self {
            value,
            permanent: value.abs(),
            units: 0.0,
        }
    }
    fn difference(a: f64, b: f64) -> Self {
        let value = a - b;
        Self {
            value,
            permanent: value.abs(),
            units: 1.0,
        }
    }
    fn sum(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            permanent: self.permanent + other.permanent,
            units: self.units.max(other.units) + 1.0,
        }
    }
    fn minus(self, other: Self) -> Self {
        self.sum(other.negated())
    }
    fn times(&self, other: &Self) -> Self {
        Self {
            value: self.value * other.value,
            permanent: self.permanent * other.permanent,
            units: self.units + other.units + 1.0,
        }
    }
    fn negated(self) -> Self {
        Self {
            value: -self.value,
            ..self
        }
    }
    fn compact(self) -> Self {
        self
    }
    fn certain_sign(&self) -> Option<Ordering> {
        let bound = (self.units + 1.0) * HALF_EPSILON * self.permanent;
        if self.value > bound {
            Some(Ordering::Greater)
        } else if -self.value > bound {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}
