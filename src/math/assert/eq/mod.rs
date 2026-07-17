mod owned;

use super::{Assert, AssertionError};
use crate::math::Tensor;
use std::fmt::Display;

/// Equality assertions, overloaded across owned and borrowed operands.
pub trait AssertEq<Rhs = Self> {
    fn eq(a: Self, b: Rhs) -> Result<(), AssertionError>;
    fn eq_within_tols(tols: &Assert, a: Self, b: Rhs) -> Result<(), AssertionError>;
}

pub(super) fn eq_impl<T: Display + PartialEq>(a: &T, b: &T) -> Result<(), AssertionError> {
    if a == b {
        Ok(())
    } else {
        Err(AssertionError {
            message: format!(
                "\n\x1b[1;91mAssertion `left == right` failed.\n\x1b[0;91m  left: {a}\n right: {b}\x1b[0m"
            ),
        })
    }
}

pub(super) fn eq_within_tols_impl<T: Display + Tensor>(
    tols: &Assert,
    a: &T,
    b: &T,
) -> Result<(), AssertionError> {
    if let Some(count) = a.error_count(b, tols.abs_tol, tols.rel_tol) {
        let abs = a.sub_abs(b);
        let rel = a.sub_rel(b);
        Err(AssertionError {
            message: format!(
                "\n\x1b[1;91mAssertion `left ≈= right` failed in {count} places.\n\x1b[0;91m  left: {a}\n right: {b}\n   abs: {abs}\n   rel: {rel}\x1b[0m"
            ),
        })
    } else {
        Ok(())
    }
}

pub(super) fn zero_impl<T: Display + Tensor>(a: &T) -> Result<(), AssertionError> {
    if a.is_zero() {
        Ok(())
    } else {
        Err(AssertionError {
            message: format!(
                "\n\x1b[1;91mAssertion `left == right` failed.\n\x1b[0;91m  left: {a}\n right: 0\x1b[0m"
            ),
        })
    }
}

pub(super) fn zero_within_tols_impl<T: Display + Tensor>(
    tols: &Assert,
    a: &T,
) -> Result<(), AssertionError> {
    if let Some(count) = a.error_count_zero(tols.abs_tol, tols.rel_tol) {
        Err(AssertionError {
            message: format!(
                "\n\x1b[1;91mAssertion `left ≈= right` failed in {count} places.\n\x1b[0;91m  left: {a}\n right: 0\x1b[0m"
            ),
        })
    } else {
        Ok(())
    }
}

impl<T> AssertEq<T> for &T
where
    T: Display + PartialEq + Tensor,
{
    fn eq(a: Self, b: T) -> Result<(), AssertionError> {
        eq_impl(a, &b)
    }
    fn eq_within_tols(tols: &Assert, a: Self, b: T) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, a, &b)
    }
}

impl<'a, T> AssertEq<&'a T> for T
where
    T: Display + PartialEq + Tensor,
{
    fn eq(a: Self, b: &'a T) -> Result<(), AssertionError> {
        eq_impl(&a, b)
    }
    fn eq_within_tols(tols: &Assert, a: Self, b: &'a T) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, b)
    }
}

impl<'a, T> AssertEq<&'a T> for &'a T
where
    T: Display + PartialEq + Tensor,
{
    fn eq(a: Self, b: &'a T) -> Result<(), AssertionError> {
        eq_impl(a, b)
    }
    fn eq_within_tols(tols: &Assert, a: Self, b: &'a T) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, a, b)
    }
}
