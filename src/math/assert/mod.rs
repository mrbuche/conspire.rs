mod error;

#[cfg(test)]
mod test;

pub use error::AssertionError;

use crate::{
    ABS_TOL, EPSILON, REL_TOL,
    math::{Scalar, Tensor},
};
use std::fmt::Display;

/// Types that can report a finite-difference comparison error against themselves.
pub trait ErrorTensor {
    fn error_fd(&self, comparator: &Self, epsilon: Scalar) -> Option<(bool, usize)>;
}

/// Specifies tolerances used by [`AssertEq`] functionalities.
pub struct Assert {
    pub abs_tol: Scalar,
    pub rel_tol: Scalar,
    pub fd_tol: Scalar,
}

impl Default for Assert {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            rel_tol: REL_TOL,
            fd_tol: 3.0 * EPSILON,
        }
    }
}

impl Assert {
    /// Asserts exact equality.
    pub fn eq<T, Rhs>(a: T, b: Rhs) -> Result<(), AssertionError>
    where
        T: AssertEq<Rhs>,
    {
        T::eq(a, b)
    }
    /// Asserts equality within `self.abs_tol` and `self.rel_tol`.
    pub fn eq_within_tols<T, Rhs>(&self, a: T, b: Rhs) -> Result<(), AssertionError>
    where
        T: AssertEq<Rhs>,
    {
        T::eq_within_tols(self, a, b)
    }
    /// Asserts finite-difference equality within `self.fd_tol`.
    pub fn eq_within_fd_tol<T, Rhs>(&self, a: T, b: Rhs) -> Result<(), AssertionError>
    where
        T: AssertEq<Rhs>,
    {
        T::eq_within_fd_tol(self, a, b)
    }
}

/// Equality assertions, overloaded across owned and borrowed operands.
pub trait AssertEq<Rhs = Self> {
    fn eq(a: Self, b: Rhs) -> Result<(), AssertionError>;
    fn eq_within_tols(tols: &Assert, a: Self, b: Rhs) -> Result<(), AssertionError>;
    fn eq_within_fd_tol(tols: &Assert, a: Self, b: Rhs) -> Result<(), AssertionError>;
}

impl<'a, T> AssertEq<&'a T> for &'a T
where
    T: Display + PartialEq + Tensor + ErrorTensor,
{
    fn eq(a: Self, b: &'a T) -> Result<(), AssertionError> {
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
    fn eq_within_tols(tols: &Assert, a: Self, b: &'a T) -> Result<(), AssertionError> {
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
    fn eq_within_fd_tol(tols: &Assert, a: Self, b: &'a T) -> Result<(), AssertionError> {
        if let Some((failed, count)) = a.error_fd(b, tols.fd_tol) {
            if failed {
                let abs = a.sub_abs(b);
                let rel = a.sub_rel(b);
                Err(AssertionError {
                    message: format!(
                        "\n\x1b[1;91mAssertion `left ≈= right` failed in {count} places.\n\x1b[0;91m  left: {a}\n right: {b}\n   abs: {abs}\n   rel: {rel}\x1b[0m"
                    ),
                })
            } else {
                println!(
                    "Warning: \n\x1b[1;93mAssertion `left ≈= right` was weak in {count} places.\x1b[0m"
                );
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
pub fn assert_eq<'a, T>(value_1: &'a T, value_2: &'a T) -> Result<(), AssertionError>
where
    T: Display + PartialEq,
{
    if value_1 == value_2 {
        Ok(())
    } else {
        Err(AssertionError {
            message: format!(
                "\n\x1b[1;91mAssertion `left == right` failed.\n\x1b[0;91m  left: {value_1}\n right: {value_2}\x1b[0m"
            ),
        })
    }
}

#[cfg(test)]
pub fn assert_eq_from_fd<'a, T>(value: &'a T, value_fd: &'a T) -> Result<(), AssertionError>
where
    T: Display + ErrorTensor + Tensor,
{
    assert_eq_from_fd_within(value, value_fd, 3.0 * EPSILON)
}

#[cfg(test)]
pub fn assert_eq_from_fd_within<'a, T>(
    value: &'a T,
    value_fd: &'a T,
    tol: Scalar,
) -> Result<(), AssertionError>
where
    T: Display + ErrorTensor + Tensor,
{
    if let Some((failed, count)) = value.error_fd(value_fd, tol) {
        if failed {
            let abs = value.sub_abs(value_fd);
            let rel = value.sub_rel(value_fd);
            Err(AssertionError {
                message: format!(
                    "\n\x1b[1;91mAssertion `left ≈= right` failed in {count} places.\n\x1b[0;91m  left: {value}\n right: {value_fd}\n   abs: {abs}\n   rel: {rel}\x1b[0m"
                ),
            })
        } else {
            println!(
                "Warning: \n\x1b[1;93mAssertion `left ≈= right` was weak in {count} places.\x1b[0m"
            );
            Ok(())
        }
    } else {
        Ok(())
    }
}

pub fn assert_eq_within<'a, T>(
    value_1: &'a T,
    value_2: &'a T,
    tol_abs: Scalar,
    tol_rel: Scalar,
) -> Result<(), AssertionError>
where
    T: Display + Tensor,
{
    if let Some(count) = value_1.error_count(value_2, tol_abs, tol_rel) {
        let abs = value_1.sub_abs(value_2);
        let rel = value_1.sub_rel(value_2);
        Err(AssertionError {
            message: format!(
                "\n\x1b[1;91mAssertion `left ≈= right` failed in {count} places.\n\x1b[0;91m  left: {value_1}\n right: {value_2}\n   abs: {abs}\n   rel: {rel}\x1b[0m"
            ),
        })
    } else {
        Ok(())
    }
}

pub fn assert_eq_within_tols<'a, T>(value_1: &'a T, value_2: &'a T) -> Result<(), AssertionError>
where
    T: Display + Tensor,
{
    assert_eq_within(value_1, value_2, ABS_TOL, REL_TOL)
}
