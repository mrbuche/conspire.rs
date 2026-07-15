mod eq;
mod error;
mod fd;
mod fd_eq;

#[cfg(test)]
mod test;

pub use self::{eq::AssertEq, error::AssertionError, fd::FiniteDifference, fd_eq::AssertFd};

use crate::{
    ABS_TOL, EPSILON, REL_TOL,
    math::{Scalar, Tensor},
};
use std::fmt::Display;

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
        T: AssertFd<Rhs>,
    {
        T::eq_within_fd_tol(self, a, b)
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
    T: Display + FiniteDifference + Tensor,
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
    T: Display + FiniteDifference + Tensor,
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
