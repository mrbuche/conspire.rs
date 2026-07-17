mod owned;

use super::{Assert, AssertionError, FiniteDifference};
use crate::math::Tensor;
use std::fmt::Display;

/// Finite-difference equality assertions, overloaded across owned and borrowed operands.
pub trait AssertFd<Rhs = Self> {
    fn eq_within_fd_tol(tols: &Assert, a: Self, b: Rhs) -> Result<(), AssertionError>;
}

pub(super) fn eq_within_fd_tol_impl<T: Display + FiniteDifference + Tensor>(
    tols: &Assert,
    a: &T,
    b: &T,
) -> Result<(), AssertionError> {
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

impl<T> AssertFd<T> for &T
where
    T: Display + PartialEq + Tensor + FiniteDifference,
{
    fn eq_within_fd_tol(tols: &Assert, a: Self, b: T) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, a, &b)
    }
}

impl<'a, T> AssertFd<&'a T> for T
where
    T: Display + PartialEq + Tensor + FiniteDifference,
{
    fn eq_within_fd_tol(tols: &Assert, a: Self, b: &'a T) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, b)
    }
}

impl<'a, T> AssertFd<&'a T> for &'a T
where
    T: Display + PartialEq + Tensor + FiniteDifference,
{
    fn eq_within_fd_tol(tols: &Assert, a: Self, b: &'a T) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, a, b)
    }
}
