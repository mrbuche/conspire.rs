mod eq;
mod error;
mod fd;
mod fd_eq;

#[cfg(test)]
mod test;

pub use self::{eq::AssertEq, error::AssertionError, fd::FiniteDifference, fd_eq::AssertFd};

use self::eq::{zero_impl, zero_within_tols_impl};
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
    /// Asserts exact equality with zero.
    pub fn zero<T>(a: &T) -> Result<(), AssertionError>
    where
        T: Display + Tensor,
    {
        zero_impl(a)
    }
    /// Asserts equality with zero within `self.abs_tol` and `self.rel_tol`.
    pub fn zero_within_tols<T>(&self, a: &T) -> Result<(), AssertionError>
    where
        T: Display + Tensor,
    {
        zero_within_tols_impl(self, a)
    }
}
