use super::super::{AssertFd, eq_within_fd_tol_impl};
use crate::math::assert::{Assert, AssertionError};
use crate::math::{SquareMatrix, Vector};

impl AssertFd<Vector> for Vector {
    fn eq_within_fd_tol(tols: &Assert, a: Self, b: Vector) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}

impl AssertFd<SquareMatrix> for SquareMatrix {
    fn eq_within_fd_tol(tols: &Assert, a: Self, b: SquareMatrix) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}
