use super::super::{AssertFd, eq_within_fd_tol_impl};
use crate::math::assert::{Assert, AssertionError};
use crate::math::{TensorRank0, TensorRank0List, TensorRank0List2D};

impl AssertFd<TensorRank0> for TensorRank0 {
    fn eq_within_fd_tol(tols: &Assert, a: Self, b: TensorRank0) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}

impl<const N: usize> AssertFd<TensorRank0List<N>> for TensorRank0List<N> {
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank0List<N>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}

impl<const N: usize> AssertFd<TensorRank0List2D<N>> for TensorRank0List2D<N> {
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank0List2D<N>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}
