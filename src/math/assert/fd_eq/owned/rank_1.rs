use super::super::{AssertFd, eq_within_fd_tol_impl};
use crate::math::assert::{Assert, AssertionError};
use crate::math::{TensorRank1, TensorRank1List, TensorRank1Vec};

impl<const D: usize, const I: usize> AssertFd<TensorRank1<D, I>> for TensorRank1<D, I> {
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank1<D, I>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}

impl<const D: usize, const I: usize, const W: usize> AssertFd<TensorRank1List<D, I, W>>
    for TensorRank1List<D, I, W>
{
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank1List<D, I, W>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}

impl<const D: usize, const I: usize> AssertFd<TensorRank1Vec<D, I>> for TensorRank1Vec<D, I> {
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank1Vec<D, I>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}
