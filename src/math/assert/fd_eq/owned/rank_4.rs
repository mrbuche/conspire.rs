use super::super::{AssertFd, eq_within_fd_tol_impl};
use crate::math::TensorRank4;
use crate::math::assert::{Assert, AssertionError};

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    AssertFd<TensorRank4<D, I, J, K, L>> for TensorRank4<D, I, J, K, L>
{
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank4<D, I, J, K, L>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}
