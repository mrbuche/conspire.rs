use super::super::{AssertFd, eq_within_fd_tol_impl};
use crate::math::TensorRank3;
use crate::math::assert::{Assert, AssertionError};

impl<const D: usize, const I: usize, const J: usize, const K: usize>
    AssertFd<TensorRank3<D, I, J, K>> for TensorRank3<D, I, J, K>
{
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank3<D, I, J, K>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}
