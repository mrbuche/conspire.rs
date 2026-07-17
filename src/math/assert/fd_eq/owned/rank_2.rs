use super::super::{AssertFd, eq_within_fd_tol_impl};
use crate::math::assert::{Assert, AssertionError};
use crate::math::{
    TensorRank2, TensorRank2SparseVec2D, TensorRank2SparseVec2DSymmetric, TensorRank2Vec,
    TensorRank2Vec2D,
};

impl<const D: usize, const I: usize, const J: usize> AssertFd<TensorRank2<D, I, J>>
    for TensorRank2<D, I, J>
{
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank2<D, I, J>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}

impl<const D: usize, const I: usize, const J: usize> AssertFd<TensorRank2Vec<D, I, J>>
    for TensorRank2Vec<D, I, J>
{
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank2Vec<D, I, J>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}

impl<const D: usize, const I: usize, const J: usize> AssertFd<TensorRank2Vec2D<D, I, J>>
    for TensorRank2Vec2D<D, I, J>
{
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank2Vec2D<D, I, J>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}

impl<const D: usize, const I: usize, const J: usize> AssertFd<TensorRank2SparseVec2D<D, I, J>>
    for TensorRank2SparseVec2D<D, I, J>
{
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank2SparseVec2D<D, I, J>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}

impl<const D: usize, const I: usize, const J: usize>
    AssertFd<TensorRank2SparseVec2DSymmetric<D, I, J>>
    for TensorRank2SparseVec2DSymmetric<D, I, J>
{
    fn eq_within_fd_tol(
        tols: &Assert,
        a: Self,
        b: TensorRank2SparseVec2DSymmetric<D, I, J>,
    ) -> Result<(), AssertionError> {
        eq_within_fd_tol_impl(tols, &a, &b)
    }
}
