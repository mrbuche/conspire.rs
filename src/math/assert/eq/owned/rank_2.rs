use super::super::{AssertEq, eq_impl, eq_within_tols_impl};
use crate::math::assert::{Assert, AssertionError};
use crate::math::{TensorRank2, TensorRank2SparseVec, TensorRank2SparseVec2DSymmetric};

impl<const D: usize, const I: usize, const J: usize> AssertEq<TensorRank2<D, I, J>>
    for TensorRank2<D, I, J>
{
    fn eq(a: Self, b: TensorRank2<D, I, J>) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(
        tols: &Assert,
        a: Self,
        b: TensorRank2<D, I, J>,
    ) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}

impl<const D: usize, const I: usize, const J: usize> AssertEq<TensorRank2SparseVec<D, I, J>>
    for TensorRank2SparseVec<D, I, J>
{
    fn eq(a: Self, b: TensorRank2SparseVec<D, I, J>) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(
        tols: &Assert,
        a: Self,
        b: TensorRank2SparseVec<D, I, J>,
    ) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}

impl<const D: usize, const I: usize, const J: usize>
    AssertEq<TensorRank2SparseVec2DSymmetric<D, I, J>>
    for TensorRank2SparseVec2DSymmetric<D, I, J>
{
    fn eq(a: Self, b: TensorRank2SparseVec2DSymmetric<D, I, J>) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(
        tols: &Assert,
        a: Self,
        b: TensorRank2SparseVec2DSymmetric<D, I, J>,
    ) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}
