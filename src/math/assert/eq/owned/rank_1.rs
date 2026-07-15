use super::super::{AssertEq, eq_impl, eq_within_tols_impl};
use crate::math::TensorRank1;
use crate::math::assert::{Assert, AssertionError};

impl<const D: usize, const I: usize> AssertEq<TensorRank1<D, I>> for TensorRank1<D, I> {
    fn eq(a: Self, b: TensorRank1<D, I>) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(tols: &Assert, a: Self, b: TensorRank1<D, I>) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}
