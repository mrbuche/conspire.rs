use super::super::{AssertEq, eq_impl, eq_within_tols_impl};
use crate::math::TensorRank3;
use crate::math::assert::{Assert, AssertionError};

impl<const D: usize, const I: usize, const J: usize, const K: usize>
    AssertEq<TensorRank3<D, I, J, K>> for TensorRank3<D, I, J, K>
{
    fn eq(a: Self, b: TensorRank3<D, I, J, K>) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(
        tols: &Assert,
        a: Self,
        b: TensorRank3<D, I, J, K>,
    ) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}
