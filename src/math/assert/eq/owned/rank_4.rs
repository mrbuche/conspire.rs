use super::super::{AssertEq, eq_impl, eq_within_tols_impl};
use crate::math::TensorRank4;
use crate::math::assert::{Assert, AssertionError};

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    AssertEq<TensorRank4<D, I, J, K, L>> for TensorRank4<D, I, J, K, L>
{
    fn eq(a: Self, b: TensorRank4<D, I, J, K, L>) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(
        tols: &Assert,
        a: Self,
        b: TensorRank4<D, I, J, K, L>,
    ) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}
