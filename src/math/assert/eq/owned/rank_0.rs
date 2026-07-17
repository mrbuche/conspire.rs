use super::super::{AssertEq, eq_impl, eq_within_tols_impl};
use crate::math::TensorRank0;
use crate::math::assert::{Assert, AssertionError};

impl AssertEq<TensorRank0> for TensorRank0 {
    fn eq(a: Self, b: TensorRank0) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(tols: &Assert, a: Self, b: TensorRank0) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}
