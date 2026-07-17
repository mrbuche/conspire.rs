use super::super::{AssertEq, eq_impl, eq_within_tols_impl};
use crate::math::assert::{Assert, AssertionError};
use crate::math::{SquareMatrix, Vector};

impl AssertEq<Vector> for Vector {
    fn eq(a: Self, b: Vector) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(tols: &Assert, a: Self, b: Vector) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}

impl AssertEq<SquareMatrix> for SquareMatrix {
    fn eq(a: Self, b: SquareMatrix) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(tols: &Assert, a: Self, b: SquareMatrix) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}
