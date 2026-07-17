use super::super::{AssertEq, eq_impl, eq_within_tols_impl};
use crate::math::assert::{Assert, AssertionError};
use crate::math::{Tensor, TensorList, TensorTuple, TensorVector};
use std::fmt::Display;

impl<T, const N: usize> AssertEq<TensorList<T, N>> for TensorList<T, N>
where
    T: Display + PartialEq + Tensor,
{
    fn eq(a: Self, b: TensorList<T, N>) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(tols: &Assert, a: Self, b: TensorList<T, N>) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}

impl<T> AssertEq<TensorVector<T>> for TensorVector<T>
where
    T: Display + PartialEq + Tensor,
{
    fn eq(a: Self, b: TensorVector<T>) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(tols: &Assert, a: Self, b: TensorVector<T>) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}

impl<T1, T2> AssertEq<TensorTuple<T1, T2>> for TensorTuple<T1, T2>
where
    T1: Display + PartialEq + Tensor,
    T2: Display + PartialEq + Tensor,
{
    fn eq(a: Self, b: TensorTuple<T1, T2>) -> Result<(), AssertionError> {
        eq_impl(&a, &b)
    }
    fn eq_within_tols(
        tols: &Assert,
        a: Self,
        b: TensorTuple<T1, T2>,
    ) -> Result<(), AssertionError> {
        eq_within_tols_impl(tols, &a, &b)
    }
}
