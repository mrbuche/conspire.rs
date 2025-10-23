use crate::math::{Tensor, TensorRank0};
use std::{
    fmt::{Display, Formatter, Result},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

#[derive(Clone, Debug)]
pub struct TensorTuple<T1, T2>(T1, T2)
where
    T1: Tensor,
    T2: Tensor;

impl<T1, T2> Default for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn default() -> Self {
        Self(T1::default(), T2::default())
    }
}

impl<T1, T2> From<(T1, T2)> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn from(tuple: (T1, T2)) -> Self {
        Self(tuple.0, tuple.1)
    }
}

impl<'a, T1, T2> From<&'a TensorTuple<T1, T2>> for (&'a T1, &'a T2)
where
    T1: Tensor,
    T2: Tensor,
{
    fn from(tensor_tuple: &'a TensorTuple<T1, T2>) -> Self {
        (&tensor_tuple.0, &tensor_tuple.1)
    }
}

impl<T1, T2> From<TensorTuple<T1, T2>> for (T1, T2)
where
    T1: Tensor,
    T2: Tensor,
{
    fn from(tensor_tuple: TensorTuple<T1, T2>) -> Self {
        (tensor_tuple.0, tensor_tuple.1)
    }
}

impl<T1, T2> Display for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Need to implement Display")
    }
}

impl<T1, T2> Tensor for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Item = T1::Item;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        if self.size() == 0 {
            self.0.iter()
        } else {
            unimplemented!()
        }
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        if self.size() == 0 {
            self.0.iter_mut()
        } else {
            unimplemented!()
        }
    }
    fn len(&self) -> usize {
        unimplemented!()
    }
    fn size(&self) -> usize {
        unimplemented!()
    }
}

impl<T1, T2> Sum for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        iter.reduce(|mut acc, item| {
            acc.0 += item.0;
            acc.1 += item.1;
            acc
        })
        .unwrap_or_else(Self::default)
    }
}

impl<T1, T2> Div<TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<T1, T2> Div<&TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<T1, T2> DivAssign<TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.0 /= &tensor_rank_0;
        self.1 /= tensor_rank_0;
    }
}

impl<T1, T2> DivAssign<&TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.0 /= tensor_rank_0;
        self.1 /= tensor_rank_0;
    }
}

impl<T1, T2> Mul<TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<T1, T2> Mul<TensorRank0> for &TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
    for<'a> &'a T1: Mul<&'a TensorRank0, Output = T1>,
    for<'a> &'a T2: Mul<&'a TensorRank0, Output = T2>,
{
    type Output = TensorTuple<T1, T2>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        TensorTuple(&self.0 * &tensor_rank_0, &self.1 * &tensor_rank_0)
    }
}

impl<T1, T2> Mul<&TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

// impl<T1, T2> Mul<&TensorRank0> for &TensorTuple<T1, T2>
// where
//     T1: Tensor,
//     T2: Tensor,
//     for<'a> &'a T1: Mul<&'a TensorRank0, Output = T1>,
//     for<'a> &'a T2: Mul<&'a TensorRank0, Output = T2>,
// {
//     type Output = TensorTuple<T1, T2>;
//     fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
//         TensorTuple(&self.0 * tensor_rank_0, &self.1 * tensor_rank_0)
//     }
// }

impl<T1, T2> MulAssign<TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.0 *= &tensor_rank_0;
        self.1 *= tensor_rank_0;
    }
}

impl<T1, T2> MulAssign<&TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.0 *= tensor_rank_0;
        self.1 *= tensor_rank_0;
    }
}

impl<T1, T2> Add for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_tuple: Self) -> Self::Output {
        self += tensor_tuple;
        self
    }
}

impl<T1, T2> Add<&Self> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_tuple: &Self) -> Self::Output {
        self += tensor_tuple;
        self
    }
}

impl<T1, T2> AddAssign for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn add_assign(&mut self, tensor_tuple: Self) {
        self.0 += tensor_tuple.0;
        self.1 += tensor_tuple.1;
    }
}

impl<T1, T2> AddAssign<&Self> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn add_assign(&mut self, tensor_tuple: &Self) {
        self.0 += &tensor_tuple.0;
        self.1 += &tensor_tuple.1;
    }
}

impl<T1, T2> Sub for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_tuple: Self) -> Self::Output {
        self -= tensor_tuple;
        self
    }
}

// impl<T1, T2> Sub<TensorTuple<T1, T2>> for &TensorTuple<T1, T2>
// where
//     T1: Tensor,
//     T2: Tensor,
// {
//     type Output = TensorTuple<T1, T2>;
//     fn sub(self, tensor_tuple: TensorTuple<T1, T2>) -> Self::Output {
//         todo!()
//     }
// }

impl<T1, T2> Sub<&Self> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_tuple: &Self) -> Self::Output {
        self -= tensor_tuple;
        self
    }
}

impl<T1, T2> Sub for &TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = TensorTuple<T1, T2>;
    fn sub(self, tensor_tuple: Self) -> Self::Output {
        todo!()
    }
}

impl<T1, T2> SubAssign for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn sub_assign(&mut self, tensor_tuple: Self) {
        self.0 += tensor_tuple.0;
        self.1 += tensor_tuple.1;
    }
}

impl<T1, T2> SubAssign<&Self> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn sub_assign(&mut self, tensor_tuple: &Self) {
        self.0 += &tensor_tuple.0;
        self.1 += &tensor_tuple.1;
    }
}
