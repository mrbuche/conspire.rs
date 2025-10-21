use crate::math::{Tensor, TensorArray, TensorRank0};
use std::{
    array::from_fn,
    fmt::{Display, Formatter, Result},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

#[derive(Clone, Debug)]
pub struct TensorList<const N: usize, T>([T; N])
where
    T: Tensor;

impl<const N: usize, T> Default for TensorList<N, T>
where
    T: Tensor,
{
    fn default() -> Self {
        Self(from_fn(|_| T::default()))
    }
}

impl<const N: usize, T> TensorList<N, T>
where
    T: Tensor,
{
    /// Associated function for const type conversion.
    pub const fn const_from(array: [T; N]) -> Self {
        Self(array)
    }
}

impl<const N: usize, T> From<[T; N]> for TensorList<N, T>
where
    T: Tensor,
{
    fn from(tensor_array: [T; N]) -> Self {
        Self(tensor_array)
    }
}

impl<const N: usize, T> Display for TensorList<N, T>
where
    T: Tensor,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        todo!()
    }
}

impl<const N: usize, T> Index<usize> for TensorList<N, T>
where
    T: Tensor,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const N: usize, T> IndexMut<usize> for TensorList<N, T>
where
    T: Tensor,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const N: usize, T> Tensor for TensorList<N, T>
where
    T: Tensor,
{
    type Item = T;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
}

impl<const N: usize, T> FromIterator<T> for TensorList<N, T>
where
    T: Tensor + TensorArray,
{
    fn from_iter<Ii: IntoIterator<Item = T>>(into_iterator: Ii) -> Self {
        let mut tensor_list = Self::zero();
        tensor_list
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_list_entry, entry)| *tensor_list_entry = entry);
        tensor_list
    }
}

impl<const N: usize, T> Sum for TensorList<N, T>
where
    T: Tensor + TensorArray, {
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        let mut output = Self::zero();
        iter.for_each(|item| output += item);
        output
    }
}

impl<const N: usize, T> TensorArray for TensorList<N, T>
where
    T: Tensor + TensorArray,
{
    type Array = [T::Array; N];
    type Item = T;
    fn as_array(&self) -> Self::Array {
        todo!()
        // let mut array = [[0.0; D]; W];
        // array
        //     .iter_mut()
        //     .zip(self.iter())
        //     .for_each(|(entry, tensor_rank_1)| *entry = tensor_rank_1.as_array());
        // array
    }
    fn identity() -> Self {
        Self(from_fn(|_| Self::Item::identity()))
    }
    fn new(array: Self::Array) -> Self {
        array.into_iter().map(Self::Item::new).collect()
    }
    fn zero() -> Self {
        Self(from_fn(|_| Self::Item::zero()))
    }
}

impl<const N: usize, T> Div<TensorRank0> for TensorList<N, T>
where
    T: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const N: usize, T> Div<&TensorRank0> for TensorList<N, T>
where
    T: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const N: usize, T> DivAssign<TensorRank0> for TensorList<N, T>
where
    T: Tensor,
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl<const N: usize, T> DivAssign<&TensorRank0> for TensorList<N, T>
where
    T: Tensor,
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl<const N: usize, T> Mul<TensorRank0> for TensorList<N, T>
where
    T: Tensor,
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}
impl<const N: usize, T> Mul<&TensorRank0> for TensorList<N, T>
where
    T: Tensor,
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const N: usize, T> Mul<&TensorRank0> for &TensorList<N, T>
where
    T: Tensor,
{
    type Output = TensorList<N, T>;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        todo!()
        // self.iter().map(|entry| entry * tensor_rank_0).collect()
    }
}

impl<const N: usize, T> MulAssign<TensorRank0> for TensorList<N, T>
where
    T: Tensor,
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl<const N: usize, T> MulAssign<&TensorRank0> for TensorList<N, T>
where
    T: Tensor,
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl<const N: usize, T> Add for TensorList<N, T>
where
    T: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_list: Self) -> Self::Output {
        self += tensor_list;
        self
    }
}

impl<const N: usize, T> Add<&Self> for TensorList<N, T>
where
    T: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_list: &Self) -> Self::Output {
        self += tensor_list;
        self
    }
}

impl<const N: usize, T> AddAssign for TensorList<N, T>
where
    T: Tensor,
{
    fn add_assign(&mut self, tensor_list: Self) {
        self.iter_mut()
            .zip(tensor_list.iter())
            .for_each(|(self_entry, entry)| *self_entry += entry);
    }
}

impl<const N: usize, T> AddAssign<&Self> for TensorList<N, T>
where
    T: Tensor,
{
    fn add_assign(&mut self, tensor_list: &Self) {
        self.iter_mut()
            .zip(tensor_list.iter())
            .for_each(|(self_entry, entry)| *self_entry += entry);
    }
}

impl<const N: usize, T> Sub for TensorList<N, T>
where
    T: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_list: Self) -> Self::Output {
        self -= tensor_list;
        self
    }
}

impl<const N: usize, T> Sub<&Self> for TensorList<N, T>
where
    T: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_list: &Self) -> Self::Output {
        self -= tensor_list;
        self
    }
}

impl<const N: usize, T> Sub<TensorList<N, T>> for &TensorList<N, T>
where
    T: Tensor,
{
    type Output = TensorList<N, T>;
    fn sub(self, tensor_list: TensorList<N, T>) -> Self::Output {
        todo!()
    }
}

impl<const N: usize, T> Sub for &TensorList<N, T>
where
    T: Tensor,
{
    type Output = TensorList<N, T>;
    fn sub(self, tensor_list: Self) -> Self::Output {
        todo!()
        // self
        //     .iter()
        //     .zip(tensor_list.iter())
        //     .map(|(self_entry, entry)| self_entry - entry)
        //     .collect()
    }
}

impl<const N: usize, T> SubAssign for TensorList<N, T>
where
    T: Tensor,
{
    fn sub_assign(&mut self, tensor_list: Self) {
        self.iter_mut()
            .zip(tensor_list.iter())
            .for_each(|(self_entry, entry)| *self_entry -= entry);
    }
}

impl<const N: usize, T> SubAssign<&Self> for TensorList<N, T>
where
    T: Tensor,
{
    fn sub_assign(&mut self, tensor_list: &Self) {
        self.iter_mut()
            .zip(tensor_list.iter())
            .for_each(|(self_entry, entry)| *self_entry -= entry);
    }
}
