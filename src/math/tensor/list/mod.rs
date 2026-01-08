use crate::math::{Tensor, TensorArray, TensorRank0};
use std::{
    array::{self, from_fn},
    fmt::{Display, Formatter, Result},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    slice,
};

#[derive(Clone, Debug)]
pub struct TensorList<T, const N: usize>([T; N])
where
    T: Tensor;

impl<T, const N: usize> Default for TensorList<T, N>
where
    T: Tensor,
{
    fn default() -> Self {
        Self(from_fn(|_| T::default()))
    }
}

impl<T, const N: usize> From<[T; N]> for TensorList<T, N>
where
    T: Tensor,
{
    fn from(tensor_array: [T; N]) -> Self {
        Self(tensor_array)
    }
}

impl<T, const N: usize> From<TensorList<T, N>> for [T; N]
where
    T: Tensor,
{
    fn from(tensor_list: TensorList<T, N>) -> Self {
        tensor_list.0
    }
}

impl<T, const N: usize> Display for TensorList<T, N>
where
    T: Tensor,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Need to implement Display")
        // write!(f, "\x1B[s")?;
        // write!(f, "[[")?;
        // self.iter().enumerate().try_for_each(|(i, tensor_rank_1)| {
        //     tensor_rank_1
        //         .iter()
        //         .try_for_each(|entry| write_tensor_rank_0(f, entry))?;
        //     if i + 1 < W {
        //         writeln!(f, "\x1B[2D],")?;
        //         write!(f, "\x1B[u")?;
        //         write!(f, "\x1B[{}B [", i + 1)?;
        //     }
        //     Ok(())
        // })?;
        // write!(f, "\x1B[2D]]")
    }
}

impl<T, const N: usize> Index<usize> for TensorList<T, N>
where
    T: Tensor,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for TensorList<T, N>
where
    T: Tensor,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T, const N: usize> Tensor for TensorList<T, N>
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
    fn len(&self) -> usize {
        N
    }
    fn size(&self) -> usize {
        N * self[0].size() // fine if T is TensorArray
    }
}

impl<T, const N: usize> FromIterator<T> for TensorList<T, N>
where
    T: Tensor,
{
    fn from_iter<Ii: IntoIterator<Item = T>>(into_iterator: Ii) -> Self {
        let mut tensor_list = Self::default();
        tensor_list
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_list_entry, entry)| *tensor_list_entry = entry);
        tensor_list
    }
}

impl<T, const N: usize> IntoIterator for TensorList<T, N>
where
    T: Tensor,
{
    type Item = T;
    type IntoIter = array::IntoIter<Self::Item, N>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a TensorList<T, N>
where
    T: Tensor,
{
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T, const N: usize> Sum for TensorList<T, N>
where
    T: Tensor,
{
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        iter.reduce(|mut acc, item| {
            acc += item;
            acc
        })
        .unwrap_or_else(Self::default)
    }
}

impl<T, const N: usize> TensorArray for TensorList<T, N>
where
    T: Tensor + TensorArray,
{
    type Array = [T::Array; N];
    type Item = T;
    fn as_array(&self) -> Self::Array {
        from_fn(|i| self[i].as_array())
    }
    fn identity() -> Self {
        Self(from_fn(|_| Self::Item::identity()))
    }
    fn zero() -> Self {
        Self(from_fn(|_| Self::Item::zero()))
    }
}

impl<T, const N: usize> Div<TensorRank0> for TensorList<T, N>
where
    T: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<T, const N: usize> Div<&TensorRank0> for TensorList<T, N>
where
    T: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<T, const N: usize> DivAssign<TensorRank0> for TensorList<T, N>
where
    T: Tensor,
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl<T, const N: usize> DivAssign<&TensorRank0> for TensorList<T, N>
where
    T: Tensor,
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl<T, const N: usize> Mul<TensorRank0> for TensorList<T, N>
where
    T: Tensor,
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<T, const N: usize> Mul<&TensorRank0> for TensorList<T, N>
where
    T: Tensor,
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<T, const N: usize> Mul<TensorRank0> for &TensorList<T, N>
where
    T: Tensor,
    for<'a> &'a T: Mul<&'a TensorRank0, Output = T>,
{
    type Output = TensorList<T, N>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * &tensor_rank_0).collect()
    }
}

impl<T, const N: usize> Mul<&TensorRank0> for &TensorList<T, N>
where
    T: Tensor,
{
    type Output = TensorList<T, N>;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        //
        // Cloning for now to avoid trait recursion nightmare.
        //
        self.clone() * tensor_rank_0
    }
}

impl<T, const N: usize> MulAssign<TensorRank0> for TensorList<T, N>
where
    T: Tensor,
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl<T, const N: usize> MulAssign<&TensorRank0> for TensorList<T, N>
where
    T: Tensor,
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl<T, const N: usize> Add for TensorList<T, N>
where
    T: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_list: Self) -> Self::Output {
        self += tensor_list;
        self
    }
}

impl<T, const N: usize> Add<&Self> for TensorList<T, N>
where
    T: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_list: &Self) -> Self::Output {
        self += tensor_list;
        self
    }
}

impl<T, const N: usize> AddAssign for TensorList<T, N>
where
    T: Tensor,
{
    fn add_assign(&mut self, tensor_list: Self) {
        self.iter_mut()
            .zip(tensor_list)
            .for_each(|(self_entry, entry)| *self_entry += entry);
    }
}

impl<T, const N: usize> AddAssign<&Self> for TensorList<T, N>
where
    T: Tensor,
{
    fn add_assign(&mut self, tensor_list: &Self) {
        self.iter_mut()
            .zip(tensor_list.iter())
            .for_each(|(self_entry, entry)| *self_entry += entry);
    }
}

impl<T, const N: usize> Sub for TensorList<T, N>
where
    T: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_list: Self) -> Self::Output {
        self -= tensor_list;
        self
    }
}

impl<T, const N: usize> Sub<&Self> for TensorList<T, N>
where
    T: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_list: &Self) -> Self::Output {
        self -= tensor_list;
        self
    }
}

impl<T, const N: usize> SubAssign for TensorList<T, N>
where
    T: Tensor,
{
    fn sub_assign(&mut self, tensor_list: Self) {
        self.iter_mut()
            .zip(tensor_list)
            .for_each(|(self_entry, entry)| *self_entry -= entry);
    }
}

impl<T, const N: usize> SubAssign<&Self> for TensorList<T, N>
where
    T: Tensor,
{
    fn sub_assign(&mut self, tensor_list: &Self) {
        self.iter_mut()
            .zip(tensor_list.iter())
            .for_each(|(self_entry, entry)| *self_entry -= entry);
    }
}
