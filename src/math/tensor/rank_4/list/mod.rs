#[cfg(test)]
mod test;

use std::{
    array::from_fn,
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use super::{Tensor, TensorArray, TensorRank0, TensorRank4};

/// A list of *d*-dimensional tensor of rank 4.
///
/// `D` is the dimension, `I`, `J`, `K`, `L` are the configurations, `W` is the list length.
#[derive(Debug)]
pub struct TensorRank4List<
    const D: usize,
    const I: usize,
    const J: usize,
    const K: usize,
    const L: usize,
    const W: usize,
>([TensorRank4<D, I, J, K, L>; W]);

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Display for TensorRank4List<D, I, J, K, L, W>
{
    fn fmt(&self, _f: &mut Formatter) -> Result {
        Ok(())
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Tensor for TensorRank4List<D, I, J, K, L, W>
{
    type Item = TensorRank4<D, I, J, K, L>;
    fn copy(&self) -> Self {
        self.iter().map(|entry| entry.copy()).collect()
    }
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > TensorArray for TensorRank4List<D, I, J, K, L, W>
{
    type Array = [[[[[TensorRank0; D]; D]; D]; D]; W];
    type Item = TensorRank4<D, I, J, K, L>;
    fn as_array(&self) -> Self::Array {
        let mut array = [[[[[0.0; D]; D]; D]; D]; W];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry_rank_4, tensor_rank_4)| *entry_rank_4 = tensor_rank_4.as_array());
        array
    }
    fn identity() -> Self {
        Self(from_fn(|_| Self::Item::identity()))
    }
    fn new(array: Self::Array) -> Self {
        array
            .iter()
            .map(|array_i| Self::Item::new(*array_i))
            .collect()
    }
    fn zero() -> Self {
        Self(from_fn(|_| Self::Item::zero()))
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > FromIterator<TensorRank4<D, I, J, K, L>> for TensorRank4List<D, I, J, K, L, W>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank4<D, I, J, K, L>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_4_list = Self::zero();
        tensor_rank_4_list
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_4_list_entry, entry)| *tensor_rank_4_list_entry = entry);
        tensor_rank_4_list
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Index<usize> for TensorRank4List<D, I, J, K, L, W>
{
    type Output = TensorRank4<D, I, J, K, L>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > IndexMut<usize> for TensorRank4List<D, I, J, K, L, W>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Add for TensorRank4List<D, I, J, K, L, W>
{
    type Output = Self;
    fn add(mut self, tensor_rank_4: Self) -> Self::Output {
        self += tensor_rank_4;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Add<&Self> for TensorRank4List<D, I, J, K, L, W>
{
    type Output = Self;
    fn add(mut self, tensor_rank_4_list: &Self) -> Self::Output {
        self += tensor_rank_4_list;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > AddAssign for TensorRank4List<D, I, J, K, L, W>
{
    fn add_assign(&mut self, tensor_rank_4_list: Self) {
        self.iter_mut()
            .zip(tensor_rank_4_list.iter())
            .for_each(|(self_entry, tensor_rank_4)| *self_entry += tensor_rank_4);
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > AddAssign<&Self> for TensorRank4List<D, I, J, K, L, W>
{
    fn add_assign(&mut self, tensor_rank_4_list: &Self) {
        self.iter_mut()
            .zip(tensor_rank_4_list.iter())
            .for_each(|(self_entry, tensor_rank_4)| *self_entry += tensor_rank_4);
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Div<TensorRank0> for TensorRank4List<D, I, J, K, L, W>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Div<&TensorRank0> for TensorRank4List<D, I, J, K, L, W>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > DivAssign<TensorRank0> for TensorRank4List<D, I, J, K, L, W>
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > DivAssign<&TensorRank0> for TensorRank4List<D, I, J, K, L, W>
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Mul<TensorRank0> for TensorRank4List<D, I, J, K, L, W>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Mul<&TensorRank0> for TensorRank4List<D, I, J, K, L, W>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > MulAssign<TensorRank0> for TensorRank4List<D, I, J, K, L, W>
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > MulAssign<&TensorRank0> for TensorRank4List<D, I, J, K, L, W>
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Sub for TensorRank4List<D, I, J, K, L, W>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_4_list: Self) -> Self::Output {
        self -= tensor_rank_4_list;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > Sub<&Self> for TensorRank4List<D, I, J, K, L, W>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_4_list: &Self) -> Self::Output {
        self -= tensor_rank_4_list;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > SubAssign for TensorRank4List<D, I, J, K, L, W>
{
    fn sub_assign(&mut self, tensor_rank_4_list: Self) {
        self.iter_mut()
            .zip(tensor_rank_4_list.iter())
            .for_each(|(self_entry, tensor_rank_4)| *self_entry -= tensor_rank_4);
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const L: usize,
        const W: usize,
    > SubAssign<&Self> for TensorRank4List<D, I, J, K, L, W>
{
    fn sub_assign(&mut self, tensor_rank_4_list: &Self) {
        self.iter_mut()
            .zip(tensor_rank_4_list.iter())
            .for_each(|(self_entry, tensor_rank_4)| *self_entry -= tensor_rank_4);
    }
}
