#[cfg(test)]
mod test;

use super::{
    super::{Tensor, TensorArray},
    list::TensorRank1List,
    TensorRank0,
};
use std::array::from_fn;
use std::{
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A 2D list of *d*-dimensional tensors of rank 1.
///
/// `D` is the dimension, `I` is the configuration, `W` and `X` are the list lengths.
#[derive(Debug)]
pub struct TensorRank1List2D<const D: usize, const I: usize, const W: usize, const X: usize>(
    pub [TensorRank1List<D, I, W>; X],
);

impl<const D: usize, const I: usize, const W: usize, const X: usize> Display
    for TensorRank1List2D<D, I, W, X>
{
    fn fmt(&self, _f: &mut Formatter) -> Result {
        Ok(())
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> Tensor
    for TensorRank1List2D<D, I, W, X>
{
    type Item = TensorRank1List<D, I, W>;
    fn copy(&self) -> Self {
        self.iter().map(|entry| entry.copy()).collect()
    }
    fn iter(&self) -> impl Iterator<Item = &TensorRank1List<D, I, W>> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> TensorArray
    for TensorRank1List2D<D, I, W, X>
{
    type Array = [[[TensorRank0; D]; W]; X];
    type Item = TensorRank1List<D, I, W>;
    fn as_array(&self) -> Self::Array {
        let mut array = [[[0.0; D]; W]; X];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry, tensor_rank_1_list)| *entry = tensor_rank_1_list.as_array());
        array
    }
    fn identity() -> Self {
        Self(from_fn(|_| Self::Item::identity()))
    }
    fn new(array: Self::Array) -> Self {
        array
            .iter()
            .map(|array_i| TensorRank1List::new(*array_i))
            .collect()
    }
    fn zero() -> Self {
        Self(from_fn(|_| TensorRank1List::zero()))
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize>
    FromIterator<TensorRank1List<D, I, W>> for TensorRank1List2D<D, I, W, X>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank1List<D, I, W>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_1_list_2d = Self::zero();
        tensor_rank_1_list_2d
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_1_list, entry)| *tensor_rank_1_list = entry);
        tensor_rank_1_list_2d
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> Index<usize>
    for TensorRank1List2D<D, I, W, X>
{
    type Output = TensorRank1List<D, I, W>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> IndexMut<usize>
    for TensorRank1List2D<D, I, W, X>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> Add
    for TensorRank1List2D<D, I, W, X>
{
    type Output = Self;
    fn add(mut self, tensor_rank_1_list_2d: Self) -> Self::Output {
        self += tensor_rank_1_list_2d;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> Add<&Self>
    for TensorRank1List2D<D, I, W, X>
{
    type Output = Self;
    fn add(mut self, tensor_rank_1_list_2d: &Self) -> Self::Output {
        self += tensor_rank_1_list_2d;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> AddAssign
    for TensorRank1List2D<D, I, W, X>
{
    fn add_assign(&mut self, tensor_rank_1_list_2d: Self) {
        self.iter_mut()
            .zip(tensor_rank_1_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_1_list)| *self_entry += tensor_rank_1_list);
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> AddAssign<&Self>
    for TensorRank1List2D<D, I, W, X>
{
    fn add_assign(&mut self, tensor_rank_1_list_2d: &Self) {
        self.iter_mut()
            .zip(tensor_rank_1_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_1_list)| *self_entry += tensor_rank_1_list);
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> Div<TensorRank0>
    for TensorRank1List2D<D, I, W, X>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> Div<&TensorRank0>
    for TensorRank1List2D<D, I, W, X>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> DivAssign<TensorRank0>
    for TensorRank1List2D<D, I, W, X>
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> DivAssign<&TensorRank0>
    for TensorRank1List2D<D, I, W, X>
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> Mul<TensorRank0>
    for TensorRank1List2D<D, I, W, X>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> Mul<&TensorRank0>
    for TensorRank1List2D<D, I, W, X>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> MulAssign<TensorRank0>
    for TensorRank1List2D<D, I, W, X>
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> MulAssign<&TensorRank0>
    for TensorRank1List2D<D, I, W, X>
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> Sub
    for TensorRank1List2D<D, I, W, X>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_1_list_2d: Self) -> Self::Output {
        self -= tensor_rank_1_list_2d;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> Sub<&Self>
    for TensorRank1List2D<D, I, W, X>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_1_list_2d: &Self) -> Self::Output {
        self -= tensor_rank_1_list_2d;
        self
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> SubAssign
    for TensorRank1List2D<D, I, W, X>
{
    fn sub_assign(&mut self, tensor_rank_1_list_2d: Self) {
        self.iter_mut()
            .zip(tensor_rank_1_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_1_list)| *self_entry -= tensor_rank_1_list);
    }
}

impl<const D: usize, const I: usize, const W: usize, const X: usize> SubAssign<&Self>
    for TensorRank1List2D<D, I, W, X>
{
    fn sub_assign(&mut self, tensor_rank_1_list_2d: &Self) {
        self.iter_mut()
            .zip(tensor_rank_1_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_1_list)| *self_entry -= tensor_rank_1_list);
    }
}
