#[cfg(test)]
mod test;

#[cfg(test)]
use super::super::test::ErrorTensor;

use std::{
    array::from_fn,
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use super::{Tensor, TensorArray, TensorRank0, TensorRank2};

/// A list of *d*-dimensional tensors of rank 2.
///
/// `D` is the dimension, `I`, `J` are the configurations `W` is the list length.
#[derive(Clone, Debug)]
pub struct TensorRank2List<const D: usize, const I: usize, const J: usize, const W: usize>(
    [TensorRank2<D, I, J>; W],
);

pub const fn tensor_rank_2_list<const D: usize, const I: usize, const J: usize, const W: usize>(
    array: [TensorRank2<D, I, J>; W],
) -> TensorRank2List<D, I, J, W> {
    TensorRank2List(array)
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Display
    for TensorRank2List<D, I, J, W>
{
    fn fmt(&self, _f: &mut Formatter) -> Result {
        Ok(())
    }
}

#[cfg(test)]
impl<const D: usize, const I: usize, const J: usize, const W: usize> ErrorTensor
    for TensorRank2List<D, I, J, W>
{
    fn error_fd(&self, comparator: &Self, epsilon: &TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_a, comparator_a)| {
                self_a
                    .iter()
                    .zip(comparator_a.iter())
                    .map(|(self_a_i, comparator_a_i)| {
                        self_a_i
                            .iter()
                            .zip(comparator_a_i.iter())
                            .filter(|&(&self_a_ij, &comparator_a_ij)| {
                                &(self_a_ij / comparator_a_ij - 1.0).abs() >= epsilon
                                    && (&self_a_ij.abs() >= epsilon
                                        || &comparator_a_ij.abs() >= epsilon)
                            })
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum();
        if error_count > 0 {
            Some((true, error_count))
        } else {
            None
        }
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Tensor
    for TensorRank2List<D, I, J, W>
{
    type Item = TensorRank2<D, I, J>;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> TensorArray
    for TensorRank2List<D, I, J, W>
{
    type Array = [[[TensorRank0; D]; D]; W];
    type Item = TensorRank2<D, I, J>;
    fn as_array(&self) -> Self::Array {
        let mut array = [[[0.0; D]; D]; W];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry_rank_2, tensor_rank_2)| *entry_rank_2 = tensor_rank_2.as_array());
        array
    }
    fn identity() -> Self {
        Self(from_fn(|_| Self::Item::identity()))
    }
    fn new(array: Self::Array) -> Self {
        array.into_iter().map(TensorRank2::new).collect()
    }
    fn zero() -> Self {
        Self(from_fn(|_| TensorRank2::zero()))
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize>
    FromIterator<TensorRank2<D, I, J>> for TensorRank2List<D, I, J, W>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank2<D, I, J>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_2_list = Self::zero();
        tensor_rank_2_list
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_2_list_entry, entry)| *tensor_rank_2_list_entry = entry);
        tensor_rank_2_list
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Index<usize>
    for TensorRank2List<D, I, J, W>
{
    type Output = TensorRank2<D, I, J>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> IndexMut<usize>
    for TensorRank2List<D, I, J, W>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Add
    for TensorRank2List<D, I, J, W>
{
    type Output = Self;
    fn add(mut self, tensor_rank_2_list: Self) -> Self::Output {
        self += tensor_rank_2_list;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Add<&Self>
    for TensorRank2List<D, I, J, W>
{
    type Output = Self;
    fn add(mut self, tensor_rank_2_list: &Self) -> Self::Output {
        self += tensor_rank_2_list;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> AddAssign
    for TensorRank2List<D, I, J, W>
{
    fn add_assign(&mut self, tensor_rank_2_list: Self) {
        self.iter_mut()
            .zip(tensor_rank_2_list.iter())
            .for_each(|(self_entry, tensor_rank_2)| *self_entry += tensor_rank_2);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> AddAssign<&Self>
    for TensorRank2List<D, I, J, W>
{
    fn add_assign(&mut self, tensor_rank_2_list: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2_list.iter())
            .for_each(|(self_entry, tensor_rank_2)| *self_entry += tensor_rank_2);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Div<TensorRank0>
    for TensorRank2List<D, I, J, W>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Div<&TensorRank0>
    for TensorRank2List<D, I, J, W>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> DivAssign<TensorRank0>
    for TensorRank2List<D, I, J, W>
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> DivAssign<&TensorRank0>
    for TensorRank2List<D, I, J, W>
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<TensorRank0>
    for TensorRank2List<D, I, J, W>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Mul<&TensorRank0>
    for TensorRank2List<D, I, J, W>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> MulAssign<TensorRank0>
    for TensorRank2List<D, I, J, W>
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> MulAssign<&TensorRank0>
    for TensorRank2List<D, I, J, W>
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Sub
    for TensorRank2List<D, I, J, W>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_2_list: Self) -> Self::Output {
        self -= tensor_rank_2_list;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Sub<&Self>
    for TensorRank2List<D, I, J, W>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_2_list: &Self) -> Self::Output {
        self -= tensor_rank_2_list;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> Sub
    for &TensorRank2List<D, I, J, W>
{
    type Output = TensorRank2List<D, I, J, W>;
    fn sub(self, tensor_rank_2_list: Self) -> Self::Output {
        tensor_rank_2_list
            .iter()
            .zip(self.iter())
            .map(|(tensor_rank_2_list_a, self_a)| self_a - tensor_rank_2_list_a)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> SubAssign
    for TensorRank2List<D, I, J, W>
{
    fn sub_assign(&mut self, tensor_rank_2_list: Self) {
        self.iter_mut()
            .zip(tensor_rank_2_list.iter())
            .for_each(|(self_entry, tensor_rank_2)| *self_entry -= tensor_rank_2);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize> SubAssign<&Self>
    for TensorRank2List<D, I, J, W>
{
    fn sub_assign(&mut self, tensor_rank_2_list: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2_list.iter())
            .for_each(|(self_entry, tensor_rank_2)| *self_entry -= tensor_rank_2);
    }
}
