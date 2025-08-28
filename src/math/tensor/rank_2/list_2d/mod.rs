#[cfg(test)]
mod test;

#[cfg(test)]
use super::super::test::ErrorTensor;

use std::{
    array::from_fn,
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use super::{
    super::{Tensor, TensorArray},
    TensorRank0, TensorRank2,
    list::TensorRank2List,
};

/// A 2D list of *d*-dimensional tensors of rank 2.
///
/// `D` is the dimension, `I`, `J` are the configurations, `W` and `X` are the list lengths.
#[derive(Clone, Debug)]
pub struct TensorRank2List2D<
    const D: usize,
    const I: usize,
    const J: usize,
    const W: usize,
    const X: usize,
>([TensorRank2List<D, I, J, W>; X]);

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> Display
    for TensorRank2List2D<D, I, J, W, X>
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "[")?;
        self.iter()
            .enumerate()
            .try_for_each(|(i, entry)| write!(f, "{entry},\n\x1B[u\x1B[{}B\x1B[2D", i + 1))?;
        write!(f, "\x1B[u\x1B[1A\x1B[{}C]", 16 * D + 2)
    }
}

#[cfg(test)]
impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> ErrorTensor
    for TensorRank2List2D<D, I, J, W, X>
{
    fn error_fd(&self, comparator: &Self, epsilon: &TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_a, comparator_a)| {
                self_a
                    .iter()
                    .zip(comparator_a.iter())
                    .map(|(self_ab, comparator_ab)| {
                        self_ab
                            .iter()
                            .zip(comparator_ab.iter())
                            .map(|(self_ab_i, comparator_ab_i)| {
                                self_ab_i
                                    .iter()
                                    .zip(comparator_ab_i.iter())
                                    .filter(|&(&self_ab_ij, &comparator_ab_ij)| {
                                        &(self_ab_ij / comparator_ab_ij - 1.0).abs() >= epsilon
                                            && (&self_ab_ij.abs() >= epsilon
                                                || &comparator_ab_ij.abs() >= epsilon)
                                    })
                                    .count()
                            })
                            .sum::<usize>()
                    })
                    .sum::<usize>()
            })
            .sum();
        if error_count > 0 {
            let auxiliary = self
                .iter()
                .zip(comparator.iter())
                .map(|(self_a, comparator_a)| {
                    self_a
                        .iter()
                        .zip(comparator_a.iter())
                        .map(|(self_ab, comparator_ab)| {
                            self_ab
                                .iter()
                                .zip(comparator_ab.iter())
                                .map(|(self_ab_i, comparator_ab_i)| {
                                    self_ab_i
                                        .iter()
                                        .zip(comparator_ab_i.iter())
                                        .filter(|&(&self_ab_ij, &comparator_ab_ij)| {
                                            &(self_ab_ij / comparator_ab_ij - 1.0).abs() >= epsilon
                                                && &(self_ab_ij - comparator_ab_ij).abs() >= epsilon
                                                && (&self_ab_ij.abs() >= epsilon
                                                    || &comparator_ab_ij.abs() >= epsilon)
                                        })
                                        .count()
                                })
                                .sum::<usize>()
                        })
                        .sum::<usize>()
                })
                .sum::<usize>()
                > 0;
            Some((auxiliary, error_count))
        } else {
            None
        }
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> Tensor
    for TensorRank2List2D<D, I, J, W, X>
{
    type Item = TensorRank2List<D, I, J, W>;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> TensorArray
    for TensorRank2List2D<D, I, J, W, X>
{
    type Array = [[[[TensorRank0; D]; D]; W]; X];
    type Item = TensorRank2List<D, I, J, W>;
    fn as_array(&self) -> Self::Array {
        let mut array = [[[[0.0; D]; D]; W]; X];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry_rank_2_list, tensor_rank_2_list)| {
                *entry_rank_2_list = tensor_rank_2_list.as_array()
            });
        array
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

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    FromIterator<TensorRank2List<D, I, J, W>> for TensorRank2List2D<D, I, J, W, X>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank2List<D, I, J, W>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_2_list_2d = Self::zero();
        tensor_rank_2_list_2d
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_2_list, entry)| *tensor_rank_2_list = entry);
        tensor_rank_2_list_2d
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> Index<usize>
    for TensorRank2List2D<D, I, J, W, X>
{
    type Output = TensorRank2List<D, I, J, W>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> IndexMut<usize>
    for TensorRank2List2D<D, I, J, W, X>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> std::iter::Sum
    for TensorRank2List2D<D, I, J, W, X>
{
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        let mut output = TensorRank2List2D::zero();
        iter.for_each(|item| output += item);
        output
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const W: usize, const X: usize>
    Mul<TensorRank2<D, J, K>> for TensorRank2List2D<D, I, J, W, X>
{
    type Output = TensorRank2List2D<D, I, K, W, X>;
    fn mul(self, tensor_rank_2: TensorRank2<D, J, K>) -> Self::Output {
        self.iter()
            .map(|self_entry| {
                self_entry
                    .iter()
                    .map(|self_tensor_rank_2| self_tensor_rank_2 * &tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const W: usize, const X: usize>
    Mul<&TensorRank2<D, J, K>> for TensorRank2List2D<D, I, J, W, X>
{
    type Output = TensorRank2List2D<D, I, K, W, X>;
    fn mul(self, tensor_rank_2: &TensorRank2<D, J, K>) -> Self::Output {
        self.iter()
            .map(|self_entry| {
                self_entry
                    .iter()
                    .map(|self_tensor_rank_2| self_tensor_rank_2 * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> Add
    for TensorRank2List2D<D, I, J, W, X>
{
    type Output = Self;
    fn add(mut self, tensor_rank_2_list_2d: Self) -> Self::Output {
        self += tensor_rank_2_list_2d;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> Add<&Self>
    for TensorRank2List2D<D, I, J, W, X>
{
    type Output = Self;
    fn add(mut self, tensor_rank_2_list_2d: &Self) -> Self::Output {
        self += tensor_rank_2_list_2d;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> AddAssign
    for TensorRank2List2D<D, I, J, W, X>
{
    fn add_assign(&mut self, tensor_rank_2_list_2d: Self) {
        self.iter_mut()
            .zip(tensor_rank_2_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_2_list)| *self_entry += tensor_rank_2_list);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    AddAssign<&Self> for TensorRank2List2D<D, I, J, W, X>
{
    fn add_assign(&mut self, tensor_rank_2_list_2d: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_2_list)| *self_entry += tensor_rank_2_list);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    Div<TensorRank0> for TensorRank2List2D<D, I, J, W, X>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    Div<&TensorRank0> for TensorRank2List2D<D, I, J, W, X>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    DivAssign<TensorRank0> for TensorRank2List2D<D, I, J, W, X>
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    DivAssign<&TensorRank0> for TensorRank2List2D<D, I, J, W, X>
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    Mul<TensorRank0> for TensorRank2List2D<D, I, J, W, X>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    Mul<&TensorRank0> for TensorRank2List2D<D, I, J, W, X>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    MulAssign<TensorRank0> for TensorRank2List2D<D, I, J, W, X>
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    MulAssign<&TensorRank0> for TensorRank2List2D<D, I, J, W, X>
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> Sub
    for TensorRank2List2D<D, I, J, W, X>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_2_list_2d: Self) -> Self::Output {
        self -= tensor_rank_2_list_2d;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> Sub<&Self>
    for TensorRank2List2D<D, I, J, W, X>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_2_list_2d: &Self) -> Self::Output {
        self -= tensor_rank_2_list_2d;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> Sub
    for &TensorRank2List2D<D, I, J, W, X>
{
    type Output = TensorRank2List2D<D, I, J, W, X>;
    fn sub(self, tensor_rank_2_list_2d: Self) -> Self::Output {
        tensor_rank_2_list_2d
            .iter()
            .zip(self.iter())
            .map(|(tensor_rank_2_list_2d_a, self_a)| self_a - tensor_rank_2_list_2d_a)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> SubAssign
    for TensorRank2List2D<D, I, J, W, X>
{
    fn sub_assign(&mut self, tensor_rank_2_list_2d: Self) {
        self.iter_mut()
            .zip(tensor_rank_2_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_2_list)| *self_entry -= tensor_rank_2_list);
    }
}

impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize>
    SubAssign<&Self> for TensorRank2List2D<D, I, J, W, X>
{
    fn sub_assign(&mut self, tensor_rank_2_list_2d: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_2_list)| *self_entry -= tensor_rank_2_list);
    }
}
