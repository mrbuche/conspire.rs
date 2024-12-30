#[cfg(test)]
pub mod test;

#[cfg(test)]
use super::super::test::ErrorTensor;

use super::{
    super::{Tensor, TensorArray},
    list::TensorRank3List,
    TensorRank0,
};
use std::{
    array::from_fn,
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A 2D list of *d*-dimensional tensors of rank 3.
///
/// `D` is the dimension, `I`, `J`, `K` are the configurations `W` and `X` are the list lengths.
#[derive(Debug)]
pub struct TensorRank3List2D<
    const D: usize,
    const I: usize,
    const J: usize,
    const K: usize,
    const W: usize,
    const X: usize,
>([TensorRank3List<D, I, J, K, W>; X]);

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > Display for TensorRank3List2D<D, I, J, K, W, X>
{
    fn fmt(&self, _f: &mut Formatter) -> Result {
        Ok(())
    }
}

#[cfg(test)]
impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > ErrorTensor for TensorRank3List2D<D, I, J, K, W, X>
{
    fn error(
        &self,
        comparator: &Self,
        tol_abs: &TensorRank0,
        tol_rel: &TensorRank0,
    ) -> Option<usize> {
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
                                    .map(|(self_ab_ij, comparator_ab_ij)| {
                                        self_ab_ij
                                            .iter()
                                            .zip(comparator_ab_ij.iter())
                                            .filter(|(&self_ab_ijk, &comparator_ab_ijk)| {
                                                &(self_ab_ijk - comparator_ab_ijk).abs() >= tol_abs
                                                    && &(self_ab_ijk / comparator_ab_ijk - 1.0)
                                                        .abs()
                                                        >= tol_rel
                                            })
                                            .count()
                                    })
                                    .sum::<usize>()
                            })
                            .sum::<usize>()
                    })
                    .sum::<usize>()
            })
            .sum();
        if error_count > 0 {
            Some(error_count)
        } else {
            None
        }
    }
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
                                    .map(|(self_ab_ij, comparator_ab_ij)| {
                                        self_ab_ij
                                            .iter()
                                            .zip(comparator_ab_ij.iter())
                                            .filter(|(&self_ab_ijk, &comparator_ab_ijk)| {
                                                &(self_ab_ijk / comparator_ab_ijk - 1.0).abs()
                                                    >= epsilon
                                                    && (&self_ab_ijk.abs() >= epsilon
                                                        || &comparator_ab_ijk.abs() >= epsilon)
                                            })
                                            .count()
                                    })
                                    .sum::<usize>()
                            })
                            .sum::<usize>()
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

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > Tensor for TensorRank3List2D<D, I, J, K, W, X>
{
    type Item = TensorRank3List<D, I, J, K, W>;
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
        const W: usize,
        const X: usize,
    > TensorArray for TensorRank3List2D<D, I, J, K, W, X>
{
    type Array = [[[[[TensorRank0; D]; D]; D]; W]; X];
    type Item = TensorRank3List<D, I, J, K, W>;
    fn as_array(&self) -> Self::Array {
        let mut array = [[[[[0.0; D]; D]; D]; W]; X];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry_rank_4_list, tensor_rank_4_list)| {
                *entry_rank_4_list = tensor_rank_4_list.as_array()
            });
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
        const W: usize,
        const X: usize,
    > FromIterator<TensorRank3List<D, I, J, K, W>> for TensorRank3List2D<D, I, J, K, W, X>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank3List<D, I, J, K, W>>>(
        into_iterator: Ii,
    ) -> Self {
        let mut tensor_rank_3_list_2d = Self::zero();
        tensor_rank_3_list_2d
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_3_list, entry)| *tensor_rank_3_list = entry);
        tensor_rank_3_list_2d
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > Index<usize> for TensorRank3List2D<D, I, J, K, W, X>
{
    type Output = TensorRank3List<D, I, J, K, W>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > IndexMut<usize> for TensorRank3List2D<D, I, J, K, W, X>
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
        const W: usize,
        const X: usize,
    > Add for TensorRank3List2D<D, I, J, K, W, X>
{
    type Output = Self;
    fn add(mut self, tensor_rank_3_list_2d: Self) -> Self::Output {
        self += tensor_rank_3_list_2d;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > Add<&Self> for TensorRank3List2D<D, I, J, K, W, X>
{
    type Output = Self;
    fn add(mut self, tensor_rank_3_list_2d: &Self) -> Self::Output {
        self += tensor_rank_3_list_2d;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > AddAssign for TensorRank3List2D<D, I, J, K, W, X>
{
    fn add_assign(&mut self, tensor_rank_3_list_2d: Self) {
        self.iter_mut()
            .zip(tensor_rank_3_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_3_list_2d)| *self_entry += tensor_rank_3_list_2d);
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > AddAssign<&Self> for TensorRank3List2D<D, I, J, K, W, X>
{
    fn add_assign(&mut self, tensor_rank_3_list_2d: &Self) {
        self.iter_mut()
            .zip(tensor_rank_3_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_3_list_2d)| *self_entry += tensor_rank_3_list_2d);
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > Div<TensorRank0> for TensorRank3List2D<D, I, J, K, W, X>
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
        const W: usize,
        const X: usize,
    > Div<&TensorRank0> for TensorRank3List2D<D, I, J, K, W, X>
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
        const W: usize,
        const X: usize,
    > DivAssign<TensorRank0> for TensorRank3List2D<D, I, J, K, W, X>
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
        const W: usize,
        const X: usize,
    > DivAssign<&TensorRank0> for TensorRank3List2D<D, I, J, K, W, X>
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
        const W: usize,
        const X: usize,
    > Mul<TensorRank0> for TensorRank3List2D<D, I, J, K, W, X>
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
        const W: usize,
        const X: usize,
    > Mul<&TensorRank0> for TensorRank3List2D<D, I, J, K, W, X>
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
        const W: usize,
        const X: usize,
    > MulAssign<TensorRank0> for TensorRank3List2D<D, I, J, K, W, X>
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
        const W: usize,
        const X: usize,
    > MulAssign<&TensorRank0> for TensorRank3List2D<D, I, J, K, W, X>
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
        const W: usize,
        const X: usize,
    > Sub for TensorRank3List2D<D, I, J, K, W, X>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_3_list_2d: Self) -> Self::Output {
        self -= tensor_rank_3_list_2d;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > Sub<&Self> for TensorRank3List2D<D, I, J, K, W, X>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_3_list_2d: &Self) -> Self::Output {
        self -= tensor_rank_3_list_2d;
        self
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > SubAssign for TensorRank3List2D<D, I, J, K, W, X>
{
    fn sub_assign(&mut self, tensor_rank_3_list_2d: Self) {
        self.iter_mut()
            .zip(tensor_rank_3_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_3_list)| *self_entry -= tensor_rank_3_list);
    }
}

impl<
        const D: usize,
        const I: usize,
        const J: usize,
        const K: usize,
        const W: usize,
        const X: usize,
    > SubAssign<&Self> for TensorRank3List2D<D, I, J, K, W, X>
{
    fn sub_assign(&mut self, tensor_rank_3_list_2d: &Self) {
        self.iter_mut()
            .zip(tensor_rank_3_list_2d.iter())
            .for_each(|(self_entry, tensor_rank_3_list)| *self_entry -= tensor_rank_3_list);
    }
}
