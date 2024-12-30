#[cfg(test)]
use super::super::test::ErrorTensor;

use crate::math::{Tensor, TensorRank0, TensorRank2, TensorRank2Vec, TensorVec};
use std::{
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A 2D vector of *d*-dimensional tensors of rank 2.
///
/// `D` is the dimension, `I`, `J` are the configurations.
#[derive(Debug)]
pub struct TensorRank2Vec2D<const D: usize, const I: usize, const J: usize>(
    pub Vec<TensorRank2Vec<D, I, J>>,
);

impl<const D: usize, const I: usize, const J: usize> Display for TensorRank2Vec2D<D, I, J> {
    fn fmt(&self, _f: &mut Formatter) -> Result {
        Ok(())
    }
}

#[cfg(test)]
impl<const D: usize, const I: usize, const J: usize> ErrorTensor for TensorRank2Vec2D<D, I, J> {
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
                                    .filter(|(&self_ab_ij, &comparator_ab_ij)| {
                                        &(self_ab_ij - comparator_ab_ij).abs() >= tol_abs
                                            && &(self_ab_ij / comparator_ab_ij - 1.0).abs()
                                                >= tol_rel
                                    })
                                    .count()
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
                                    .filter(|(&self_ab_ij, &comparator_ab_ij)| {
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
            let auxillary = self
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
                                        .filter(|(&self_ab_ij, &comparator_ab_ij)| {
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
            Some((auxillary, error_count))
        } else {
            None
        }
    }
}

impl<const D: usize, const I: usize, const J: usize> FromIterator<TensorRank2Vec<D, I, J>>
    for TensorRank2Vec2D<D, I, J>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank2Vec<D, I, J>>>(into_iterator: Ii) -> Self {
        Self(Vec::from_iter(into_iterator))
    }
}

impl<const D: usize, const I: usize, const J: usize> Index<usize> for TensorRank2Vec2D<D, I, J> {
    type Output = TensorRank2Vec<D, I, J>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize> IndexMut<usize> for TensorRank2Vec2D<D, I, J> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<'a, const D: usize, const I: usize, const J: usize> TensorVec<'a>
    for TensorRank2Vec2D<D, I, J>
{
    type Item = TensorRank2Vec<D, I, J>;
    type Slice = &'a [&'a [[[TensorRank0; D]; D]]];
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn new(slice: Self::Slice) -> Self {
        slice
            .iter()
            .map(|slice_entry| Self::Item::new(slice_entry))
            .collect()
    }
    fn zero(len: usize) -> Self {
        (0..len).map(|_| Self::Item::zero(len)).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Tensor for TensorRank2Vec2D<D, I, J> {
    type Item = TensorRank2Vec<D, I, J>;
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

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<TensorRank2<D, J, K>>
    for TensorRank2Vec2D<D, I, J>
{
    type Output = TensorRank2Vec2D<D, I, K>;
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

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<&TensorRank2<D, J, K>>
    for TensorRank2Vec2D<D, I, J>
{
    type Output = TensorRank2Vec2D<D, I, K>;
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

impl<const D: usize, const I: usize, const J: usize> Add for TensorRank2Vec2D<D, I, J> {
    type Output = Self;
    fn add(mut self, tensor_rank_2_vec_2d: Self) -> Self::Output {
        self += tensor_rank_2_vec_2d;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Add<&Self> for TensorRank2Vec2D<D, I, J> {
    type Output = Self;
    fn add(mut self, tensor_rank_2_vec_2d: &Self) -> Self::Output {
        self += tensor_rank_2_vec_2d;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> AddAssign for TensorRank2Vec2D<D, I, J> {
    fn add_assign(&mut self, tensor_rank_2_vec_2d: Self) {
        self.iter_mut()
            .zip(tensor_rank_2_vec_2d.iter())
            .for_each(|(self_entry, tensor_rank_2_vec)| *self_entry += tensor_rank_2_vec);
    }
}

impl<const D: usize, const I: usize, const J: usize> AddAssign<&Self>
    for TensorRank2Vec2D<D, I, J>
{
    fn add_assign(&mut self, tensor_rank_2_vec_2d: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2_vec_2d.iter())
            .for_each(|(self_entry, tensor_rank_2_vec)| *self_entry += tensor_rank_2_vec);
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank0>
    for TensorRank2Vec2D<D, I, J>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<&TensorRank0>
    for TensorRank2Vec2D<D, I, J>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> DivAssign<TensorRank0>
    for TensorRank2Vec2D<D, I, J>
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> DivAssign<&TensorRank0>
    for TensorRank2Vec2D<D, I, J>
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry /= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank0>
    for TensorRank2Vec2D<D, I, J>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank0>
    for TensorRank2Vec2D<D, I, J>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<TensorRank0>
    for TensorRank2Vec2D<D, I, J>
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<&TensorRank0>
    for TensorRank2Vec2D<D, I, J>
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|entry| *entry *= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub for TensorRank2Vec2D<D, I, J> {
    type Output = Self;
    fn sub(mut self, tensor_rank_2_vec_2d: Self) -> Self::Output {
        self -= tensor_rank_2_vec_2d;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub<&Self> for TensorRank2Vec2D<D, I, J> {
    type Output = Self;
    fn sub(mut self, tensor_rank_2_vec_2d: &Self) -> Self::Output {
        self -= tensor_rank_2_vec_2d;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> SubAssign for TensorRank2Vec2D<D, I, J> {
    fn sub_assign(&mut self, tensor_rank_2_vec_2d: Self) {
        self.iter_mut()
            .zip(tensor_rank_2_vec_2d.iter())
            .for_each(|(self_entry, tensor_rank_2_vec)| *self_entry -= tensor_rank_2_vec);
    }
}

impl<const D: usize, const I: usize, const J: usize> SubAssign<&Self>
    for TensorRank2Vec2D<D, I, J>
{
    fn sub_assign(&mut self, tensor_rank_2_vec_2d: &Self) {
        self.iter_mut()
            .zip(tensor_rank_2_vec_2d.iter())
            .for_each(|(self_entry, tensor_rank_2_vec)| *self_entry -= tensor_rank_2_vec);
    }
}
