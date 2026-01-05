#[cfg(test)]
mod test;

use crate::math::{Tensor, TensorRank0, TensorRank2, TensorRank2List, tensor::list::TensorList};
use std::ops::Mul;

#[cfg(test)]
use crate::math::tensor::test::ErrorTensor;

pub type TensorRank2List2D<
    const D: usize,
    const I: usize,
    const J: usize,
    const M: usize,
    const N: usize,
> = TensorList<TensorRank2List<D, I, J, M>, N>;

impl<const D: usize, const I: usize, const J: usize, const M: usize, const N: usize>
    From<[[[[TensorRank0; D]; D]; M]; N]> for TensorRank2List2D<D, I, J, M, N>
{
    fn from(array: [[[[TensorRank0; D]; D]; M]; N]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
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

#[cfg(test)]
impl<const D: usize, const I: usize, const J: usize, const W: usize, const X: usize> ErrorTensor
    for TensorRank2List2D<D, I, J, W, X>
{
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
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
                                        (self_ab_ij / comparator_ab_ij - 1.0).abs() >= epsilon
                                            && (self_ab_ij.abs() >= epsilon
                                                || comparator_ab_ij.abs() >= epsilon)
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
                                            (self_ab_ij / comparator_ab_ij - 1.0).abs() >= epsilon
                                                && (self_ab_ij - comparator_ab_ij).abs() >= epsilon
                                                && (self_ab_ij.abs() >= epsilon
                                                    || comparator_ab_ij.abs() >= epsilon)
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
