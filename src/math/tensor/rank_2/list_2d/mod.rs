#[cfg(test)]
mod test;

use crate::math::{Tensor, TensorRank0, TensorRank2, TensorRank2List, tensor::list::TensorList};
use crate::units::{Dimensionless, UnitMul};
use std::ops::Mul;

use crate::math::assert::FiniteDifference;

/// A list of lists of rank-2 tensors.
pub type TensorRank2List2D<
    const D: usize,
    I,
    J,
    const M: usize,
    const N: usize,
    U = Dimensionless,
> = TensorList<TensorRank2List<D, I, J, M, U>, N>;

impl<const D: usize, I, J, const M: usize, const N: usize, U> From<[[[[TensorRank0; D]; D]; M]; N]>
    for TensorRank2List2D<D, I, J, M, N, U>
{
    fn from(array: [[[[TensorRank0; D]; D]; M]; N]) -> Self {
        array.into_iter().map(|entry| entry.into()).collect()
    }
}

impl<const D: usize, I, J, K, const W: usize, const X: usize, U, V> Mul<TensorRank2<D, J, K, V>>
    for TensorRank2List2D<D, I, J, W, X, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2List2D<D, I, K, W, X, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2: TensorRank2<D, J, K, V>) -> Self::Output {
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

impl<const D: usize, I, J, K, const W: usize, const X: usize, U, V> Mul<&TensorRank2<D, J, K, V>>
    for TensorRank2List2D<D, I, J, W, X, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2List2D<D, I, K, W, X, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2: &TensorRank2<D, J, K, V>) -> Self::Output {
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

impl<const D: usize, I, J, const W: usize, const X: usize, U> FiniteDifference
    for TensorRank2List2D<D, I, J, W, X, U>
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
                                        self_ab_ij.differs(comparator_ab_ij, epsilon)
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
                                            self_ab_ij.differs_severely(comparator_ab_ij, epsilon)
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
