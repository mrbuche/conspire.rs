use crate::math::{
    Hessian, SquareMatrix, Tensor, TensorRank0, TensorRank2, TensorRank2Vec,
    tensor::vec::TensorVector,
};
use std::ops::Mul;

#[cfg(test)]
use crate::math::tensor::test::ErrorTensor;

pub type TensorRank2Vec2D<const D: usize, const I: usize, const J: usize> =
    TensorVector<TensorRank2Vec<D, I, J>>;

impl<const D: usize, const I: usize, const J: usize> TensorRank2Vec2D<D, I, J> {
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| TensorRank2Vec::zero(len)).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> From<TensorRank2Vec2D<D, I, J>>
    for Vec<TensorRank0>
{
    fn from(tensor_rank_2_vec_2d: TensorRank2Vec2D<D, I, J>) -> Self {
        tensor_rank_2_vec_2d
            .into_iter()
            .flat_map(|tensor_rank_2_vec_1d| {
                tensor_rank_2_vec_1d.into_iter().flat_map(|tensor_rank_2| {
                    tensor_rank_2
                        .into_iter()
                        .flat_map(|tensor_rank_1| tensor_rank_1.into_iter())
                })
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Hessian for TensorRank2Vec2D<D, I, J> {
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.into_iter().enumerate().for_each(|(a, entry_a)| {
            entry_a.into_iter().enumerate().for_each(|(b, entry_ab)| {
                entry_ab
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, entry_ab_i)| {
                        entry_ab_i
                            .into_iter()
                            .enumerate()
                            .for_each(|(j, entry_ab_ij)| {
                                square_matrix[D * a + i][D * b + j] = entry_ab_ij
                            })
                    })
            })
        });
    }
    fn retain_from(self, retained: &[bool]) -> SquareMatrix {
        SquareMatrix::from(self)
            .into_iter()
            .zip(retained.iter())
            .filter(|(_, retained)| **retained)
            .map(|(self_i, _)| {
                self_i
                    .into_iter()
                    .zip(retained.iter())
                    .filter(|(_, retained)| **retained)
                    .map(|(self_ij, _)| self_ij)
                    .collect()
            })
            .collect()
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

#[cfg(test)]
impl<const D: usize, const I: usize, const J: usize> ErrorTensor for TensorRank2Vec2D<D, I, J> {
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
