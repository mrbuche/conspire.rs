#[cfg(test)]
mod test;

use super::TensorRank2;
use crate::math::unit::{Dimensionless, UnitMul};
use crate::math::{
    Hessian, HessianAccumulate, Rank2, Scalar, SquareMatrix, Tensor, Vector,
    tensor::vec::TensorVector,
};
use std::ops::Mul;

use super::sparse_vec::TensorRank2SparseVec;

use crate::math::{TensorArray, TensorRank0, assert::FiniteDifference};

/// A vector of sparse vectors of rank-2 tensors, storing only inserted entries.
pub type TensorRank2SparseVec2D<const D: usize, I, J, U = Dimensionless> =
    TensorVector<TensorRank2SparseVec<D, I, J, U>>;

impl<const D: usize, I, J, U> TensorRank2SparseVec2D<D, I, J, U> {
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| TensorRank2SparseVec::default()).collect()
    }
}

impl<const D: usize, I, U> HessianAccumulate<D, I, U> for TensorRank2SparseVec2D<D, I, I, U> {
    fn accumulate(&mut self, a: usize, b: usize, block: TensorRank2<D, I, I, U>) {
        if a == b {
            self[a][b] += block;
        } else {
            self[b][a] += block.transpose();
            self[a][b] += block;
        }
    }
}

impl<const D: usize, I, J, K, U, V> Mul<TensorRank2SparseVec2D<D, J, K, V>>
    for TensorRank2<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2SparseVec2D<D, I, K, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2_sparse_vec_2d: TensorRank2SparseVec2D<D, J, K, V>) -> Self::Output {
        tensor_rank_2_sparse_vec_2d
            .into_iter()
            .map(|row| {
                TensorRank2SparseVec(
                    row.0
                        .into_iter()
                        .map(|(column, block)| (column, &self * block))
                        .collect(),
                )
            })
            .collect()
    }
}

impl<const D: usize, I, J, K, U, V> Mul<TensorRank2<D, J, K, V>>
    for TensorRank2SparseVec2D<D, I, J, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank2SparseVec2D<D, I, K, <U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_2: TensorRank2<D, J, K, V>) -> Self::Output {
        self.into_iter()
            .map(|row| {
                TensorRank2SparseVec(
                    row.0
                        .into_iter()
                        .map(|(column, block)| (column, block * &tensor_rank_2))
                        .collect(),
                )
            })
            .collect()
    }
}

impl<const D: usize, I, J, U> Hessian for TensorRank2SparseVec2D<D, I, J, U> {
    fn quadratic_form(&self, vector: &Vector) -> Scalar {
        self.iter()
            .enumerate()
            .map(|(a, row)| {
                row.entries()
                    .map(|(b, block)| {
                        block
                            .iter()
                            .enumerate()
                            .map(|(i, block_i)| {
                                block_i
                                    .iter()
                                    .enumerate()
                                    .map(|(j, block_ij)| {
                                        block_ij.value() * vector[D * a + i] * vector[D * b + j]
                                    })
                                    .sum::<Scalar>()
                            })
                            .sum::<Scalar>()
                    })
                    .sum::<Scalar>()
            })
            .sum()
    }
    fn entry(&self, row: usize, column: usize) -> Scalar {
        match self[row / D]
            .0
            .binary_search_by_key(&(column / D), |&(b, _)| b)
        {
            Ok(k) => self[row / D].0[k].1[row % D][column % D].value(),
            Err(_) => 0.0,
        }
    }
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.iter().enumerate().for_each(|(a, row)| {
            row.entries().for_each(|(b, block)| {
                block.iter().enumerate().for_each(|(i, block_i)| {
                    block_i.iter().enumerate().for_each(|(j, block_ij)| {
                        square_matrix[D * a + i][D * b + j] = block_ij.value()
                    })
                })
            })
        });
    }
    fn retain_from(self, retained: &[bool]) -> SquareMatrix {
        let mut remap = vec![0; retained.len()];
        let mut count = 0;
        retained.iter().enumerate().for_each(|(p, &keep)| {
            if keep {
                remap[p] = count;
                count += 1;
            }
        });
        let mut square_matrix = SquareMatrix::zero(count);
        self.iter().enumerate().for_each(|(a, row)| {
            row.entries().for_each(|(b, block)| {
                block.iter().enumerate().for_each(|(i, block_i)| {
                    block_i.iter().enumerate().for_each(|(j, block_ij)| {
                        if retained[D * a + i] && retained[D * b + j] {
                            square_matrix[remap[D * a + i]][remap[D * b + j]] = block_ij.value()
                        }
                    })
                })
            })
        });
        square_matrix
    }
}

impl<const D: usize, I, J, U> FiniteDifference for TensorRank2SparseVec2D<D, I, J, U> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let zero = TensorRank2::zero();
        let block_errors =
            |self_ab: &TensorRank2<D, I, J, U>, comparator_ab: &TensorRank2<D, I, J, U>| {
                let mut errors = (0, 0);
                self_ab.iter().zip(comparator_ab.iter()).for_each(
                    |(self_ab_i, comparator_ab_i)| {
                        self_ab_i.iter().zip(comparator_ab_i.iter()).for_each(
                            |(&self_ab_ij, &comparator_ab_ij)| {
                                if self_ab_ij.differs(comparator_ab_ij, epsilon) {
                                    errors.0 += 1;
                                    if self_ab_ij.differs_severely(comparator_ab_ij, epsilon) {
                                        errors.1 += 1;
                                    }
                                }
                            },
                        )
                    },
                );
                errors
            };
        let (error_count, severe_count) = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_a, comparator_a)| {
                let mut errors = (0, 0);
                let (mut p, mut q) = (0, 0);
                while p < self_a.0.len() || q < comparator_a.0.len() {
                    let b = self_a.0.get(p).map(|&(b, _)| b);
                    let c = comparator_a.0.get(q).map(|&(c, _)| c);
                    let block = match (b, c) {
                        (Some(b), Some(c)) if b == c => {
                            p += 1;
                            q += 1;
                            block_errors(&self_a.0[p - 1].1, &comparator_a.0[q - 1].1)
                        }
                        (Some(b), Some(c)) if b < c => {
                            p += 1;
                            block_errors(&self_a.0[p - 1].1, &zero)
                        }
                        (Some(_), None) => {
                            p += 1;
                            block_errors(&self_a.0[p - 1].1, &zero)
                        }
                        _ => {
                            q += 1;
                            block_errors(&zero, &comparator_a.0[q - 1].1)
                        }
                    };
                    errors.0 += block.0;
                    errors.1 += block.1;
                }
                errors
            })
            .fold((0, 0), |sum, errors| (sum.0 + errors.0, sum.1 + errors.1));
        if error_count > 0 {
            Some((severe_count > 0, error_count))
        } else {
            None
        }
    }
}
