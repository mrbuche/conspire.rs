#[cfg(test)]
mod test;

use super::TensorRank2;
use crate::math::{
    Hessian, HessianAccumulate, Rank2, Scalar, SquareMatrix, Tensor, TensorRank0, Vector,
};
use crate::units::Dimensionless;
use std::{
    fmt::{self, Debug, Display, Formatter},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use super::sparse_vec::TensorRank2SparseVec;
use super::sparse_vec_2d::TensorRank2SparseVec2D;

use crate::math::{TensorArray, assert::FiniteDifference};

/// A vector of sparse vectors of rank-2 tensors, storing only the symmetric half.
///
/// The underlying block matrix is known to be symmetric under index-pair
/// transpose, meaning block(a, b) == block(b, a)ᵀ for every pair of block
/// indices. Only the canonical (row <= column) half of the blocks is stored;
/// entries on the other side are reconstructed by transposing on lookup
/// instead of being duplicated in memory.
pub struct TensorRank2SparseVec2DSymmetric<const D: usize, I, J, U = Dimensionless>(
    TensorRank2SparseVec2D<D, I, J, U>,
);

impl<const D: usize, I, J, U> Clone for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<const D: usize, I, J, U> Debug for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl<const D: usize, I, J, U> Default for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<const D: usize, I, J, U> PartialEq for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const D: usize, I, J, U> TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    pub fn zero(len: usize) -> Self {
        Self(TensorRank2SparseVec2D::zero(len))
    }
}

impl<const D: usize, I, J, U> Display for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Need to implement Display")
    }
}

impl<const D: usize, I, J, U> Tensor for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    type Item = TensorRank2SparseVec<D, I, J, U>;
    type Unit = U;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<const D: usize, I, J, U> Add for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl<const D: usize, I, J, U> Add<&Self> for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        Self(self.0 + &other.0)
    }
}

impl<const D: usize, I, J, U> AddAssign for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl<const D: usize, I, J, U> AddAssign<&Self> for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn add_assign(&mut self, other: &Self) {
        self.0 += &other.0;
    }
}

impl<const D: usize, I, J, U> Sub for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl<const D: usize, I, J, U> Sub<&Self> for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        Self(self.0 - &other.0)
    }
}

impl<const D: usize, I, J, U> SubAssign for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl<const D: usize, I, J, U> SubAssign<&Self> for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn sub_assign(&mut self, other: &Self) {
        self.0 -= &other.0;
    }
}

impl<const D: usize, I, J, U> Mul<TensorRank0> for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    type Output = Self;
    fn mul(self, scalar: TensorRank0) -> Self {
        Self(self.0 * scalar)
    }
}

impl<const D: usize, I, J, U> MulAssign<TensorRank0>
    for TensorRank2SparseVec2DSymmetric<D, I, J, U>
{
    fn mul_assign(&mut self, scalar: TensorRank0) {
        self.0 *= scalar;
    }
}

impl<const D: usize, I, J, U> MulAssign<&TensorRank0>
    for TensorRank2SparseVec2DSymmetric<D, I, J, U>
{
    fn mul_assign(&mut self, scalar: &TensorRank0) {
        self.0 *= scalar;
    }
}

impl<const D: usize, I, J, U> Div<TensorRank0> for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    type Output = Self;
    fn div(self, scalar: TensorRank0) -> Self {
        Self(self.0 / scalar)
    }
}

impl<const D: usize, I, J, U> DivAssign<TensorRank0>
    for TensorRank2SparseVec2DSymmetric<D, I, J, U>
{
    fn div_assign(&mut self, scalar: TensorRank0) {
        self.0 /= scalar;
    }
}

impl<const D: usize, I, J, U> DivAssign<&TensorRank0>
    for TensorRank2SparseVec2DSymmetric<D, I, J, U>
{
    fn div_assign(&mut self, scalar: &TensorRank0) {
        self.0 /= scalar;
    }
}

impl<const D: usize, I, J, U> Sum for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn sum<T>(iter: T) -> Self
    where
        T: Iterator<Item = Self>,
    {
        iter.fold(Self::default(), |sum, entry| sum + entry)
    }
}

impl<const D: usize, I, U> HessianAccumulate<D, I, U>
    for TensorRank2SparseVec2DSymmetric<D, I, I, U>
{
    fn accumulate(&mut self, a: usize, b: usize, block: TensorRank2<D, I, I, U>) {
        if a <= b {
            self.0[a][b] += block;
        } else {
            self.0[b][a] += block.transpose();
        }
    }
}

impl<const D: usize, I, J, U> Hessian for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
    fn entry(&self, row: usize, column: usize) -> Scalar {
        let (a, b, i, j) = if row / D <= column / D {
            (row / D, column / D, row % D, column % D)
        } else {
            (column / D, row / D, column % D, row % D)
        };
        match self.0[a].0.binary_search_by_key(&b, |&(c, _)| c) {
            Ok(k) => self.0[a].0[k].1[i][j].value(),
            Err(_) => 0.0,
        }
    }
    fn quadratic_form(&self, vector: &Vector) -> Scalar {
        //
        // Only one triangle is stored, and the entry mirroring a stored one
        // contributes the very same product, so an off-diagonal block counts
        // twice. A diagonal block is stored whole and counts once.
        //
        self.0
            .iter()
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
                            * if a == b { 1.0 } else { 2.0 }
                    })
                    .sum::<Scalar>()
            })
            .sum()
    }
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.0.iter().enumerate().for_each(|(a, row)| {
            row.entries().for_each(|(b, block)| {
                block.iter().enumerate().for_each(|(i, block_i)| {
                    block_i.iter().enumerate().for_each(|(j, block_ij)| {
                        square_matrix[D * a + i][D * b + j] = block_ij.value();
                        if a != b {
                            square_matrix[D * b + j][D * a + i] = block_ij.value();
                        }
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
        self.0.iter().enumerate().for_each(|(a, row)| {
            row.entries().for_each(|(b, block)| {
                block.iter().enumerate().for_each(|(i, block_i)| {
                    block_i.iter().enumerate().for_each(|(j, block_ij)| {
                        if retained[D * a + i] && retained[D * b + j] {
                            square_matrix[remap[D * a + i]][remap[D * b + j]] = block_ij.value();
                            if a != b {
                                square_matrix[remap[D * b + j]][remap[D * a + i]] =
                                    block_ij.value();
                            }
                        }
                    })
                })
            })
        });
        square_matrix
    }
}

impl<const D: usize, I, J, U> FiniteDifference for TensorRank2SparseVec2DSymmetric<D, I, J, U> {
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
            .0
            .iter()
            .zip(comparator.0.iter())
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
