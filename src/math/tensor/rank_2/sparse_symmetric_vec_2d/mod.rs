#[cfg(test)]
mod test;

use super::TensorRank2;
use crate::math::{Hessian, HessianAccumulate, Rank2, Scalar, SquareMatrix, Tensor, TensorRank0};
use std::{
    fmt::{self, Display, Formatter},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use super::sparse_vec::TensorRank2SparseVec;
use super::sparse_vec_2d::TensorRank2SparseVec2D;

#[cfg(test)]
use crate::math::{TensorArray, tensor::test::ErrorTensor};

/// A vector of sparse vectors of rank-2 tensors, storing only the canonical
/// (row <= column) half of a matrix known to be symmetric under node-pair
/// transpose: block(a,b) == block(b,a)ᵀ.
#[derive(Clone, Debug, Default)]
pub struct TensorRank2SparseSymmetricVec2D<const D: usize, const I: usize, const J: usize>(
    TensorRank2SparseVec2D<D, I, J>,
);

impl<const D: usize, const I: usize, const J: usize> TensorRank2SparseSymmetricVec2D<D, I, J> {
    pub fn zero(len: usize) -> Self {
        Self(TensorRank2SparseVec2D::zero(len))
    }
}

impl<const D: usize, const I: usize, const J: usize> Display
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Need to implement Display")
    }
}

impl<const D: usize, const I: usize, const J: usize> Tensor
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    type Item = TensorRank2SparseVec<D, I, J>;
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

impl<const D: usize, const I: usize, const J: usize> Add
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl<const D: usize, const I: usize, const J: usize> Add<&Self>
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        Self(self.0 + &other.0)
    }
}

impl<const D: usize, const I: usize, const J: usize> AddAssign
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl<const D: usize, const I: usize, const J: usize> AddAssign<&Self>
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn add_assign(&mut self, other: &Self) {
        self.0 += &other.0;
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub<&Self>
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        Self(self.0 - &other.0)
    }
}

impl<const D: usize, const I: usize, const J: usize> SubAssign
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl<const D: usize, const I: usize, const J: usize> SubAssign<&Self>
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn sub_assign(&mut self, other: &Self) {
        self.0 -= &other.0;
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank0>
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    type Output = Self;
    fn mul(self, scalar: TensorRank0) -> Self {
        Self(self.0 * scalar)
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<TensorRank0>
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn mul_assign(&mut self, scalar: TensorRank0) {
        self.0 *= scalar;
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<&TensorRank0>
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn mul_assign(&mut self, scalar: &TensorRank0) {
        self.0 *= scalar;
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank0>
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    type Output = Self;
    fn div(self, scalar: TensorRank0) -> Self {
        Self(self.0 / scalar)
    }
}

impl<const D: usize, const I: usize, const J: usize> DivAssign<TensorRank0>
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn div_assign(&mut self, scalar: TensorRank0) {
        self.0 /= scalar;
    }
}

impl<const D: usize, const I: usize, const J: usize> DivAssign<&TensorRank0>
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn div_assign(&mut self, scalar: &TensorRank0) {
        self.0 /= scalar;
    }
}

impl<const D: usize, const I: usize, const J: usize> Sum
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn sum<T>(iter: T) -> Self
    where
        T: Iterator<Item = Self>,
    {
        iter.fold(Self::default(), |sum, entry| sum + entry)
    }
}

impl<const D: usize, const I: usize> HessianAccumulate<D, I>
    for TensorRank2SparseSymmetricVec2D<D, I, I>
{
    fn accumulate(&mut self, a: usize, b: usize, block: TensorRank2<D, I, I>) {
        if a <= b {
            self.0[a][b] += block;
        } else {
            self.0[b][a] += block.transpose();
        }
    }
}

impl<const D: usize, const I: usize, const J: usize> Hessian
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn entry(&self, row: usize, column: usize) -> Scalar {
        let (a, b, i, j) = if row / D <= column / D {
            (row / D, column / D, row % D, column % D)
        } else {
            (column / D, row / D, column % D, row % D)
        };
        match self.0[a].0.binary_search_by_key(&b, |&(c, _)| c) {
            Ok(k) => self.0[a].0[k].1[i][j],
            Err(_) => 0.0,
        }
    }
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.0.iter().enumerate().for_each(|(a, row)| {
            row.entries().for_each(|(b, block)| {
                block.iter().enumerate().for_each(|(i, block_i)| {
                    block_i.iter().enumerate().for_each(|(j, block_ij)| {
                        square_matrix[D * a + i][D * b + j] = *block_ij;
                        if a != b {
                            square_matrix[D * b + j][D * a + i] = *block_ij;
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
                            square_matrix[remap[D * a + i]][remap[D * b + j]] = *block_ij;
                            if a != b {
                                square_matrix[remap[D * b + j]][remap[D * a + i]] = *block_ij;
                            }
                        }
                    })
                })
            })
        });
        square_matrix
    }
}

#[cfg(test)]
impl<const D: usize, const I: usize, const J: usize> ErrorTensor
    for TensorRank2SparseSymmetricVec2D<D, I, J>
{
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let zero = TensorRank2::zero();
        let block_errors =
            |self_ab: &TensorRank2<D, I, J>, comparator_ab: &TensorRank2<D, I, J>| {
                let mut errors = (0, 0);
                self_ab.iter().zip(comparator_ab.iter()).for_each(
                    |(self_ab_i, comparator_ab_i)| {
                        self_ab_i.iter().zip(comparator_ab_i.iter()).for_each(
                            |(&self_ab_ij, &comparator_ab_ij)| {
                                if (self_ab_ij / comparator_ab_ij - 1.0).abs() >= epsilon
                                    && (self_ab_ij.abs() >= epsilon
                                        || comparator_ab_ij.abs() >= epsilon)
                                {
                                    errors.0 += 1;
                                    if (self_ab_ij - comparator_ab_ij).abs() >= epsilon {
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
