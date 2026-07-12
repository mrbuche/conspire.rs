#[cfg(test)]
mod test;

use super::TensorRank2;
use crate::math::{
    Hessian, Scalar, SquareMatrix, Tensor, TensorArray, TensorRank0, tensor::vec::TensorVector,
};
use std::{
    fmt::{Display, Formatter, Result},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

#[cfg(test)]
use crate::math::tensor::test::ErrorTensor;

/// A vector of sparse vectors of rank-2 tensors, storing only inserted entries.
pub type TensorRank2SparseVec2D<const D: usize, const I: usize, const J: usize> =
    TensorVector<TensorRank2SparseVec<D, I, J>>;

impl<const D: usize, const I: usize, const J: usize> TensorRank2SparseVec2D<D, I, J> {
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| TensorRank2SparseVec::default()).collect()
    }
}

/// A sparse vector of rank-2 tensors, storing only inserted entries.
#[derive(Clone, Debug, Default)]
pub struct TensorRank2SparseVec<const D: usize, const I: usize, const J: usize>(
    Vec<(usize, TensorRank2<D, I, J>)>,
);

impl<const D: usize, const I: usize, const J: usize> TensorRank2SparseVec<D, I, J> {
    pub fn entries(&self) -> impl Iterator<Item = (usize, &TensorRank2<D, I, J>)> {
        self.0.iter().map(|(column, entry)| (*column, entry))
    }
}

impl<const D: usize, const I: usize, const J: usize> FromIterator<TensorRank2<D, I, J>>
    for TensorRank2SparseVec<D, I, J>
{
    fn from_iter<T>(into_iterator: T) -> Self
    where
        T: IntoIterator<Item = TensorRank2<D, I, J>>,
    {
        Self(into_iterator.into_iter().enumerate().collect())
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize>
    Mul<TensorRank2SparseVec2D<D, J, K>> for TensorRank2<D, I, J>
{
    type Output = TensorRank2SparseVec2D<D, I, K>;
    fn mul(self, tensor_rank_2_sparse_vec_2d: TensorRank2SparseVec2D<D, J, K>) -> Self::Output {
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

impl<const D: usize, const I: usize, const J: usize, const K: usize> Mul<TensorRank2<D, J, K>>
    for TensorRank2SparseVec2D<D, I, J>
{
    type Output = TensorRank2SparseVec2D<D, I, K>;
    fn mul(self, tensor_rank_2: TensorRank2<D, J, K>) -> Self::Output {
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

impl<const D: usize, const I: usize, const J: usize> Index<usize>
    for TensorRank2SparseVec<D, I, J>
{
    type Output = TensorRank2<D, I, J>;
    fn index(&self, index: usize) -> &Self::Output {
        match self.0.binary_search_by_key(&index, |&(column, _)| column) {
            Ok(k) => &self.0[k].1,
            Err(_) => panic!("Entry ({index}) not present."),
        }
    }
}

impl<const D: usize, const I: usize, const J: usize> IndexMut<usize>
    for TensorRank2SparseVec<D, I, J>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let k = match self.0.binary_search_by_key(&index, |&(column, _)| column) {
            Ok(k) => k,
            Err(k) => {
                self.0.insert(k, (index, TensorRank2::zero()));
                k
            }
        };
        &mut self.0[k].1
    }
}

impl<const D: usize, const I: usize, const J: usize> Display for TensorRank2SparseVec<D, I, J> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Need to implement Display")
    }
}

impl<const D: usize, const I: usize, const J: usize> Tensor for TensorRank2SparseVec<D, I, J> {
    type Item = TensorRank2<D, I, J>;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter().map(|(_, entry)| entry)
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut().map(|(_, entry)| entry)
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn size(&self) -> usize {
        self.0.len() * D * D
    }
}

fn merge<const D: usize, const I: usize, const J: usize>(
    a: TensorRank2SparseVec<D, I, J>,
    b: &TensorRank2SparseVec<D, I, J>,
    sign: Scalar,
) -> TensorRank2SparseVec<D, I, J> {
    let mut merged = a;
    b.0.iter()
        .for_each(|(column, entry)| merged[*column] += entry * sign);
    merged
}

impl<const D: usize, const I: usize, const J: usize> Add for TensorRank2SparseVec<D, I, J> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        merge(self, &other, 1.0)
    }
}

impl<const D: usize, const I: usize, const J: usize> Add<&Self> for TensorRank2SparseVec<D, I, J> {
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        merge(self, other, 1.0)
    }
}

impl<const D: usize, const I: usize, const J: usize> AddAssign for TensorRank2SparseVec<D, I, J> {
    fn add_assign(&mut self, other: Self) {
        other
            .0
            .into_iter()
            .for_each(|(column, entry)| self[column] += entry);
    }
}

impl<const D: usize, const I: usize, const J: usize> AddAssign<&Self>
    for TensorRank2SparseVec<D, I, J>
{
    fn add_assign(&mut self, other: &Self) {
        other
            .0
            .iter()
            .for_each(|(column, entry)| self[*column] += entry);
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub for TensorRank2SparseVec<D, I, J> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        merge(self, &other, -1.0)
    }
}

impl<const D: usize, const I: usize, const J: usize> Sub<&Self> for TensorRank2SparseVec<D, I, J> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        merge(self, other, -1.0)
    }
}

impl<const D: usize, const I: usize, const J: usize> SubAssign for TensorRank2SparseVec<D, I, J> {
    fn sub_assign(&mut self, other: Self) {
        other
            .0
            .into_iter()
            .for_each(|(column, entry)| self[column] -= entry);
    }
}

impl<const D: usize, const I: usize, const J: usize> SubAssign<&Self>
    for TensorRank2SparseVec<D, I, J>
{
    fn sub_assign(&mut self, other: &Self) {
        other
            .0
            .iter()
            .for_each(|(column, entry)| self[*column] -= entry);
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<TensorRank0>
    for TensorRank2SparseVec<D, I, J>
{
    type Output = Self;
    fn mul(mut self, scalar: TensorRank0) -> Self {
        self *= &scalar;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<TensorRank0>
    for TensorRank2SparseVec<D, I, J>
{
    fn mul_assign(&mut self, scalar: TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry *= &scalar);
    }
}

impl<const D: usize, const I: usize, const J: usize> MulAssign<&TensorRank0>
    for TensorRank2SparseVec<D, I, J>
{
    fn mul_assign(&mut self, scalar: &TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry *= scalar);
    }
}

impl<const D: usize, const I: usize, const J: usize> Div<TensorRank0>
    for TensorRank2SparseVec<D, I, J>
{
    type Output = Self;
    fn div(mut self, scalar: TensorRank0) -> Self {
        self /= &scalar;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize> DivAssign<TensorRank0>
    for TensorRank2SparseVec<D, I, J>
{
    fn div_assign(&mut self, scalar: TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry /= &scalar);
    }
}

impl<const D: usize, const I: usize, const J: usize> DivAssign<&TensorRank0>
    for TensorRank2SparseVec<D, I, J>
{
    fn div_assign(&mut self, scalar: &TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry /= scalar);
    }
}

impl<const D: usize, const I: usize, const J: usize> Sum for TensorRank2SparseVec<D, I, J> {
    fn sum<T>(iter: T) -> Self
    where
        T: Iterator<Item = Self>,
    {
        iter.fold(Self::default(), |sum, entry| sum + entry)
    }
}

impl<const D: usize, const I: usize, const J: usize> Hessian for TensorRank2SparseVec2D<D, I, J> {
    fn entry(&self, row: usize, column: usize) -> Scalar {
        match self[row / D]
            .0
            .binary_search_by_key(&(column / D), |&(b, _)| b)
        {
            Ok(k) => self[row / D].0[k].1[row % D][column % D],
            Err(_) => 0.0,
        }
    }
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.iter().enumerate().for_each(|(a, row)| {
            row.entries().for_each(|(b, block)| {
                block.iter().enumerate().for_each(|(i, block_i)| {
                    block_i
                        .iter()
                        .enumerate()
                        .for_each(|(j, block_ij)| square_matrix[D * a + i][D * b + j] = *block_ij)
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
                            square_matrix[remap[D * a + i]][remap[D * b + j]] = *block_ij
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
    for TensorRank2SparseVec2D<D, I, J>
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
