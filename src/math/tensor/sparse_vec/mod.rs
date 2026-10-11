#[cfg(test)]
mod test;

use crate::math::{
    Hessian, HessianBlock, Scalar, SquareMatrix, Tensor, TensorRank0, tensor::vec::TensorVector,
};
use std::{
    fmt::{self, Debug, Display, Formatter, Result},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A sparse vector of tensors, storing only inserted entries.
pub struct TensorSparseVec<T>(Vec<(usize, T)>)
where
    T: Tensor;

/// A vector of sparse vectors of tensors, storing only inserted entries.
pub type TensorSparseVec2D<T> = TensorVector<TensorSparseVec<T>>;

impl<T> TensorSparseVec2D<T>
where
    T: Tensor,
{
    pub fn zero(len: usize) -> Self {
        (0..len).map(|_| TensorSparseVec::default()).collect()
    }
}

impl<T> TensorSparseVec<T>
where
    T: Tensor,
{
    pub fn entries(&self) -> impl Iterator<Item = (usize, &T)> {
        self.0.iter().map(|(column, entry)| (*column, entry))
    }
}

impl<T> Clone for TensorSparseVec<T>
where
    T: Tensor,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Debug for TensorSparseVec<T>
where
    T: Tensor,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl<T> Default for TensorSparseVec<T>
where
    T: Tensor,
{
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> PartialEq for TensorSparseVec<T>
where
    T: Tensor + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> FromIterator<T> for TensorSparseVec<T>
where
    T: Tensor,
{
    fn from_iter<I>(into_iterator: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self(into_iterator.into_iter().enumerate().collect())
    }
}

impl<T> Index<usize> for TensorSparseVec<T>
where
    T: Tensor,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        match self.0.binary_search_by_key(&index, |&(column, _)| column) {
            Ok(k) => &self.0[k].1,
            Err(_) => panic!("Entry ({index}) not present."),
        }
    }
}

impl<T> IndexMut<usize> for TensorSparseVec<T>
where
    T: Tensor,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let k = match self.0.binary_search_by_key(&index, |&(column, _)| column) {
            Ok(k) => k,
            Err(k) => {
                self.0.insert(k, (index, T::default()));
                k
            }
        };
        &mut self.0[k].1
    }
}

impl<T> Display for TensorSparseVec<T>
where
    T: Tensor,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Need to implement Display")
    }
}

impl<T> Tensor for TensorSparseVec<T>
where
    T: Tensor,
{
    type Item = T;
    type Unit = T::Unit;
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
        self.0.iter().map(|(_, entry)| entry.size()).sum()
    }
}

fn merge<T>(a: TensorSparseVec<T>, b: &TensorSparseVec<T>, sign: TensorRank0) -> TensorSparseVec<T>
where
    T: Tensor,
{
    let mut merged = a;
    b.0.iter()
        .for_each(|(column, entry)| merged[*column] += entry.clone() * sign);
    merged
}

impl<T> Add for TensorSparseVec<T>
where
    T: Tensor,
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        merge(self, &other, 1.0)
    }
}

impl<T> Add<&Self> for TensorSparseVec<T>
where
    T: Tensor,
{
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        merge(self, other, 1.0)
    }
}

impl<T> AddAssign for TensorSparseVec<T>
where
    T: Tensor,
{
    fn add_assign(&mut self, other: Self) {
        other
            .0
            .into_iter()
            .for_each(|(column, entry)| self[column] += entry);
    }
}

impl<T> AddAssign<&Self> for TensorSparseVec<T>
where
    T: Tensor,
{
    fn add_assign(&mut self, other: &Self) {
        other
            .0
            .iter()
            .for_each(|(column, entry)| self[*column] += entry);
    }
}

impl<T> Sub for TensorSparseVec<T>
where
    T: Tensor,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        merge(self, &other, -1.0)
    }
}

impl<T> Sub<&Self> for TensorSparseVec<T>
where
    T: Tensor,
{
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        merge(self, other, -1.0)
    }
}

impl<T> SubAssign for TensorSparseVec<T>
where
    T: Tensor,
{
    fn sub_assign(&mut self, other: Self) {
        other
            .0
            .into_iter()
            .for_each(|(column, entry)| self[column] -= entry);
    }
}

impl<T> SubAssign<&Self> for TensorSparseVec<T>
where
    T: Tensor,
{
    fn sub_assign(&mut self, other: &Self) {
        other
            .0
            .iter()
            .for_each(|(column, entry)| self[*column] -= entry);
    }
}

impl<T> Mul<TensorRank0> for TensorSparseVec<T>
where
    T: Tensor,
{
    type Output = Self;
    fn mul(mut self, scalar: TensorRank0) -> Self {
        self *= &scalar;
        self
    }
}

impl<T> MulAssign<TensorRank0> for TensorSparseVec<T>
where
    T: Tensor,
{
    fn mul_assign(&mut self, scalar: TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry *= &scalar);
    }
}

impl<T> MulAssign<&TensorRank0> for TensorSparseVec<T>
where
    T: Tensor,
{
    fn mul_assign(&mut self, scalar: &TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry *= scalar);
    }
}

impl<T> Div<TensorRank0> for TensorSparseVec<T>
where
    T: Tensor,
{
    type Output = Self;
    fn div(mut self, scalar: TensorRank0) -> Self {
        self /= &scalar;
        self
    }
}

impl<T> DivAssign<TensorRank0> for TensorSparseVec<T>
where
    T: Tensor,
{
    fn div_assign(&mut self, scalar: TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry /= &scalar);
    }
}

impl<T> DivAssign<&TensorRank0> for TensorSparseVec<T>
where
    T: Tensor,
{
    fn div_assign(&mut self, scalar: &TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry /= scalar);
    }
}

impl<T> Sum for TensorSparseVec<T>
where
    T: Tensor,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::default(), |sum, entry| sum + entry)
    }
}

impl<T> Hessian for TensorSparseVec2D<T>
where
    T: HessianBlock + Tensor,
{
    fn entry(&self, row: usize, column: usize) -> Scalar {
        let block = T::default();
        let (height, width) = (block.height(), block.width());
        match self[row / height]
            .0
            .binary_search_by_key(&(column / width), |&(b, _)| b)
        {
            Ok(k) => self[row / height].0[k]
                .1
                .entry(row % height, column % width),
            Err(_) => 0.0,
        }
    }
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        let block = T::default();
        let (height, width) = (block.height(), block.width());
        self.iter().enumerate().for_each(|(a, row)| {
            row.entries()
                .for_each(|(b, block)| block.fill_into_block(square_matrix, height * a, width * b))
        });
    }
    fn retain_from(self, retained: &[bool]) -> SquareMatrix {
        let block = T::default();
        let (height, width) = (block.height(), block.width());
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
                (0..height).for_each(|i| {
                    (0..width).for_each(|j| {
                        if retained[height * a + i] && retained[width * b + j] {
                            square_matrix[remap[height * a + i]][remap[width * b + j]] =
                                block.entry(i, j)
                        }
                    })
                })
            })
        });
        square_matrix
    }
}
