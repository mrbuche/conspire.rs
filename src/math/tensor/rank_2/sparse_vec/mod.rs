#[cfg(test)]
mod test;

use super::TensorRank2;
use crate::math::{Tensor, TensorArray, TensorRank0};
use std::{
    fmt::{self, Debug, Display, Formatter, Result},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A sparse vector of rank-2 tensors, storing only inserted entries.
pub struct TensorRank2SparseVec<const D: usize, I, J>(
    pub(super) Vec<(usize, TensorRank2<D, I, J>)>,
);

impl<const D: usize, I, J> Clone for TensorRank2SparseVec<D, I, J> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<const D: usize, I, J> Debug for TensorRank2SparseVec<D, I, J> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl<const D: usize, I, J> Default for TensorRank2SparseVec<D, I, J> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<const D: usize, I, J> PartialEq for TensorRank2SparseVec<D, I, J> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const D: usize, I, J> TensorRank2SparseVec<D, I, J> {
    pub fn entries(&self) -> impl Iterator<Item = (usize, &TensorRank2<D, I, J>)> {
        self.0.iter().map(|(column, entry)| (*column, entry))
    }
}

impl<const D: usize, I, J> FromIterator<TensorRank2<D, I, J>> for TensorRank2SparseVec<D, I, J> {
    fn from_iter<T>(into_iterator: T) -> Self
    where
        T: IntoIterator<Item = TensorRank2<D, I, J>>,
    {
        Self(into_iterator.into_iter().enumerate().collect())
    }
}

impl<const D: usize, I, J> Index<usize> for TensorRank2SparseVec<D, I, J> {
    type Output = TensorRank2<D, I, J>;
    fn index(&self, index: usize) -> &Self::Output {
        match self.0.binary_search_by_key(&index, |&(column, _)| column) {
            Ok(k) => &self.0[k].1,
            Err(_) => panic!("Entry ({index}) not present."),
        }
    }
}

impl<const D: usize, I, J> IndexMut<usize> for TensorRank2SparseVec<D, I, J> {
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

impl<const D: usize, I, J> Display for TensorRank2SparseVec<D, I, J> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Need to implement Display")
    }
}

impl<const D: usize, I, J> Tensor for TensorRank2SparseVec<D, I, J> {
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

fn merge<const D: usize, I, J>(
    a: TensorRank2SparseVec<D, I, J>,
    b: &TensorRank2SparseVec<D, I, J>,
    sign: TensorRank0,
) -> TensorRank2SparseVec<D, I, J> {
    let mut merged = a;
    b.0.iter()
        .for_each(|(column, entry)| merged[*column] += entry * sign);
    merged
}

impl<const D: usize, I, J> Add for TensorRank2SparseVec<D, I, J> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        merge(self, &other, 1.0)
    }
}

impl<const D: usize, I, J> Add<&Self> for TensorRank2SparseVec<D, I, J> {
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        merge(self, other, 1.0)
    }
}

impl<const D: usize, I, J> AddAssign for TensorRank2SparseVec<D, I, J> {
    fn add_assign(&mut self, other: Self) {
        other
            .0
            .into_iter()
            .for_each(|(column, entry)| self[column] += entry);
    }
}

impl<const D: usize, I, J> AddAssign<&Self> for TensorRank2SparseVec<D, I, J> {
    fn add_assign(&mut self, other: &Self) {
        other
            .0
            .iter()
            .for_each(|(column, entry)| self[*column] += entry);
    }
}

impl<const D: usize, I, J> Sub for TensorRank2SparseVec<D, I, J> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        merge(self, &other, -1.0)
    }
}

impl<const D: usize, I, J> Sub<&Self> for TensorRank2SparseVec<D, I, J> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        merge(self, other, -1.0)
    }
}

impl<const D: usize, I, J> SubAssign for TensorRank2SparseVec<D, I, J> {
    fn sub_assign(&mut self, other: Self) {
        other
            .0
            .into_iter()
            .for_each(|(column, entry)| self[column] -= entry);
    }
}

impl<const D: usize, I, J> SubAssign<&Self> for TensorRank2SparseVec<D, I, J> {
    fn sub_assign(&mut self, other: &Self) {
        other
            .0
            .iter()
            .for_each(|(column, entry)| self[*column] -= entry);
    }
}

impl<const D: usize, I, J> Mul<TensorRank0> for TensorRank2SparseVec<D, I, J> {
    type Output = Self;
    fn mul(mut self, scalar: TensorRank0) -> Self {
        self *= &scalar;
        self
    }
}

impl<const D: usize, I, J> MulAssign<TensorRank0> for TensorRank2SparseVec<D, I, J> {
    fn mul_assign(&mut self, scalar: TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry *= &scalar);
    }
}

impl<const D: usize, I, J> MulAssign<&TensorRank0> for TensorRank2SparseVec<D, I, J> {
    fn mul_assign(&mut self, scalar: &TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry *= scalar);
    }
}

impl<const D: usize, I, J> Div<TensorRank0> for TensorRank2SparseVec<D, I, J> {
    type Output = Self;
    fn div(mut self, scalar: TensorRank0) -> Self {
        self /= &scalar;
        self
    }
}

impl<const D: usize, I, J> DivAssign<TensorRank0> for TensorRank2SparseVec<D, I, J> {
    fn div_assign(&mut self, scalar: TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry /= &scalar);
    }
}

impl<const D: usize, I, J> DivAssign<&TensorRank0> for TensorRank2SparseVec<D, I, J> {
    fn div_assign(&mut self, scalar: &TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry /= scalar);
    }
}

impl<const D: usize, I, J> Sum for TensorRank2SparseVec<D, I, J> {
    fn sum<T>(iter: T) -> Self
    where
        T: Iterator<Item = Self>,
    {
        iter.fold(Self::default(), |sum, entry| sum + entry)
    }
}
