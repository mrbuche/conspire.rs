use crate::math::Dimensionless;
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
pub struct TensorRank2SparseVec<const D: usize, I, J, U = Dimensionless>(
    pub(super) Vec<(usize, TensorRank2<D, I, J, U>)>,
);

impl<const D: usize, I, J, U> Clone for TensorRank2SparseVec<D, I, J, U> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<const D: usize, I, J, U> Debug for TensorRank2SparseVec<D, I, J, U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl<const D: usize, I, J, U> Default for TensorRank2SparseVec<D, I, J, U> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<const D: usize, I, J, U> PartialEq for TensorRank2SparseVec<D, I, J, U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const D: usize, I, J, U> TensorRank2SparseVec<D, I, J, U> {
    pub fn entries(&self) -> impl Iterator<Item = (usize, &TensorRank2<D, I, J, U>)> {
        self.0.iter().map(|(column, entry)| (*column, entry))
    }
}

impl<const D: usize, I, J, U> FromIterator<TensorRank2<D, I, J, U>>
    for TensorRank2SparseVec<D, I, J, U>
{
    fn from_iter<T>(into_iterator: T) -> Self
    where
        T: IntoIterator<Item = TensorRank2<D, I, J, U>>,
    {
        Self(into_iterator.into_iter().enumerate().collect())
    }
}

impl<const D: usize, I, J, U> Index<usize> for TensorRank2SparseVec<D, I, J, U> {
    type Output = TensorRank2<D, I, J, U>;
    fn index(&self, index: usize) -> &Self::Output {
        match self.0.binary_search_by_key(&index, |&(column, _)| column) {
            Ok(k) => &self.0[k].1,
            Err(_) => panic!("Entry ({index}) not present."),
        }
    }
}

impl<const D: usize, I, J, U> IndexMut<usize> for TensorRank2SparseVec<D, I, J, U> {
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

impl<const D: usize, I, J, U> Display for TensorRank2SparseVec<D, I, J, U> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Need to implement Display")
    }
}

impl<const D: usize, I, J, U> Tensor for TensorRank2SparseVec<D, I, J, U> {
    type Item = TensorRank2<D, I, J, U>;
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

fn merge<const D: usize, I, J, U>(
    a: TensorRank2SparseVec<D, I, J, U>,
    b: &TensorRank2SparseVec<D, I, J, U>,
    sign: TensorRank0,
) -> TensorRank2SparseVec<D, I, J, U> {
    let mut merged = a;
    b.0.iter()
        .for_each(|(column, entry)| merged[*column] += entry * sign);
    merged
}

impl<const D: usize, I, J, U> Add for TensorRank2SparseVec<D, I, J, U> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        merge(self, &other, 1.0)
    }
}

impl<const D: usize, I, J, U> Add<&Self> for TensorRank2SparseVec<D, I, J, U> {
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        merge(self, other, 1.0)
    }
}

impl<const D: usize, I, J, U> AddAssign for TensorRank2SparseVec<D, I, J, U> {
    fn add_assign(&mut self, other: Self) {
        other
            .0
            .into_iter()
            .for_each(|(column, entry)| self[column] += entry);
    }
}

impl<const D: usize, I, J, U> AddAssign<&Self> for TensorRank2SparseVec<D, I, J, U> {
    fn add_assign(&mut self, other: &Self) {
        other
            .0
            .iter()
            .for_each(|(column, entry)| self[*column] += entry);
    }
}

impl<const D: usize, I, J, U> Sub for TensorRank2SparseVec<D, I, J, U> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        merge(self, &other, -1.0)
    }
}

impl<const D: usize, I, J, U> Sub<&Self> for TensorRank2SparseVec<D, I, J, U> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        merge(self, other, -1.0)
    }
}

impl<const D: usize, I, J, U> SubAssign for TensorRank2SparseVec<D, I, J, U> {
    fn sub_assign(&mut self, other: Self) {
        other
            .0
            .into_iter()
            .for_each(|(column, entry)| self[column] -= entry);
    }
}

impl<const D: usize, I, J, U> SubAssign<&Self> for TensorRank2SparseVec<D, I, J, U> {
    fn sub_assign(&mut self, other: &Self) {
        other
            .0
            .iter()
            .for_each(|(column, entry)| self[*column] -= entry);
    }
}

impl<const D: usize, I, J, U> Mul<TensorRank0> for TensorRank2SparseVec<D, I, J, U> {
    type Output = Self;
    fn mul(mut self, scalar: TensorRank0) -> Self {
        self *= &scalar;
        self
    }
}

impl<const D: usize, I, J, U> MulAssign<TensorRank0> for TensorRank2SparseVec<D, I, J, U> {
    fn mul_assign(&mut self, scalar: TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry *= &scalar);
    }
}

impl<const D: usize, I, J, U> MulAssign<&TensorRank0> for TensorRank2SparseVec<D, I, J, U> {
    fn mul_assign(&mut self, scalar: &TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry *= scalar);
    }
}

impl<const D: usize, I, J, U> Div<TensorRank0> for TensorRank2SparseVec<D, I, J, U> {
    type Output = Self;
    fn div(mut self, scalar: TensorRank0) -> Self {
        self /= &scalar;
        self
    }
}

impl<const D: usize, I, J, U> DivAssign<TensorRank0> for TensorRank2SparseVec<D, I, J, U> {
    fn div_assign(&mut self, scalar: TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry /= &scalar);
    }
}

impl<const D: usize, I, J, U> DivAssign<&TensorRank0> for TensorRank2SparseVec<D, I, J, U> {
    fn div_assign(&mut self, scalar: &TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry /= scalar);
    }
}

impl<const D: usize, I, J, U> Sum for TensorRank2SparseVec<D, I, J, U> {
    fn sum<T>(iter: T) -> Self
    where
        T: Iterator<Item = Self>,
    {
        iter.fold(Self::default(), |sum, entry| sum + entry)
    }
}
