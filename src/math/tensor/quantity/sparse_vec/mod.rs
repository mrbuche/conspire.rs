use super::Quantity;
use crate::math::{Tensor, TensorRank0};
use crate::units::Dimensionless;
use std::{
    fmt::{self, Debug, Display, Formatter, Result},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// A sparse vector of quantities, storing only inserted entries.
pub struct QuantitySparseVec<U = Dimensionless>(pub(super) Vec<(usize, Quantity<U>)>);

impl<U> Clone for QuantitySparseVec<U> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<U> Debug for QuantitySparseVec<U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl<U> Default for QuantitySparseVec<U> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<U> PartialEq for QuantitySparseVec<U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<U> QuantitySparseVec<U> {
    pub fn entries(&self) -> impl Iterator<Item = (usize, &Quantity<U>)> {
        self.0.iter().map(|(column, entry)| (*column, entry))
    }
}

impl<U> FromIterator<Quantity<U>> for QuantitySparseVec<U> {
    fn from_iter<T>(into_iterator: T) -> Self
    where
        T: IntoIterator<Item = Quantity<U>>,
    {
        Self(into_iterator.into_iter().enumerate().collect())
    }
}

impl<U> Index<usize> for QuantitySparseVec<U> {
    type Output = Quantity<U>;
    fn index(&self, index: usize) -> &Self::Output {
        match self.0.binary_search_by_key(&index, |&(column, _)| column) {
            Ok(k) => &self.0[k].1,
            Err(_) => panic!("Entry ({index}) not present."),
        }
    }
}

impl<U> IndexMut<usize> for QuantitySparseVec<U> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let k = match self.0.binary_search_by_key(&index, |&(column, _)| column) {
            Ok(k) => k,
            Err(k) => {
                self.0.insert(k, (index, Quantity::new(0.0)));
                k
            }
        };
        &mut self.0[k].1
    }
}

impl<U> Display for QuantitySparseVec<U> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Need to implement Display")
    }
}

impl<U> Tensor for QuantitySparseVec<U> {
    type Item = Quantity<U>;
    type Unit = U;
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
        self.0.len()
    }
}

fn merge<U>(
    a: QuantitySparseVec<U>,
    b: &QuantitySparseVec<U>,
    sign: TensorRank0,
) -> QuantitySparseVec<U> {
    let mut merged = a;
    b.0.iter()
        .for_each(|(column, entry)| merged[*column] += entry * sign);
    merged
}

impl<U> Add for QuantitySparseVec<U> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        merge(self, &other, 1.0)
    }
}

impl<U> Add<&Self> for QuantitySparseVec<U> {
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        merge(self, other, 1.0)
    }
}

impl<U> AddAssign for QuantitySparseVec<U> {
    fn add_assign(&mut self, other: Self) {
        other
            .0
            .into_iter()
            .for_each(|(column, entry)| self[column] += entry);
    }
}

impl<U> AddAssign<&Self> for QuantitySparseVec<U> {
    fn add_assign(&mut self, other: &Self) {
        other
            .0
            .iter()
            .for_each(|(column, entry)| self[*column] += entry);
    }
}

impl<U> Sub for QuantitySparseVec<U> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        merge(self, &other, -1.0)
    }
}

impl<U> Sub<&Self> for QuantitySparseVec<U> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        merge(self, other, -1.0)
    }
}

impl<U> SubAssign for QuantitySparseVec<U> {
    fn sub_assign(&mut self, other: Self) {
        other
            .0
            .into_iter()
            .for_each(|(column, entry)| self[column] -= entry);
    }
}

impl<U> SubAssign<&Self> for QuantitySparseVec<U> {
    fn sub_assign(&mut self, other: &Self) {
        other
            .0
            .iter()
            .for_each(|(column, entry)| self[*column] -= entry);
    }
}

impl<U> Mul<TensorRank0> for QuantitySparseVec<U> {
    type Output = Self;
    fn mul(mut self, scalar: TensorRank0) -> Self {
        self *= &scalar;
        self
    }
}

impl<U> MulAssign<TensorRank0> for QuantitySparseVec<U> {
    fn mul_assign(&mut self, scalar: TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry *= &scalar);
    }
}

impl<U> MulAssign<&TensorRank0> for QuantitySparseVec<U> {
    fn mul_assign(&mut self, scalar: &TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry *= scalar);
    }
}

impl<U> Div<TensorRank0> for QuantitySparseVec<U> {
    type Output = Self;
    fn div(mut self, scalar: TensorRank0) -> Self {
        self /= &scalar;
        self
    }
}

impl<U> DivAssign<TensorRank0> for QuantitySparseVec<U> {
    fn div_assign(&mut self, scalar: TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry /= &scalar);
    }
}

impl<U> DivAssign<&TensorRank0> for QuantitySparseVec<U> {
    fn div_assign(&mut self, scalar: &TensorRank0) {
        self.0.iter_mut().for_each(|(_, entry)| *entry /= scalar);
    }
}

impl<U> Sum for QuantitySparseVec<U> {
    fn sum<T>(iter: T) -> Self
    where
        T: Iterator<Item = Self>,
    {
        iter.fold(Self::default(), |sum, entry| sum + entry)
    }
}
