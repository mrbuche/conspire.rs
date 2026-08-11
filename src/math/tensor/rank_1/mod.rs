use crate::math::Dimensionless;
use crate::math::{Current, Projection, Reference};
#[cfg(test)]
mod test;

pub(crate) mod cross;
pub(crate) mod list;
pub(crate) mod list_2d;
pub(crate) mod vec;
pub(crate) mod vec_2d;

use std::{
    array::from_fn,
    fmt::{self, Debug, Display, Formatter},
    iter::Sum,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{
    ABS_TOL,
    math::{
        matrix::vector::Vector,
        tensor::{
            Jacobian, Solution, Tensor, TensorArray, rank_0::TensorRank0,
            rank_1::list::TensorRank1List, rank_2::TensorRank2,
        },
        write_tensor_rank_0,
    },
};

use crate::math::assert::FiniteDifference;

/// A *d*-dimensional tensor of rank 1.
///
/// `D` is the dimension, `I` is the configuration.
#[repr(transparent)]
pub struct TensorRank1<const D: usize, I, U = Dimensionless>(
    pub(super) [TensorRank0; D],
    pub(super) PhantomData<(I, U)>,
);

impl<const D: usize, I, U> Clone for TensorRank1<D, I, U> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<const D: usize, I, U> Debug for TensorRank1<D, I, U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<const D: usize, I, U> PartialEq for TensorRank1<D, I, U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const D: usize, I, U> TensorRank1<D, I, U> {
    /// Views the tensor with its configuration and unit discarded, so that
    /// arithmetic is compiled once per dimension rather than once per either.
    pub(super) fn canonical(&self) -> &TensorRank1<D, Reference, Dimensionless> {
        unsafe { &*(self as *const Self as *const TensorRank1<D, Reference, Dimensionless>) }
    }
}

impl<const D: usize, I, U> TensorRank1<D, I, U> {
    /// Associated function for const type conversion.
    pub const fn const_from(array: [TensorRank0; D]) -> Self {
        Self(array, PhantomData)
    }
}

impl<const D: usize, I, U> Default for TensorRank1<D, I, U> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const D: usize, U> From<TensorRank1<D, Reference, U>> for TensorRank1<D, Current, U> {
    fn from(tensor_rank_1: TensorRank1<D, Reference, U>) -> Self {
        Self(tensor_rank_1.0, PhantomData)
    }
}

impl<const D: usize, U> From<&TensorRank1<D, Reference, U>> for TensorRank1<D, Current, U> {
    fn from(tensor_rank_1: &TensorRank1<D, Reference, U>) -> Self {
        Self(tensor_rank_1.0, PhantomData)
    }
}

impl<const D: usize, U> From<TensorRank1<D, Current, U>> for TensorRank1<D, Reference, U> {
    fn from(tensor_rank_1: TensorRank1<D, Current, U>) -> Self {
        Self(tensor_rank_1.0, PhantomData)
    }
}

impl<const D: usize, U> From<&TensorRank1<D, Current, U>> for TensorRank1<D, Reference, U> {
    fn from(tensor_rank_1: &TensorRank1<D, Current, U>) -> Self {
        Self(tensor_rank_1.0, PhantomData)
    }
}

impl<const D: usize, U> From<TensorRank1<D, Projection, U>> for TensorRank1<D, Reference, U> {
    fn from(tensor_rank_1: TensorRank1<D, Projection, U>) -> Self {
        Self(tensor_rank_1.0, PhantomData)
    }
}

impl<const D: usize, U> From<&TensorRank1<D, Projection, U>> for TensorRank1<D, Reference, U> {
    fn from(tensor_rank_1: &TensorRank1<D, Projection, U>) -> Self {
        Self(tensor_rank_1.0, PhantomData)
    }
}

impl<const D: usize, I, U> Display for TensorRank1<D, I, U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "\x1B[s")?;
        write!(f, "[")?;
        self.iter()
            .try_for_each(|entry| write_tensor_rank_0(f, entry))?;
        write!(f, "\x1B[2D]")
    }
}

impl<const D: usize, I, U> TensorRank1<D, I, U> {
    /// Returns a raw pointer to the slice’s buffer.
    pub const fn as_ptr(&self) -> *const TensorRank0 {
        self.0.as_ptr()
    }
    pub fn orthonormal_basis(&self) -> TensorRank1List<D, I, D, U> {
        let norm = self.norm();
        assert!(
            norm > ABS_TOL,
            "Cannot build an orthonormal basis from the zero vector"
        );
        let mut basis = TensorRank1List::zero();
        basis[0] = self / norm;
        let mut filled = 1;
        for i in 0..D {
            if filled == D {
                break;
            }
            let mut v = zero();
            v[i] = 1.0;
            basis.iter().take(filled).for_each(|q| v -= q * (&v * q));
            let v_norm = v.norm();
            if v_norm > ABS_TOL {
                basis[filled] = v / v_norm;
                filled += 1;
            }
        }
        assert!(filled == D, "Failed to construct full orthonormal basis");
        basis
    }
}

impl<const D: usize, I, U> FiniteDifference for TensorRank1<D, I, U> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .filter(|&(&self_i, &comparator_i)| {
                (self_i / comparator_i - 1.0).abs() >= epsilon
                    && (self_i.abs() >= epsilon || comparator_i.abs() >= epsilon)
            })
            .count();
        if error_count > 0 {
            Some((true, error_count))
        } else {
            None
        }
    }
}

impl<const D: usize, I, U> Solution for TensorRank1<D, I, U> {
    fn decrement_from(&mut self, _other: &Vector) {
        unimplemented!()
    }
    fn decrement_from_chained(&mut self, _other: &mut Vector, _vector: &Vector) {
        unimplemented!()
    }
}

impl<const D: usize, I, U> Jacobian for TensorRank1<D, I, U> {
    fn fill_into(&self, _vector: &mut Vector) {
        unimplemented!()
    }
    fn fill_into_chained(self, _other: Vector, _vector: &mut Vector) {
        unimplemented!()
    }
}

impl<const D: usize, I, U> Sub<Vector> for TensorRank1<D, I, U> {
    type Output = Self;
    fn sub(self, _vector: Vector) -> Self::Output {
        unimplemented!()
    }
}

impl<const D: usize, I, U> Sub<&Vector> for TensorRank1<D, I, U> {
    type Output = Self;
    fn sub(self, _vector: &Vector) -> Self::Output {
        unimplemented!()
    }
}

impl<const D: usize, I, U> Tensor for TensorRank1<D, I, U> {
    type Item = TensorRank0;
    fn full_contraction(&self, tensor_rank_1: &Self) -> TensorRank0 {
        self * tensor_rank_1
    }
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
    fn len(&self) -> usize {
        D
    }
    fn size(&self) -> usize {
        D
    }
}

impl<const D: usize, I, U> IntoIterator for TensorRank1<D, I, U> {
    type Item = TensorRank0;
    type IntoIter = std::array::IntoIter<Self::Item, D>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize, I, U> TensorArray for TensorRank1<D, I, U> {
    type Array = [TensorRank0; D];
    type Item = TensorRank0;
    fn as_array(&self) -> Self::Array {
        self.0
    }
    fn identity() -> Self {
        ones()
    }
    fn zero() -> Self {
        zero()
    }
}

/// Returns the rank-1 tensor of ones as a constant.
pub(crate) const fn ones<const D: usize, I, U>() -> TensorRank1<D, I, U> {
    TensorRank1([1.0; D], PhantomData)
}

/// Returns the rank-1 zero tensor as a constant.
pub const fn zero<const D: usize, I, U>() -> TensorRank1<D, I, U> {
    TensorRank1([0.0; D], PhantomData)
}

impl<const D: usize, I, U> From<[TensorRank0; D]> for TensorRank1<D, I, U> {
    fn from(array: [TensorRank0; D]) -> Self {
        Self(array, PhantomData)
    }
}

impl<const D: usize, I, U> From<TensorRank1<D, I, U>> for [TensorRank0; D] {
    fn from(tensor_rank_1: TensorRank1<D, I, U>) -> Self {
        tensor_rank_1.0
    }
}

impl<const D: usize, I, U> From<Vec<TensorRank0>> for TensorRank1<D, I, U> {
    fn from(vec: Vec<TensorRank0>) -> Self {
        Self(vec.try_into().unwrap(), PhantomData)
    }
}

impl<const D: usize, I, U> From<TensorRank1<D, I, U>> for Vec<TensorRank0> {
    fn from(tensor_rank_1: TensorRank1<D, I, U>) -> Self {
        tensor_rank_1.0.to_vec()
    }
}

impl<const D: usize, I, U> From<Vector> for TensorRank1<D, I, U> {
    fn from(_vector: Vector) -> Self {
        unimplemented!()
    }
}

impl<const D: usize, I, U> FromIterator<TensorRank0> for TensorRank1<D, I, U> {
    fn from_iter<Ii: IntoIterator<Item = TensorRank0>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_1 = zero();
        tensor_rank_1
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_1_i, value_i)| *tensor_rank_1_i = value_i);
        tensor_rank_1
    }
}

impl<const D: usize, I, U> Index<usize> for TensorRank1<D, I, U> {
    type Output = TensorRank0;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, I, U> IndexMut<usize> for TensorRank1<D, I, U> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, I, U> Sum for TensorRank1<D, I, U> {
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        iter.reduce(|mut acc, item| {
            acc += item;
            acc
        })
        .unwrap_or_else(Self::default)
    }
}

impl<'a, const D: usize, I, U> Sum<&'a Self> for TensorRank1<D, I, U> {
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::default(), |mut acc, item| {
            acc += item;
            acc
        })
    }
}

impl<const D: usize, I, U> Neg for TensorRank1<D, I, U> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        from_fn(|i| -self[i]).into()
    }
}

impl<const D: usize, I, U> Neg for &TensorRank1<D, I, U> {
    type Output = TensorRank1<D, I, U>;
    fn neg(self) -> Self::Output {
        from_fn(|i| -self[i]).into()
    }
}

impl<const D: usize, I, U> Div<TensorRank0> for TensorRank1<D, I, U> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, U> Div<TensorRank0> for &TensorRank1<D, I, U> {
    type Output = TensorRank1<D, I, U>;
    fn div(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, I, U> Div<&TensorRank0> for TensorRank1<D, I, U> {
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, U> Div<&TensorRank0> for &TensorRank1<D, I, U> {
    type Output = TensorRank1<D, I, U>;
    fn div(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, I, U> DivAssign<TensorRank0> for TensorRank1<D, I, U> {
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= &tensor_rank_0);
    }
}

impl<const D: usize, I, U> DivAssign<&TensorRank0> for TensorRank1<D, I, U> {
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= tensor_rank_0);
    }
}

impl<const D: usize, I, U> Mul<TensorRank0> for TensorRank1<D, I, U> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, U> Mul<TensorRank0> for &TensorRank1<D, I, U> {
    type Output = TensorRank1<D, I, U>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, I, U> Mul<&TensorRank0> for TensorRank1<D, I, U> {
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, I, U> Mul<&TensorRank0> for &TensorRank1<D, I, U> {
    type Output = TensorRank1<D, I, U>;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_0).collect()
    }
}

impl<const D: usize, I, U> MulAssign<TensorRank0> for TensorRank1<D, I, U> {
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= &tensor_rank_0);
    }
}

impl<const D: usize, I, U> MulAssign<&TensorRank0> for TensorRank1<D, I, U> {
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= tensor_rank_0);
    }
}

impl<const D: usize, I, U> Add for TensorRank1<D, I, U> {
    type Output = Self;
    fn add(mut self, tensor_rank_1: Self) -> Self::Output {
        self += tensor_rank_1;
        self
    }
}

impl<const D: usize, I, U> Add<&Self> for TensorRank1<D, I, U> {
    type Output = Self;
    fn add(mut self, tensor_rank_1: &Self) -> Self::Output {
        self += tensor_rank_1;
        self
    }
}

impl<const D: usize, I, U> Add<TensorRank1<D, I, U>> for &TensorRank1<D, I, U> {
    type Output = TensorRank1<D, I, U>;
    fn add(self, mut tensor_rank_1: TensorRank1<D, I, U>) -> Self::Output {
        tensor_rank_1 += self;
        tensor_rank_1
    }
}

impl<const D: usize, I, U> Add<Self> for &TensorRank1<D, I, U> {
    type Output = TensorRank1<D, I, U>;
    fn add(self, tensor_rank_1: Self) -> Self::Output {
        tensor_rank_1
            .iter()
            .zip(self.iter())
            .map(|(tensor_rank_1_i, self_i)| self_i + *tensor_rank_1_i)
            .collect()
    }
}

impl<const D: usize, I, U> AddAssign for TensorRank1<D, I, U> {
    fn add_assign(&mut self, tensor_rank_1: Self) {
        self.iter_mut()
            .zip(tensor_rank_1)
            .for_each(|(self_i, tensor_rank_1_i)| *self_i += tensor_rank_1_i);
    }
}

impl<const D: usize, I, U> AddAssign<&Self> for TensorRank1<D, I, U> {
    fn add_assign(&mut self, tensor_rank_1: &Self) {
        self.iter_mut()
            .zip(tensor_rank_1.iter())
            .for_each(|(self_i, tensor_rank_1_i)| *self_i += tensor_rank_1_i);
    }
}

impl<const D: usize, I, U> Sub for TensorRank1<D, I, U> {
    type Output = Self;
    fn sub(mut self, tensor_rank_1: Self) -> Self::Output {
        self -= tensor_rank_1;
        self
    }
}

impl<const D: usize, I, U> Sub<&Self> for TensorRank1<D, I, U> {
    type Output = Self;
    fn sub(mut self, tensor_rank_1: &Self) -> Self::Output {
        self -= tensor_rank_1;
        self
    }
}

impl<const D: usize, I, U> Sub<TensorRank1<D, I, U>> for &TensorRank1<D, I, U> {
    type Output = TensorRank1<D, I, U>;
    fn sub(self, mut tensor_rank_1: TensorRank1<D, I, U>) -> Self::Output {
        tensor_rank_1
            .iter_mut()
            .zip(self.iter())
            .for_each(|(tensor_rank_1_i, self_i)| *tensor_rank_1_i = self_i - *tensor_rank_1_i);
        tensor_rank_1
    }
}

impl<const D: usize, I, U> Sub<Self> for &TensorRank1<D, I, U> {
    type Output = TensorRank1<D, I, U>;
    fn sub(self, tensor_rank_1: Self) -> Self::Output {
        tensor_rank_1
            .iter()
            .zip(self.iter())
            .map(|(tensor_rank_1_i, self_i)| self_i - *tensor_rank_1_i)
            .collect()
    }
}

impl<const D: usize, I, U> SubAssign for TensorRank1<D, I, U> {
    fn sub_assign(&mut self, tensor_rank_1: Self) {
        self.iter_mut()
            .zip(tensor_rank_1)
            .for_each(|(self_i, tensor_rank_1_i)| *self_i -= tensor_rank_1_i);
    }
}

impl<const D: usize, I, U> SubAssign<&Self> for TensorRank1<D, I, U> {
    fn sub_assign(&mut self, tensor_rank_1: &Self) {
        self.iter_mut()
            .zip(tensor_rank_1.iter())
            .for_each(|(self_i, tensor_rank_1_i)| *self_i -= tensor_rank_1_i);
    }
}

impl<const D: usize, I, U> Mul for TensorRank1<D, I, U> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_1: Self) -> Self::Output {
        self.into_iter()
            .zip(tensor_rank_1)
            .map(|(self_i, tensor_rank_1_i)| self_i * tensor_rank_1_i)
            .sum()
    }
}

impl<const D: usize, I, U> Mul<&Self> for TensorRank1<D, I, U> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_1: &Self) -> Self::Output {
        self.into_iter()
            .zip(tensor_rank_1.iter())
            .map(|(self_i, tensor_rank_1_i)| self_i * tensor_rank_1_i)
            .sum()
    }
}

impl<const D: usize, I, U> Mul<TensorRank1<D, I, U>> for &TensorRank1<D, I, U> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_1: TensorRank1<D, I, U>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1)
            .map(|(self_i, tensor_rank_1_i)| self_i * tensor_rank_1_i)
            .sum()
    }
}

impl<const D: usize, I, U> Mul for &TensorRank1<D, I, U> {
    type Output = TensorRank0;
    fn mul(self, tensor_rank_1: Self) -> Self::Output {
        self.iter()
            .zip(tensor_rank_1.iter())
            .map(|(self_i, tensor_rank_1_i)| self_i * tensor_rank_1_i)
            .sum()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const D: usize, I, J, U> Div<TensorRank2<D, I, J, U>> for &TensorRank1<D, I, U> {
    type Output = TensorRank1<D, J, U>;
    fn div(self, tensor_rank_2: TensorRank2<D, I, J, U>) -> Self::Output {
        tensor_rank_2.inverse() * self
    }
}
