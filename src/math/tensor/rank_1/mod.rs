#[cfg(test)]
mod test;
use super::{ContractWith, Differentiate, Erase};
use crate::math::{Current, Projection, Reference};

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

use crate::units::{UnitDiv, UnitMul};
use crate::{
    ABS_TOL,
    math::{
        matrix::vector::Vector,
        tensor::{
            Jacobian, Quantity, Solution, Tensor, TensorArray, rank_0::TensorRank0,
            rank_1::list::TensorRank1List, rank_2::TensorRank2,
        },
        write_tensor_rank_0,
    },
    units::Dimensionless,
};

use crate::math::assert::FiniteDifference;

/// A *d*-dimensional tensor of rank 1.
///
/// `D` is the dimension, `I` is the configuration.
#[repr(transparent)]
pub struct TensorRank1<const D: usize, I, U = Dimensionless>(
    pub(super) [Quantity<U>; D],
    pub(super) PhantomData<I>,
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
    pub(super) fn canonical(&self) -> &TensorRank1<D, Reference, Dimensionless> {
        unsafe { &*(self as *const Self as *const TensorRank1<D, Reference, Dimensionless>) }
    }
    /// Asserts that the tensor carries the given unit.
    pub fn with_unit<V>(self) -> TensorRank1<D, I, V> {
        relabel(self.into_canonical())
    }
    /// Returns the direction the tensor points in.
    pub fn normalized(self) -> TensorRank1<D, I, Dimensionless> {
        let norm = self.norm().value();
        (self / norm).with_unit()
    }
    fn into_canonical(self) -> TensorRank1<D, Reference, Dimensionless> {
        unsafe {
            (&self as *const Self)
                .cast::<TensorRank1<D, Reference, Dimensionless>>()
                .read()
        }
    }
}

pub(super) fn relabel<const D: usize, I, U>(
    tensor: TensorRank1<D, Reference, Dimensionless>,
) -> TensorRank1<D, I, U> {
    unsafe {
        (&tensor as *const TensorRank1<D, Reference, Dimensionless>)
            .cast::<TensorRank1<D, I, U>>()
            .read()
    }
}

impl<const D: usize, I, U> TensorRank1<D, I, U> {
    /// Associated function for const type conversion.
    pub const fn const_from(array: [TensorRank0; D]) -> Self {
        let mut entries = [Quantity::new(0.0); D];
        let mut i = 0;
        while i < D {
            entries[i] = Quantity::new(array[i]);
            i += 1;
        }
        Self(entries, PhantomData)
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
            .try_for_each(|entry| write_tensor_rank_0(f, &entry.value()))?;
        write!(f, "\x1B[2D]")
    }
}

impl<const D: usize, I, U> TensorRank1<D, I, U> {
    /// Returns a raw pointer to the slice’s buffer.
    pub const fn as_ptr(&self) -> *const TensorRank0 {
        self.0.as_ptr().cast()
    }
    /// Returns an orthonormal basis whose first vector is this one's direction.
    pub fn orthonormal_basis(&self) -> TensorRank1List<D, I, D, Dimensionless> {
        let norm = self.norm().value();
        assert!(
            norm > ABS_TOL,
            "Cannot build an orthonormal basis from the zero vector"
        );
        let mut basis = TensorRank1List::zero();
        basis[0] = (self / norm).with_unit();
        let mut filled = 1;
        for i in 0..D {
            if filled == D {
                break;
            }
            let mut v: TensorRank1<D, I, Dimensionless> = zero();
            v[i] = Quantity::new(1.0);
            basis.iter().take(filled).for_each(|q| v -= q * (&v * q));
            let v_norm = v.norm().value();
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
            .filter(|&(&self_i, &comparator_i)| self_i.differs(comparator_i, epsilon))
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

impl<const D: usize, I, U> Erase for TensorRank1<D, I, U> {
    type Erased = TensorRank1<D, Reference, Dimensionless>;
    fn erase(&self) -> &Self::Erased {
        self.canonical()
    }
}

impl<const D: usize, I, U, V> Mul<Quantity<V>> for TensorRank1<D, I, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, quantity: Quantity<V>) -> Self::Output {
        relabel(self.into_canonical() * quantity.value())
    }
}

impl<const D: usize, I, U, V> Mul<Quantity<V>> for &TensorRank1<D, I, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, quantity: Quantity<V>) -> Self::Output {
        relabel(self.canonical() * quantity.value())
    }
}

impl<const D: usize, I, U, V> Mul<&Quantity<V>> for TensorRank1<D, I, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, quantity: &Quantity<V>) -> Self::Output {
        self * *quantity
    }
}

impl<const D: usize, I, U, V> Mul<&Quantity<V>> for &TensorRank1<D, I, U>
where
    U: UnitMul<V>,
{
    type Output = TensorRank1<D, I, <U as UnitMul<V>>::Output>;
    fn mul(self, quantity: &Quantity<V>) -> Self::Output {
        self * *quantity
    }
}

impl<const D: usize, I, U, V> Div<Quantity<V>> for TensorRank1<D, I, U>
where
    U: UnitDiv<V>,
{
    type Output = TensorRank1<D, I, <U as UnitDiv<V>>::Output>;
    fn div(self, quantity: Quantity<V>) -> Self::Output {
        relabel(self.into_canonical() / quantity.value())
    }
}

impl<const D: usize, I, U, V> Div<Quantity<V>> for &TensorRank1<D, I, U>
where
    U: UnitDiv<V>,
{
    type Output = TensorRank1<D, I, <U as UnitDiv<V>>::Output>;
    fn div(self, quantity: Quantity<V>) -> Self::Output {
        relabel(self.canonical() / quantity.value())
    }
}

impl<const D: usize, I, U> Tensor for TensorRank1<D, I, U> {
    type Item = Quantity<U>;
    type Unit = U;
    fn full_contraction(&self, tensor_rank_1: &Self) -> TensorRank0 {
        self.iter()
            .zip(tensor_rank_1.iter())
            .map(|(self_i, other_i)| self_i.value() * other_i.value())
            .sum()
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
    type Item = Quantity<U>;
    type IntoIter = std::array::IntoIter<Self::Item, D>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize, I, U> TensorArray for TensorRank1<D, I, U> {
    type Array = [Quantity<U>; D];
    type Item = Quantity<U>;
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
    TensorRank1([Quantity::new(1.0); D], PhantomData)
}

/// Returns the rank-1 zero tensor as a constant.
pub const fn zero<const D: usize, I, U>() -> TensorRank1<D, I, U> {
    TensorRank1([Quantity::new(0.0); D], PhantomData)
}

impl<const D: usize, I, U> From<[Quantity<U>; D]> for TensorRank1<D, I, U> {
    fn from(array: [Quantity<U>; D]) -> Self {
        Self(array, PhantomData)
    }
}

impl<const D: usize, I, U> From<[TensorRank0; D]> for TensorRank1<D, I, U> {
    fn from(array: [TensorRank0; D]) -> Self {
        Self(array.map(Quantity::new), PhantomData)
    }
}

impl<const D: usize, I, U> From<TensorRank1<D, I, U>> for [TensorRank0; D] {
    fn from(tensor_rank_1: TensorRank1<D, I, U>) -> Self {
        tensor_rank_1.0.map(|entry| entry.value())
    }
}

impl<const D: usize, I, U> From<Vec<TensorRank0>> for TensorRank1<D, I, U> {
    fn from(vec: Vec<TensorRank0>) -> Self {
        Self(
            TryInto::<[TensorRank0; D]>::try_into(vec)
                .unwrap()
                .map(Quantity::new),
            PhantomData,
        )
    }
}

impl<const D: usize, I, U> From<TensorRank1<D, I, U>> for Vec<TensorRank0> {
    fn from(tensor_rank_1: TensorRank1<D, I, U>) -> Self {
        tensor_rank_1.0.iter().map(|entry| entry.value()).collect()
    }
}

impl<const D: usize, I, U> From<Vector> for TensorRank1<D, I, U> {
    fn from(_vector: Vector) -> Self {
        unimplemented!()
    }
}

impl<const D: usize, I, U> FromIterator<TensorRank0> for TensorRank1<D, I, U> {
    fn from_iter<Ii: IntoIterator<Item = TensorRank0>>(into_iterator: Ii) -> Self {
        into_iterator.into_iter().map(Quantity::new).collect()
    }
}

impl<const D: usize, I, U> FromIterator<Quantity<U>> for TensorRank1<D, I, U> {
    fn from_iter<Ii: IntoIterator<Item = Quantity<U>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_1 = zero();
        tensor_rank_1
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_1_i, value_i)| *tensor_rank_1_i = value_i);
        tensor_rank_1
    }
}

impl<const D: usize, I, U> Index<usize> for TensorRank1<D, I, U> {
    type Output = Quantity<U>;
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
            .map(|(self_i, tensor_rank_1_i)| self_i.value() * tensor_rank_1_i.value())
            .sum()
    }
}

impl<const D: usize, I, U, V> Mul<&TensorRank1<D, I, V>> for TensorRank1<D, I, U>
where
    U: UnitMul<V>,
{
    type Output = Quantity<<U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1: &TensorRank1<D, I, V>) -> Self::Output {
        Quantity::new(
            self.into_iter()
                .zip(tensor_rank_1.iter())
                .map(|(self_i, tensor_rank_1_i)| self_i.value() * tensor_rank_1_i.value())
                .sum(),
        )
    }
}

impl<const D: usize, I, U, V> Mul<TensorRank1<D, I, V>> for &TensorRank1<D, I, U>
where
    U: UnitMul<V>,
{
    type Output = Quantity<<U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1: TensorRank1<D, I, V>) -> Self::Output {
        Quantity::new(
            self.iter()
                .zip(tensor_rank_1)
                .map(|(self_i, tensor_rank_1_i)| self_i.value() * tensor_rank_1_i.value())
                .sum(),
        )
    }
}

impl<const D: usize, I, U, V> Mul<&TensorRank1<D, I, V>> for &TensorRank1<D, I, U>
where
    U: UnitMul<V>,
{
    type Output = Quantity<<U as UnitMul<V>>::Output>;
    fn mul(self, tensor_rank_1: &TensorRank1<D, I, V>) -> Self::Output {
        Quantity::new(
            self.iter()
                .zip(tensor_rank_1.iter())
                .map(|(self_i, tensor_rank_1_i)| self_i.value() * tensor_rank_1_i.value())
                .sum(),
        )
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const D: usize, I, J, U, V> Div<TensorRank2<D, I, J, V>> for &TensorRank1<D, I, U>
where
    U: UnitDiv<V>,
{
    type Output = TensorRank1<D, J, <U as UnitDiv<V>>::Output>;
    fn div(self, tensor_rank_2: TensorRank2<D, I, J, V>) -> Self::Output {
        relabel(tensor_rank_2.canonical().clone().inverse() * self.canonical())
    }
}

impl<const D: usize, I, U, V> ContractWith<TensorRank1<D, I, V>> for TensorRank1<D, I, U>
where
    U: UnitMul<V>,
{
    type Output = Quantity<<U as UnitMul<V>>::Output>;
    fn contract_with(&self, tensor_rank_1: &TensorRank1<D, I, V>) -> Self::Output {
        Quantity::new(self.canonical().full_contraction(tensor_rank_1.canonical()))
    }
}

impl<const D: usize, I, U, T> Differentiate<T> for TensorRank1<D, I, U>
where
    U: UnitDiv<T>,
{
    type Derivative = TensorRank1<D, I, <U as UnitDiv<T>>::Output>;
}
