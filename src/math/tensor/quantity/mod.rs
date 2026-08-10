#[cfg(test)]
mod test;

use super::{
    rank_0::TensorRank0,
    unit::{Dimensionless, UnitDiv, UnitMul},
};
use std::{
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A scalar carrying a physical unit.
///
/// The dimensions of a constitutive law live in its material parameters rather
/// than in the tensors they multiply, so a modulus is a [`Quantity`] while a
/// count, a ratio, or a tolerance stays a bare [`TensorRank0`].
#[repr(transparent)]
pub struct Quantity<U = Dimensionless>(TensorRank0, PhantomData<U>);

impl<U> Quantity<U> {
    /// Associated function for const type conversion.
    pub const fn new(value: TensorRank0) -> Self {
        Self(value, PhantomData)
    }
    /// Returns the value with its unit discarded.
    pub const fn value(&self) -> TensorRank0 {
        self.0
    }
}

impl<U> Clone for Quantity<U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<U> Copy for Quantity<U> {}

impl<U> fmt::Debug for Quantity<U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<U> Display for Quantity<U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl<U> PartialEq for Quantity<U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<U> PartialOrd for Quantity<U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<U> From<TensorRank0> for Quantity<U> {
    fn from(value: TensorRank0) -> Self {
        Self::new(value)
    }
}

impl<U> From<Quantity<U>> for TensorRank0 {
    fn from(quantity: Quantity<U>) -> Self {
        quantity.0
    }
}

impl<U> Neg for Quantity<U> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.0)
    }
}

impl<U> Add for Quantity<U> {
    type Output = Self;
    fn add(self, quantity: Self) -> Self::Output {
        Self::new(self.0 + quantity.0)
    }
}

impl<U> AddAssign for Quantity<U> {
    fn add_assign(&mut self, quantity: Self) {
        self.0 += quantity.0
    }
}

impl<U> Sub for Quantity<U> {
    type Output = Self;
    fn sub(self, quantity: Self) -> Self::Output {
        Self::new(self.0 - quantity.0)
    }
}

impl<U> SubAssign for Quantity<U> {
    fn sub_assign(&mut self, quantity: Self) {
        self.0 -= quantity.0
    }
}

// Scaling by a bare scalar leaves the unit alone, in either order.

impl<U> Mul<TensorRank0> for Quantity<U> {
    type Output = Self;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        Self::new(self.0 * tensor_rank_0)
    }
}

impl<U> MulAssign<TensorRank0> for Quantity<U> {
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.0 *= tensor_rank_0
    }
}

impl<U> Div<TensorRank0> for Quantity<U> {
    type Output = Self;
    fn div(self, tensor_rank_0: TensorRank0) -> Self::Output {
        Self::new(self.0 / tensor_rank_0)
    }
}

impl<U> DivAssign<TensorRank0> for Quantity<U> {
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.0 /= tensor_rank_0
    }
}

impl<U> Mul<Quantity<U>> for TensorRank0 {
    type Output = Quantity<U>;
    fn mul(self, quantity: Quantity<U>) -> Self::Output {
        Quantity::new(self * quantity.0)
    }
}

// Combining two quantities combines their units.

impl<U, V> Mul<Quantity<V>> for Quantity<U>
where
    U: UnitMul<V>,
{
    type Output = Quantity<<U as UnitMul<V>>::Output>;
    fn mul(self, quantity: Quantity<V>) -> Self::Output {
        Quantity::new(self.0 * quantity.0)
    }
}

impl<U, V> Div<Quantity<V>> for Quantity<U>
where
    U: UnitDiv<V>,
{
    type Output = Quantity<<U as UnitDiv<V>>::Output>;
    fn div(self, quantity: Quantity<V>) -> Self::Output {
        Quantity::new(self.0 / quantity.0)
    }
}
