#[cfg(test)]
mod test;

use super::{
    Differentiate, Erase, Hessian, Jacobian, Solution, SquareMatrix, Tensor, TensorArray, Vector,
    rank_0::TensorRank0,
    unit::{Dimensionless, UnitDiv, UnitInv, UnitMul},
};
use crate::math::{TensorList, assert::FiniteDifference};
use std::{
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Implemented only where the two types are the same, so that a unit may be
/// named where it is discarded without allowing a different one.
pub trait Is<T> {}

impl<T> Is<T> for T {}

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
    /// Returns the value, stating the unit being discarded.
    ///
    /// Compiles only when the quantity carries that unit, so a synonym for it
    /// is accepted and anything else is not.
    pub const fn value_as<V>(&self) -> TensorRank0
    where
        U: Is<V>,
    {
        self.0
    }
}

impl<U> Quantity<U> {
    /// Returns the absolute value, which leaves the unit alone.
    pub fn abs(self) -> Self {
        Self::new(self.0.abs())
    }
    /// Returns whether the value is not a number.
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }
    /// Returns how many of another quantity of the same unit this one is.
    ///
    /// A count of steps carries no unit whatever the step is measured in, so
    /// the ratio needs no unit named for it.
    pub fn ratio(self, quantity: Self) -> TensorRank0 {
        self.0 / quantity.0
    }
    /// Returns the lesser of two quantities of the same unit.
    pub fn min(self, quantity: Self) -> Self {
        Self::new(self.0.min(quantity.0))
    }
    /// Returns the greater of two quantities of the same unit.
    pub fn max(self, quantity: Self) -> Self {
        Self::new(self.0.max(quantity.0))
    }
}

impl Quantity<Dimensionless> {
    /// Returns the smallest integer greater than or equal to the value.
    pub fn ceil(self) -> Self {
        Self::new(self.0.ceil())
    }
    /// Raises to an integer power.
    pub fn powi(self, n: i32) -> Self {
        Self::new(self.0.powi(n))
    }
    /// Raises to a power.
    pub fn powf(self, n: TensorRank0) -> Self {
        Self::new(self.0.powf(n))
    }
    /// Returns the square root.
    pub fn sqrt(self) -> Self {
        Self::new(self.0.sqrt())
    }
    /// Returns the natural logarithm.
    pub fn ln(self) -> Self {
        Self::new(self.0.ln())
    }
    /// Returns the exponential.
    pub fn exp(self) -> Self {
        Self::new(self.0.exp())
    }
    /// Returns the sine.
    pub fn sin(self) -> Self {
        Self::new(self.0.sin())
    }
    /// Returns the cosine.
    pub fn cos(self) -> Self {
        Self::new(self.0.cos())
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

impl<U> Default for Quantity<U> {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl<U> Add for Quantity<U> {
    type Output = Self;
    fn add(self, quantity: Self) -> Self::Output {
        Self::new(self.0 + quantity.0)
    }
}

impl<U> Add<&Self> for Quantity<U> {
    type Output = Self;
    fn add(self, quantity: &Self) -> Self::Output {
        Self::new(self.0 + quantity.0)
    }
}

impl<U> AddAssign<&Self> for Quantity<U> {
    fn add_assign(&mut self, quantity: &Self) {
        self.0 += quantity.0
    }
}

impl<U> Sub<&Self> for Quantity<U> {
    type Output = Self;
    fn sub(self, quantity: &Self) -> Self::Output {
        Self::new(self.0 - quantity.0)
    }
}

impl<U> SubAssign<&Self> for Quantity<U> {
    fn sub_assign(&mut self, quantity: &Self) {
        self.0 -= quantity.0
    }
}

// Reference-taking forms, which the integrators and solvers reach for.

impl<U> Add for &Quantity<U> {
    type Output = Quantity<U>;
    fn add(self, quantity: Self) -> Self::Output {
        Quantity::new(self.0 + quantity.0)
    }
}

impl<U> Add<Quantity<U>> for &Quantity<U> {
    type Output = Quantity<U>;
    fn add(self, quantity: Quantity<U>) -> Self::Output {
        Quantity::new(self.0 + quantity.0)
    }
}

impl<U> Sub for &Quantity<U> {
    type Output = Quantity<U>;
    fn sub(self, quantity: Self) -> Self::Output {
        Quantity::new(self.0 - quantity.0)
    }
}

impl<U> Mul<TensorRank0> for &Quantity<U> {
    type Output = Quantity<U>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        Quantity::new(self.0 * tensor_rank_0)
    }
}

impl<U> MulAssign<&TensorRank0> for Quantity<U> {
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.0 *= tensor_rank_0
    }
}

impl<U> DivAssign<&TensorRank0> for Quantity<U> {
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.0 /= tensor_rank_0
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

impl<U> Mul<&TensorRank0> for Quantity<U> {
    type Output = Self;
    fn mul(self, tensor_rank_0: &TensorRank0) -> Self::Output {
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

/// Scaling a bare scalar by a dimensionless quantity leaves it bare.
///
/// Sound rather than a loophole: the unit is named, and naming any other fails
/// to compile. It is how an unknown that is a bare scalar takes a step.
impl Mul<Quantity<Dimensionless>> for &TensorRank0 {
    type Output = TensorRank0;
    fn mul(self, quantity: Quantity<Dimensionless>) -> Self::Output {
        self * quantity.0
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

impl<U, V> Mul<Quantity<V>> for &Quantity<U>
where
    U: UnitMul<V>,
{
    type Output = Quantity<<U as UnitMul<V>>::Output>;
    fn mul(self, quantity: Quantity<V>) -> Self::Output {
        Quantity::new(self.0 * quantity.0)
    }
}

impl<U, V> Div<Quantity<V>> for &Quantity<U>
where
    U: UnitDiv<V>,
{
    type Output = Quantity<<U as UnitDiv<V>>::Output>;
    fn div(self, quantity: Quantity<V>) -> Self::Output {
        Quantity::new(self.0 / quantity.value())
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

// The weights an element hands over are iterated by reference, so a quantity
// scales the same way whether it is owned or borrowed.

impl<U, V> Mul<&Quantity<V>> for Quantity<U>
where
    U: UnitMul<V>,
{
    type Output = Quantity<<U as UnitMul<V>>::Output>;
    fn mul(self, quantity: &Quantity<V>) -> Self::Output {
        self * *quantity
    }
}

impl<U, V> Mul<&Quantity<V>> for &Quantity<U>
where
    U: UnitMul<V>,
{
    type Output = Quantity<<U as UnitMul<V>>::Output>;
    fn mul(self, quantity: &Quantity<V>) -> Self::Output {
        *self * *quantity
    }
}

impl<V> Mul<&Quantity<V>> for TensorRank0 {
    type Output = Quantity<V>;
    fn mul(self, quantity: &Quantity<V>) -> Self::Output {
        Quantity::new(self * quantity.0)
    }
}

impl<V> Mul<&Quantity<V>> for &TensorRank0 {
    type Output = Quantity<V>;
    fn mul(self, quantity: &Quantity<V>) -> Self::Output {
        Quantity::new(self * quantity.0)
    }
}

// A dimensionless quantity is a number, and mixes with one freely.

impl Add<TensorRank0> for Quantity<Dimensionless> {
    type Output = Self;
    fn add(self, tensor_rank_0: TensorRank0) -> Self::Output {
        Self::new(self.0 + tensor_rank_0)
    }
}

impl Add<Quantity<Dimensionless>> for TensorRank0 {
    type Output = Quantity<Dimensionless>;
    fn add(self, quantity: Quantity<Dimensionless>) -> Self::Output {
        Quantity::new(self + quantity.0)
    }
}

impl Sub<TensorRank0> for Quantity<Dimensionless> {
    type Output = Self;
    fn sub(self, tensor_rank_0: TensorRank0) -> Self::Output {
        Self::new(self.0 - tensor_rank_0)
    }
}

impl Sub<Quantity<Dimensionless>> for TensorRank0 {
    type Output = Quantity<Dimensionless>;
    fn sub(self, quantity: Quantity<Dimensionless>) -> Self::Output {
        Quantity::new(self - quantity.0)
    }
}

impl PartialEq<TensorRank0> for Quantity<Dimensionless> {
    fn eq(&self, tensor_rank_0: &TensorRank0) -> bool {
        &self.0 == tensor_rank_0
    }
}

impl PartialOrd<TensorRank0> for Quantity<Dimensionless> {
    fn partial_cmp(&self, tensor_rank_0: &TensorRank0) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(tensor_rank_0)
    }
}

impl<U> Div<Quantity<U>> for TensorRank0
where
    U: UnitInv,
{
    type Output = Quantity<<U as UnitInv>::Output>;
    fn div(self, quantity: Quantity<U>) -> Self::Output {
        Quantity::new(self / quantity.0)
    }
}

impl<U> std::iter::Sum for Quantity<U> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        Self::new(iter.map(|quantity| quantity.0).sum())
    }
}

impl<'a, U> std::iter::Sum<&'a Quantity<U>> for Quantity<U> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Quantity<U>>,
    {
        Self::new(iter.map(|quantity| quantity.0).sum())
    }
}

// A quantity is a scalar, so it is a tensor of rank zero that happens to carry
// a unit. Every default that iterates is overridden, an iterator over a scalar
// yielding the scalar itself.

impl<U> Erase for Quantity<U> {
    type Erased = TensorRank0;
    fn erase(&self) -> &Self::Erased {
        &self.0
    }
}

impl<U> Tensor for Quantity<U> {
    type Item = Self;
    type Unit = U;
    fn error_count_zero(&self, tol_abs: TensorRank0, tol_rel: TensorRank0) -> Option<usize> {
        self.0.error_count_zero(tol_abs, tol_rel)
    }
    fn error_count(
        &self,
        other: &Self,
        tol_abs: TensorRank0,
        tol_rel: TensorRank0,
    ) -> Option<usize> {
        self.0.error_count(&other.0, tol_abs, tol_rel)
    }
    fn full_contraction(&self, quantity: &Self) -> TensorRank0 {
        self.0 * quantity.0
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        std::slice::from_ref(self).iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        std::slice::from_mut(self).iter_mut()
    }
    fn len(&self) -> usize {
        1
    }
    fn norm_inf(&self) -> TensorRank0 {
        self.0.abs()
    }
    fn norm_l1(&self) -> TensorRank0 {
        self.0.abs()
    }
    fn norm_p_sum(&self, p: TensorRank0) -> TensorRank0 {
        self.0.abs().powf(p)
    }
    fn normalized(self) -> Self {
        Self::new(1.0)
    }
    fn size(&self) -> usize {
        1
    }
    fn sub_abs(&self, other: &Self) -> Self {
        Self::new((self.0 - other.0).abs())
    }
    fn sub_rel(&self, other: &Self) -> Self {
        Self::new(self.0.sub_rel(&other.0))
    }
}

impl<U> TensorArray for Quantity<U> {
    type Array = TensorRank0;
    type Item = Self;
    fn as_array(&self) -> Self::Array {
        self.0
    }
    fn identity() -> Self {
        Self::new(1.0)
    }
    fn zero() -> Self {
        Self::new(0.0)
    }
}

impl<U> FiniteDifference for Quantity<U> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        self.0.error_fd(&comparator.0, epsilon)
    }
}

// A list of quantities compares entry by entry, as a list of scalars does.

impl<U, const N: usize> FiniteDifference for TensorList<Quantity<U>, N> {
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        error_fd_over(self.iter().zip(comparator.iter()), epsilon)
    }
}

impl<U, const M: usize, const N: usize> FiniteDifference
    for TensorList<TensorList<Quantity<U>, N>, M>
{
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        error_fd_over(
            self.iter()
                .zip(comparator.iter())
                .flat_map(|(entry, comparator_entry)| entry.iter().zip(comparator_entry.iter())),
            epsilon,
        )
    }
}

fn error_fd_over<'a, U: 'a>(
    entries: impl Iterator<Item = (&'a Quantity<U>, &'a Quantity<U>)>,
    epsilon: TensorRank0,
) -> Option<(bool, usize)> {
    let error_count = entries
        .filter_map(|(entry, comparator_entry)| entry.error_fd(comparator_entry, epsilon))
        .map(|(_, count)| count)
        .sum();
    if error_count > 0 {
        Some((true, error_count))
    } else {
        None
    }
}

// A scalar unknown has no vector to be filled into or decremented from.

impl<U> Solution for Quantity<U> {
    fn decrement_from(&mut self, _other: &Vector) {
        unimplemented!()
    }
    fn decrement_from_chained(&mut self, _other: &mut Vector, _vector: &Vector) {
        unimplemented!()
    }
}

// A scalar Hessian is the scalar itself, with no matrix to be filled into.

impl<U> Hessian for Quantity<U> {
    fn quadratic_form(&self, vector: &Vector) -> TensorRank0 {
        self.0 * vector[0] * vector[0]
    }
    fn entry(&self, _row: usize, _column: usize) -> TensorRank0 {
        unimplemented!()
    }
    fn fill_into(self, _square_matrix: &mut SquareMatrix) {
        unimplemented!()
    }
}

impl<U> Jacobian for Quantity<U> {
    fn fill_into(&self, _vector: &mut Vector) {
        unimplemented!()
    }
    fn fill_into_chained(self, _other: Vector, _vector: &mut Vector) {
        unimplemented!()
    }
}

impl<U> Sub<Vector> for Quantity<U> {
    type Output = Self;
    fn sub(self, _vector: Vector) -> Self::Output {
        unimplemented!()
    }
}

impl<U> Sub<&Vector> for Quantity<U> {
    type Output = Self;
    fn sub(self, _vector: &Vector) -> Self::Output {
        unimplemented!()
    }
}

impl<U> From<Vector> for Quantity<U> {
    fn from(_vector: Vector) -> Self {
        unimplemented!()
    }
}

impl<U, T> Differentiate<T> for Quantity<U>
where
    U: UnitDiv<T>,
{
    type Derivative = Quantity<<U as UnitDiv<T>>::Output>;
}
