#[cfg(test)]
mod test;

pub(crate) mod list;
pub(crate) mod vec;

use crate::math::{
    Differentiable, Erase, Jacobian, Quantity, Scalar, Solution, Tensor, TensorRank0, Vector,
};
use crate::units::UnitHalves;
use std::{
    fmt::{Display, Formatter, Result},
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

/// A fixed-size nested collection of different tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorTuple<T1, T2>(pub T1, pub T2)
where
    T1: Tensor,
    T2: Tensor;

type First<V> = <V as UnitHalves>::First;
type Second<V> = <V as UnitHalves>::Second;

impl<T1, T2, V> Mul<Quantity<V>> for TensorTuple<T1, T2>
where
    V: UnitHalves,
    T1: Mul<Quantity<First<V>>> + Tensor,
    T2: Mul<Quantity<Second<V>>> + Tensor,
    <T1 as Mul<Quantity<First<V>>>>::Output: Tensor,
    <T2 as Mul<Quantity<Second<V>>>>::Output: Tensor,
{
    type Output = TensorTuple<
        <T1 as Mul<Quantity<First<V>>>>::Output,
        <T2 as Mul<Quantity<Second<V>>>>::Output,
    >;
    fn mul(self, quantity: Quantity<V>) -> Self::Output {
        let (first, second) = quantity.halves();
        TensorTuple(self.0 * first, self.1 * second)
    }
}

impl<T1, T2, V> Mul<Quantity<V>> for &TensorTuple<T1, T2>
where
    V: UnitHalves,
    T1: Clone + Mul<Quantity<First<V>>> + Tensor,
    T2: Clone + Mul<Quantity<Second<V>>> + Tensor,
    <T1 as Mul<Quantity<First<V>>>>::Output: Tensor,
    <T2 as Mul<Quantity<Second<V>>>>::Output: Tensor,
{
    type Output = TensorTuple<
        <T1 as Mul<Quantity<First<V>>>>::Output,
        <T2 as Mul<Quantity<Second<V>>>>::Output,
    >;
    fn mul(self, quantity: Quantity<V>) -> Self::Output {
        let (first, second) = quantity.halves();
        TensorTuple(self.0.clone() * first, self.1.clone() * second)
    }
}

impl<T1, T2, V> Div<Quantity<V>> for TensorTuple<T1, T2>
where
    V: UnitHalves,
    T1: Div<Quantity<First<V>>> + Tensor,
    T2: Div<Quantity<Second<V>>> + Tensor,
    <T1 as Div<Quantity<First<V>>>>::Output: Tensor,
    <T2 as Div<Quantity<Second<V>>>>::Output: Tensor,
{
    type Output = TensorTuple<
        <T1 as Div<Quantity<First<V>>>>::Output,
        <T2 as Div<Quantity<Second<V>>>>::Output,
    >;
    fn div(self, quantity: Quantity<V>) -> Self::Output {
        let (first, second) = quantity.halves();
        TensorTuple(self.0 / first, self.1 / second)
    }
}

impl<T1, T2> Erase for TensorTuple<T1, T2>
where
    T1: Erase + Tensor,
    T2: Erase + Tensor,
    <T1 as Erase>::Erased: Tensor,
    <T2 as Erase>::Erased: Tensor,
{
    type Erased = TensorTuple<<T1 as Erase>::Erased, <T2 as Erase>::Erased>;
    fn erase(&self) -> &Self::Erased {
        unsafe { &*(self as *const Self as *const Self::Erased) }
    }
}

impl<T1, T2> Default for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn default() -> Self {
        Self(T1::default(), T2::default())
    }
}

impl<T1, T2> From<(T1, T2)> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn from(tuple: (T1, T2)) -> Self {
        Self(tuple.0, tuple.1)
    }
}

impl<'a, T1, T2> From<&'a TensorTuple<T1, T2>> for (&'a T1, &'a T2)
where
    T1: Tensor,
    T2: Tensor,
{
    fn from(tensor_tuple: &'a TensorTuple<T1, T2>) -> Self {
        (&tensor_tuple.0, &tensor_tuple.1)
    }
}

impl<T1, T2> From<TensorTuple<T1, T2>> for (T1, T2)
where
    T1: Tensor,
    T2: Tensor,
{
    fn from(tensor_tuple: TensorTuple<T1, T2>) -> Self {
        (tensor_tuple.0, tensor_tuple.1)
    }
}

impl<T1, T2> From<Vector> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn from(_vector: Vector) -> Self {
        unimplemented!()
    }
}

impl<T1, T2> Display for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

impl<T1, T2> Tensor for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Item = T1::Item;
    type Unit = (<T1 as Tensor>::Unit, <T2 as Tensor>::Unit);
    fn error_count_zero(&self, tol_abs: Scalar, tol_rel: Scalar) -> Option<usize> {
        let error_count = self.0.error_count_zero(tol_abs, tol_rel).unwrap_or(0)
            + self.1.error_count_zero(tol_abs, tol_rel).unwrap_or(0);
        if error_count > 0 {
            Some(error_count)
        } else {
            None
        }
    }
    fn error_count(&self, tensor_tuple: &Self, tol_abs: Scalar, tol_rel: Scalar) -> Option<usize> {
        let error_count = self
            .0
            .error_count(&tensor_tuple.0, tol_abs, tol_rel)
            .unwrap_or(0)
            + self
                .1
                .error_count(&tensor_tuple.1, tol_abs, tol_rel)
                .unwrap_or(0);
        if error_count > 0 {
            Some(error_count)
        } else {
            None
        }
    }
    fn full_contraction(&self, tensor_tuple: &Self) -> TensorRank0 {
        self.0.full_contraction(&tensor_tuple.0) + self.1.full_contraction(&tensor_tuple.1)
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        if self.size() == 0 {
            self.0.iter()
        } else {
            unimplemented!()
        }
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        if self.size() == 0 {
            self.0.iter_mut()
        } else {
            unimplemented!()
        }
    }
    fn len(&self) -> usize {
        unimplemented!()
    }
    fn norm_inf(&self) -> Quantity<Self::Unit> {
        Quantity::new(self.0.norm_inf().value().max(self.1.norm_inf().value()))
    }
    fn norm_l1(&self) -> Quantity<Self::Unit> {
        Quantity::new(self.0.norm_l1().value() + self.1.norm_l1().value())
    }
    fn norm_p_sum(&self, p: TensorRank0) -> TensorRank0 {
        self.0.norm_p_sum(p) + self.1.norm_p_sum(p)
    }
    fn size(&self) -> usize {
        self.0.size() + self.1.size()
    }
    fn sub_abs(&self, tensor_tuple: &Self) -> Self {
        Self(
            self.0.sub_abs(&tensor_tuple.0),
            self.1.sub_abs(&tensor_tuple.1),
        )
    }
    fn sub_rel(&self, tensor_tuple: &Self) -> Self {
        Self(
            self.0.sub_rel(&tensor_tuple.0),
            self.1.sub_rel(&tensor_tuple.1),
        )
    }
}

impl<T1, T2> Jacobian for TensorTuple<T1, T2>
where
    T1: Jacobian,
    T2: Jacobian,
{
    fn fill_into(&self, vector: &mut Vector) {
        let mut head = Vector::zero(self.0.size());
        self.0.fill_into(&mut head);
        let mut tail = Vector::zero(self.1.size());
        self.1.fill_into(&mut tail);
        head.into_iter()
            .chain(tail)
            .zip(vector.iter_mut())
            .for_each(|(entry, vector_i)| *vector_i = entry)
    }
    fn fill_into_chained(self, other: Vector, vector: &mut Vector) {
        let mut head = Vector::zero(self.0.size());
        self.0.fill_into(&mut head);
        let mut tail = Vector::zero(self.1.size());
        self.1.fill_into(&mut tail);
        head.into_iter()
            .chain(tail)
            .chain(other)
            .zip(vector.iter_mut())
            .for_each(|(entry, vector_i)| *vector_i = entry)
    }
}

impl<T1, T2> Solution for TensorTuple<T1, T2>
where
    T1: Solution,
    T2: Solution,
{
    fn decrement_from(&mut self, other: &Vector) {
        let split = self.0.size();
        let head: Vector = other.iter().take(split).copied().collect();
        let tail: Vector = other.iter().skip(split).copied().collect();
        self.0.decrement_from(&head);
        self.1.decrement_from(&tail);
    }
    fn decrement_from_chained(&mut self, other: &mut Vector, vector: &Vector) {
        let split = self.0.size();
        let tail_len = self.1.size();
        let head: Vector = vector.iter().take(split).copied().collect();
        let tail: Vector = vector.iter().skip(split).take(tail_len).copied().collect();
        self.0.decrement_from(&head);
        self.1.decrement_from(&tail);
        other
            .iter_mut()
            .zip(vector.iter().skip(split + tail_len))
            .for_each(|(entry_i, vector_i)| *entry_i -= vector_i)
    }
}

impl<T1, T2> Sum for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        iter.reduce(|mut acc, item| {
            acc.0 += item.0;
            acc.1 += item.1;
            acc
        })
        .unwrap_or_else(Self::default)
    }
}

impl<T1, T2> Div<TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<T1, T2> Div<&TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<T1, T2> DivAssign<TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.0 /= &tensor_rank_0;
        self.1 /= tensor_rank_0;
    }
}

impl<T1, T2> DivAssign<&TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.0 /= tensor_rank_0;
        self.1 /= tensor_rank_0;
    }
}

impl<T1, T2> Mul<TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<T1, T2> Mul<&TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<T1, T2> Mul<TensorRank0> for &TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = TensorTuple<T1, T2>;
    fn mul(self, tensor_rank_0: TensorRank0) -> Self::Output {
        //
        // Cloning for now to avoid trait recursion nightmare.
        //
        TensorTuple(
            self.0.clone() * tensor_rank_0,
            self.1.clone() * tensor_rank_0,
        )
    }
}

impl<T1, T2> MulAssign<TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.0 *= &tensor_rank_0;
        self.1 *= tensor_rank_0;
    }
}

impl<T1, T2> MulAssign<&TensorRank0> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.0 *= tensor_rank_0;
        self.1 *= tensor_rank_0;
    }
}

impl<T1, T2> Add for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_tuple: Self) -> Self::Output {
        self += tensor_tuple;
        self
    }
}

impl<T1, T2> Add<&Self> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn add(mut self, tensor_tuple: &Self) -> Self::Output {
        self += tensor_tuple;
        self
    }
}

impl<T1, T2> AddAssign for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn add_assign(&mut self, tensor_tuple: Self) {
        self.0 += tensor_tuple.0;
        self.1 += tensor_tuple.1;
    }
}

impl<T1, T2> AddAssign<&Self> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn add_assign(&mut self, tensor_tuple: &Self) {
        self.0 += &tensor_tuple.0;
        self.1 += &tensor_tuple.1;
    }
}

impl<T1, T2> Sub for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_tuple: Self) -> Self::Output {
        self -= tensor_tuple;
        self
    }
}

impl<T1, T2> Sub<&Self> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = Self;
    fn sub(mut self, tensor_tuple: &Self) -> Self::Output {
        self -= tensor_tuple;
        self
    }
}

impl<T1, T2> Sub for &TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    type Output = TensorTuple<T1, T2>;
    fn sub(self, tensor_tuple: Self) -> Self::Output {
        self.clone() - tensor_tuple
    }
}

impl<T1, T2> SubAssign for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn sub_assign(&mut self, tensor_tuple: Self) {
        self.0 -= tensor_tuple.0;
        self.1 -= tensor_tuple.1;
    }
}

impl<T1, T2> SubAssign<&Self> for TensorTuple<T1, T2>
where
    T1: Tensor,
    T2: Tensor,
{
    fn sub_assign(&mut self, tensor_tuple: &Self) {
        self.0 -= &tensor_tuple.0;
        self.1 -= &tensor_tuple.1;
    }
}

impl<T1, T2> Sub<Vector> for TensorTuple<T1, T2>
where
    T1: Tensor + Sub<Vector, Output = T1>,
    T2: Tensor + Sub<Vector, Output = T2>,
{
    type Output = Self;
    fn sub(mut self, vector: Vector) -> Self::Output {
        let split = self.0.size();
        self.0 = self.0 - vector.iter().take(split).copied().collect::<Vector>();
        self.1 = self.1 - vector.iter().skip(split).copied().collect::<Vector>();
        self
    }
}

impl<T1, T2> Sub<&Vector> for TensorTuple<T1, T2>
where
    T1: Tensor + Sub<Vector, Output = T1>,
    T2: Tensor + Sub<Vector, Output = T2>,
{
    type Output = Self;
    fn sub(mut self, vector: &Vector) -> Self::Output {
        let split = self.0.size();
        self.0 = self.0 - vector.iter().take(split).copied().collect::<Vector>();
        self.1 = self.1 - vector.iter().skip(split).copied().collect::<Vector>();
        self
    }
}

impl<T0, T1, T4, T5> Div<TensorTuple<T0, T1>> for &TensorTuple<T4, T5>
where
    T0: Tensor,
    T1: Tensor,
    T4: Tensor,
    T5: Tensor,
{
    type Output = TensorTuple<T4, T5>;
    fn div(self, _tensor_tuple: TensorTuple<T0, T1>) -> Self::Output {
        unimplemented!()
    }
}

impl<T1, T2, T> Differentiable<T> for TensorTuple<T1, T2>
where
    T1: Differentiable<T> + Tensor,
    T2: Differentiable<T> + Tensor,
    <T1 as Differentiable<T>>::Derivative: Tensor,
    <T2 as Differentiable<T>>::Derivative: Tensor,
{
    type Derivative =
        TensorTuple<<T1 as Differentiable<T>>::Derivative, <T2 as Differentiable<T>>::Derivative>;
}
