#[cfg(test)]
mod test;

#[cfg(test)]
use super::test::ErrorTensor;

pub mod list;

use super::{Hessian, Jacobian, Solution, SquareMatrix, Tensor, TensorArray, Vector};
use std::ops::Sub;

/// A tensor of rank 0 (a scalar).
pub type TensorRank0 = f64;

#[cfg(test)]
impl ErrorTensor for TensorRank0 {
    fn error_fd(&self, comparator: &Self, epsilon: &TensorRank0) -> Option<(bool, usize)> {
        if &(self / comparator - 1.0).abs() >= epsilon {
            Some((true, 1))
        } else {
            None
        }
    }
}

impl Solution for TensorRank0 {
    fn decrement_from(&mut self, _other: &Vector) {
        unimplemented!()
    }
    fn decrement_from_chained(&mut self, _other: &mut Vector, _vector: Vector) {
        unimplemented!()
    }
}

impl Jacobian for TensorRank0 {
    fn fill_into(self, _vector: &mut Vector) {
        unimplemented!()
    }
    fn fill_into_chained(self, _other: Vector, _vector: &mut Vector) {
        unimplemented!()
    }
}

impl Sub<Vector> for TensorRank0 {
    type Output = Self;
    fn sub(self, _vector: Vector) -> Self::Output {
        unimplemented!()
    }
}

impl Sub<&Vector> for TensorRank0 {
    type Output = Self;
    fn sub(self, _vector: &Vector) -> Self::Output {
        unimplemented!()
    }
}

impl Hessian for TensorRank0 {
    fn fill_into(self, _square_matrix: &mut SquareMatrix) {
        unimplemented!()
    }
}

impl Tensor for TensorRank0 {
    type Item = TensorRank0;
    fn error_count(
        &self,
        other: &Self,
        tol_abs: &TensorRank0,
        tol_rel: &TensorRank0,
    ) -> Option<usize> {
        if &self.sub_abs(other) < tol_abs || &self.sub_rel(other) < tol_rel {
            None
        } else {
            Some(1)
        }
    }
    fn full_contraction(&self, tensor_rank_0: &Self) -> TensorRank0 {
        self * tensor_rank_0
    }
    fn is_zero(&self) -> bool {
        self == &0.0
    }
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        [0.0].iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        [self].into_iter()
    }
    fn len(&self) -> usize {
        1
    }
    fn norm_inf(&self) -> TensorRank0 {
        self.abs()
    }
    fn normalized(self) -> Self {
        1.0
    }
    fn size(&self) -> usize {
        1
    }
    fn sub_abs(&self, other: &Self) -> Self {
        (self - other).abs()
    }
    fn sub_rel(&self, other: &Self) -> Self {
        if other == &0.0 {
            if self == &0.0 { 0.0 } else { 1.0 }
        } else {
            (self / other - 1.0).abs()
        }
    }
}

impl TensorArray for TensorRank0 {
    type Array = Self;
    type Item = TensorRank0;
    fn as_array(&self) -> Self::Array {
        *self
    }
    fn identity() -> Self {
        1.0
    }
    fn new(array: Self::Array) -> Self {
        array
    }
    fn zero() -> Self {
        0.0
    }
}

impl From<Vector> for TensorRank0 {
    fn from(_vector: Vector) -> Self {
        unimplemented!()
    }
}
