#[cfg(test)]
mod test;

#[cfg(test)]
use super::test::ErrorTensor;

pub mod list;

use super::{Hessian, SquareMatrix, Tensor, TensorArray, TensorVec, Vector};

/// A tensor of rank 0 (a scalar).
pub type TensorRank0 = f64;

#[cfg(test)]
impl ErrorTensor for TensorRank0 {
    fn error(
        &self,
        comparator: &Self,
        tol_abs: &TensorRank0,
        tol_rel: &TensorRank0,
    ) -> Option<usize> {
        if &(self - comparator).abs() >= tol_abs && &(self / comparator - 1.0).abs() >= tol_rel {
            Some(1)
        } else {
            None
        }
    }
    fn error_fd(&self, comparator: &Self, epsilon: &TensorRank0) -> Option<(bool, usize)> {
        if &(self / comparator - 1.0).abs() >= epsilon {
            Some((true, 1))
        } else {
            None
        }
    }
}

impl Hessian for TensorRank0 {
    fn fill_into(self, _square_matrix: &mut SquareMatrix) {
        panic!()
    }
    fn into_matrix(self) -> SquareMatrix {
        panic!()
    }
    fn is_positive_definite(&self) -> bool {
        self > &0.0
    }
}

impl Tensor for TensorRank0 {
    type Item = TensorRank0;
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
    fn normalized(self) -> Self {
        1.0
    }
}

impl TensorArray for TensorRank0 {
    type Array = [Self; 1];
    type Item = TensorRank0;
    fn as_array(&self) -> Self::Array {
        [*self]
    }
    fn identity() -> Self {
        1.0
    }
    fn new(array: Self::Array) -> Self {
        array[0]
    }
    fn zero() -> Self {
        0.0
    }
}

impl From<TensorRank0> for Vector {
    fn from(tensor_rank_0: TensorRank0) -> Self {
        Vector::new(&[tensor_rank_0])
    }
}
