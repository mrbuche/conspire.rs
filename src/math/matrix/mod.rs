pub mod square;
pub mod vector;

use crate::math::{TensorRank0, TensorRank1Vec, TensorRank2, TensorVec};
use std::ops::{Index, IndexMut, Mul};
use vector::Vector;

/// A matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix(Vec<Vector>);

impl Matrix {
    pub fn height(&self) -> usize {
        self.0.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = &Vector> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Vector> {
        self.0.iter_mut()
    }
    pub fn width(&self) -> usize {
        self.0[0].len()
    }
    pub fn zero(height: usize, width: usize) -> Self {
        (0..height).map(|_| Vector::zero(width)).collect()
    }
}

impl FromIterator<Vector> for Matrix {
    fn from_iter<Ii: IntoIterator<Item = Vector>>(into_iterator: Ii) -> Self {
        Self(Vec::from_iter(into_iterator))
    }
}

impl Index<usize> for Matrix {
    type Output = Vector;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Mul<Vector> for &Matrix {
    type Output = Vector;
    fn mul(self, vector: Vector) -> Self::Output {
        self.iter().map(|self_i| self_i * &vector).collect()
    }
}

impl Mul<&Vector> for &Matrix {
    type Output = Vector;
    fn mul(self, vector: &Vector) -> Self::Output {
        self.iter().map(|self_i| self_i * vector).collect()
    }
}

impl Mul<&TensorRank0> for &Matrix {
    type Output = Vector;
    fn mul(self, _tensor_rank_0: &TensorRank0) -> Self::Output {
        panic!()
    }
}

impl<const D: usize, const I: usize> Mul<&TensorRank1Vec<D, I>> for &Matrix {
    type Output = Vector;
    fn mul(self, tensor_rank_1_vec: &TensorRank1Vec<D, I>) -> Self::Output {
        self.iter()
            .map(|self_i| self_i * tensor_rank_1_vec)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize> Mul<&TensorRank2<D, I, J>> for &Matrix {
    type Output = Vector;
    fn mul(self, tensor_rank_2: &TensorRank2<D, I, J>) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_2).collect()
    }
}
