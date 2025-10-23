pub mod square;
pub mod vector;

use crate::math::{Scalar, Tensor, TensorRank1, TensorRank1Vec, TensorRank2};
use std::ops::{Index, IndexMut, Mul};
use vector::Vector;

/// A matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix(Vec<Vector>);

impl Matrix {
    pub fn height(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = &Vector> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Vector> {
        self.0.iter_mut()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn transpose(&self) -> Self {
        (0..self.width())
            .map(|i| (0..self.len()).map(|j| self[j][i]).collect())
            .collect()
        // let mut transpose = Self::zero(self.width(), self.len());
        // self.iter().enumerate().for_each(|(i, self_i)|
        //     self_i.iter().zip(transpose.iter_mut()).for_each(|(self_ij, transpose_j)|
        //         transpose_j[i] = *self_ij
        //     )
        // );
        // transpose
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

impl IntoIterator for Matrix {
    type Item = Vector;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
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

impl Mul<&Scalar> for &Matrix {
    type Output = Vector;
    fn mul(self, _tensor_rank_0: &Scalar) -> Self::Output {
        unimplemented!()
    }
}

impl<const D: usize, const I: usize> Mul<&TensorRank1<D, I>> for &Matrix {
    type Output = Vector;
    fn mul(self, _tensor_rank_1: &TensorRank1<D, I>) -> Self::Output {
        unimplemented!()
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

impl Mul for Matrix {
    type Output = Self;
    fn mul(self, matrix: Self) -> Self::Output {
        let mut output = Self::zero(self.len(), matrix.width());
        self.iter()
            .zip(output.iter_mut())
            .for_each(|(self_i, output_i)| {
                self_i
                    .iter()
                    .zip(matrix.iter())
                    .for_each(|(self_ij, matrix_j)| *output_i += matrix_j * self_ij)
            });
        output
    }
}
