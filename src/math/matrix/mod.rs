pub mod square;
pub mod vector;

use crate::math::{TensorVec, TensorRank0};
use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};
use vector::Vector;

delete Matrix probably

/// A matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix(Vec<Vector>);

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
