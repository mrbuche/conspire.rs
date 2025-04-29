pub mod square;
pub mod vector;

use crate::math::{TensorRank0, TensorVec};
use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};
use vector::Vector;

/// A matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix(Vec<Vector>);

impl Matrix {
    pub fn iter(&self) -> impl Iterator<Item = &Vector> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Vector> {
        self.0.iter_mut()
    }
}

impl Mul<&Vector> for &Matrix {
    type Output = Vector;
    fn mul(self, vector: &Vector) -> Self::Output {
        self.iter().map(|self_i| self_i * vector).collect()
    }
}
