#[cfg(test)]
mod test;

use crate::math::{HessianBlock, Tensor, TensorRank0, TensorTuple, Vector};
use std::ops::IndexMut;

/// A block matrix of four blocks that may differ in type, as `[[A, B], [C, D]]`.
///
/// `C` is held as its transpose, so a row is stored as the column it is
/// the transpose of.
pub type TensorBlock2x2<A, B, C, D> = TensorTuple<TensorTuple<A, B>, TensorTuple<C, D>>;

impl<A, B, C, D> HessianBlock for TensorBlock2x2<A, B, C, D>
where
    A: HessianBlock + Tensor,
    B: HessianBlock + Tensor,
    C: HessianBlock + Tensor,
    D: HessianBlock + Tensor,
{
    fn entry(&self, row: usize, column: usize) -> TensorRank0 {
        let (top, left) = (self.0.0.height(), self.0.0.width());
        match (row < top, column < left) {
            (true, true) => self.0.0.entry(row, column),
            (true, false) => self.0.1.entry(row, column - left),
            (false, true) => self.1.0.entry(column, row - top),
            (false, false) => self.1.1.entry(row - top, column - left),
        }
    }
    fn height(&self) -> usize {
        self.0.0.height() + self.1.1.height()
    }
    fn width(&self) -> usize {
        self.0.0.width() + self.0.1.width()
    }
    fn fill_into_block<M>(&self, matrix: &mut M, row: usize, column: usize)
    where
        M: IndexMut<usize, Output = Vector>,
    {
        let (top, left) = (self.0.0.height(), self.0.0.width());
        self.0.0.fill_into_block(matrix, row, column);
        self.0.1.fill_into_block(matrix, row, column + left);
        (0..self.1.0.height()).for_each(|i| {
            (0..self.1.0.width())
                .for_each(|j| matrix[row + top + j][column + i] = self.1.0.entry(i, j))
        });
        self.1.1.fill_into_block(matrix, row + top, column + left);
    }
}
