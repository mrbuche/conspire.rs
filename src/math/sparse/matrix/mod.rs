#[cfg(test)]
mod test;

use crate::math::{Scalar, Vector};
use std::ops::Mul;

/// A sparse matrix in compressed sparse column format.
#[derive(Clone, Debug, PartialEq)]
pub struct CscMartix {
    height: usize,
    width: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<Scalar>,
    pattern: Vec<(usize, usize)>,
    scatter: Vec<usize>,
}

impl CscMartix {
    /// Builds the sparsity structure from a list of nonzero (row, column) positions,
    /// with all values initialized to zero.
    pub fn from_pattern(height: usize, width: usize, pattern: Vec<(usize, usize)>) -> Self {
        assert!(!pattern.is_empty(), "Matrix must have at least one entry.");
        let mut order: Vec<usize> = (0..pattern.len()).collect();
        order.sort_unstable_by_key(|&k| (pattern[k].1, pattern[k].0));
        let mut col_ptr = vec![0; width + 1];
        let mut row_idx = Vec::with_capacity(pattern.len());
        let mut scatter = vec![0; pattern.len()];
        let mut last = (usize::MAX, usize::MAX);
        order.into_iter().for_each(|k| {
            let (i, j) = pattern[k];
            assert!(i < height && j < width, "Position out of bounds.");
            if (j, i) != last {
                last = (j, i);
                row_idx.push(i);
                col_ptr[j + 1] += 1;
            }
            scatter[k] = row_idx.len() - 1;
        });
        (0..width).for_each(|j| col_ptr[j + 1] += col_ptr[j]);
        let values = vec![0.0; row_idx.len()];
        Self {
            height,
            width,
            col_ptr,
            row_idx,
            values,
            pattern,
            scatter,
        }
    }
    /// Fills the values from a source, summing duplicate positions in the pattern.
    pub fn fill(&mut self, mut source: impl FnMut(usize, usize) -> Scalar) {
        self.values.fill(0.0);
        self.pattern
            .iter()
            .zip(self.scatter.iter())
            .for_each(|(&(i, j), &k)| self.values[k] += source(i, j));
    }
    pub fn height(&self) -> usize {
        self.height
    }
    /// Iterates over the nonzero entries as (row, column, value), in column-major order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &Scalar)> {
        (0..self.width).flat_map(move |j| {
            (self.col_ptr[j]..self.col_ptr[j + 1])
                .map(move |k| (self.row_idx[k], j, &self.values[k]))
        })
    }
    pub fn nonzeros(&self) -> usize {
        self.row_idx.len()
    }
    /// The nonzero (row, column) positions this structure was built from.
    pub fn pattern(&self) -> &[(usize, usize)] {
        &self.pattern
    }
    pub fn transpose(&self) -> Self {
        let nnz = self.row_idx.len();
        let mut col_ptr = vec![0; self.height + 1];
        self.row_idx.iter().for_each(|&i| col_ptr[i + 1] += 1);
        (0..self.height).for_each(|i| col_ptr[i + 1] += col_ptr[i]);
        let mut next = col_ptr.clone();
        let mut row_idx = vec![0; nnz];
        let mut values = vec![0.0; nnz];
        let mut perm = vec![0; nnz];
        (0..self.width).for_each(|j| {
            (self.col_ptr[j]..self.col_ptr[j + 1]).for_each(|k| {
                let p = next[self.row_idx[k]];
                next[self.row_idx[k]] += 1;
                row_idx[p] = j;
                values[p] = self.values[k];
                perm[k] = p;
            })
        });
        Self {
            height: self.width,
            width: self.height,
            col_ptr,
            row_idx,
            values,
            pattern: self.pattern.iter().map(|&(i, j)| (j, i)).collect(),
            scatter: self.scatter.iter().map(|&k| perm[k]).collect(),
        }
    }
    pub fn width(&self) -> usize {
        self.width
    }
}

impl Mul<&Vector> for &CscMartix {
    type Output = Vector;
    fn mul(self, vector: &Vector) -> Self::Output {
        let mut output = Vector::zero(self.height);
        (0..self.width).for_each(|j| {
            (self.col_ptr[j]..self.col_ptr[j + 1])
                .for_each(|k| output[self.row_idx[k]] += self.values[k] * vector[j])
        });
        output
    }
}
