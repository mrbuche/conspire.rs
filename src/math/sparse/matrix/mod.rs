#[cfg(test)]
mod test;

use crate::math::{Scalar, Vector};
use std::ops::Mul;

/// A sparse matrix in compressed sparse column format.
#[derive(Clone, Debug, PartialEq)]
pub struct CscMatrix {
    height: usize,
    width: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<Scalar>,
    pattern: Vec<(usize, usize)>,
    scatter: Vec<usize>,
}

impl CscMatrix {
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
    /// Iterates over the nonzero entries of a column as (row, value).
    pub fn column(&self, j: usize) -> impl Iterator<Item = (usize, &Scalar)> {
        (self.col_ptr[j]..self.col_ptr[j + 1]).map(move |k| (self.row_idx[k], &self.values[k]))
    }
    pub fn height(&self) -> usize {
        self.height
    }
    /// Iterates over the nonzero entries as (row, column, value), in column-major order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &Scalar)> {
        (0..self.width).flat_map(move |j| self.column(j).map(move |(i, value)| (i, j, value)))
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
    /// A column-to-row matching pairing every column with a structurally
    /// nonzero row (a maximum transversal), or None if structurally singular.
    pub(crate) fn maxtrans(&self) -> Option<Vec<usize>> {
        const NONE: usize = usize::MAX;
        let n = self.width;
        assert_eq!(n, self.height);
        if (0..n).all(|c| self.row_idx[self.col_ptr[c]..self.col_ptr[c + 1]].contains(&c)) {
            return Some((0..n).collect());
        }
        let mut cmatch = vec![NONE; n];
        let mut rmatch = vec![NONE; n];
        (0..n).for_each(|c| {
            if self.row_idx[self.col_ptr[c]..self.col_ptr[c + 1]].contains(&c) {
                rmatch[c] = c;
                cmatch[c] = c;
            }
        });
        (0..n).for_each(|c| {
            if cmatch[c] == NONE {
                for &r in &self.row_idx[self.col_ptr[c]..self.col_ptr[c + 1]] {
                    if rmatch[r] == NONE {
                        rmatch[r] = c;
                        cmatch[c] = r;
                        break;
                    }
                }
            }
        });
        let mut visited = vec![NONE; n];
        let mut cstack = vec![0; n];
        let mut rstack = vec![0; n];
        let mut estack = vec![0; n];
        for root in 0..n {
            if cmatch[root] != NONE {
                continue;
            }
            let mut head = 0;
            cstack[0] = root;
            estack[0] = self.col_ptr[root];
            let mut found = false;
            'dfs: loop {
                let c = cstack[head];
                let end = self.col_ptr[c + 1];
                if estack[head] == self.col_ptr[c] {
                    for &r in &self.row_idx[self.col_ptr[c]..end] {
                        if rmatch[r] == NONE {
                            rstack[head] = r;
                            (0..=head).for_each(|level| {
                                cmatch[cstack[level]] = rstack[level];
                                rmatch[rstack[level]] = cstack[level];
                            });
                            found = true;
                            break 'dfs;
                        }
                    }
                }
                let mut descended = false;
                while estack[head] < end {
                    let r = self.row_idx[estack[head]];
                    estack[head] += 1;
                    if visited[r] == root || rmatch[r] == NONE {
                        continue;
                    }
                    visited[r] = root;
                    rstack[head] = r;
                    head += 1;
                    cstack[head] = rmatch[r];
                    estack[head] = self.col_ptr[rmatch[r]];
                    descended = true;
                    break;
                }
                if !descended {
                    if head == 0 {
                        break 'dfs;
                    }
                    head -= 1;
                }
            }
            if !found {
                return None;
            }
        }
        Some(cmatch)
    }
    pub fn width(&self) -> usize {
        self.width
    }
}

impl Mul<&Vector> for &CscMatrix {
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
