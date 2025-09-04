#[cfg(test)]
mod test;

use crate::math::{Scalar, write_tensor_rank_0};
use std::{
    fmt::{self, Display, Formatter},
    ops::{Index, Mul},
};

/// A sparse matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseMatrix {
    rows: usize,
    cols: usize,
    values: Vec<Scalar>,
    col_indices: Vec<usize>,
    row_pointers: Vec<usize>,
}

impl SparseMatrix {
    /// Adds a value to the sparse matrix at the specified row and column.
    pub fn add_value(&mut self, row: usize, col: usize, val: Scalar) {
        self.values.push(val);
        self.col_indices.push(col);
        (row + 1..=self.rows).for_each(|i| self.row_pointers[i] += 1)
    }
    /// Creates a new sparse matrix with the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_pointers: vec![0; rows + 1],
        }
    }
    /// Solve a system of linear equations using the LU decomposition.
    pub fn solve_lu(&self, b: &Vector) -> Vector {
    // pub fn solve_lu(&self, b: &Vector) -> Result<Vector, SquareMatrixError> {
    //
    // just make this SparseSquareMatrix? And put in square/sparse?
    //
        let n = self.rows;
        let mut p: Vec<usize> = (0..n).collect();
        let mut factor;
        let mut lu = self.clone();
        let mut max_row;
        let mut max_val;
        let mut pivot;
        for i in 0..n {
            max_row = i;
            max_val = lu[[max_row, i]].abs();
            for k in i + 1..n {
                if lu[[k, i]].abs() > max_val {
                    max_row = k;
                    max_val = lu[[max_row, i]].abs();
                }
            }
            // if max_row != i {
            //     lu.0.swap(i, max_row);
            //     p.swap(i, max_row);
            // }
            pivot = lu[[i, i]];
            if pivot.abs() < ABS_TOL {
                return Err(SquareMatrixError::Singular);
            }
            for j in i + 1..n {
                if lu[[j, i]] != 0.0 {
                    lu[[j, i]] /= pivot;
                    factor = lu[[j, i]];
                    for k in i + 1..n {
                        lu[[j, k]] -= factor * lu[[i, k]];
                    }
                }
            }
        }
        let mut x: Vector = p.into_iter().map(|p_i| b[p_i]).collect();
        forward_substitution(&mut x, &lu);
        backward_substitution(&mut x, &lu);
        Ok(x)
    }
}

impl Display for SparseMatrix {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "\x1B[s")?;
        write!(f, "[[")?;
        (0..self.rows).try_for_each(|row| {
            (0..self.cols).try_for_each(|col| write_tensor_rank_0(f, &self[[row, col]]))?;
            if row + 1 < self.rows {
                writeln!(f, "\x1B[2D],")?;
                write!(f, "\x1B[u")?;
                write!(f, "\x1B[{}B [", row + 1)?;
            }
            Ok(())
        })?;
        write!(f, "\x1B[2D]]")
    }
}

impl Index<[usize; 2]> for SparseMatrix {
    type Output = Scalar;
    fn index(&self, [row, col]: [usize; 2]) -> &Self::Output {
        for i in self.row_pointers[row]..self.row_pointers[row + 1] {
            if self.col_indices[i] == col {
                return &self.values[i];
            }
        }
        &0.0
    }
}

impl Mul<&Self> for SparseMatrix {
    type Output = Self;
    fn mul(self, other: &Self) -> Self::Output {
        assert!(
            self.cols == other.rows,
            "Matrix dimensions do not match for multiplication"
        );

        let mut result_values = Vec::new();
        let mut result_col_indices = Vec::new();
        let mut result_row_pointers = vec![0; self.rows + 1];

        for row in 0..self.rows {
            let mut row_result = std::collections::HashMap::new();

            let start_a = self.row_pointers[row];
            let end_a = self.row_pointers[row + 1];

            for i in start_a..end_a {
                let col_a = self.col_indices[i];
                let value_a = self.values[i];

                let start_b = other.row_pointers[col_a];
                let end_b = other.row_pointers[col_a + 1];

                for j in start_b..end_b {
                    let col_b = other.col_indices[j];
                    let value_b = other.values[j];

                    let entry = row_result.entry(col_b).or_insert(Scalar::default());
                    *entry += value_a * value_b;
                }
            }

            for (&col, &value) in row_result.iter() {
                if value != Scalar::default() {
                    result_values.push(value);
                    result_col_indices.push(col);
                }
            }

            result_row_pointers[row + 1] = result_values.len();
        }

        SparseMatrix {
            rows: self.rows,
            cols: other.cols,
            values: result_values,
            col_indices: result_col_indices,
            row_pointers: result_row_pointers,
        }
    }
}
