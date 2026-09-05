#[cfg(test)]
mod test;

use super::{SquareMatrix, SquareMatrixError};
use crate::{
    ABS_TOL, REL_TOL,
    math::{Tensor, Vector, simd},
};

impl SquareMatrix {
    /// Factorize the matrix using the LU decomposition.
    pub fn factorize_lu(&self) -> Result<LuDecomposition, SquareMatrixError> {
        let mut decomposition = LuDecomposition {
            lu: self.clone(),
            permutation: (0..self.len()).collect(),
        };
        decomposition.factorize()?;
        Ok(decomposition)
    }
    /// Factorize the matrix into an existing LU decomposition of the same size.
    pub fn factorize_lu_into(
        &self,
        decomposition: &mut LuDecomposition,
    ) -> Result<(), SquareMatrixError> {
        let LuDecomposition { lu, permutation } = decomposition;
        lu.iter_mut().zip(self.iter()).for_each(|(lu_i, self_i)| {
            lu_i.iter_mut()
                .zip(self_i.iter())
                .for_each(|(a, b)| *a = *b)
        });
        permutation
            .iter_mut()
            .enumerate()
            .for_each(|(i, permutation_i)| *permutation_i = i);
        decomposition.factorize()
    }
    /// Solve a system of linear equations using the LU decomposition.
    pub fn solve_lu(&self, b: &Vector) -> Result<Vector, SquareMatrixError> {
        Ok(self.factorize_lu()?.solve(b))
    }
}

/// The LU decomposition of a square matrix.
pub struct LuDecomposition {
    lu: SquareMatrix,
    permutation: Vec<usize>,
}

impl LuDecomposition {
    fn factorize(&mut self) -> Result<(), SquareMatrixError> {
        let Self { lu, permutation } = self;
        let n = lu.len();
        let mut largest = 0.0;
        for i in 0..n {
            let mut max_row = i;
            let mut max_val = lu[i][i].abs();
            for k in i + 1..n {
                let candidate = lu[k][i].abs();
                if candidate > max_val {
                    max_row = k;
                    max_val = candidate;
                }
            }
            if max_row != i {
                lu.0.swap(i, max_row);
                permutation.swap(i, max_row);
            }
            largest = max_val.max(largest);
            if max_val < ABS_TOL && max_val <= REL_TOL * largest {
                return Err(SquareMatrixError::Singular);
            }
            let pivot = lu[i][i];
            let (front, back) = lu.0.split_at_mut(i + 1);
            let column = &front[i].as_slice()[i + 1..];
            let mut count = 0;
            for row in 0..back.len() {
                let factor = back[row][i];
                if factor != 0.0 {
                    back[row][i] = factor / pivot;
                    if row != count {
                        back.swap(row, count);
                        permutation.swap(i + 1 + row, i + 1 + count)
                    }
                    count += 1
                }
            }
            if column.len() < 4 {
                back[..count].iter_mut().for_each(|row| {
                    let factor = row[i];
                    row.as_mut_slice()[i + 1..]
                        .iter_mut()
                        .zip(column.iter())
                        .for_each(|(row_k, column_k)| *row_k -= factor * column_k)
                })
            } else {
                back[..count].chunks_mut(4).for_each(|chunk| {
                    if let [a, b, c, d] = chunk {
                        let u = [-a[i], -b[i], -c[i], -d[i]];
                        simd::rank_one_quad(
                            &mut a.as_mut_slice()[i + 1..],
                            &mut b.as_mut_slice()[i + 1..],
                            &mut c.as_mut_slice()[i + 1..],
                            &mut d.as_mut_slice()[i + 1..],
                            column,
                            u,
                        )
                    } else {
                        chunk.iter_mut().for_each(|row| {
                            let factor = row[i];
                            simd::axpy(&mut row.as_mut_slice()[i + 1..], column, factor)
                        })
                    }
                })
            }
        }
        Ok(())
    }
    /// An unfactorized decomposition sized to hold that of a matrix of the given length.
    pub fn zero(len: usize) -> Self {
        Self {
            lu: SquareMatrix::zero(len),
            permutation: (0..len).collect(),
        }
    }
    /// Solve a system of linear equations for another right-hand side.
    pub fn solve(&self, b: &Vector) -> Vector {
        let mut x = Vector::zero(self.permutation.len());
        self.solve_into(b, &mut x);
        x
    }
    /// Solve a system of linear equations into an existing vector.
    pub fn solve_into(&self, b: &Vector, x: &mut Vector) {
        self.permutation
            .iter()
            .zip(x.iter_mut())
            .for_each(|(&p_i, x_i)| *x_i = b[p_i]);
        forward_substitution(x, &self.lu);
        backward_substitution(x, &self.lu)
    }
}

fn forward_substitution(x: &mut Vector, a: &SquareMatrix) {
    a.iter().enumerate().for_each(|(i, a_i)| {
        x[i] -= a_i
            .iter()
            .take(i)
            .zip(x.iter().take(i))
            .map(|(a_ij, x_j)| a_ij.algebraic_mul(*x_j))
            .fold(0.0, f64::algebraic_add)
    })
}

fn backward_substitution(x: &mut Vector, a: &SquareMatrix) {
    a.0.iter().enumerate().rev().for_each(|(i, a_i)| {
        x[i] -= a_i
            .iter()
            .skip(i + 1)
            .zip(x.iter().skip(i + 1))
            .map(|(a_ij, x_j)| a_ij.algebraic_mul(*x_j))
            .fold(0.0, f64::algebraic_add);
        x[i] /= a_i[i];
    })
}
