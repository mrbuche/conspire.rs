#[cfg(test)]
mod test;

use super::{SquareMatrix, SquareMatrixError};
use crate::{
    ABS_TOL, REL_TOL,
    math::{Scalar, Tensor, Vector, simd},
};

impl SquareMatrix {
    /// Factorize the symmetric matrix using the LDLᵀ decomposition.
    pub fn factorize_ldl(&self) -> Result<LdlDecomposition, SquareMatrixError> {
        let mut decomposition = LdlDecomposition {
            ldl: self.clone(),
            permutation: (0..self.len()).collect(),
            pair: vec![false; self.len()],
            column: vec![0.0; self.len()],
            other: vec![0.0; self.len()],
        };
        decomposition.factorize()?;
        Ok(decomposition)
    }
    /// Factorize the symmetric matrix into an existing LDLᵀ decomposition of the same size.
    pub fn factorize_ldl_into(
        &self,
        decomposition: &mut LdlDecomposition,
    ) -> Result<(), SquareMatrixError> {
        let LdlDecomposition {
            ldl, permutation, ..
        } = decomposition;
        ldl.iter_mut().zip(self.iter()).for_each(|(ldl_i, self_i)| {
            ldl_i
                .iter_mut()
                .zip(self_i.iter())
                .for_each(|(a, b)| *a = *b)
        });
        permutation
            .iter_mut()
            .enumerate()
            .for_each(|(i, permutation_i)| *permutation_i = i);
        decomposition.factorize()
    }
    /// Solve a system of linear equations using the LDLᵀ decomposition.
    pub fn solve_ldl(&self, b: &Vector) -> Result<Vector, SquareMatrixError> {
        Ok(self.factorize_ldl()?.solve(b))
    }
}

/// The LDLᵀ decomposition of a symmetric matrix.
pub struct LdlDecomposition {
    ldl: SquareMatrix,
    permutation: Vec<usize>,
    pair: Vec<bool>,
    column: Vec<Scalar>,
    other: Vec<Scalar>,
}

/// The Bunch-Kaufman threshold, balancing the growth a one-by-one pivot
/// admits against that of a two-by-two.
const BUNCH_KAUFMAN: Scalar = 0.640_388_203_202_207_8;

impl LdlDecomposition {
    /// An unfactorized decomposition sized to hold that of a matrix of the given length.
    pub fn zero(len: usize) -> Self {
        Self {
            ldl: SquareMatrix::zero(len),
            permutation: (0..len).collect(),
            pair: vec![false; len],
            column: vec![0.0; len],
            other: vec![0.0; len],
        }
    }
    /// The entry of the symmetric matrix held only in the lower triangle.
    fn at(&self, i: usize, j: usize) -> Scalar {
        if i >= j {
            self.ldl[i][j]
        } else {
            self.ldl[j][i]
        }
    }
    /// Symmetrically permute two indices, the lower triangle making each of the
    /// four affected stretches its own.
    fn exchange(&mut self, p: usize, q: usize) {
        if p == q {
            return;
        }
        let (p, q) = (p.min(q), p.max(q));
        for j in 0..p {
            let temp = self.ldl[p][j];
            self.ldl[p][j] = self.ldl[q][j];
            self.ldl[q][j] = temp
        }
        for i in p + 1..q {
            let temp = self.ldl[i][p];
            self.ldl[i][p] = self.ldl[q][i];
            self.ldl[q][i] = temp
        }
        for i in q + 1..self.ldl.len() {
            let temp = self.ldl[i][p];
            self.ldl[i][p] = self.ldl[i][q];
            self.ldl[i][q] = temp
        }
        let temp = self.ldl[p][p];
        self.ldl[p][p] = self.ldl[q][q];
        self.ldl[q][q] = temp;
        self.permutation.swap(p, q)
    }
    fn factorize(&mut self) -> Result<(), SquareMatrixError> {
        let n = self.ldl.len();
        self.pair.iter_mut().for_each(|paired| *paired = false);
        let mut largest = 0.0;
        let mut k = 0;
        while k < n {
            let mut omega = 0.0;
            let mut r = k;
            for i in k + 1..n {
                let candidate = self.ldl[i][k].abs();
                if candidate > omega {
                    omega = candidate;
                    r = i
                }
            }
            let mut block = 1;
            if omega > 0.0 && self.ldl[k][k].abs() < BUNCH_KAUFMAN * omega {
                let mut omega_r = 0.0;
                for i in k..n {
                    if i != r {
                        omega_r = self.at(i, r).abs().max(omega_r)
                    }
                }
                if self.ldl[k][k].abs() * omega_r < BUNCH_KAUFMAN * omega * omega {
                    if self.at(r, r).abs() >= BUNCH_KAUFMAN * omega_r {
                        self.exchange(k, r)
                    } else {
                        self.exchange(k + 1, r);
                        block = 2
                    }
                }
            }
            if block == 1 {
                let pivot = self.ldl[k][k];
                largest = pivot.abs().max(largest);
                if pivot.abs() < ABS_TOL && pivot.abs() <= REL_TOL * largest {
                    return Err(SquareMatrixError::Singular);
                }
                (k + 1..n).for_each(|i| self.column[i] = self.ldl[i][k]);
                (k + 1..n).for_each(|i| self.ldl[i][k] /= pivot);
                self.update_one(k, n)
            } else {
                let (a, b, c) = (self.ldl[k][k], self.at(k + 1, k), self.ldl[k + 1][k + 1]);
                let determinant = a * c - b * b;
                largest = a.abs().max(c.abs()).max(largest);
                if determinant.abs() < ABS_TOL && determinant.abs() <= (REL_TOL * largest).powi(2) {
                    return Err(SquareMatrixError::Singular);
                }
                (k + 2..n).for_each(|i| {
                    self.column[i] = self.ldl[i][k];
                    self.other[i] = self.ldl[i][k + 1]
                });
                (k + 2..n).for_each(|i| {
                    let (w_0, w_1) = (self.column[i], self.other[i]);
                    self.ldl[i][k] = (c * w_0 - b * w_1) / determinant;
                    self.ldl[i][k + 1] = (a * w_1 - b * w_0) / determinant
                });
                self.ldl[k + 1][k] = b;
                self.pair[k] = true;
                self.update_two(k, n)
            }
            k += block
        }
        Ok(())
    }
    /// Applies a one-by-one pivot to the trailing lower triangle.
    fn update_one(&mut self, k: usize, n: usize) {
        let Self { ldl, column, .. } = self;
        let source = &column[k + 1..n];
        let back = &mut ldl.0[k + 1..];
        let (quads, tail) = back.as_chunks_mut::<4>();
        quads
            .iter_mut()
            .enumerate()
            .for_each(|(chunk, [a, b, c, d])| {
                let base = 4 * chunk;
                let u = [-a[k], -b[k], -c[k], -d[k]];
                let common = base + 1;
                simd::rank_one_quad(
                    &mut a.as_mut_slice()[k + 1..k + 1 + common],
                    &mut b.as_mut_slice()[k + 1..k + 1 + common],
                    &mut c.as_mut_slice()[k + 1..k + 1 + common],
                    &mut d.as_mut_slice()[k + 1..k + 1 + common],
                    &source[..common],
                    u,
                );
                b[k + 1 + common] += u[1] * source[common];
                c[k + 1 + common] += u[2] * source[common];
                c[k + 2 + common] += u[2] * source[common + 1];
                d[k + 1 + common] += u[3] * source[common];
                d[k + 2 + common] += u[3] * source[common + 1];
                d[k + 3 + common] += u[3] * source[common + 2]
            });
        let base = 4 * quads.len();
        tail.iter_mut().enumerate().for_each(|(row, entries)| {
            let factor = entries[k];
            simd::axpy(
                &mut entries.as_mut_slice()[k + 1..k + 2 + base + row],
                &source[..base + row + 1],
                factor,
            )
        })
    }
    /// Applies a two-by-two pivot to the trailing lower triangle.
    fn update_two(&mut self, k: usize, n: usize) {
        let Self {
            ldl, column, other, ..
        } = self;
        let source = &column[k + 2..n];
        let paired = &other[k + 2..n];
        let back = &mut ldl.0[k + 2..];
        let (quads, tail) = back.as_chunks_mut::<4>();
        quads
            .iter_mut()
            .enumerate()
            .for_each(|(chunk, [a, b, c, d])| {
                let base = 4 * chunk;
                let u = [-a[k], -b[k], -c[k], -d[k]];
                let w = [-a[k + 1], -b[k + 1], -c[k + 1], -d[k + 1]];
                let common = base + 1;
                simd::rank_two_quad(
                    &mut a.as_mut_slice()[k + 2..k + 2 + common],
                    &mut b.as_mut_slice()[k + 2..k + 2 + common],
                    &mut c.as_mut_slice()[k + 2..k + 2 + common],
                    &mut d.as_mut_slice()[k + 2..k + 2 + common],
                    &source[..common],
                    &paired[..common],
                    u,
                    w,
                );
                b[k + 2 + common] += u[1] * source[common] + w[1] * paired[common];
                c[k + 2 + common] += u[2] * source[common] + w[2] * paired[common];
                c[k + 3 + common] += u[2] * source[common + 1] + w[2] * paired[common + 1];
                d[k + 2 + common] += u[3] * source[common] + w[3] * paired[common];
                d[k + 3 + common] += u[3] * source[common + 1] + w[3] * paired[common + 1];
                d[k + 4 + common] += u[3] * source[common + 2] + w[3] * paired[common + 2]
            });
        let base = 4 * quads.len();
        tail.iter_mut().enumerate().for_each(|(row, entries)| {
            let (u, w) = (entries[k], entries[k + 1]);
            entries.as_mut_slice()[k + 2..k + 3 + base + row]
                .iter_mut()
                .zip(source[..base + row + 1].iter())
                .zip(paired[..base + row + 1].iter())
                .for_each(|((entry, source_j), paired_j)| *entry -= u * source_j + w * paired_j)
        })
    }
    /// Solve a system of linear equations for another right-hand side.
    pub fn solve(&self, b: &Vector) -> Vector {
        let mut x = Vector::zero(self.permutation.len());
        self.solve_into(b, &mut x);
        x
    }
    /// Solve a system of linear equations into an existing vector.
    pub fn solve_into(&self, b: &Vector, x: &mut Vector) {
        let n = self.permutation.len();
        let mut y = Vector::zero(n);
        self.permutation
            .iter()
            .zip(y.iter_mut())
            .for_each(|(&p_i, y_i)| *y_i = b[p_i]);
        for i in 0..n {
            let stop = if i > 0 && self.pair[i - 1] { i - 1 } else { i };
            let sum = (0..stop)
                .map(|j| self.ldl[i][j].algebraic_mul(y[j]))
                .fold(0.0, f64::algebraic_add);
            y[i] -= sum
        }
        let mut k = 0;
        while k < n {
            if self.pair[k] {
                let (a, b_off, c) = (self.ldl[k][k], self.ldl[k + 1][k], self.ldl[k + 1][k + 1]);
                let determinant = a * c - b_off * b_off;
                let (y_0, y_1) = (y[k], y[k + 1]);
                y[k] = (c * y_0 - b_off * y_1) / determinant;
                y[k + 1] = (a * y_1 - b_off * y_0) / determinant;
                k += 2
            } else {
                y[k] /= self.ldl[k][k];
                k += 1
            }
        }
        for i in (0..n).rev() {
            let start = if self.pair[i] { i + 2 } else { i + 1 };
            let sum = (start..n)
                .map(|j| self.ldl[j][i].algebraic_mul(y[j]))
                .fold(0.0, f64::algebraic_add);
            y[i] -= sum
        }
        self.permutation
            .iter()
            .zip(y.iter())
            .for_each(|(&p_i, y_i)| x[p_i] = *y_i)
    }
}
