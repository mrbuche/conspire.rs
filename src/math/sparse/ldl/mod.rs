#[cfg(test)]
mod test;

use super::{
    SparseError,
    lu::{CHUNK, NONE, gemm},
    matrix::CscMatrix,
};
use crate::{
    ABS_TOL,
    math::{Scalar, Vector},
};

/// A sparse LDLᵀ factorization, PAPᵀ = LDLᵀ, for symmetric matrices with a
/// structurally full diagonal, with L stored as dense supernodal panels.
pub struct CscLdl {
    fill: usize,
    sn_of: Vec<usize>,
    sn_start: Vec<usize>,
    sn_rows_ptr: Vec<usize>,
    sn_rows: Vec<usize>,
    sn_panel_ptr: Vec<usize>,
    sn_values: Vec<Scalar>,
    row_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    row_pos: Vec<usize>,
    d: Vec<Scalar>,
    pinv: Vec<usize>,
    q: Vec<usize>,
}

impl CscMatrix {
    /// Builds the LDLᵀ factorization pattern symbolically for the fill-reducing
    /// ordering, requiring a structurally full diagonal; the values must be
    /// symmetric and are all zero until a refactorization supplies them.
    pub fn ldl_symbolic(&self) -> Result<CscLdl, SparseError> {
        let lu = self.lu_symbolic()?;
        if lu.q.iter().enumerate().any(|(j, &q_j)| lu.pinv[q_j] != j) {
            return Err(SparseError::Unsymmetric);
        }
        let mut row_pos = Vec::with_capacity(lu.u_row_idx.len());
        lu.u_col_ptr.windows(2).enumerate().for_each(|(j, window)| {
            lu.u_row_idx[window[0]..window[1]].iter().for_each(|&k| {
                if k == j {
                    row_pos.push(NONE);
                } else {
                    let t = lu.sn_of[k];
                    let rows = &lu.sn_rows[lu.sn_rows_ptr[t]..lu.sn_rows_ptr[t + 1]];
                    row_pos.push(
                        lu.sn_panel_ptr[t]
                            + (k - lu.sn_start[t]) * rows.len()
                            + rows.binary_search(&j).expect("Row not in supernode."),
                    );
                }
            });
        });
        let n = lu.pinv.len();
        Ok(CscLdl {
            fill: lu.sn_panel_ptr[lu.sn_start.len() - 1] + n,
            sn_of: lu.sn_of,
            sn_start: lu.sn_start,
            sn_rows_ptr: lu.sn_rows_ptr,
            sn_rows: lu.sn_rows,
            sn_panel_ptr: lu.sn_panel_ptr,
            sn_values: lu.sn_values,
            row_ptr: lu.u_col_ptr,
            row_idx: lu.u_row_idx,
            row_pos,
            d: vec![0.0; n],
            pinv: lu.pinv,
            q: lu.q,
        })
    }
}

impl CscLdl {
    /// Solve a system of linear equations using the factorization.
    pub fn solve(&self, b: &Vector) -> Vector {
        let n = self.pinv.len();
        let mut x = vec![0.0; n];
        let mut below = vec![0.0; self.max_below()];
        self.pinv
            .iter()
            .enumerate()
            .for_each(|(i, &p_i)| x[p_i] = b[i]);
        (0..self.sn_start.len() - 1).for_each(|s| {
            let t1 = self.sn_start[s];
            let t2 = self.sn_start[s + 1];
            let width = t2 - t1;
            let rows = &self.sn_rows[self.sn_rows_ptr[s]..self.sn_rows_ptr[s + 1]];
            let m = rows.len();
            let panel = &self.sn_values[self.sn_panel_ptr[s]..self.sn_panel_ptr[s + 1]];
            (0..width).for_each(|c| {
                let x_c = x[t1 + c];
                if x_c != 0.0 {
                    let column = &panel[c * m..(c + 1) * m];
                    x[t1 + c + 1..t2]
                        .iter_mut()
                        .zip(column[c + 1..width].iter())
                        .for_each(|(x_r, value)| *x_r -= value * x_c);
                    below[..m - width]
                        .iter_mut()
                        .zip(column[width..].iter())
                        .for_each(|(below_r, value)| *below_r += value * x_c);
                }
            });
            rows[width..]
                .iter()
                .zip(below[..m - width].iter_mut())
                .for_each(|(&row, below_r)| {
                    x[row] -= *below_r;
                    *below_r = 0.0;
                });
        });
        x.iter_mut().zip(self.d.iter()).for_each(|(x_j, d_j)| {
            *x_j /= d_j;
        });
        (0..self.sn_start.len() - 1).rev().for_each(|s| {
            let t1 = self.sn_start[s];
            let t2 = self.sn_start[s + 1];
            let width = t2 - t1;
            let rows = &self.sn_rows[self.sn_rows_ptr[s]..self.sn_rows_ptr[s + 1]];
            let m = rows.len();
            let panel = &self.sn_values[self.sn_panel_ptr[s]..self.sn_panel_ptr[s + 1]];
            (0..width).rev().for_each(|c| {
                let column = &panel[c * m..(c + 1) * m];
                let mut x_c = x[t1 + c];
                x[t1 + c + 1..t2]
                    .iter()
                    .zip(column[c + 1..width].iter())
                    .for_each(|(x_r, value)| x_c -= value * x_r);
                rows[width..]
                    .iter()
                    .zip(column[width..].iter())
                    .for_each(|(&row, value)| x_c -= value * x[row]);
                x[t1 + c] = x_c;
            });
        });
        let mut solution = Vector::zero(n);
        self.q
            .iter()
            .enumerate()
            .for_each(|(j, &q_j)| solution[q_j] = x[j]);
        solution
    }
    /// The number of nonzero entries in the factors.
    pub fn nonzeros(&self) -> usize {
        self.fill
    }
    /// Recomputes the factorization for new values in the same pattern, reusing
    /// the pivot order and fill pattern without any symbolic work or pivot search.
    /// The factorization is invalid if an error is returned.
    pub fn refactor(&mut self, matrix: &CscMatrix) -> Result<(), SparseError> {
        let n = self.pinv.len();
        assert_eq!(n, matrix.height());
        let mut work = vec![0.0; n * CHUNK];
        let mut temp = vec![0.0; 4 * self.max_below()];
        let mut pointers = [0; CHUNK];
        for s in 0..self.sn_start.len() - 1 {
            let s1 = self.sn_start[s];
            let s2 = self.sn_start[s + 1];
            let s_width = s2 - s1;
            let s_rows_start = self.sn_rows_ptr[s];
            let s_m = self.sn_rows_ptr[s + 1] - s_rows_start;
            let mut c1 = s1;
            while c1 < s2 {
                let c2 = s2.min(c1 + CHUNK);
                let chunk = c2 - c1;
                (c1..c2).zip(pointers.iter_mut()).for_each(|(j, pointer)| {
                    let pinv = &self.pinv;
                    let column = &mut work[(j - c1) * n..(j - c1 + 1) * n];
                    matrix
                        .column(self.q[j])
                        .for_each(|(i, value)| column[pinv[i]] = *value);
                    *pointer = self.row_ptr[j];
                });
                loop {
                    let mut t = NONE;
                    (c1..c2).zip(pointers.iter()).for_each(|(j, &pointer)| {
                        if pointer < self.row_ptr[j + 1] - 1 {
                            let row = self.row_idx[pointer];
                            if row < c1 && (t == NONE || self.sn_of[row] < t) {
                                t = self.sn_of[row];
                            }
                        }
                    });
                    if t == NONE {
                        break;
                    }
                    let t1 = self.sn_start[t];
                    let t2 = self.sn_start[t + 1];
                    let width = t2 - t1;
                    let consumed = width.min(c1 - t1);
                    let rows = &self.sn_rows[self.sn_rows_ptr[t]..self.sn_rows_ptr[t + 1]];
                    let m = rows.len();
                    let below = m - width;
                    let panel = &self.sn_values[self.sn_panel_ptr[t]..self.sn_panel_ptr[t + 1]];
                    (c1..c2).zip(pointers.iter_mut()).for_each(|(j, pointer)| {
                        let p_end = self.row_ptr[j + 1] - 1;
                        let column = &mut work[(j - c1) * n..(j - c1 + 1) * n];
                        while *pointer < p_end {
                            let k = self.row_idx[*pointer];
                            if k >= c1 || k >= t2 {
                                break;
                            }
                            let w = self.sn_values[self.row_pos[*pointer]] * self.d[k];
                            column[k] = w;
                            if t2 > c1 && w != 0.0 {
                                let c = k - t1;
                                column[c1..t2]
                                    .iter_mut()
                                    .zip(panel[c * m + c1 - t1..c * m + width].iter())
                                    .for_each(|(work_r, value)| *work_r -= value * w);
                            }
                            *pointer += 1;
                        }
                    });
                    if below > 0 {
                        let mut c = 0;
                        while c < chunk {
                            let block = (chunk - c).min(4);
                            temp[..block * below].fill(0.0);
                            gemm(
                                &mut temp[..block * below],
                                &work[c * n..],
                                n,
                                panel,
                                m,
                                width,
                                t1,
                                consumed,
                                below,
                                block,
                            );
                            (0..block).for_each(|b| {
                                let column = &mut work[(c + b) * n..(c + b + 1) * n];
                                rows[width..]
                                    .iter()
                                    .zip(temp[b * below..(b + 1) * below].iter())
                                    .for_each(|(&row, value)| column[row] -= value);
                            });
                            c += block;
                        }
                    }
                    (0..chunk).for_each(|c| work[c * n + t1..c * n + t1 + consumed].fill(0.0));
                }
                for j in c1..c2 {
                    let p_end = self.row_ptr[j + 1] - 1;
                    let offset = (j - c1) * n;
                    let mut pointer = pointers[j - c1];
                    while pointer < p_end {
                        let k = self.row_idx[pointer];
                        let c = k - s1;
                        let w = self.sn_values[self.row_pos[pointer]] * self.d[k];
                        work[offset + k] = 0.0;
                        if w != 0.0 {
                            let start = self.sn_panel_ptr[s] + c * s_m;
                            work[offset + k + 1..offset + s2]
                                .iter_mut()
                                .zip(self.sn_values[start + c + 1..start + s_width].iter())
                                .for_each(|(work_r, value)| *work_r -= value * w);
                            self.sn_rows[s_rows_start + s_width..s_rows_start + s_m]
                                .iter()
                                .zip(self.sn_values[start + s_width..start + s_m].iter())
                                .for_each(|(&row, value)| work[offset + row] -= value * w);
                        }
                        pointer += 1;
                    }
                    let pivot = work[offset + j];
                    work[offset + j] = 0.0;
                    if pivot.abs() < ABS_TOL {
                        return Err(SparseError::Singular);
                    }
                    self.d[j] = pivot;
                    let c = j - s1;
                    let start = self.sn_panel_ptr[s] + c * s_m;
                    self.sn_values[start + c] = 1.0;
                    (c + 1..s_width).for_each(|local| {
                        self.sn_values[start + local] = work[offset + s1 + local] / pivot;
                        work[offset + s1 + local] = 0.0;
                    });
                    (s_width..s_m).for_each(|local| {
                        let row = self.sn_rows[s_rows_start + local];
                        self.sn_values[start + local] = work[offset + row] / pivot;
                        work[offset + row] = 0.0;
                    });
                }
                c1 = c2;
            }
        }
        Ok(())
    }
    fn max_below(&self) -> usize {
        (0..self.sn_start.len() - 1)
            .map(|s| {
                self.sn_rows_ptr[s + 1]
                    - self.sn_rows_ptr[s]
                    - (self.sn_start[s + 1] - self.sn_start[s])
            })
            .max()
            .unwrap_or(0)
    }
}
