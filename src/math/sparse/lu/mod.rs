#[cfg(test)]
mod test;

use super::{SparseError, matrix::CscMatrix};
use crate::{
    ABS_TOL,
    math::{Scalar, Vector},
};

const NONE: usize = usize::MAX;

/// Threshold for preferring the diagonal pivot, which preserves the
/// fill-reducing ordering when the matrix has a symmetric pattern.
const PIVOT_TOL: Scalar = 0.001;

/// Number of columns factored together when refactoring.
const CHUNK: usize = 8;

/// A sparse LU factorization, PAQ = LU, with L stored as dense supernodal panels.
pub struct CscLu {
    fill: usize,
    sn_of: Vec<usize>,
    sn_start: Vec<usize>,
    sn_rows_ptr: Vec<usize>,
    sn_rows: Vec<usize>,
    sn_panel_ptr: Vec<usize>,
    sn_values: Vec<Scalar>,
    u_col_ptr: Vec<usize>,
    u_row_idx: Vec<usize>,
    u_values: Vec<Scalar>,
    pinv: Vec<usize>,
    q: Vec<usize>,
}

impl CscMatrix {
    /// Factors PA = LU using the Gilbert-Peierls method with partial pivoting.
    pub fn lu(&self) -> Result<CscLu, SparseError> {
        self.factor((0..self.height()).collect())
    }
    /// Factors PAQ = LU using the Gilbert-Peierls method with partial pivoting,
    /// with a fill-reducing approximate minimum degree column ordering.
    pub fn lu_amd(&self) -> Result<CscLu, SparseError> {
        self.factor(self.amd())
    }
    /// Builds the factorization pattern symbolically for the fill-reducing
    /// ordering assuming pivots from a maximum transversal (the diagonal when it
    /// is structurally full), which is the exact fill pattern for a symmetric
    /// pattern and a superset otherwise; all values are zero until a
    /// refactorization supplies them.
    pub fn lu_symbolic(&self) -> Result<CscLu, SparseError> {
        let n = self.height();
        assert_eq!(n, self.width());
        let matching = self.maxtrans().ok_or(SparseError::Singular)?;
        let q = if matching.iter().enumerate().all(|(c, &r)| r == c) {
            self.amd()
        } else {
            let mut rinv = vec![0; n];
            matching.iter().enumerate().for_each(|(c, &r)| rinv[r] = c);
            let mut permuted = Vec::with_capacity(self.nonzeros());
            (0..n).for_each(|c| {
                self.column(c)
                    .for_each(|(r, _)| permuted.push((rinv[r], c)))
            });
            CscMatrix::from_pattern(n, n, permuted).amd()
        };
        let mut pinv = vec![NONE; n];
        let mut qinv = vec![NONE; n];
        q.iter().enumerate().for_each(|(j, &q_j)| {
            qinv[q_j] = j;
            pinv[matching[q_j]] = j;
        });
        let mut count = vec![0_usize; n];
        (0..n).for_each(|c| {
            let j = qinv[c];
            self.column(c).for_each(|(r, _)| {
                let i = pinv[r];
                if i != j {
                    count[i.max(j)] += 1;
                }
            });
        });
        let mut adj_ptr = Vec::with_capacity(n + 1);
        adj_ptr.push(0_usize);
        count
            .iter()
            .for_each(|c| adj_ptr.push(adj_ptr.last().unwrap() + c));
        let mut adj = vec![0; adj_ptr[n]];
        let mut next = adj_ptr[..n].to_vec();
        (0..n).for_each(|c| {
            let j = qinv[c];
            self.column(c).for_each(|(r, _)| {
                let i = pinv[r];
                if i != j {
                    adj[next[i.max(j)]] = i.min(j);
                    next[i.max(j)] += 1;
                }
            });
        });
        let upper = |k: usize| adj[adj_ptr[k]..adj_ptr[k + 1]].iter();
        let mut parent = vec![NONE; n];
        let mut ancestor = vec![NONE; n];
        (0..n).for_each(|k| {
            upper(k).for_each(|&start| {
                let mut i = start;
                while i != NONE && i != k {
                    let next = ancestor[i];
                    ancestor[i] = k;
                    if next == NONE {
                        parent[i] = k;
                    }
                    i = next;
                }
            });
        });
        let mut mark = vec![NONE; n];
        let mut row = Vec::new();
        let mut l_count = vec![1_usize; n];
        let mut u_col_ptr = Vec::with_capacity(n + 1);
        let mut u_row_idx = Vec::new();
        u_col_ptr.push(0);
        (0..n).for_each(|k| {
            mark[k] = k;
            row.clear();
            upper(k).for_each(|&start| {
                let mut i = start;
                while mark[i] != k {
                    mark[i] = k;
                    row.push(i);
                    i = parent[i];
                }
            });
            row.sort_unstable();
            row.iter().for_each(|&j| l_count[j] += 1);
            u_row_idx.extend_from_slice(&row);
            u_row_idx.push(k);
            u_col_ptr.push(u_row_idx.len());
        });
        let mut l_col_ptr = Vec::with_capacity(n + 1);
        l_col_ptr.push(0);
        l_count.iter().for_each(|count| {
            l_col_ptr.push(l_col_ptr.last().unwrap() + count);
        });
        let mut l_row_idx = vec![0; l_col_ptr[n]];
        let mut next = vec![0; n];
        (0..n).for_each(|j| {
            l_row_idx[l_col_ptr[j]] = j;
            next[j] = l_col_ptr[j] + 1;
        });
        (0..n).for_each(|k| {
            u_row_idx[u_col_ptr[k]..u_col_ptr[k + 1] - 1]
                .iter()
                .for_each(|&j| {
                    l_row_idx[next[j]] = k;
                    next[j] += 1;
                });
        });
        let fill = l_row_idx.len() + u_row_idx.len();
        let l_values = vec![0.0; l_row_idx.len()];
        let (sn_of, sn_start, sn_rows_ptr, sn_rows, sn_panel_ptr, sn_values) =
            supernodes(&l_col_ptr, &l_row_idx, &l_values, n);
        Ok(CscLu {
            fill,
            sn_of,
            sn_start,
            sn_rows_ptr,
            sn_rows,
            sn_panel_ptr,
            sn_values,
            u_col_ptr,
            u_row_idx,
            u_values: vec![0.0; fill - l_values.len()],
            pinv,
            q,
        })
    }
    fn factor(&self, q: Vec<usize>) -> Result<CscLu, SparseError> {
        let n = self.height();
        assert_eq!(n, self.width());
        let mut pinv = vec![NONE; n];
        let mut l_cols = Vec::<Vec<(usize, Scalar)>>::with_capacity(n);
        let mut u_cols = Vec::<Vec<(usize, Scalar)>>::with_capacity(n);
        let mut x = vec![0.0; n];
        let mut mark = vec![0; n];
        let mut order = vec![0; n];
        let mut stack = vec![0; n];
        let mut pstack = vec![0; n];
        let mut top;
        for (j, &q_j) in q.iter().enumerate() {
            top = reach(
                self.column(q_j).map(|(i, _)| i),
                &l_cols,
                &pinv,
                j + 1,
                &mut mark,
                &mut order,
                &mut stack,
                &mut pstack,
            );
            self.column(q_j).for_each(|(i, value)| x[i] = *value);
            order[top..n].iter().for_each(|&i| {
                if pinv[i] != NONE {
                    let x_i = x[i];
                    l_cols[pinv[i]]
                        .iter()
                        .skip(1)
                        .for_each(|&(row, value)| x[row] -= value * x_i);
                }
            });
            let mut pivot_row = NONE;
            let mut pivot_abs = 0.0;
            order[top..n].iter().for_each(|&i| {
                if pinv[i] == NONE && x[i].abs() > pivot_abs {
                    pivot_abs = x[i].abs();
                    pivot_row = i;
                }
            });
            if pivot_row == NONE || pivot_abs < ABS_TOL {
                return Err(SparseError::Singular);
            }
            if pinv[q_j] == NONE && x[q_j].abs() >= PIVOT_TOL * pivot_abs {
                pivot_row = q_j;
            }
            let pivot = x[pivot_row];
            pinv[pivot_row] = j;
            let mut l_col = vec![(pivot_row, 1.0)];
            let mut u_col = Vec::new();
            order[top..n].iter().for_each(|&i| {
                if pinv[i] == NONE {
                    l_col.push((i, x[i] / pivot));
                } else if i != pivot_row {
                    u_col.push((pinv[i], x[i]));
                }
                x[i] = 0.0;
            });
            u_col.push((j, pivot));
            u_col.sort_unstable_by_key(|&(row, _)| row);
            l_cols.push(l_col);
            u_cols.push(u_col);
        }
        let (l_col_ptr, l_row_idx, l_values) = compress(l_cols, &pinv);
        let (u_col_ptr, u_row_idx, u_values) = compress(u_cols, &(0..n).collect::<Vec<usize>>());
        let fill = l_values.len() + u_values.len();
        let (sn_of, sn_start, sn_rows_ptr, sn_rows, sn_panel_ptr, sn_values) =
            supernodes(&l_col_ptr, &l_row_idx, &l_values, n);
        Ok(CscLu {
            fill,
            sn_of,
            sn_start,
            sn_rows_ptr,
            sn_rows,
            sn_panel_ptr,
            sn_values,
            u_col_ptr,
            u_row_idx,
            u_values,
            pinv,
            q,
        })
    }
}

impl CscLu {
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
        (0..n).rev().for_each(|j| {
            let end = self.u_col_ptr[j + 1];
            x[j] /= self.u_values[end - 1];
            let x_j = x[j];
            if x_j != 0.0 {
                self.u_row_idx[self.u_col_ptr[j]..end - 1]
                    .iter()
                    .zip(self.u_values[self.u_col_ptr[j]..end - 1].iter())
                    .for_each(|(&row, value)| x[row] -= value * x_j);
            }
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
        let mut tile = vec![
            0.0;
            CHUNK
                * (0..self.sn_start.len() - 1)
                    .map(|s| self.sn_start[s + 1] - self.sn_start[s])
                    .max()
                    .unwrap_or(0)
        ];
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
                    *pointer = self.u_col_ptr[j];
                });
                loop {
                    let mut t = NONE;
                    (c1..c2).zip(pointers.iter()).for_each(|(j, &pointer)| {
                        if pointer < self.u_col_ptr[j + 1] - 1 {
                            let row = self.u_row_idx[pointer];
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
                    if consumed >= 4 {
                        if chunk < CHUNK {
                            tile[..width * CHUNK].fill(0.0);
                        }
                        (0..chunk).for_each(|b| {
                            work[b * n + t1..b * n + t2]
                                .iter()
                                .zip(tile.chunks_exact_mut(CHUNK))
                                .for_each(|(&value, row)| row[b] = value);
                        });
                        trisolve(&mut tile[..width * CHUNK], panel, m, consumed, width);
                        (0..chunk).for_each(|b| {
                            work[b * n + t1..b * n + t2]
                                .iter_mut()
                                .zip(tile.chunks_exact(CHUNK))
                                .for_each(|(value, row)| *value = row[b]);
                        });
                        (c1..c2).zip(pointers.iter_mut()).for_each(|(j, pointer)| {
                            let p_end = self.u_col_ptr[j + 1] - 1;
                            let column = &work[(j - c1) * n..(j - c1 + 1) * n];
                            while *pointer < p_end {
                                let k = self.u_row_idx[*pointer];
                                if k >= t1 + consumed {
                                    break;
                                }
                                self.u_values[*pointer] = column[k];
                                *pointer += 1;
                            }
                        });
                    } else {
                        (c1..c2).zip(pointers.iter_mut()).for_each(|(j, pointer)| {
                            let p_end = self.u_col_ptr[j + 1] - 1;
                            let column = &mut work[(j - c1) * n..(j - c1 + 1) * n];
                            if consumed == width
                                && width <= 3
                                && *pointer + width <= p_end
                                && self.u_row_idx[*pointer] == t1
                                && self.u_row_idx[*pointer + width - 1] == t1 + width - 1
                            {
                                match width {
                                    1 => self.u_values[*pointer] = column[t1],
                                    2 => {
                                        let u_0 = column[t1];
                                        let u_1 = column[t1 + 1] - panel[1] * u_0;
                                        column[t1 + 1] = u_1;
                                        self.u_values[*pointer] = u_0;
                                        self.u_values[*pointer + 1] = u_1;
                                    }
                                    _ => {
                                        let u_0 = column[t1];
                                        let u_1 = column[t1 + 1] - panel[1] * u_0;
                                        let u_2 =
                                            column[t1 + 2] - panel[2] * u_0 - panel[m + 2] * u_1;
                                        column[t1 + 1] = u_1;
                                        column[t1 + 2] = u_2;
                                        self.u_values[*pointer] = u_0;
                                        self.u_values[*pointer + 1] = u_1;
                                        self.u_values[*pointer + 2] = u_2;
                                    }
                                }
                                *pointer += width;
                            } else {
                                while *pointer < p_end {
                                    let k = self.u_row_idx[*pointer];
                                    if k >= c1 || k >= t2 {
                                        break;
                                    }
                                    let c = k - t1;
                                    let u = column[k];
                                    self.u_values[*pointer] = u;
                                    if u != 0.0 {
                                        column[k + 1..t2]
                                            .iter_mut()
                                            .zip(panel[c * m + c + 1..c * m + width].iter())
                                            .for_each(|(work_r, value)| *work_r -= value * u);
                                    }
                                    *pointer += 1;
                                }
                            }
                        });
                    }
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
                    let p_end = self.u_col_ptr[j + 1] - 1;
                    let offset = (j - c1) * n;
                    let mut pointer = pointers[j - c1];
                    while pointer < p_end {
                        let k = self.u_row_idx[pointer];
                        let c = k - s1;
                        let u = work[offset + k];
                        self.u_values[pointer] = u;
                        work[offset + k] = 0.0;
                        if u != 0.0 {
                            let start = self.sn_panel_ptr[s] + c * s_m;
                            work[offset + k + 1..offset + s2]
                                .iter_mut()
                                .zip(self.sn_values[start + c + 1..start + s_width].iter())
                                .for_each(|(work_r, value)| *work_r -= value * u);
                            self.sn_rows[s_rows_start + s_width..s_rows_start + s_m]
                                .iter()
                                .zip(self.sn_values[start + s_width..start + s_m].iter())
                                .for_each(|(&row, value)| work[offset + row] -= value * u);
                        }
                        pointer += 1;
                    }
                    let pivot = work[offset + j];
                    work[offset + j] = 0.0;
                    if pivot.abs() < ABS_TOL {
                        return Err(SparseError::Singular);
                    }
                    self.u_values[p_end] = pivot;
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

/// Eliminates the first `consumed` pivot columns of a panel from a transposed
/// tile of CHUNK target columns, as a dense unit-lower triangular solve
/// vectorized across the targets.
fn trisolve(tile: &mut [Scalar], panel: &[Scalar], m: usize, consumed: usize, width: usize) {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        return unsafe { avx::trisolve(tile, panel, m, consumed, width) };
    }
    (0..consumed).for_each(|c| {
        let (row, rest) = tile[c * CHUNK..].split_at_mut(CHUNK);
        if row.iter().any(|&u| u != 0.0) {
            rest[..(width - c - 1) * CHUNK]
                .chunks_exact_mut(CHUNK)
                .zip(panel[c * m + c + 1..c * m + width].iter())
                .for_each(|(target, &value)| {
                    target
                        .iter_mut()
                        .zip(row.iter())
                        .for_each(|(entry, &u)| *entry -= value * u)
                });
        }
    });
}

#[cfg(target_arch = "x86_64")]
fn simd() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

fn rank_one_quad(
    temp_0: &mut [Scalar],
    temp_1: &mut [Scalar],
    temp_2: &mut [Scalar],
    temp_3: &mut [Scalar],
    column: &[Scalar],
    u: [Scalar; 4],
) {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        return unsafe { avx::rank_one_quad(temp_0, temp_1, temp_2, temp_3, column, u) };
    }
    column
        .iter()
        .zip(
            temp_0
                .iter_mut()
                .zip(temp_1.iter_mut())
                .zip(temp_2.iter_mut().zip(temp_3.iter_mut())),
        )
        .for_each(|(&value, ((a_0, a_1), (a_2, a_3)))| {
            *a_0 += value * u[0];
            *a_1 += value * u[1];
            *a_2 += value * u[2];
            *a_3 += value * u[3];
        });
}

#[allow(clippy::too_many_arguments)]
fn rank_two_quad(
    temp_0: &mut [Scalar],
    temp_1: &mut [Scalar],
    temp_2: &mut [Scalar],
    temp_3: &mut [Scalar],
    column: &[Scalar],
    other: &[Scalar],
    u: [Scalar; 4],
    w: [Scalar; 4],
) {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        return unsafe { avx::rank_two_quad(temp_0, temp_1, temp_2, temp_3, column, other, u, w) };
    }
    column
        .iter()
        .zip(other.iter())
        .zip(
            temp_0
                .iter_mut()
                .zip(temp_1.iter_mut())
                .zip(temp_2.iter_mut().zip(temp_3.iter_mut())),
        )
        .for_each(|((&value, &second), ((a_0, a_1), (a_2, a_3)))| {
            *a_0 += value * u[0] + second * w[0];
            *a_1 += value * u[1] + second * w[1];
            *a_2 += value * u[2] + second * w[2];
            *a_3 += value * u[3] + second * w[3];
        });
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use super::{CHUNK, Scalar};
    use std::arch::x86_64::{
        _mm256_castpd_si256, _mm256_fmadd_pd, _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_or_pd,
        _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_testz_si256,
    };

    const LANES: usize = CHUNK / 4;

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn trisolve(
        tile: &mut [Scalar],
        panel: &[Scalar],
        m: usize,
        consumed: usize,
        width: usize,
    ) {
        unsafe {
            let mut c = 0;
            while c + 4 <= consumed {
                let mut u = [_mm256_setzero_pd(); LANES];
                let mut v = [_mm256_setzero_pd(); LANES];
                let mut x = [_mm256_setzero_pd(); LANES];
                let mut y = [_mm256_setzero_pd(); LANES];
                let value_uv = _mm256_set1_pd(panel[c * m + c + 1]);
                let value_ux = _mm256_set1_pd(panel[c * m + c + 2]);
                let value_uy = _mm256_set1_pd(panel[c * m + c + 3]);
                let value_vx = _mm256_set1_pd(panel[(c + 1) * m + c + 2]);
                let value_vy = _mm256_set1_pd(panel[(c + 1) * m + c + 3]);
                let value_xy = _mm256_set1_pd(panel[(c + 2) * m + c + 3]);
                let mut bits = _mm256_setzero_pd();
                for l in 0..LANES {
                    u[l] = _mm256_loadu_pd(tile.as_ptr().add(c * CHUNK + 4 * l));
                    let row_v = tile.as_mut_ptr().add((c + 1) * CHUNK + 4 * l);
                    v[l] = _mm256_fnmadd_pd(value_uv, u[l], _mm256_loadu_pd(row_v));
                    _mm256_storeu_pd(row_v, v[l]);
                    let row_x = tile.as_mut_ptr().add((c + 2) * CHUNK + 4 * l);
                    x[l] = _mm256_fnmadd_pd(
                        value_vx,
                        v[l],
                        _mm256_fnmadd_pd(value_ux, u[l], _mm256_loadu_pd(row_x)),
                    );
                    _mm256_storeu_pd(row_x, x[l]);
                    let row_y = tile.as_mut_ptr().add((c + 3) * CHUNK + 4 * l);
                    y[l] = _mm256_fnmadd_pd(
                        value_xy,
                        x[l],
                        _mm256_fnmadd_pd(
                            value_vy,
                            v[l],
                            _mm256_fnmadd_pd(value_uy, u[l], _mm256_loadu_pd(row_y)),
                        ),
                    );
                    _mm256_storeu_pd(row_y, y[l]);
                    bits = _mm256_or_pd(
                        bits,
                        _mm256_or_pd(_mm256_or_pd(u[l], v[l]), _mm256_or_pd(x[l], y[l])),
                    );
                }
                let bits = _mm256_castpd_si256(bits);
                if _mm256_testz_si256(bits, bits) == 1 {
                    c += 4;
                    continue;
                }
                for r in c + 4..width {
                    let first = _mm256_set1_pd(panel[c * m + r]);
                    let second = _mm256_set1_pd(panel[(c + 1) * m + r]);
                    let third = _mm256_set1_pd(panel[(c + 2) * m + r]);
                    let fourth = _mm256_set1_pd(panel[(c + 3) * m + r]);
                    for l in 0..LANES {
                        let entry = tile.as_mut_ptr().add(r * CHUNK + 4 * l);
                        _mm256_storeu_pd(
                            entry,
                            _mm256_fnmadd_pd(
                                fourth,
                                y[l],
                                _mm256_fnmadd_pd(
                                    third,
                                    x[l],
                                    _mm256_fnmadd_pd(
                                        second,
                                        v[l],
                                        _mm256_fnmadd_pd(first, u[l], _mm256_loadu_pd(entry)),
                                    ),
                                ),
                            ),
                        );
                    }
                }
                c += 4;
            }
            while c + 2 <= consumed {
                let mut u = [_mm256_setzero_pd(); LANES];
                let mut v = [_mm256_setzero_pd(); LANES];
                let value = _mm256_set1_pd(panel[c * m + c + 1]);
                let mut bits = _mm256_setzero_pd();
                for l in 0..LANES {
                    u[l] = _mm256_loadu_pd(tile.as_ptr().add(c * CHUNK + 4 * l));
                    let next = tile.as_mut_ptr().add((c + 1) * CHUNK + 4 * l);
                    v[l] = _mm256_fnmadd_pd(value, u[l], _mm256_loadu_pd(next));
                    _mm256_storeu_pd(next, v[l]);
                    bits = _mm256_or_pd(bits, _mm256_or_pd(u[l], v[l]));
                }
                let bits = _mm256_castpd_si256(bits);
                if _mm256_testz_si256(bits, bits) == 1 {
                    c += 2;
                    continue;
                }
                for r in c + 2..width {
                    let first = _mm256_set1_pd(panel[c * m + r]);
                    let second = _mm256_set1_pd(panel[(c + 1) * m + r]);
                    for l in 0..LANES {
                        let entry = tile.as_mut_ptr().add(r * CHUNK + 4 * l);
                        _mm256_storeu_pd(
                            entry,
                            _mm256_fnmadd_pd(
                                second,
                                v[l],
                                _mm256_fnmadd_pd(first, u[l], _mm256_loadu_pd(entry)),
                            ),
                        );
                    }
                }
                c += 2;
            }
            if c < consumed {
                let mut u = [_mm256_setzero_pd(); LANES];
                let mut bits = _mm256_setzero_pd();
                for (l, u_l) in u.iter_mut().enumerate() {
                    *u_l = _mm256_loadu_pd(tile.as_ptr().add(c * CHUNK + 4 * l));
                    bits = _mm256_or_pd(bits, *u_l);
                }
                let bits = _mm256_castpd_si256(bits);
                if _mm256_testz_si256(bits, bits) == 0 {
                    for r in c + 1..width {
                        let value = _mm256_set1_pd(panel[c * m + r]);
                        for (l, &u_l) in u.iter().enumerate() {
                            let entry = tile.as_mut_ptr().add(r * CHUNK + 4 * l);
                            _mm256_storeu_pd(
                                entry,
                                _mm256_fnmadd_pd(value, u_l, _mm256_loadu_pd(entry)),
                            );
                        }
                    }
                }
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn rank_one_quad(
        temp_0: &mut [Scalar],
        temp_1: &mut [Scalar],
        temp_2: &mut [Scalar],
        temp_3: &mut [Scalar],
        column: &[Scalar],
        u: [Scalar; 4],
    ) {
        let len = column.len();
        let u_0 = _mm256_set1_pd(u[0]);
        let u_1 = _mm256_set1_pd(u[1]);
        let u_2 = _mm256_set1_pd(u[2]);
        let u_3 = _mm256_set1_pd(u[3]);
        let mut r = 0;
        unsafe {
            while r + 4 <= len {
                let value = _mm256_loadu_pd(column.as_ptr().add(r));
                _mm256_storeu_pd(
                    temp_0.as_mut_ptr().add(r),
                    _mm256_fmadd_pd(value, u_0, _mm256_loadu_pd(temp_0.as_ptr().add(r))),
                );
                _mm256_storeu_pd(
                    temp_1.as_mut_ptr().add(r),
                    _mm256_fmadd_pd(value, u_1, _mm256_loadu_pd(temp_1.as_ptr().add(r))),
                );
                _mm256_storeu_pd(
                    temp_2.as_mut_ptr().add(r),
                    _mm256_fmadd_pd(value, u_2, _mm256_loadu_pd(temp_2.as_ptr().add(r))),
                );
                _mm256_storeu_pd(
                    temp_3.as_mut_ptr().add(r),
                    _mm256_fmadd_pd(value, u_3, _mm256_loadu_pd(temp_3.as_ptr().add(r))),
                );
                r += 4;
            }
        }
        (r..len).for_each(|i| {
            temp_0[i] += column[i] * u[0];
            temp_1[i] += column[i] * u[1];
            temp_2[i] += column[i] * u[2];
            temp_3[i] += column[i] * u[3];
        });
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn rank_two_quad(
        temp_0: &mut [Scalar],
        temp_1: &mut [Scalar],
        temp_2: &mut [Scalar],
        temp_3: &mut [Scalar],
        column: &[Scalar],
        other: &[Scalar],
        u: [Scalar; 4],
        w: [Scalar; 4],
    ) {
        let len = column.len();
        let u_0 = _mm256_set1_pd(u[0]);
        let u_1 = _mm256_set1_pd(u[1]);
        let u_2 = _mm256_set1_pd(u[2]);
        let u_3 = _mm256_set1_pd(u[3]);
        let w_0 = _mm256_set1_pd(w[0]);
        let w_1 = _mm256_set1_pd(w[1]);
        let w_2 = _mm256_set1_pd(w[2]);
        let w_3 = _mm256_set1_pd(w[3]);
        let mut r = 0;
        unsafe {
            while r + 4 <= len {
                let value = _mm256_loadu_pd(column.as_ptr().add(r));
                let second = _mm256_loadu_pd(other.as_ptr().add(r));
                _mm256_storeu_pd(
                    temp_0.as_mut_ptr().add(r),
                    _mm256_fmadd_pd(
                        second,
                        w_0,
                        _mm256_fmadd_pd(value, u_0, _mm256_loadu_pd(temp_0.as_ptr().add(r))),
                    ),
                );
                _mm256_storeu_pd(
                    temp_1.as_mut_ptr().add(r),
                    _mm256_fmadd_pd(
                        second,
                        w_1,
                        _mm256_fmadd_pd(value, u_1, _mm256_loadu_pd(temp_1.as_ptr().add(r))),
                    ),
                );
                _mm256_storeu_pd(
                    temp_2.as_mut_ptr().add(r),
                    _mm256_fmadd_pd(
                        second,
                        w_2,
                        _mm256_fmadd_pd(value, u_2, _mm256_loadu_pd(temp_2.as_ptr().add(r))),
                    ),
                );
                _mm256_storeu_pd(
                    temp_3.as_mut_ptr().add(r),
                    _mm256_fmadd_pd(
                        second,
                        w_3,
                        _mm256_fmadd_pd(value, u_3, _mm256_loadu_pd(temp_3.as_ptr().add(r))),
                    ),
                );
                r += 4;
            }
        }
        (r..len).for_each(|i| {
            temp_0[i] += column[i] * u[0] + other[i] * w[0];
            temp_1[i] += column[i] * u[1] + other[i] * w[1];
            temp_2[i] += column[i] * u[2] + other[i] * w[2];
            temp_3[i] += column[i] * u[3] + other[i] * w[3];
        });
    }
}

/// Below-block contributions of the first `consumed` columns of a panel to a
/// block of up to four dense target columns, streaming the panel once.
#[allow(clippy::too_many_arguments)]
fn gemm(
    temp: &mut [Scalar],
    work: &[Scalar],
    n: usize,
    panel: &[Scalar],
    m: usize,
    width: usize,
    t1: usize,
    consumed: usize,
    below: usize,
    block: usize,
) {
    if block == 4 {
        let (half_a, half_b) = temp.split_at_mut(2 * below);
        let (temp_0, temp_1) = half_a.split_at_mut(below);
        let (temp_2, temp_3) = half_b.split_at_mut(below);
        let mut c = 0;
        while c + 2 <= consumed {
            let u_0 = work[t1 + c];
            let u_1 = work[n + t1 + c];
            let u_2 = work[2 * n + t1 + c];
            let u_3 = work[3 * n + t1 + c];
            let w_0 = work[t1 + c + 1];
            let w_1 = work[n + t1 + c + 1];
            let w_2 = work[2 * n + t1 + c + 1];
            let w_3 = work[3 * n + t1 + c + 1];
            if u_0 != 0.0 || u_1 != 0.0 || u_2 != 0.0 || u_3 != 0.0 {
                if w_0 != 0.0 || w_1 != 0.0 || w_2 != 0.0 || w_3 != 0.0 {
                    rank_two_quad(
                        temp_0,
                        temp_1,
                        temp_2,
                        temp_3,
                        &panel[c * m + width..(c + 1) * m],
                        &panel[(c + 1) * m + width..(c + 2) * m],
                        [u_0, u_1, u_2, u_3],
                        [w_0, w_1, w_2, w_3],
                    );
                } else {
                    rank_one_quad(
                        temp_0,
                        temp_1,
                        temp_2,
                        temp_3,
                        &panel[c * m + width..(c + 1) * m],
                        [u_0, u_1, u_2, u_3],
                    );
                }
            } else if w_0 != 0.0 || w_1 != 0.0 || w_2 != 0.0 || w_3 != 0.0 {
                rank_one_quad(
                    temp_0,
                    temp_1,
                    temp_2,
                    temp_3,
                    &panel[(c + 1) * m + width..(c + 2) * m],
                    [w_0, w_1, w_2, w_3],
                );
            }
            c += 2;
        }
        if c < consumed {
            let u_0 = work[t1 + c];
            let u_1 = work[n + t1 + c];
            let u_2 = work[2 * n + t1 + c];
            let u_3 = work[3 * n + t1 + c];
            if u_0 != 0.0 || u_1 != 0.0 || u_2 != 0.0 || u_3 != 0.0 {
                rank_one_quad(
                    temp_0,
                    temp_1,
                    temp_2,
                    temp_3,
                    &panel[c * m + width..(c + 1) * m],
                    [u_0, u_1, u_2, u_3],
                );
            }
        }
    } else {
        (0..block).for_each(|b| {
            let target = &mut temp[b * below..(b + 1) * below];
            (0..consumed).for_each(|c| {
                let u = work[b * n + t1 + c];
                if u != 0.0 {
                    panel[c * m + width..(c + 1) * m]
                        .iter()
                        .zip(target.iter_mut())
                        .for_each(|(&value, target_r)| *target_r += value * u);
                }
            });
        });
    }
}

/// Partition of the pivot columns into supernodes: maximal runs of consecutive
/// columns with nested row structure, stored as dense column-major panels.
type Supernodes = (
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<Scalar>,
);

fn supernodes(col_ptr: &[usize], row_idx: &[usize], values: &[Scalar], n: usize) -> Supernodes {
    let mut sn_start = vec![0];
    let mut sn_rows_ptr = vec![0];
    let mut sn_rows = Vec::new();
    let mut union: Vec<usize> = row_idx[col_ptr[0]..col_ptr[1]].to_vec();
    let mut true_nnz = union.len();
    let mut start = 0;
    (1..n).for_each(|j| {
        let curr = &row_idx[col_ptr[j]..col_ptr[j + 1]];
        let merged = merge_sorted(&union, curr);
        let width = j - start + 1;
        let stored = width * merged.len() - width * (width - 1) / 2;
        let padding = stored - true_nnz - curr.len();
        if 8 * padding <= stored + 256 {
            union = merged;
            true_nnz += curr.len();
        } else {
            sn_start.push(j);
            sn_rows.append(&mut union);
            sn_rows_ptr.push(sn_rows.len());
            union = curr.to_vec();
            true_nnz = curr.len();
            start = j;
        }
    });
    sn_start.push(n);
    sn_rows.append(&mut union);
    sn_rows_ptr.push(sn_rows.len());
    let num = sn_start.len() - 1;
    let mut sn_of = vec![0; n];
    let mut sn_panel_ptr = Vec::with_capacity(num + 1);
    sn_panel_ptr.push(0);
    let mut sn_values = Vec::new();
    let mut pos = vec![0; n];
    (0..num).for_each(|s| {
        let t1 = sn_start[s];
        let t2 = sn_start[s + 1];
        let width = t2 - t1;
        (t1..t2).for_each(|j| sn_of[j] = s);
        let rows = &sn_rows[sn_rows_ptr[s]..sn_rows_ptr[s + 1]];
        let m = rows.len();
        debug_assert!(
            rows[..width]
                .iter()
                .enumerate()
                .all(|(c, &row)| row == t1 + c)
        );
        rows.iter()
            .enumerate()
            .for_each(|(local, &row)| pos[row] = local);
        let panel_start = sn_values.len();
        sn_values.resize(panel_start + m * width, 0.0);
        (0..width).for_each(|c| {
            row_idx[col_ptr[t1 + c]..col_ptr[t1 + c + 1]]
                .iter()
                .zip(values[col_ptr[t1 + c]..col_ptr[t1 + c + 1]].iter())
                .for_each(|(&row, &value)| sn_values[panel_start + c * m + pos[row]] = value);
        });
        sn_panel_ptr.push(sn_values.len());
    });
    (
        sn_of,
        sn_start,
        sn_rows_ptr,
        sn_rows,
        sn_panel_ptr,
        sn_values,
    )
}

fn merge_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut merged = Vec::with_capacity(a.len() + b.len());
    let (mut p, mut q) = (0, 0);
    while p < a.len() && q < b.len() {
        if a[p] < b[q] {
            merged.push(a[p]);
            p += 1;
        } else {
            if a[p] == b[q] {
                p += 1;
            }
            merged.push(b[q]);
            q += 1;
        }
    }
    merged.extend_from_slice(&a[p..]);
    merged.extend_from_slice(&b[q..]);
    merged
}

/// Nonzero pattern of the solution to Lx = b, as the topologically ordered reach
/// of the pattern of b in the graph of L, placed in order[top..] with top returned.
#[allow(clippy::too_many_arguments)]
fn reach(
    starts: impl Iterator<Item = usize>,
    l_cols: &[Vec<(usize, Scalar)>],
    pinv: &[usize],
    tag: usize,
    mark: &mut [usize],
    order: &mut [usize],
    stack: &mut [usize],
    pstack: &mut [usize],
) -> usize {
    let mut top = order.len();
    starts.for_each(|start| {
        if mark[start] != tag {
            mark[start] = tag;
            stack[0] = start;
            pstack[0] = 0;
            let mut head = 0;
            loop {
                let i = stack[head];
                let mut descended = false;
                if pinv[i] != NONE {
                    let column = &l_cols[pinv[i]];
                    while pstack[head] < column.len() {
                        let child = column[pstack[head]].0;
                        pstack[head] += 1;
                        if mark[child] != tag {
                            mark[child] = tag;
                            head += 1;
                            stack[head] = child;
                            pstack[head] = 0;
                            descended = true;
                            break;
                        }
                    }
                }
                if !descended {
                    top -= 1;
                    order[top] = i;
                    if head == 0 {
                        break;
                    }
                    head -= 1;
                }
            }
        }
    });
    top
}

fn compress(
    cols: Vec<Vec<(usize, Scalar)>>,
    row_map: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<Scalar>) {
    let nnz = cols.iter().map(|col| col.len()).sum();
    let mut col_ptr = Vec::with_capacity(cols.len() + 1);
    let mut row_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    col_ptr.push(0);
    cols.into_iter().for_each(|col| {
        let mut mapped: Vec<(usize, Scalar)> = col
            .into_iter()
            .map(|(i, value)| (row_map[i], value))
            .collect();
        mapped.sort_unstable_by_key(|&(row, _)| row);
        mapped.into_iter().for_each(|(row, value)| {
            row_idx.push(row);
            values.push(value);
        });
        col_ptr.push(row_idx.len());
    });
    (col_ptr, row_idx, values)
}
