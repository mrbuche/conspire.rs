#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(test)]
mod test;

use super::super::{SparseError, matrix::CscMatrix};
use super::gemm::{CHUNK, NONE, axpy, etree, gemm_wide, max_below, reach_sorted, supernodes};
use crate::{
    ABS_TOL,
    math::{Scalar, Vector},
};

/// Threshold for preferring the diagonal pivot, which preserves the
/// fill-reducing ordering when the matrix has a symmetric pattern.
const PIVOT_TOL: Scalar = 0.001;

/// A sparse LU factorization, PAQ = LU, with L stored as dense supernodal panels.
pub struct CscLu {
    pub(super) fill: usize,
    pub(super) sn_of: Vec<usize>,
    pub(super) sn_start: Vec<usize>,
    pub(super) sn_rows_ptr: Vec<usize>,
    pub(super) sn_rows: Vec<usize>,
    pub(super) sn_panel_ptr: Vec<usize>,
    pub(super) sn_values: Vec<Scalar>,
    pub(super) u_col_ptr: Vec<usize>,
    pub(super) u_row_idx: Vec<usize>,
    pub(super) u_values: Vec<Scalar>,
    pub(super) pinv: Vec<usize>,
    pub(super) q: Vec<usize>,
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
        q.iter()
            .enumerate()
            .for_each(|(j, &q_j)| pinv[matching[q_j]] = j);
        Ok(self.symbolic(q, pinv, &vec![false; n]))
    }
    pub(super) fn symbolic(&self, q: Vec<usize>, pinv: Vec<usize>, locked: &[bool]) -> CscLu {
        let n = self.height();
        let mut qinv = vec![NONE; n];
        q.iter().enumerate().for_each(|(j, &q_j)| qinv[q_j] = j);
        let (parent, adj_ptr, adj) = etree(self, n, n, |c| qinv[c], |r| pinv[r]);
        let mut mark = vec![NONE; n];
        let mut row = Vec::new();
        let mut l_count = vec![1_usize; n];
        let mut u_col_ptr = Vec::with_capacity(n + 1);
        let mut u_row_idx = Vec::new();
        u_col_ptr.push(0);
        (0..n).for_each(|k| {
            reach_sorted(k, &adj_ptr, &adj, &parent, &mut mark, &mut row);
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
            supernodes(&l_col_ptr, &l_row_idx, &l_values, n, locked);
        CscLu {
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
        }
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
            supernodes(&l_col_ptr, &l_row_idx, &l_values, n, &vec![false; n]);
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
        let mut temp = vec![0.0; CHUNK * self.max_below()];
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
                        temp[..CHUNK * below].fill(0.0);
                        gemm_wide(
                            &mut temp[..CHUNK * below],
                            &work,
                            n,
                            panel,
                            m,
                            width,
                            t1,
                            consumed,
                            below,
                            chunk,
                        );
                        (0..chunk).for_each(|b| {
                            let column = &mut work[b * n..(b + 1) * n];
                            rows[width..]
                                .iter()
                                .zip(temp[b * below..(b + 1) * below].iter())
                                .for_each(|(&row, value)| column[row] -= value);
                        });
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
                            axpy(
                                &mut work[offset + k + 1..offset + s2],
                                &self.sn_values[start + c + 1..start + s_width],
                                u,
                            );
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
    pub(super) fn max_below(&self) -> usize {
        max_below(&self.sn_start, &self.sn_rows_ptr)
    }
}

/// Eliminates the first `consumed` pivot columns of a panel from a transposed
/// tile of CHUNK target columns, as a dense unit-lower triangular solve
/// vectorized across the targets.
fn trisolve(tile: &mut [Scalar], panel: &[Scalar], m: usize, consumed: usize, width: usize) {
    #[cfg(target_arch = "x86_64")]
    if super::simd() {
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
