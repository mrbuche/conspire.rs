#[cfg(test)]
mod test;

use super::super::{SparseError, matrix::CscMatrix};
use super::gemm::{CHUNK, NONE, axpy, etree, gemm_wide, max_below, reach_sorted, supernodes};
use crate::{
    ABS_TOL,
    math::{Scalar, Vector},
};

/// A sparse LDLᵀ factorization for symmetric matrices with a structurally full diagonal.
///
/// Computes PAPᵀ = LDLᵀ, with L stored as dense supernodal panels.
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
    e: Vec<Scalar>,
    pair: Vec<usize>,
    pinv: Vec<usize>,
    q: Vec<usize>,
}

impl CscMatrix {
    /// Builds the LDLᵀ factorization pattern symbolically for the fill-reducing
    /// ordering, requiring a structurally full diagonal; the values must be
    /// symmetric and are all zero until a refactorization supplies them.
    pub fn ldl_symbolic(&self) -> Result<CscLdl, SparseError> {
        let n = self.height();
        assert_eq!(n, self.width());
        let matching = self.maxtrans().ok_or(SparseError::Singular)?;
        let mut partner = vec![NONE; n];
        for (c, &r) in matching.iter().enumerate() {
            if r != c {
                if matching[r] != c {
                    return Err(SparseError::Unsymmetric);
                }
                partner[c] = r;
            }
        }
        let has_diag: Vec<bool> = (0..n)
            .map(|c| self.column(c).any(|(r, _)| r == c))
            .collect();
        let mut comp_of = vec![NONE; n];
        let mut members = Vec::with_capacity(n);
        (0..n).for_each(|c| {
            if partner[c] == NONE {
                comp_of[c] = members.len();
                members.push((c, NONE));
            } else if c < partner[c] {
                let (first, second) = if has_diag[c] || !has_diag[partner[c]] {
                    (c, partner[c])
                } else {
                    (partner[c], c)
                };
                comp_of[c] = members.len();
                comp_of[partner[c]] = members.len();
                members.push((first, second));
            }
        });
        let order = if members.len() == n {
            self.amd()
        } else {
            let mut compressed = Vec::with_capacity(self.nonzeros());
            (0..n).for_each(|c| {
                self.column(c)
                    .for_each(|(r, _)| compressed.push((comp_of[r], comp_of[c])))
            });
            CscMatrix::from_pattern(members.len(), members.len(), compressed).amd()
        };
        let nc = members.len();
        let mut q = Vec::with_capacity(n);
        let mut offset = Vec::with_capacity(nc);
        let mut locked = vec![false; n];
        order.iter().for_each(|&node| {
            let (first, second) = members[node];
            offset.push(q.len());
            q.push(first);
            if second != NONE {
                locked[q.len()] = true;
                q.push(second);
            }
        });
        let mut cpos = vec![NONE; nc];
        order
            .iter()
            .enumerate()
            .for_each(|(v, &node)| cpos[node] = v);
        let mut pinv = vec![NONE; n];
        let mut pair = vec![NONE; n];
        q.iter().enumerate().for_each(|(j, &q_j)| pinv[q_j] = j);
        (0..n).for_each(|j| {
            if locked[j] {
                pair[j] = j - 1;
                pair[j - 1] = j;
            }
        });
        let (parent, adj_ptr, adj) = etree(self, n, nc, |c| cpos[comp_of[c]], |r| cpos[comp_of[r]]);
        let mut mark = vec![NONE; nc];
        let mut reach = Vec::new();
        let mut l_count = vec![1_usize; n];
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        row_ptr.push(0);
        (0..nc).for_each(|v| {
            reach_sorted(v, &adj_ptr, &adj, &parent, &mut mark, &mut reach);
            let j = offset[v];
            let paired = pair[j] == j + 1;
            reach.iter().for_each(|&u| {
                let k = offset[u];
                row_idx.push(k);
                l_count[k] += 1;
                if locked.get(k + 1) == Some(&true) {
                    row_idx.push(k + 1);
                    l_count[k + 1] += 1;
                }
            });
            row_idx.push(j);
            row_ptr.push(row_idx.len());
            if paired {
                let start = row_ptr[j];
                let mut expanded: Vec<usize> = row_idx[start..row_idx.len() - 1].to_vec();
                expanded.push(j);
                expanded.iter().for_each(|&k| l_count[k] += 1);
                row_idx.extend_from_slice(&expanded);
                row_idx.push(j + 1);
                row_ptr.push(row_idx.len());
            }
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
            row_idx[row_ptr[k]..row_ptr[k + 1] - 1]
                .iter()
                .for_each(|&j| {
                    l_row_idx[next[j]] = k;
                    next[j] += 1;
                });
        });
        let l_values = vec![0.0; l_col_ptr[n]];
        let (sn_of, sn_start, sn_rows_ptr, sn_rows, sn_panel_ptr, sn_values) =
            supernodes(&l_col_ptr, &l_row_idx, &l_values, n, &locked);
        let mut row_pos = Vec::with_capacity(row_idx.len());
        row_ptr.windows(2).enumerate().for_each(|(j, window)| {
            row_idx[window[0]..window[1]].iter().for_each(|&k| {
                if k == j {
                    row_pos.push(NONE);
                } else {
                    let t = sn_of[k];
                    let rows = &sn_rows[sn_rows_ptr[t]..sn_rows_ptr[t + 1]];
                    row_pos.push(
                        sn_panel_ptr[t]
                            + (k - sn_start[t]) * rows.len()
                            + rows.binary_search(&j).expect("Row not in supernode."),
                    );
                }
            });
        });
        Ok(CscLdl {
            fill: sn_panel_ptr[sn_start.len() - 1] + n,
            sn_of,
            sn_start,
            sn_rows_ptr,
            sn_rows,
            sn_panel_ptr,
            sn_values,
            row_ptr,
            row_idx,
            row_pos,
            d: vec![0.0; n],
            e: vec![0.0; n],
            pair,
            pinv,
            q,
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
        let mut j = 0;
        while j < n {
            if self.pair[j] == j + 1 {
                let det = self.d[j] * self.d[j + 1] - self.e[j] * self.e[j];
                let (x_0, x_1) = (x[j], x[j + 1]);
                x[j] = (self.d[j + 1] * x_0 - self.e[j] * x_1) / det;
                x[j + 1] = (self.d[j] * x_1 - self.e[j] * x_0) / det;
                j += 2;
            } else {
                x[j] /= self.d[j];
                j += 1;
            }
        }
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
        let width_max = (0..self.sn_start.len() - 1)
            .map(|s| self.sn_start[s + 1] - self.sn_start[s])
            .max()
            .unwrap_or(0);
        let mut temp = vec![0.0; CHUNK * self.max_below().max(width_max)];
        let mut pointers = [0; CHUNK];
        for s in 0..self.sn_start.len() - 1 {
            let s1 = self.sn_start[s];
            let s2 = self.sn_start[s + 1];
            let s_width = s2 - s1;
            let s_rows_start = self.sn_rows_ptr[s];
            let s_m = self.sn_rows_ptr[s + 1] - s_rows_start;
            let mut c1 = s1;
            while c1 < s2 {
                let mut c2 = s2.min(c1 + CHUNK);
                if c2 < s2 && self.pair[c2] == c2 - 1 {
                    c2 -= 1;
                }
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
                            let value = self.sn_values[self.row_pos[*pointer]];
                            *pointer += 1;
                            let (base, w_0, w_1, paired) = if self.pair[k] == k + 1 {
                                if *pointer < p_end && self.row_idx[*pointer] == k + 1 {
                                    let other = self.sn_values[self.row_pos[*pointer]];
                                    *pointer += 1;
                                    (
                                        k,
                                        self.d[k] * value + self.e[k] * other,
                                        self.e[k] * value + self.d[k + 1] * other,
                                        true,
                                    )
                                } else {
                                    (k, self.d[k] * value, self.e[k] * value, true)
                                }
                            } else if self.pair[k] != NONE {
                                (k - 1, self.e[k - 1] * value, self.d[k] * value, true)
                            } else {
                                (k, self.d[k] * value, 0.0, false)
                            };
                            column[base] = w_0;
                            if paired {
                                column[base + 1] = w_1;
                            }
                        }
                    });
                    if t2 > c1 {
                        let inside = t2 - c1;
                        temp[..CHUNK * inside].fill(0.0);
                        gemm_wide(
                            &mut temp[..CHUNK * inside],
                            &work,
                            n,
                            panel,
                            m,
                            c1 - t1,
                            t1,
                            consumed,
                            inside,
                            chunk,
                        );
                        (0..chunk).for_each(|b| {
                            work[b * n + c1..b * n + t2]
                                .iter_mut()
                                .zip(temp[b * inside..(b + 1) * inside].iter())
                                .for_each(|(work_r, value)| *work_r -= value);
                        });
                    }
                    if below > 0 {
                        let mut offsets = [0; CHUNK];
                        offsets[0] = rows[width..].partition_point(|&row| row < c1);
                        (1..chunk).for_each(|b| {
                            let mut o = offsets[b - 1];
                            while o < below && rows[width + o] < c1 + b {
                                o += 1;
                            }
                            offsets[b] = o;
                        });
                        let shared = offsets[0];
                        let ahead = below - shared;
                        if ahead > 0 {
                            temp[..CHUNK * ahead].fill(0.0);
                            gemm_wide(
                                &mut temp[..CHUNK * ahead],
                                &work,
                                n,
                                panel,
                                m,
                                width + shared,
                                t1,
                                consumed,
                                ahead,
                                chunk,
                            );
                            (0..chunk).for_each(|b| {
                                let skip = offsets[b] - shared;
                                let column = &mut work[b * n..(b + 1) * n];
                                rows[width + shared + skip..]
                                    .iter()
                                    .zip(temp[b * ahead + skip..(b + 1) * ahead].iter())
                                    .for_each(|(&row, value)| column[row] -= value);
                            });
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
                        if k + 1 == j && self.pair[k] == j {
                            break;
                        }
                        let value = self.sn_values[self.row_pos[pointer]];
                        pointer += 1;
                        let (base, w_0, w_1, paired) = if self.pair[k] == k + 1 {
                            if pointer < p_end && self.row_idx[pointer] == k + 1 {
                                let other = self.sn_values[self.row_pos[pointer]];
                                pointer += 1;
                                (
                                    k,
                                    self.d[k] * value + self.e[k] * other,
                                    self.e[k] * value + self.d[k + 1] * other,
                                    true,
                                )
                            } else {
                                (k, self.d[k] * value, self.e[k] * value, true)
                            }
                        } else if self.pair[k] != NONE {
                            (k - 1, self.e[k - 1] * value, self.d[k] * value, true)
                        } else {
                            (k, self.d[k] * value, 0.0, false)
                        };
                        work[offset + base] = 0.0;
                        finalize_update(
                            &mut work[offset..offset + n],
                            &self.sn_values,
                            &self.sn_rows,
                            self.sn_panel_ptr[s],
                            s_rows_start,
                            s_m,
                            s_width,
                            s1,
                            s2,
                            base,
                            w_0,
                        );
                        if paired {
                            work[offset + base + 1] = 0.0;
                            finalize_update(
                                &mut work[offset..offset + n],
                                &self.sn_values,
                                &self.sn_rows,
                                self.sn_panel_ptr[s],
                                s_rows_start,
                                s_m,
                                s_width,
                                s1,
                                s2,
                                base + 1,
                                w_1,
                            );
                        }
                    }
                    if self.pair[j] == j + 1 {
                        continue;
                    }
                    if j > 0 && self.pair[j] == j - 1 {
                        let p = j - 1;
                        let off_0 = (p - c1) * n;
                        let b_00 = work[off_0 + p];
                        let b_01 = work[offset + p];
                        let b_11 = work[offset + j];
                        work[off_0 + p] = 0.0;
                        work[off_0 + j] = 0.0;
                        work[offset + p] = 0.0;
                        work[offset + j] = 0.0;
                        let det = b_00 * b_11 - b_01 * b_01;
                        if det.abs() < ABS_TOL {
                            return Err(SparseError::Singular);
                        }
                        self.d[p] = b_00;
                        self.e[p] = b_01;
                        self.d[j] = b_11;
                        let c_0 = p - s1;
                        let start_0 = self.sn_panel_ptr[s] + c_0 * s_m;
                        let start_1 = self.sn_panel_ptr[s] + (c_0 + 1) * s_m;
                        self.sn_values[start_0 + c_0] = 1.0;
                        self.sn_values[start_0 + c_0 + 1] = 0.0;
                        self.sn_values[start_1 + c_0 + 1] = 1.0;
                        for local in c_0 + 2..s_m {
                            let position = if local < s_width {
                                s1 + local
                            } else {
                                self.sn_rows[s_rows_start + local]
                            };
                            let w_0 = work[off_0 + position];
                            let w_1 = work[offset + position];
                            self.sn_values[start_0 + local] = (w_0 * b_11 - w_1 * b_01) / det;
                            self.sn_values[start_1 + local] = (w_1 * b_00 - w_0 * b_01) / det;
                            work[off_0 + position] = 0.0;
                            work[offset + position] = 0.0;
                        }
                    } else {
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
                }
                c1 = c2;
            }
        }
        Ok(())
    }
    fn max_below(&self) -> usize {
        max_below(&self.sn_start, &self.sn_rows_ptr)
    }
}

/// Applies a finalized own-supernode column's update to a work column.
#[allow(clippy::too_many_arguments)]
fn finalize_update(
    column: &mut [Scalar],
    sn_values: &[Scalar],
    sn_rows: &[usize],
    panel_ptr: usize,
    s_rows_start: usize,
    s_m: usize,
    s_width: usize,
    s1: usize,
    s2: usize,
    base: usize,
    w: Scalar,
) {
    if w != 0.0 {
        let c = base - s1;
        let start = panel_ptr + c * s_m;
        axpy(
            &mut column[base + 1..s2],
            &sn_values[start + c + 1..start + s_width],
            w,
        );
        sn_rows[s_rows_start + s_width..s_rows_start + s_m]
            .iter()
            .zip(sn_values[start + s_width..start + s_m].iter())
            .for_each(|(&row, value)| column[row] -= value * w);
    }
}
