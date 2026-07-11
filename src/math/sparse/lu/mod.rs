#[cfg(test)]
mod test;

use super::{SparseError, matrix::CscMatrix};
use crate::{
    ABS_TOL,
    math::{Scalar, Vector},
};

const NONE: usize = usize::MAX;

/// A sparse LU factorization, PAQ = LU, in compressed sparse column format.
pub struct CscLu {
    l_col_ptr: Vec<usize>,
    l_row_idx: Vec<usize>,
    l_values: Vec<Scalar>,
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
        Ok(CscLu {
            l_col_ptr,
            l_row_idx,
            l_values,
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
        let mut x = Vector::zero(n);
        self.pinv
            .iter()
            .enumerate()
            .for_each(|(i, &p_i)| x[p_i] = b[i]);
        (0..n).for_each(|j| {
            let x_j = x[j];
            if x_j != 0.0 {
                (self.l_col_ptr[j] + 1..self.l_col_ptr[j + 1])
                    .for_each(|k| x[self.l_row_idx[k]] -= self.l_values[k] * x_j);
            }
        });
        (0..n).rev().for_each(|j| {
            let end = self.u_col_ptr[j + 1];
            x[j] /= self.u_values[end - 1];
            let x_j = x[j];
            if x_j != 0.0 {
                (self.u_col_ptr[j]..end - 1)
                    .for_each(|k| x[self.u_row_idx[k]] -= self.u_values[k] * x_j);
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
        self.l_values.len() + self.u_values.len()
    }
    /// Recomputes the factorization for new values in the same pattern, reusing
    /// the pivot order and fill pattern without any symbolic work or pivot search.
    /// The factorization is invalid if an error is returned.
    pub fn refactor(&mut self, matrix: &CscMatrix) -> Result<(), SparseError> {
        let Self {
            l_col_ptr,
            l_row_idx,
            l_values,
            u_col_ptr,
            u_row_idx,
            u_values,
            pinv,
            q,
        } = self;
        let n = pinv.len();
        assert_eq!(n, matrix.height());
        let mut x = vec![0.0; n];
        for (j, &q_j) in q.iter().enumerate() {
            matrix
                .column(q_j)
                .for_each(|(i, value)| x[pinv[i]] = *value);
            (u_col_ptr[j]..u_col_ptr[j + 1] - 1).for_each(|p| {
                let k = u_row_idx[p];
                let x_k = x[k];
                u_values[p] = x_k;
                x[k] = 0.0;
                if x_k != 0.0 {
                    (l_col_ptr[k] + 1..l_col_ptr[k + 1])
                        .for_each(|r| x[l_row_idx[r]] -= l_values[r] * x_k);
                }
            });
            let pivot = x[j];
            x[j] = 0.0;
            if pivot.abs() < ABS_TOL {
                return Err(SparseError::Singular);
            }
            u_values[u_col_ptr[j + 1] - 1] = pivot;
            (l_col_ptr[j] + 1..l_col_ptr[j + 1]).for_each(|p| {
                l_values[p] = x[l_row_idx[p]] / pivot;
                x[l_row_idx[p]] = 0.0;
            });
        }
        Ok(())
    }
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
        col.into_iter().for_each(|(i, value)| {
            row_idx.push(row_map[i]);
            values.push(value);
        });
        col_ptr.push(row_idx.len());
    });
    (col_ptr, row_idx, values)
}
