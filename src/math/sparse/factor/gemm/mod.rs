#[cfg(target_arch = "x86_64")]
mod avx;

use super::super::matrix::CscMatrix;
use crate::math::Scalar;

pub(super) const NONE: usize = usize::MAX;

/// Number of columns factored together when refactoring.
pub(super) const CHUNK: usize = 8;

/// Widest below-diagonal supernode extent, the largest scratch buffer
/// a chunked refactor needs for below-block updates.
pub(super) fn max_below(sn_start: &[usize], sn_rows_ptr: &[usize]) -> usize {
    (0..sn_start.len() - 1)
        .map(|s| sn_rows_ptr[s + 1] - sn_rows_ptr[s] - (sn_start[s + 1] - sn_start[s]))
        .max()
        .unwrap_or(0)
}

/// Elimination tree over `num_nodes` graph nodes, built from the upper
/// triangle induced by mapping each of `matrix`'s `n` columns and their row
/// positions through `col_node`/`row_node`. Returns `(parent, adj_ptr, adj)`,
/// the tree's parent pointers and a CSR adjacency of each node's upper
/// neighbors.
pub(super) fn etree(
    matrix: &CscMatrix,
    n: usize,
    num_nodes: usize,
    col_node: impl Fn(usize) -> usize,
    row_node: impl Fn(usize) -> usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut count = vec![0_usize; num_nodes];
    (0..n).for_each(|c| {
        let j = col_node(c);
        matrix.column(c).for_each(|(r, _)| {
            let i = row_node(r);
            if i != j {
                count[i.max(j)] += 1;
            }
        });
    });
    let mut adj_ptr = Vec::with_capacity(num_nodes + 1);
    adj_ptr.push(0_usize);
    count
        .iter()
        .for_each(|c| adj_ptr.push(adj_ptr.last().unwrap() + c));
    let mut adj = vec![0; adj_ptr[num_nodes]];
    let mut next = adj_ptr[..num_nodes].to_vec();
    (0..n).for_each(|c| {
        let j = col_node(c);
        matrix.column(c).for_each(|(r, _)| {
            let i = row_node(r);
            if i != j {
                adj[next[i.max(j)]] = i.min(j);
                next[i.max(j)] += 1;
            }
        });
    });
    let mut parent = vec![NONE; num_nodes];
    let mut ancestor = vec![NONE; num_nodes];
    (0..num_nodes).for_each(|k| {
        adj[adj_ptr[k]..adj_ptr[k + 1]].iter().for_each(|&start| {
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
    (parent, adj_ptr, adj)
}

/// Sorted reach set of node `k` in the elimination tree given by
/// `parent`/`adj_ptr`/`adj`, via path compression through `mark`.
pub(super) fn reach_sorted(
    k: usize,
    adj_ptr: &[usize],
    adj: &[usize],
    parent: &[usize],
    mark: &mut [usize],
    reach: &mut Vec<usize>,
) {
    mark[k] = k;
    reach.clear();
    adj[adj_ptr[k]..adj_ptr[k + 1]].iter().for_each(|&start| {
        let mut i = start;
        while mark[i] != k {
            mark[i] = k;
            reach.push(i);
            i = parent[i];
        }
    });
    reach.sort_unstable();
}

/// Subtracts `w` times a column slice from an equal-length target slice.
pub(super) fn axpy(target: &mut [Scalar], column: &[Scalar], w: Scalar) {
    #[cfg(target_arch = "x86_64")]
    if super::simd() {
        return unsafe { avx::axpy(target, column, w) };
    }
    target
        .iter_mut()
        .zip(column.iter())
        .for_each(|(target_r, value)| *target_r -= value * w);
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
    if super::simd() {
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
    if super::simd() {
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

/// Applies a source column pair to four target slices with multipliers `u`
/// and `w`, skipping all-zero multiplier quads.
#[allow(clippy::too_many_arguments)]
fn pair_quad(
    temp_0: &mut [Scalar],
    temp_1: &mut [Scalar],
    temp_2: &mut [Scalar],
    temp_3: &mut [Scalar],
    column: &[Scalar],
    other: &[Scalar],
    u: [Scalar; 4],
    w: [Scalar; 4],
) {
    let any_u = u.iter().any(|&value| value != 0.0);
    let any_w = w.iter().any(|&value| value != 0.0);
    if any_u && any_w {
        rank_two_quad(temp_0, temp_1, temp_2, temp_3, column, other, u, w);
    } else if any_u {
        rank_one_quad(temp_0, temp_1, temp_2, temp_3, column, u);
    } else if any_w {
        rank_one_quad(temp_0, temp_1, temp_2, temp_3, other, w);
    }
}

/// Contributions of the first `consumed` columns of a panel's rows
/// `[lo, lo + count)` to up to CHUNK dense target columns, streaming each
/// panel column pair from memory once for both target quads.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_wide(
    temp: &mut [Scalar],
    work: &[Scalar],
    n: usize,
    panel: &[Scalar],
    m: usize,
    lo: usize,
    t1: usize,
    consumed: usize,
    count: usize,
    chunk: usize,
) {
    let (half, rest) = temp.split_at_mut(2 * count);
    let (temp_0, temp_1) = half.split_at_mut(count);
    let (half, rest) = rest.split_at_mut(2 * count);
    let (temp_2, temp_3) = half.split_at_mut(count);
    let (half, rest) = rest.split_at_mut(2 * count);
    let (temp_4, temp_5) = half.split_at_mut(count);
    let (temp_6, temp_7) = rest[..2 * count].split_at_mut(count);
    let read = |b: usize, c: usize| {
        if b < chunk { work[b * n + t1 + c] } else { 0.0 }
    };
    let mut c = 0;
    while c + 2 <= consumed {
        let column = &panel[c * m + lo..c * m + lo + count];
        let other = &panel[(c + 1) * m + lo..(c + 1) * m + lo + count];
        pair_quad(
            temp_0,
            temp_1,
            temp_2,
            temp_3,
            column,
            other,
            [read(0, c), read(1, c), read(2, c), read(3, c)],
            [
                read(0, c + 1),
                read(1, c + 1),
                read(2, c + 1),
                read(3, c + 1),
            ],
        );
        if chunk > 4 {
            pair_quad(
                temp_4,
                temp_5,
                temp_6,
                temp_7,
                column,
                other,
                [read(4, c), read(5, c), read(6, c), read(7, c)],
                [
                    read(4, c + 1),
                    read(5, c + 1),
                    read(6, c + 1),
                    read(7, c + 1),
                ],
            );
        }
        c += 2;
    }
    if c < consumed {
        let column = &panel[c * m + lo..c * m + lo + count];
        let u = [read(0, c), read(1, c), read(2, c), read(3, c)];
        if u.iter().any(|&value| value != 0.0) {
            rank_one_quad(temp_0, temp_1, temp_2, temp_3, column, u);
        }
        if chunk > 4 {
            let u = [read(4, c), read(5, c), read(6, c), read(7, c)];
            if u.iter().any(|&value| value != 0.0) {
                rank_one_quad(temp_4, temp_5, temp_6, temp_7, column, u);
            }
        }
    }
}

/// Below-block contributions of the first `consumed` columns of a panel to a
/// block of up to four dense target columns, streaming the panel once.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm(
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
                        &panel[c * m + width..c * m + width + below],
                        &panel[(c + 1) * m + width..(c + 1) * m + width + below],
                        [u_0, u_1, u_2, u_3],
                        [w_0, w_1, w_2, w_3],
                    );
                } else {
                    rank_one_quad(
                        temp_0,
                        temp_1,
                        temp_2,
                        temp_3,
                        &panel[c * m + width..c * m + width + below],
                        [u_0, u_1, u_2, u_3],
                    );
                }
            } else if w_0 != 0.0 || w_1 != 0.0 || w_2 != 0.0 || w_3 != 0.0 {
                rank_one_quad(
                    temp_0,
                    temp_1,
                    temp_2,
                    temp_3,
                    &panel[(c + 1) * m + width..(c + 1) * m + width + below],
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
                    &panel[c * m + width..c * m + width + below],
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
                    panel[c * m + width..c * m + width + below]
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

pub(super) fn supernodes(
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[Scalar],
    n: usize,
    locked: &[bool],
) -> Supernodes {
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
        if locked[j] || 8 * padding <= stored + 256 {
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
