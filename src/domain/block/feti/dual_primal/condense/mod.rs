#[cfg(test)]
mod test;

use super::DualPrimalSplit;
use crate::math::{Matrix, Scalar, SquareMatrix, Vector};

pub(crate) struct Condensed {
    pub(crate) schur: SquareMatrix,
    pub(crate) reduced_force: Vector,
    pub(crate) dual_map: Matrix,
}

fn extract_vector(source: &Vector, indices: &[usize]) -> Vector {
    indices.iter().map(|&i| source[i]).collect()
}

fn extract_square(source: &SquareMatrix, indices: &[usize]) -> SquareMatrix {
    indices
        .iter()
        .map(|&row| indices.iter().map(|&col| source[row][col]).collect())
        .collect()
}

fn extract_rectangular(source: &SquareMatrix, rows: &[usize], columns: &[usize]) -> Matrix {
    rows.iter()
        .map(|&row| columns.iter().map(|&column| source[row][column]).collect())
        .collect()
}

/// Statically condenses the dual (remainder boundary) DOFs out of a
/// subdomain's local stiffness — they are never shared beyond this
/// subdomain, so this elimination is purely local and embarrassingly
/// parallel across subdomains — leaving a non-singular Schur complement
/// on the corner DOFs, which is what gets assembled into the global
/// coarse problem. dual_map = K_dd^-1 K_dp recovers the eliminated
/// dual solution once the corner (and multiplier) unknowns are known.
pub(crate) fn condense(
    local_stiffness: &SquareMatrix,
    local_force: &Vector,
    split: &DualPrimalSplit,
) -> Condensed {
    let primal = split.primal();
    let dual = split.dual();
    let k_pp = extract_square(local_stiffness, primal);
    let k_pd = extract_rectangular(local_stiffness, primal, dual);
    let k_dp = extract_rectangular(local_stiffness, dual, primal);
    let k_dd = extract_square(local_stiffness, dual);
    let f_p = extract_vector(local_force, primal);
    let f_d = extract_vector(local_force, dual);
    let columns: Matrix = (0..primal.len())
        .map(|column| {
            let rhs: Vector = k_dp.iter().map(|row| row[column]).collect();
            k_dd.solve_lu(&rhs)
                .expect("remainder block K_dd is singular")
        })
        .collect();
    let dual_map = columns.transpose();
    let schur = primal
        .iter()
        .enumerate()
        .map(|(row, _)| {
            primal
                .iter()
                .enumerate()
                .map(|(column, _)| {
                    k_pp[row][column]
                        - dual
                            .iter()
                            .enumerate()
                            .map(|(d, _)| k_pd[row][d] * dual_map[d][column])
                            .sum::<Scalar>()
                })
                .collect()
        })
        .collect();
    let y = k_dd
        .solve_lu(&f_d)
        .expect("remainder block K_dd is singular");
    let reduced_force = primal
        .iter()
        .enumerate()
        .map(|(row, _)| {
            f_p[row]
                - dual
                    .iter()
                    .enumerate()
                    .map(|(d, _)| k_pd[row][d] * y[d])
                    .sum::<Scalar>()
        })
        .collect();
    Condensed {
        schur,
        reduced_force,
        dual_map,
    }
}
