#[cfg(test)]
mod test;

use super::DualPrimalSplit;
use crate::math::{Matrix, Scalar, SquareMatrix, Vector};

pub(crate) struct Condensed {
    pub(crate) schur: SquareMatrix,
    pub(crate) reduced_force: Vector,
    pub(crate) primal_map: Matrix,
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

/// Statically condenses the primal (corner) DOFs out of a subdomain's local
/// stiffness, leaving a non-singular Schur complement on the dual DOFs and
/// the map back from a coarse primal solution to this subdomain's interior.
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
    let columns: Matrix = (0..dual.len())
        .map(|column| {
            let rhs: Vector = k_pd.iter().map(|row| row[column]).collect();
            k_pp.solve_lu(&rhs).expect("corner block K_pp is singular")
        })
        .collect();
    let primal_map = columns.transpose();
    let schur = dual
        .iter()
        .enumerate()
        .map(|(row, _)| {
            dual.iter()
                .enumerate()
                .map(|(column, _)| {
                    k_dd[row][column]
                        - primal
                            .iter()
                            .enumerate()
                            .map(|(p, _)| k_dp[row][p] * primal_map[p][column])
                            .sum::<Scalar>()
                })
                .collect()
        })
        .collect();
    let y = k_pp.solve_lu(&f_p).expect("corner block K_pp is singular");
    let reduced_force = dual
        .iter()
        .enumerate()
        .map(|(row, _)| {
            f_d[row]
                - primal
                    .iter()
                    .enumerate()
                    .map(|(p, _)| k_dp[row][p] * y[p])
                    .sum::<Scalar>()
        })
        .collect();
    Condensed {
        schur,
        reduced_force,
        primal_map,
    }
}
