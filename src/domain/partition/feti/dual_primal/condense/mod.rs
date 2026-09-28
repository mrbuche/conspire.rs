#[cfg(test)]
mod test;

use crate::math::{Matrix, Scalar, SquareMatrix, Vector};

const RELATIVE_PIVOT: Scalar = 1e-10;

/// A subdomain's corner (primal) DOFs after its dual DOFs are eliminated.
///
/// `dual_map` maps a corner solution back to this subdomain's dual
/// correction, which is what carries the coarse-grid coupling term into the
/// dual operator.
pub(crate) struct Condensed {
    pub(crate) schur: SquareMatrix,
    pub(crate) reduced_force: Vector,
    pub(crate) dual_map: Matrix,
}

impl Condensed {
    pub(crate) fn try_condense(
        local_stiffness: &SquareMatrix,
        local_force: &Vector,
        primal: &[usize],
        dual: &[usize],
    ) -> Option<Self> {
        let k_pp = extract_square(local_stiffness, primal);
        let k_pd = extract_rectangular(local_stiffness, primal, dual);
        let k_dp = extract_rectangular(local_stiffness, dual, primal);
        let k_dd = extract_square(local_stiffness, dual);
        let f_p = extract_vector(local_force, primal);
        let f_d = extract_vector(local_force, dual);
        let factor = k_dd.factorize_lu().ok()?;
        if factor.near_zero_pivots(RELATIVE_PIVOT) > 0 {
            return None;
        }
        let columns: Matrix = (0..primal.len())
            .map(|column| {
                let rhs: Vector = k_dp.iter().map(|row| row[column]).collect();
                factor.solve(&rhs)
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
        let y = factor.solve(&f_d);
        let reduced_force = &f_p - &(&k_pd * &y);
        Some(Self {
            schur,
            reduced_force,
            dual_map,
        })
    }
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
