#[cfg(test)]
mod test;

use super::{CornerSelection, DualPrimalSplit, condense::Condensed};
use crate::math::{SquareMatrix, Vector};

/// Assembles the global corner (coarse) problem by scatter-adding each
/// subdomain's local corner Schur complement and reduced force at the
/// shared global corner DOFs, exactly like standard finite element
/// assembly restricted to the primal DOFs.
pub(crate) fn assemble(
    condensed: &[Condensed],
    splits: &[DualPrimalSplit],
    corners: &CornerSelection,
    dimension: usize,
) -> (SquareMatrix, Vector) {
    let num_corner_dofs = corners.num_corners() * dimension;
    let mut schur = SquareMatrix::zero(num_corner_dofs);
    let mut force = Vector::zero(num_corner_dofs);
    condensed
        .iter()
        .zip(splits.iter())
        .for_each(|(local, split)| {
            let global = split.primal_global();
            global.iter().enumerate().for_each(|(i, &row)| {
                force[row] += local.reduced_force[i];
                global.iter().enumerate().for_each(|(j, &column)| {
                    schur[row][column] += local.schur[i][j];
                });
            });
        });
    (schur, force)
}
