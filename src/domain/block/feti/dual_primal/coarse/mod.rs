#[cfg(test)]
mod test;

use super::{CornerDofs, DualPrimalSplit, condense::Condensed};
use crate::math::{LuDecomposition, SquareMatrix, Tensor, Vector};

/// Assembles the global corner (coarse) problem by scatter-adding each
/// subdomain's local corner Schur complement and reduced force at the
/// shared global corner DOFs, exactly like standard finite element
/// assembly restricted to the primal DOFs. Sized by `corner_dofs.count()` —
/// the count of DOFs that actually survive as free primal unknowns after
/// boundary conditions, not a raw node count — so a boundary condition on a
/// corner node's component can never leave a permanently-zero row/column.
pub(crate) fn assemble(
    condensed: &[Condensed],
    splits: &[DualPrimalSplit],
    corner_dofs: &CornerDofs,
) -> (SquareMatrix, Vector) {
    let num_corner_dofs = corner_dofs.count();
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

/// The assembled corner problem, factorized once: the dual operator solves
/// it on every application, so refactorizing per solve would dominate the
/// dual PCG once there are many corners.
pub(crate) struct Coarse {
    factor: LuDecomposition,
    len: usize,
}

impl Coarse {
    pub(crate) fn new(schur: &SquareMatrix) -> Self {
        Self {
            factor: schur
                .factorize_lu()
                .expect("assembled coarse problem is singular"),
            len: schur.len(),
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.len
    }
    pub(crate) fn solve(&self, force: &Vector) -> Vector {
        self.factor.solve(force)
    }
}
