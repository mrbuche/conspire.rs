#[cfg(test)]
mod test;

use super::{CornerDofs, DualPrimalSplit, condense::Condensed};
use crate::math::{
    Scalar, Vector,
    sparse::{CscLdl, CscMatrix},
};

/// The assembled corner (coarse) matrix as unsummed triplets, one per entry
/// of each subdomain's local corner Schur complement, so entries that land on
/// the same global position are summed only when the matrix is built.
pub(crate) struct CoarseSystem {
    len: usize,
    pattern: Vec<(usize, usize)>,
    values: Vec<Scalar>,
}

impl CoarseSystem {
    pub(crate) fn len(&self) -> usize {
        self.len
    }
    /// The summed value at `(row, column)`, scanning every triplet.
    #[cfg(test)]
    pub(crate) fn entry(&self, row: usize, column: usize) -> Scalar {
        self.pattern
            .iter()
            .zip(self.values.iter())
            .filter(|&(&position, _)| position == (row, column))
            .map(|(_, &value)| value)
            .sum()
    }
}

/// Assembles the global corner (coarse) problem by scatter-adding each
/// subdomain's local corner Schur complement and reduced force at the
/// shared global corner DOFs, exactly like standard finite element
/// assembly restricted to the primal DOFs. Sized by `corner_dofs.count()` —
/// the count of DOFs that actually survive as free primal unknowns after
/// boundary conditions, not a raw node count — so a boundary condition on a
/// corner node's component can never leave a permanently-zero row/column.
///
/// The matrix stays sparse: corners couple only to the corners of the
/// subdomains they share, so a dense `n x n` matrix would be almost all
/// zeros and, at many subdomains, the largest thing in memory.
pub(crate) fn assemble(
    condensed: &[Condensed],
    splits: &[DualPrimalSplit],
    corner_dofs: &CornerDofs,
) -> (CoarseSystem, Vector) {
    let num_corner_dofs = corner_dofs.count();
    let capacity = splits
        .iter()
        .map(|split| split.primal_global().len().pow(2))
        .sum();
    let mut pattern = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);
    let mut force = Vector::zero(num_corner_dofs);
    condensed
        .iter()
        .zip(splits.iter())
        .for_each(|(local, split)| {
            let global = split.primal_global();
            global.iter().enumerate().for_each(|(i, &row)| {
                force[row] += local.reduced_force[i];
                global.iter().enumerate().for_each(|(j, &column)| {
                    pattern.push((row, column));
                    values.push(local.schur[i][j]);
                });
            });
        });
    (
        CoarseSystem {
            len: num_corner_dofs,
            pattern,
            values,
        },
        force,
    )
}

/// The assembled corner problem, factorized once: the dual operator solves
/// it on every application, so refactorizing per solve would dominate the
/// dual PCG once there are many corners. The factorization is a sparse
/// LDLᵀ, which needs the symmetric tangent FETI-DP already requires.
pub(crate) struct Coarse {
    factor: Option<CscLdl>,
    len: usize,
}

impl Coarse {
    pub(crate) fn new(system: CoarseSystem) -> Self {
        let len = system.len;
        if len == 0 {
            return Self { factor: None, len };
        }
        let CoarseSystem {
            pattern, values, ..
        } = system;
        let mut matrix = CscMatrix::from_pattern(len, len, pattern);
        let mut next = 0;
        matrix.fill(|_, _| {
            let value = values[next];
            next += 1;
            value
        });
        let mut factor = matrix
            .ldl_symbolic()
            .expect("assembled coarse problem is singular");
        factor
            .refactor(&matrix)
            .expect("assembled coarse problem is singular");
        Self {
            factor: Some(factor),
            len,
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.len
    }
    pub(crate) fn solve(&self, force: &Vector) -> Vector {
        match &self.factor {
            Some(factor) => factor.solve(force),
            None => Vector::zero(0),
        }
    }
}
