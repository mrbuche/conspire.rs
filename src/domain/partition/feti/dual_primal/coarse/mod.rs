#[cfg(test)]
mod test;

use super::{CornerDofs, DualPrimalSplit, condense::Condensed};
use crate::math::{
    Scalar, Tensor, Vector,
    sparse::{CscLu, CscMatrix, SparseError},
};

/// The assembled corner matrix as unsummed triplets.
///
/// One triplet per entry of each subdomain's local corner Schur complement,
/// so entries landing on the same global position are summed only when the
/// matrix is built.
pub(crate) struct CoarseSystem {
    len: usize,
    pattern: Vec<(usize, usize)>,
    values: Vec<Scalar>,
}

impl CoarseSystem {
    pub(crate) fn assemble(
        condensed: &[Condensed],
        splits: &[DualPrimalSplit],
        corner_dofs: &CornerDofs,
    ) -> (Self, Vector) {
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
            Self {
                len: num_corner_dofs,
                pattern,
                values,
            },
            force,
        )
    }
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

/// The assembled corner problem, factorized once.
///
/// The dual operator solves it on every application, so refactorizing per
/// solve would dominate the dual PCG once there are many corners. The
/// factorization is a sparse LU, so the corner problem need not be
/// symmetric.
///
/// With constraint rows `G u_p = g` it is the saddle point
/// `[S G^T; G 0] [u_p; mu] = [r; g]` instead, factorized with pivoting since
/// its trailing block is zero. `solve` still returns the corner solution
/// alone, for a homogeneous constraint, which is what the dual operator needs.
pub(crate) struct Coarse {
    factor: Option<CscLu>,
    len: usize,
    num_constraints: usize,
}

/// One constraint row over the global corner DOFs, and its right-hand side.
pub(crate) struct CornerConstraint {
    pub(crate) entries: Vec<(usize, Scalar)>,
    pub(crate) value: Scalar,
}

impl TryFrom<CoarseSystem> for Coarse {
    type Error = SparseError;
    fn try_from(system: CoarseSystem) -> Result<Self, Self::Error> {
        Self::try_new(system, &[])
    }
}

impl Coarse {
    pub(crate) fn try_new(
        system: CoarseSystem,
        constraints: &[CornerConstraint],
    ) -> Result<Self, SparseError> {
        let len = system.len;
        let num_constraints = constraints.len();
        let size = len + num_constraints;
        if size == 0 {
            return Ok(Self {
                factor: None,
                len,
                num_constraints,
            });
        }
        let CoarseSystem {
            mut pattern,
            mut values,
            ..
        } = system;
        constraints
            .iter()
            .enumerate()
            .for_each(|(row, constraint)| {
                constraint
                    .entries
                    .iter()
                    .for_each(|&(corner, coefficient)| {
                        pattern.push((len + row, corner));
                        values.push(coefficient);
                        pattern.push((corner, len + row));
                        values.push(coefficient);
                    })
            });
        let mut matrix = CscMatrix::from_pattern(size, size, pattern);
        let mut next = 0;
        matrix.fill(|_, _| {
            let value = values[next];
            next += 1;
            value
        });
        let factor = if num_constraints == 0 {
            let mut factor = matrix.lu_symbolic()?;
            factor.refactor(&matrix)?;
            factor
        } else {
            matrix.lu()?
        };
        Ok(Self {
            factor: Some(factor),
            len,
            num_constraints,
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.len
    }
    pub(crate) fn solve(&self, force: &Vector) -> Vector {
        self.solve_constrained(force, &Vector::zero(self.num_constraints))
            .0
    }
    /// The corner solution and the multipliers of the constraint rows.
    pub(crate) fn solve_constrained(&self, force: &Vector, values: &Vector) -> (Vector, Vector) {
        match &self.factor {
            Some(factor) => {
                let rhs: Vector = force.iter().chain(values.iter()).copied().collect();
                let solution = factor.solve(&rhs);
                (
                    solution.iter().take(self.len).copied().collect(),
                    solution.iter().skip(self.len).copied().collect(),
                )
            }
            None => (Vector::zero(0), Vector::zero(0)),
        }
    }
}
