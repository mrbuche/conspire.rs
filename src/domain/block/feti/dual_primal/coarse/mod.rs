#[cfg(test)]
mod test;

use super::{CornerDofs, DualPrimalSplit, condense::Condensed};
use crate::math::{
    Scalar, Vector,
    sparse::{CscLdl, CscMatrix, SparseError},
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
/// factorization is a sparse LDLᵀ, which needs the symmetric tangent
/// that FETI-DP already requires.
pub(crate) struct Coarse {
    factor: Option<CscLdl>,
    len: usize,
}

impl TryFrom<CoarseSystem> for Coarse {
    type Error = SparseError;
    fn try_from(system: CoarseSystem) -> Result<Self, Self::Error> {
        let len = system.len;
        if len == 0 {
            return Ok(Self { factor: None, len });
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
        let mut factor = matrix.ldl_symbolic()?;
        factor.refactor(&matrix)?;
        Ok(Self {
            factor: Some(factor),
            len,
        })
    }
}

impl Coarse {
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
