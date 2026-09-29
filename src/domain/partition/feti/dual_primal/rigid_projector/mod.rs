#[cfg(test)]
mod test;

use crate::{
    domain::feti::subdomain::Subdomain,
    math::{LuDecomposition, Scalar, SquareMatrix, Tensor, Vector},
};

const RELATIVE_PIVOT: f64 = 1e-10;

/// The rigid-body coarse problem of classical FETI.
///
/// A floating subdomain's local solve only exists for a right-hand side
/// orthogonal to its kernel, so the dual problem is solved on the multipliers
/// with a component along the columns of `G = B R` removed, where `R` holds
/// every subdomain's kernel vectors. `project` applies
/// `P = I - G (G^T G)^-1 G^T`, and the small matrix `G^T G` is factorized once.
///
/// A column of `G` is nonzero only on the multipliers of its own subdomain's
/// interface, so the columns are kept sparse: applying `G` or its transpose
/// costs the interfaces, not every mode against every multiplier.
///
/// `G` loses rank, and so `G^T G` is singular, when the whole block still has
/// a rigid-body mode after the boundary conditions, or when the multipliers
/// cannot tell some mode of a subdomain from a motion of the rest.
pub(crate) struct RigidProjector {
    columns: Vec<Vec<(usize, Scalar)>>,
    factor: Option<LuDecomposition>,
}

impl RigidProjector {
    pub(crate) fn try_new<B>(subdomains: &[Subdomain<B>]) -> Option<Self> {
        Self::from_sparse_columns(
            subdomains
                .iter()
                .flat_map(|subdomain| {
                    subdomain
                        .kernel()
                        .iter()
                        .map(|mode| subdomain.interface().apply_sparse(mode))
                })
                .collect(),
        )
    }
    #[cfg(test)]
    pub(crate) fn from_columns(columns: Vec<Vector>) -> Option<Self> {
        Self::from_sparse_columns(
            columns
                .iter()
                .map(|column| {
                    column
                        .iter()
                        .enumerate()
                        .filter(|&(_, &value)| value != 0.0)
                        .map(|(row, &value)| (row, value))
                        .collect()
                })
                .collect(),
        )
    }
    fn from_sparse_columns(columns: Vec<Vec<(usize, Scalar)>>) -> Option<Self> {
        if columns.is_empty() {
            return Some(Self {
                columns,
                factor: None,
            });
        }
        let height = columns
            .iter()
            .flatten()
            .map(|&(row, _)| row + 1)
            .max()
            .unwrap_or(0);
        let mut rows = vec![Vec::new(); height];
        columns.iter().enumerate().for_each(|(column, entries)| {
            entries
                .iter()
                .for_each(|&(row, value)| rows[row].push((column, value)))
        });
        let mut gram = SquareMatrix::zero(columns.len());
        rows.iter().for_each(|row| {
            row.iter().for_each(|&(a, value_a)| {
                row.iter()
                    .for_each(|&(b, value_b)| gram[a][b] += value_a * value_b)
            })
        });
        let factor = gram.factorize_lu().ok()?;
        if factor.near_zero_pivots(RELATIVE_PIVOT) > 0 {
            return None;
        }
        Some(Self {
            columns,
            factor: Some(factor),
        })
    }
    /// The number of rigid-body modes across all subdomains.
    pub(crate) fn len(&self) -> usize {
        self.columns.len()
    }
    /// `G c`
    fn expand(&self, coefficients: &Vector, len: usize) -> Vector {
        let mut sum = Vector::zero(len);
        self.columns
            .iter()
            .zip(coefficients.iter())
            .for_each(|(column, &coefficient)| {
                column
                    .iter()
                    .for_each(|&(row, value)| sum[row] += coefficient * value)
            });
        sum
    }
    /// `G^T v`
    fn restrict(&self, v: &Vector) -> Vector {
        self.columns
            .iter()
            .map(|column| column.iter().map(|&(row, value)| value * v[row]).sum())
            .collect()
    }
    /// `(G^T G)^-1 G^T v`, the rigid-body amplitudes of `v`.
    pub(crate) fn amplitudes(&self, v: &Vector) -> Vector {
        match &self.factor {
            Some(factor) => factor.solve(&self.restrict(v)),
            None => Vector::zero(0),
        }
    }
    /// `P v`, which has nothing along any column of `G`.
    pub(crate) fn project(&self, v: &Vector) -> Vector {
        let mut projected = v.clone();
        let along = self.expand(&self.amplitudes(v), v.len());
        (0..v.len()).for_each(|row| projected[row] -= along[row]);
        projected
    }
    /// `G (G^T G)^-1 e`, the multipliers that satisfy `G^T lambda = e`.
    pub(crate) fn particular(&self, e: &Vector, len: usize) -> Vector {
        match &self.factor {
            Some(factor) => self.expand(&factor.solve(e), len),
            None => Vector::zero(len),
        }
    }
}

/// `R^T f`, each kernel vector against its own subdomain's force, in the same
/// order as the columns of `G`.
pub(crate) fn rigid_rhs<B>(subdomains: &[Subdomain<B>], local_forces: &[Vector]) -> Vector {
    subdomains
        .iter()
        .zip(local_forces.iter())
        .flat_map(|(subdomain, force)| {
            subdomain
                .kernel()
                .iter()
                .map(|mode| mode.full_contraction(force))
        })
        .collect()
}

/// Adds each subdomain's rigid-body motion `R_s alpha_s` to its local solution,
/// with `alpha` in the same order as the columns of `G`.
pub(crate) fn add_rigid_motion<B>(
    subdomains: &[Subdomain<B>],
    alpha: &Vector,
    solutions: &mut [Vector],
) {
    let mut next = 0;
    subdomains
        .iter()
        .zip(solutions.iter_mut())
        .for_each(|(subdomain, solution)| {
            subdomain.kernel().iter().for_each(|mode| {
                (0..solution.len()).for_each(|dof| solution[dof] += alpha[next] * mode[dof]);
                next += 1;
            })
        });
}
