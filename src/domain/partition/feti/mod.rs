#![allow(dead_code)]

#[cfg(test)]
mod test;

pub(crate) mod block;
pub(crate) mod dual;
pub(crate) mod dual_primal;
pub(crate) mod interface;
pub(crate) mod parallel;
pub(crate) mod pcg;
pub(crate) mod solid;
pub(crate) mod subdomain;
#[cfg(feature = "fem")]
pub(crate) mod thermal;

pub use block::element::{DecomposableElements, ElementSystems};
#[cfg(feature = "fem")]
pub use block::solve::SolveError;
#[cfg(feature = "fem")]
pub use dual_primal::BoundaryConditions;
#[cfg(feature = "fem")]
pub use pcg::Preconditioner;

#[cfg(feature = "fem")]
use crate::{
    geometry::mesh::Partition,
    math::{
        Scalar, Vector,
        optimize::{Krylov, LinearSolver},
    },
};
#[cfg(feature = "fem")]
use block::solve::solve_local_systems;

pub(crate) const THREADS: usize = 1;

/// FETI-DP solver for the linearized systems of a decomposable block.
///
/// The block is split by a [`Partition`], and each subdomain is solved
/// independently, tied together through the corner DOFs and Lagrange
/// multipliers on the interface. Only hyperelastic models are supported, since
/// the method needs a symmetric tangent.
///
/// Only zero-displacement boundary conditions are supported. At least enough
/// DOFs must be pinned to remove every rigid-body mode of the whole block.
///
/// As the linear solver of a [`NewtonRaphson`](crate::math::optimize::NewtonRaphson),
/// it works with a fixed equality constraint only, whose fixed DOFs are the
/// pinned ones.
#[cfg(feature = "fem")]
#[derive(Clone, Debug)]
pub struct Feti {
    pub partition: Partition,
    pub preconditioner: Preconditioner,
    pub rel_tol: Scalar,
}

#[cfg(feature = "fem")]
impl Default for Feti {
    fn default() -> Self {
        Self {
            partition: Partition::default(),
            preconditioner: Preconditioner::Dirichlet,
            rel_tol: Krylov::default().rel_tol,
        }
    }
}

#[cfg(feature = "fem")]
impl LinearSolver for Feti {
    type Tangent = ElementSystems;
    fn solve(
        &self,
        tangent: ElementSystems,
        retained: &[usize],
        _residual: &Vector,
    ) -> Result<Vector, String> {
        let (stiffnesses, forces) = tangent.subdomains(&self.partition)?;
        let mut fixed = vec![true; 3 * tangent.positions().len()];
        retained.iter().for_each(|&dof| fixed[dof] = false);
        let boundary_conditions = BoundaryConditions::new(
            fixed
                .iter()
                .enumerate()
                .filter(|&(_, &pinned)| pinned)
                .map(|(dof, _)| (dof / 3, dof % 3))
                .collect(),
        );
        let solution = solve_local_systems(
            &self.partition,
            &boundary_conditions,
            stiffnesses,
            forces,
            tangent.positions(),
            self.preconditioner,
            self.rel_tol,
        )
        .map_err(|error| error.to_string())?;
        Ok(retained.iter().map(|&dof| solution[dof]).collect())
    }
}
