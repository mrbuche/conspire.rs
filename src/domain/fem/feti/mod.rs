//! Finite element solves by dual-primal finite element tearing and interconnecting (FETI-DP).

use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    domain::block::feti::{solve_local_systems, solve_with},
    domain::fem::{
        NodalCoordinates,
        block::{Block, element::solid::hyperelastic::HyperelasticFiniteElement},
    },
    geometry::mesh::Partition,
    math::{
        Scalar, Vector,
        optimize::{Krylov, LinearSolver},
    },
};

pub use crate::domain::block::feti::{
    Preconditioner, SolveError, SolveStats,
    dual_primal::BoundaryConditions,
    element_systems::{DecomposableElements, ElementSystems},
};

/// FETI-DP solver for the linearized systems of a hyperelastic block.
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
#[derive(Clone, Debug)]
pub struct Feti {
    pub partition: Partition,
    pub preconditioner: Preconditioner,
    pub rel_tol: Scalar,
}

impl Default for Feti {
    fn default() -> Self {
        Self {
            partition: Partition::default(),
            preconditioner: Preconditioner::Dirichlet,
            rel_tol: Krylov::default().rel_tol,
        }
    }
}

impl Feti {
    /// Solves the block's linearized system at the given nodal coordinates.
    ///
    /// Returns the global vector of nodal values, ordered node by node.
    pub fn solve<C, F, const G: usize, const M: usize, const N: usize, const P: usize>(
        &self,
        block: &Block<C, F, G, M, N, P>,
        nodal_coordinates: &NodalCoordinates<3>,
        boundary_conditions: &BoundaryConditions,
    ) -> Result<Vector, SolveError>
    where
        C: Hyperelastic,
        F: HyperelasticFiniteElement<C, G, M, N, P>,
    {
        self.solve_with_stats(block, nodal_coordinates, boundary_conditions)
            .map(|(solution, _)| solution)
    }
    /// Like [`Feti::solve`], also returning the time spent in each stage.
    #[allow(clippy::type_complexity)]
    pub fn solve_with_stats<C, F, const G: usize, const M: usize, const N: usize, const P: usize>(
        &self,
        block: &Block<C, F, G, M, N, P>,
        nodal_coordinates: &NodalCoordinates<3>,
        boundary_conditions: &BoundaryConditions,
    ) -> Result<(Vector, SolveStats), SolveError>
    where
        C: Hyperelastic,
        F: HyperelasticFiniteElement<C, G, M, N, P>,
    {
        solve_with(
            block,
            nodal_coordinates,
            &self.partition,
            boundary_conditions,
            3,
            self.preconditioner,
            self.rel_tol,
        )
    }
}

impl LinearSolver for Feti {
    type Tangent = ElementSystems;
    fn solve(
        &self,
        tangent: ElementSystems,
        retained: &[usize],
        _residual: &Vector,
    ) -> Result<Vector, String> {
        let number_of_nodes = tangent.number_of_nodes();
        let (stiffnesses, forces) = tangent.subdomains(&self.partition)?;
        let mut fixed = vec![true; 3 * number_of_nodes];
        retained.iter().for_each(|&dof| fixed[dof] = false);
        let boundary_conditions = BoundaryConditions::new(
            fixed
                .iter()
                .enumerate()
                .filter(|&(_, &pinned)| pinned)
                .map(|(dof, _)| (dof / 3, dof % 3))
                .collect(),
        );
        let (solution, _) = solve_local_systems(
            &self.partition,
            &boundary_conditions,
            stiffnesses,
            forces,
            number_of_nodes,
            3,
            self.preconditioner,
            self.rel_tol,
        )
        .map_err(|error| error.to_string())?;
        Ok(retained.iter().map(|&dof| solution[dof]).collect())
    }
}
