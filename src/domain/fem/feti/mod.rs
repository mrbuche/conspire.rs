use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    domain::fem::{
        NodalCoordinates,
        block::{Block, element::solid::hyperelastic::HyperelasticFiniteElement},
    },
    domain::partition::feti::block::solve::solve_with,
    math::Vector,
};

pub use crate::domain::partition::feti::{
    Feti,
    block::{
        element::{DecomposableElements, ElementSystems},
        solve::SolveError,
    },
    dual_primal::BoundaryConditions,
    pcg::Preconditioner,
};

impl Feti {
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
        solve_with(
            block,
            nodal_coordinates,
            &self.partition,
            boundary_conditions,
            self.preconditioner,
            self.rel_tol,
        )
    }
}
