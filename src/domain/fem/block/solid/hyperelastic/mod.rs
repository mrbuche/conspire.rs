pub mod internal_variables;

use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    domain::partition::feti::{
        Feti,
        block::{
            element::{DecomposableElements, ElementSystem, ElementSystems, positions},
            solve::{SolveError, solve_with},
        },
        dual_primal::BoundaryConditions,
    },
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{
                FiniteElementError, planar::PlanarHyperelasticFiniteElement,
                solid::hyperelastic::HyperelasticFiniteElement,
            },
        },
        solid::{
            NodalStiffnessesSolidSymmetric, elastic::ElasticElements,
            hyperelastic::HyperelasticElements,
        },
    },
    math::{HessianAccumulate, Quantity, Vector},
    units::Energy,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> HyperelasticElements<3>
    for Block<C, F, G, M, N, P>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, M, N, P>,
    Self: ElasticElements<3>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                )
            })
            .sum::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_symmetric_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_stiffnesses: &mut NodalStiffnessesSolidSymmetric<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                if node_a <= node_b {
                                    nodal_stiffnesses.accumulate(node_a, node_b, nodal_stiffness)
                                }
                            })
                    });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize> DecomposableElements
    for Block<C, F, G, 3, N, P>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, 3, N, P>,
{
    fn element_systems(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementSystems, ElementModelError> {
        let elements = self
            .connectivity()
            .iter()
            .zip(self.elements())
            .map(|(nodes, element)| {
                let coordinates = Self::element_coordinates(nodal_coordinates, nodes);
                let forces = element.nodal_forces(self.constitutive_model(), &coordinates)?;
                let stiffnesses =
                    element.nodal_stiffnesses(self.constitutive_model(), &coordinates)?;
                Ok::<_, FiniteElementError>(ElementSystem::pack(
                    nodes.to_vec(),
                    |a, i| forces[a][i].value(),
                    |a, b, i, j| stiffnesses[a][b][i][j].value(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ElementModelError::upstream(error, self))?;
        Ok(ElementSystems {
            positions: positions(nodal_coordinates),
            elements,
        })
    }
}

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

impl<C, F, const G: usize, const N: usize, const P: usize> HyperelasticElements<2>
    for Block<C, F, G, 2, N, P>
where
    C: Hyperelastic,
    F: PlanarHyperelasticFiniteElement<C, G, N, P>,
    Self: ElasticElements<2>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<2>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                )
            })
            .sum::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_symmetric_into(
        &self,
        nodal_coordinates: &NodalCoordinates<2>,
        nodal_stiffnesses: &mut NodalStiffnessesSolidSymmetric<2>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                if node_a <= node_b {
                                    nodal_stiffnesses.accumulate(node_a, node_b, nodal_stiffness)
                                }
                            })
                    });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
