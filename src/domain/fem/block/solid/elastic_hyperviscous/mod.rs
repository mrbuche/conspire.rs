use crate::{
    constitutive::solid::elastic_hyperviscous::ElasticHyperviscous,
    fem::{
        ElementModelError, NodalCoordinates, NodalVelocities,
        block::{
            Block, element::FiniteElementError,
            element::solid::elastic_hyperviscous::ElasticHyperviscousFiniteElement,
        },
        solid::{
            elastic_hyperviscous::ElasticHyperviscousElements, viscoelastic::ViscoelasticElements,
        },
    },
    math::Quantity,
    units::Power,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ElasticHyperviscousElements<3> for Block<C, F, G, M, N, P>
where
    C: ElasticHyperviscous,
    F: ElasticHyperviscousFiniteElement<C, G, M, N, P>,
    Self: ViscoelasticElements<3>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.viscous_dissipation(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .sum::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.dissipation_potential(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .sum::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
