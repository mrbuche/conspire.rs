use crate::{
    constitutive::solid::elastic_hyperviscous::ElasticHyperviscous,
    domain::{
        ElementModelError,
        solid::{
            elastic_hyperviscous::ElasticHyperviscousElements, viscoelastic::ViscoelasticElements,
        },
    },
    math::Quantity,
    units::Power,
    vem::{
        NodalCoordinates, NodalVelocities,
        block::{
            Block,
            element::{
                VirtualElementError, solid::elastic_hyperviscous::ElasticHyperviscousVirtualElement,
            },
        },
    },
};

impl<C, F> ElasticHyperviscousElements<3> for Block<C, F>
where
    C: ElasticHyperviscous,
    F: ElasticHyperviscousVirtualElement<C>,
    Self: ViscoelasticElements<3>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Quantity<Power>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.viscous_dissipation(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .sum::<Result<_, VirtualElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Quantity<Power>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.dissipation_potential(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .sum::<Result<_, VirtualElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
