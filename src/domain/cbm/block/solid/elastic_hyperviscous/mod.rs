use super::super::{Block, node::solid::elastic_hyperviscous::ElasticHyperviscousElement};
use crate::{
    constitutive::{ConstitutiveError, solid::elastic_hyperviscous::ElasticHyperviscous},
    domain::{ElementModelError, NodalCoordinates, NodalVelocities},
    math::Quantity,
    units::Power,
};

pub use crate::domain::solid::elastic_hyperviscous::ElasticHyperviscousElements;

impl<C> ElasticHyperviscousElements<3> for Block<C>
where
    C: ElasticHyperviscous,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, ElementModelError> {
        self.nodes
            .iter()
            .map(|node| {
                node.viscous_dissipation(
                    &self.constitutive_model,
                    nodal_coordinates,
                    nodal_velocities,
                )
            })
            .sum::<Result<_, ConstitutiveError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, ElementModelError> {
        self.nodes
            .iter()
            .map(|node| {
                node.dissipation_potential(
                    &self.constitutive_model,
                    nodal_coordinates,
                    nodal_velocities,
                )
            })
            .sum::<Result<_, ConstitutiveError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
