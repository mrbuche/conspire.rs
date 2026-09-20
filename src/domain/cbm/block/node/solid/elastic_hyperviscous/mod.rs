use super::{super::Node, SolidElement};
use crate::{
    constitutive::{ConstitutiveError, solid::elastic_hyperviscous::ElasticHyperviscous},
    domain::{NodalCoordinates, NodalVelocities},
    math::Quantity,
    units::Power,
};

pub use crate::domain::block::element::solid::elastic_hyperviscous::ElasticHyperviscousElement;

impl<C> ElasticHyperviscousElement<C> for Node
where
    C: ElasticHyperviscous,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, ConstitutiveError> {
        Ok(constitutive_model.viscous_dissipation(
            &self.deformation_gradients(nodal_coordinates),
            &self.deformation_gradient_rates(nodal_coordinates, nodal_velocities),
        )? * self.volume)
    }
    fn dissipation_potential(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, ConstitutiveError> {
        Ok(constitutive_model.dissipation_potential(
            &self.deformation_gradients(nodal_coordinates),
            &self.deformation_gradient_rates(nodal_coordinates, nodal_velocities),
        )? * self.volume)
    }
}
