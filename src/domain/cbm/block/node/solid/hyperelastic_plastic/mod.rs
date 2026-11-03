use super::{super::Node, SolidElement};
use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic_plastic::HyperelasticPlastic},
    domain::{NodalCoordinates, block::element::solid::plastic::PlasticStateVariables},
    math::Quantity,
    units::Energy,
};

pub use crate::domain::block::element::solid::hyperelastic_plastic::HyperelasticPlasticElement;

impl<C> HyperelasticPlasticElement<C, 1> for Node
where
    C: HyperelasticPlastic,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariables<1>,
    ) -> Result<Quantity<Energy>, ConstitutiveError> {
        Ok(constitutive_model.helmholtz_free_energy_density(
            &self.deformation_gradients(nodal_coordinates),
            &state_variables[0].0,
        )? * self.volume)
    }
}
