use super::{super::Node, SolidElement};
use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic_viscoplastic::HyperelasticViscoplastic},
    domain::{NodalCoordinates, block::element::solid::viscoplastic::ViscoplasticStateVariables},
    math::{Differentiable, Quantity, Tensor},
    units::Energy,
};

pub use crate::domain::block::element::solid::hyperelastic_viscoplastic::HyperelasticViscoplasticElement;

impl<C, Y> HyperelasticViscoplasticElement<C, 1, Y> for Node
where
    C: HyperelasticViscoplastic<Y>,
    Y: Differentiable + Tensor,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<1, Y>,
    ) -> Result<Quantity<Energy>, ConstitutiveError> {
        let (deformation_gradient_p, _) = (&state_variables[0]).into();
        Ok(constitutive_model.helmholtz_free_energy_density(
            &self.deformation_gradients(nodal_coordinates),
            deformation_gradient_p,
        )? * self.volume)
    }
}
