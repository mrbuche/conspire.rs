use super::{super::Point, SolidElement};
use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic::Hyperelastic},
    domain::NodalCoordinates,
    math::Quantity,
    units::Energy,
};

pub use crate::domain::block::element::solid::hyperelastic::HyperelasticElement;

impl<C> HyperelasticElement<C> for Point
where
    C: Hyperelastic,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, ConstitutiveError> {
        Ok(constitutive_model
            .helmholtz_free_energy_density(&self.deformation_gradients(nodal_coordinates))?
            * self.weight())
    }
}
