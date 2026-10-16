#[cfg(test)]
mod test;

use crate::{
    constitutive::{ConstitutiveError, diffusion::Diffusion, electrical::Electrical},
    math::{IDENTITY_00, Quantity, Tensor},
    mechanics::{ElectricDisplacement, ElectricField, ElectricTangent},
    units::{ElectricPotential, EnergyDensity, Permittivity},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Linear {
    /// The permittivity $`\varepsilon`$.
    pub permittivity: Quantity<Permittivity>,
}

impl Electrical for Linear {}

impl Diffusion for Linear {
    type Potential = ElectricPotential;
    #[doc = include_str!("potential.md")]
    fn potential(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        Ok(potential_gradient.norm_squared() * self.permittivity * 0.5)
    }
    #[doc = include_str!("electric_displacement.md")]
    fn flux(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<ElectricDisplacement, ConstitutiveError> {
        Ok(potential_gradient * -self.permittivity)
    }
    #[doc = include_str!("electric_tangent.md")]
    fn tangent(
        &self,
        _potential_gradient: &ElectricField,
    ) -> Result<ElectricTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * -self.permittivity)
    }
}
