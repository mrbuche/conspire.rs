#[cfg(test)]
mod test;

use crate::{
    constitutive::{ConstitutiveError, diffusion::Diffusion, electrical::Electrical},
    math::{IDENTITY_00, Quantity, Tensor},
    mechanics::{CurrentDensity, CurrentTangent, ElectricField},
    units::{Current, ElectricalConductivity, PowerDensity},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Ohmic {
    /// The electrical conductivity $`\sigma`$.
    pub electrical_conductivity: Quantity<ElectricalConductivity>,
}

impl Electrical for Ohmic {}

impl Diffusion for Ohmic {
    type Balance = Current;
    #[doc = include_str!("potential.md")]
    fn potential(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<Quantity<PowerDensity>, ConstitutiveError> {
        Ok(potential_gradient.norm_squared() * self.electrical_conductivity * 0.5)
    }
    #[doc = include_str!("current_density.md")]
    fn flux(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<CurrentDensity, ConstitutiveError> {
        Ok(potential_gradient * -self.electrical_conductivity)
    }
    #[doc = include_str!("current_tangent.md")]
    fn tangent(
        &self,
        _potential_gradient: &ElectricField,
    ) -> Result<CurrentTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * -self.electrical_conductivity)
    }
}
