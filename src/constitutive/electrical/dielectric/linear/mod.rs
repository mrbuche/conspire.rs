#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        electrical::{Electrical, dielectric::Dielectric},
    },
    math::{IDENTITY_00, Quantity, Tensor},
    mechanics::{ElectricDisplacement, ElectricField, ElectricTangent},
    units::{EnergyDensity, Permittivity},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Linear {
    /// The permittivity $`\varepsilon`$.
    pub permittivity: Quantity<Permittivity>,
}

impl Electrical for Linear {}

impl Dielectric for Linear {
    #[doc = include_str!("potential.md")]
    fn potential(
        &self,
        electric_field: &ElectricField,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        Ok(electric_field.norm_squared() * self.permittivity * 0.5)
    }
    #[doc = include_str!("electric_displacement.md")]
    fn electric_displacement(
        &self,
        electric_field: &ElectricField,
    ) -> Result<ElectricDisplacement, ConstitutiveError> {
        Ok(electric_field * self.permittivity)
    }
    #[doc = include_str!("electric_tangent.md")]
    fn electric_tangent(
        &self,
        _electric_field: &ElectricField,
    ) -> Result<ElectricTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * self.permittivity)
    }
}
